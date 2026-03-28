"""Generic store for reading ADIOS BP files without PIO namespace.

For BP files that were written directly with ADIOS2 (not through PIO/SCORPIO),
variables are read as-is using their native ADIOS shapes and metadata.
"""

from __future__ import annotations

from typing import Any

import adios2
import numpy as np

from .pio_store import _ADIOS_TYPE_MAP, _parse_attr_value


class AdiosVariableInfo:
    """Metadata about a generic ADIOS variable."""

    __slots__ = ("name", "dims", "shape", "dtype", "attrs")

    def __init__(
        self,
        name: str,
        dims: tuple[str, ...],
        shape: tuple[int, ...],
        dtype: np.dtype,
        attrs: dict[str, Any],
    ):
        self.name = name
        self.dims = dims
        self.shape = shape
        self.dtype = dtype
        self.attrs = attrs


class AdiosStore:
    """Read generic ADIOS BP files (no PIO namespace).

    Variables are read directly using ADIOS2 global array selections.
    """

    def __init__(self, filename: str):
        self._filename = str(filename)
        self._adios = adios2.Adios()
        self._io = self._adios.declare_io("xarray_adios_reader")
        self._engine = self._io.open(self._filename, adios2.Mode.ReadRandomAccess)

        self._all_vars = self._io.available_variables()
        self._all_attrs = self._io.available_attributes()

        self._variable_info: dict[str, AdiosVariableInfo] | None = None

    def get_dimensions(self) -> dict[str, int]:
        """Infer dimensions from variable shapes.

        For generic ADIOS files, there's no explicit dimension metadata.
        Dimensions are generated from variable shapes.
        """
        dims: dict[str, int] = {}
        for vinfo in self.get_variables().values():
            for dname, dsize in zip(vinfo.dims, vinfo.shape):
                if dname not in dims:
                    dims[dname] = dsize
        return dims

    def get_variables(self) -> dict[str, AdiosVariableInfo]:
        """Catalog all variables in the file."""
        if self._variable_info is not None:
            return self._variable_info

        variables: dict[str, AdiosVariableInfo] = {}

        for vname, vinfo_dict in self._all_vars.items():
            var = self._io.inquire_variable(vname)
            if var is None:
                continue

            type_str = var.type()
            dtype_class = _ADIOS_TYPE_MAP.get(type_str, np.float64)
            if dtype_class is str:
                continue  # Skip string variables for now

            # Get shape from ADIOS metadata
            shape_str = vinfo_dict.get("Shape", "")
            if shape_str:
                shape = tuple(int(s) for s in shape_str.split(",") if s.strip())
            else:
                # Local variable or scalar
                single_val = vinfo_dict.get("SingleValue", "")
                if single_val:
                    shape = ()
                else:
                    # Try block info
                    bi = self._engine.blocks_info(vname, 0)
                    if bi:
                        count_str = bi[0].get("Count", "")
                        if count_str:
                            shape = tuple(int(s) for s in count_str.split(",") if s.strip())
                        else:
                            shape = (int(count_str),) if count_str else ()
                    else:
                        shape = ()

            # Generate dimension names
            dims = tuple(f"dim_{i}" for i in range(len(shape)))

            # Read attributes
            attrs = self._read_var_attrs(vname)

            variables[vname] = AdiosVariableInfo(
                name=vname,
                dims=dims,
                shape=shape,
                dtype=np.dtype(dtype_class),
                attrs=attrs,
            )

        self._variable_info = variables
        return variables

    def read_variable(self, name: str, key: tuple | None = None) -> np.ndarray:
        """Read a variable from the file.

        Parameters
        ----------
        name : str
            Variable name.
        key : tuple of slices/ints, optional
            Indexing key for slicing.

        Returns
        -------
        np.ndarray
        """
        info = self.get_variables()[name]
        var = self._io.inquire_variable(name)

        if info.shape:
            data = np.zeros(info.shape, dtype=info.dtype)
            var.set_step_selection((0, 1))
            self._engine.get(var, data)
            self._engine.perform_gets()
        else:
            # Scalar
            data = np.zeros(1, dtype=info.dtype)
            self._engine.get(var, data)
            self._engine.perform_gets()
            data = data.reshape(())

        if key is not None:
            data = data[key]

        return data

    def _read_var_attrs(self, var_name: str) -> dict[str, Any]:
        """Read attributes associated with a variable."""
        attrs: dict[str, Any] = {}
        prefix = f"{var_name}/"
        for aname, ainfo in self._all_attrs.items():
            if aname.startswith(prefix):
                attr_name = aname[len(prefix) :]
                attrs[attr_name] = _parse_attr_value(ainfo)
        return attrs

    def get_global_attrs(self) -> dict[str, Any]:
        """Read global attributes."""
        attrs: dict[str, Any] = {}
        var_names = set(self._all_vars.keys())

        for aname, ainfo in self._all_attrs.items():
            parts = aname.split("/")
            if len(parts) > 1 and parts[0] in var_names:
                continue
            attrs[aname] = _parse_attr_value(ainfo)

        return attrs

    def close(self):
        """Close the ADIOS engine."""
        self._engine.close()
