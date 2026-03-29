"""Synthetic PIO BP file generator for testing.

Provides ``PioFixtureBuilder`` — a fluent builder for creating PIO-formatted
ADIOS BP files with controllable dimensions, variables, decomposition maps,
and SCORPIO metadata.  Useful for writing targeted unit and integration tests
without shipping large binary fixture files.

Example::

    builder = PioFixtureBuilder(tmp_path / "test.bp")
    builder.add_dimension("time", 3)
    builder.add_dimension("ncol", 100)
    builder.add_variable("PS", dims=("time", "ncol"))
    builder.add_decomposition("512", [np.arange(1, 51), np.arange(51, 101)])
    builder.set_var_decomp("PS", "512")
    builder.write()

Also provides ``SimpleFixtureBuilder`` for non-PIO (generic ADIOS) BP files.
"""

from __future__ import annotations

from typing import Any

import adios2
import numpy as np


class PioFixtureBuilder:
    """Build synthetic PIO-formatted ADIOS BP files for testing.

    All data is written under the ``__pio__/`` namespace, matching the
    layout produced by SCORPIO.
    """

    def __init__(self, path: str | Any):
        self._path = str(path)
        self._dims: dict[str, int] = {}
        self._variables: dict[str, dict[str, Any]] = {}
        self._decomp_maps: dict[str, list[np.ndarray]] = {}
        self._var_decomps: dict[str, str] = {}
        self._global_attrs: dict[str, Any] = {}
        self._var_attrs: dict[str, dict[str, Any]] = {}
        self._scorpio_meta: dict[str, dict[str, Any]] = {}

    def add_dimension(self, name: str, size: int) -> PioFixtureBuilder:
        """Register a named dimension."""
        self._dims[name] = size
        return self

    def add_variable(
        self,
        name: str,
        *,
        dims: tuple[str, ...] | None = None,
        data: np.ndarray | None = None,
        dtype: np.dtype | type = np.float32,
        fill_value: float | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> PioFixtureBuilder:
        """Add a variable.

        If *data* is None, random data is generated with appropriate shape.
        If *dims* is None and data is provided, dimensions are inferred from
        shape (falls back to generated names).
        """
        self._variables[name] = {
            "dims": dims,
            "data": data,
            "dtype": np.dtype(dtype),
            "fill_value": fill_value,
        }
        if attrs:
            self._var_attrs[name] = attrs
        return self

    def add_decomposition(self, ioid: str, rank_maps: list[np.ndarray]) -> PioFixtureBuilder:
        """Add a decomposition map with per-rank 1-based global indices."""
        self._decomp_maps[ioid] = [np.asarray(m, dtype=np.int64) for m in rank_maps]
        return self

    def set_var_decomp(self, var_name: str, ioid: str) -> PioFixtureBuilder:
        """Associate a variable with a decomposition (track attribute)."""
        self._var_decomps[var_name] = ioid
        return self

    def add_scorpio_metadata(self, var_name: str, **kwargs: Any) -> PioFixtureBuilder:
        """Add SCORPIO def/* metadata for a variable.

        Keyword args become ``def/{key}`` attributes, e.g.
        ``add_scorpio_metadata("PS", dims='{ "time", "ncol" }', nctype=5)``.
        """
        self._scorpio_meta[var_name] = kwargs
        return self

    def add_global_attr(self, name: str, value: Any) -> PioFixtureBuilder:
        """Add a global attribute."""
        self._global_attrs[name] = value
        return self

    def _generate_data(self, name: str) -> np.ndarray:
        """Generate random data matching the variable's dims and dtype."""
        info = self._variables[name]
        dims = info["dims"]
        dtype = info["dtype"]

        if dims is None:
            return np.zeros(0, dtype=dtype)

        shape = tuple(self._dims[d] for d in dims)
        total = int(np.prod(shape)) if shape else 1

        rng = np.random.default_rng(hash(name) % (2**31))
        if np.issubdtype(dtype, np.floating):
            flat = rng.standard_normal(total).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            flat = rng.integers(0, 100, size=total, dtype=dtype)
        else:
            flat = np.zeros(total, dtype=dtype)

        # Inject fill values if requested
        if info["fill_value"] is not None and total > 2:
            flat[1] = dtype.type(info["fill_value"])

        return flat

    def write(self) -> str:
        """Write the BP file and return its path."""
        adios_obj = adios2.Adios()
        io = adios_obj.declare_io("pio_fixture_writer")

        # Dimension variables
        for dname, dsize in self._dims.items():
            io.define_variable(
                f"/__pio__/dim/{dname}",
                np.array([dsize], dtype=np.uint64),
                [1],
                [0],
                [1],
            )

        # Decomposition map variables (define once per ioid, write multiple blocks)
        for ioid, rank_maps in self._decomp_maps.items():
            if not rank_maps:
                continue
            first_map = rank_maps[0]
            io.define_variable(
                f"/__pio__/decomp/{ioid}",
                first_map,
                [],
                [],
                first_map.shape,
            )

        # Science variables
        var_data: dict[str, np.ndarray] = {}
        for vname, info in self._variables.items():
            data = info["data"]
            if data is None:
                data = self._generate_data(vname)
            data = np.asarray(data)
            var_data[vname] = data
            io.define_variable(
                f"/__pio__/var/{vname}",
                data,
                [],
                [],
                data.shape,
            )

        # Track attributes (var → decomp association)
        for vname, ioid in self._var_decomps.items():
            io.define_attribute(f"/__pio__/track/{vname}", str(ioid))

        # Variable attributes
        for vname, attrs in self._var_attrs.items():
            for aname, aval in attrs.items():
                io.define_attribute(f"/__pio__/var/{vname}/{aname}", aval)

        # SCORPIO def/* metadata
        for vname, meta in self._scorpio_meta.items():
            for key, val in meta.items():
                io.define_attribute(f"/__pio__/var/{vname}/def/{key}", val)

        # Global attributes
        for aname, aval in self._global_attrs.items():
            io.define_attribute(f"/__pio__/global/{aname}", aval)

        # Write data
        engine = io.open(self._path, adios2.Mode.Write)

        for dname, dsize in self._dims.items():
            var = io.inquire_variable(f"/__pio__/dim/{dname}")
            engine.put(var, np.array([dsize], dtype=np.uint64))

        for ioid, rank_maps in self._decomp_maps.items():
            for rmap in rank_maps:
                var = io.inquire_variable(f"/__pio__/decomp/{ioid}")
                engine.put(var, rmap)

        for vname, data in var_data.items():
            var = io.inquire_variable(f"/__pio__/var/{vname}")
            engine.put(var, data)

        engine.close()
        return self._path


class SimpleFixtureBuilder:
    """Build generic (non-PIO) ADIOS BP files for testing."""

    def __init__(self, path: str | Any):
        self._path = str(path)
        self._variables: dict[str, np.ndarray] = {}
        self._attrs: dict[str, Any] = {}

    def add_variable(self, name: str, data: np.ndarray) -> SimpleFixtureBuilder:
        self._variables[name] = np.asarray(data)
        return self

    def add_attr(self, name: str, value: Any) -> SimpleFixtureBuilder:
        self._attrs[name] = value
        return self

    def write(self) -> str:
        adios_obj = adios2.Adios()
        io = adios_obj.declare_io("simple_writer")

        for name, data in self._variables.items():
            io.define_variable(name, data, data.shape, [0] * data.ndim, data.shape)

        for aname, aval in self._attrs.items():
            io.define_attribute(aname, aval)

        engine = io.open(self._path, adios2.Mode.Write)
        for name, data in self._variables.items():
            var = io.inquire_variable(name)
            engine.put(var, data)
        engine.close()
        return self._path
