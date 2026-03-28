"""Test fixtures for xarray-adios."""

import pytest

# Skip all tests requiring adios2 if it's not installed
adios2 = pytest.importorskip("adios2")


@pytest.fixture
def tmp_bp_dir(tmp_path):
    """Provide a temporary directory for BP file creation."""
    return tmp_path


def write_simple_bp(path, variables, dimensions=None, attrs=None):
    """Write a simple generic ADIOS BP file for testing.

    Parameters
    ----------
    path : str
        Output path for the BP file.
    variables : dict
        Mapping of name → numpy array.
    dimensions : dict, optional
        Mapping of dim name → size (unused for generic files).
    attrs : dict, optional
        Global attributes.
    """
    import numpy as np

    adios_obj = adios2.Adios()
    io = adios_obj.declare_io("writer")

    for name, data in variables.items():
        data = np.asarray(data)
        var = io.define_variable(name, data, data.shape, [0] * data.ndim, data.shape)

    if attrs:
        for aname, aval in attrs.items():
            io.define_attribute(aname, aval)

    engine = io.open(str(path), adios2.Mode.Write)
    for name, data in variables.items():
        data = np.asarray(data)
        var = io.inquire_variable(name)
        engine.put(var, data)
    engine.close()


def write_pio_bp(path, variables, dimensions, var_attrs=None, global_attrs=None):
    """Write a PIO-formatted ADIOS BP file for testing.

    Mimics the ``__pio__`` namespace structure written by SCORPIO.

    Parameters
    ----------
    path : str
        Output path for the BP file.
    variables : dict
        Mapping of short name → numpy array (flattened, as PIO would write).
    dimensions : dict
        Mapping of dim name → size.
    var_attrs : dict of dict, optional
        Mapping of var name → {attr_name: attr_value}.
    global_attrs : dict, optional
        Global attributes.
    """
    import numpy as np

    adios_obj = adios2.Adios()
    io = adios_obj.declare_io("pio_writer")

    # Define dimension variables
    for dname, dsize in dimensions.items():
        io.define_variable(
            f"/__pio__/dim/{dname}",
            np.array([dsize], dtype=np.uint64),
            [1],
            [0],
            [1],
        )

    # Define science variables as local arrays (simulating block writes)
    for vname, data in variables.items():
        data = np.asarray(data)
        # Write as a single block (local variable) to simulate PIO output
        io.define_variable(
            f"/__pio__/var/{vname}",
            data,
            [],
            [],  # no global shape (local variable)
            data.shape,
        )

    # Define variable attributes
    if var_attrs:
        for vname, attrs in var_attrs.items():
            for aname, aval in attrs.items():
                io.define_attribute(
                    f"/__pio__/var/{vname}/{aname}",
                    aval,
                )

    # Define global attributes
    if global_attrs:
        for aname, aval in global_attrs.items():
            io.define_attribute(aname, aval)

    engine = io.open(str(path), adios2.Mode.Write)

    for dname, dsize in dimensions.items():
        var = io.inquire_variable(f"/__pio__/dim/{dname}")
        engine.put(var, np.array([dsize], dtype=np.uint64))

    for vname, data in variables.items():
        data = np.asarray(data)
        var = io.inquire_variable(f"/__pio__/var/{vname}")
        engine.put(var, data)

    engine.close()


def write_pio_bp_decomp(
    path,
    variables,
    dimensions,
    decomp_maps,
    var_decomps=None,
    var_attrs=None,
    global_attrs=None,
):
    """Write a PIO-formatted BP file **with decomposition maps** for testing.

    This writes single-block (single-rank) data with decomposition maps and
    track attributes that associate variables with their decompositions.

    Parameters
    ----------
    path : str
        Output path for the BP file.
    variables : dict
        Mapping of short name → numpy array (flattened data as one rank would
        write it — element order matches the decomp map).
    dimensions : dict
        Mapping of dim name → size.
    decomp_maps : dict
        Mapping of ioid string → 1-D numpy int64 array of 1-based global
        indices (the decomposition map for the single simulated rank).
    var_decomps : dict, optional
        Mapping of variable short name → ioid string.  When provided, each
        association is stored as a ``/__pio__/track/{varname}`` attribute so
        the reader can discover it via the attribute-based strategy.  When
        *None*, the reader will fall back to the block-matching heuristic.
    var_attrs : dict of dict, optional
        Mapping of var name → {attr_name: attr_value}.
    global_attrs : dict, optional
        Global attributes.
    """
    import numpy as np

    adios_obj = adios2.Adios()
    io = adios_obj.declare_io("pio_decomp_writer")

    # -- dimension variables --
    for dname, dsize in dimensions.items():
        io.define_variable(
            f"/__pio__/dim/{dname}",
            np.array([dsize], dtype=np.uint64),
            [1],
            [0],
            [1],
        )

    # -- decomposition map variables (local arrays) --
    for ioid, dmap in decomp_maps.items():
        dmap = np.asarray(dmap, dtype=np.int64)
        io.define_variable(
            f"/__pio__/decomp/{ioid}",
            dmap,
            [],
            [],  # local variable
            dmap.shape,
        )

    # -- science variables (local arrays) --
    for vname, data in variables.items():
        data = np.asarray(data)
        io.define_variable(
            f"/__pio__/var/{vname}",
            data,
            [],
            [],  # local variable
            data.shape,
        )

    # -- track attributes (variable → decomp association) --
    if var_decomps:
        for vname, ioid in var_decomps.items():
            io.define_attribute(f"/__pio__/track/{vname}", str(ioid))

    # -- variable attributes --
    if var_attrs:
        for vname, attrs in var_attrs.items():
            for aname, aval in attrs.items():
                io.define_attribute(f"/__pio__/var/{vname}/{aname}", aval)

    # -- global attributes --
    if global_attrs:
        for aname, aval in global_attrs.items():
            io.define_attribute(aname, aval)

    # -- write everything --
    engine = io.open(str(path), adios2.Mode.Write)

    for dname, dsize in dimensions.items():
        var = io.inquire_variable(f"/__pio__/dim/{dname}")
        engine.put(var, np.array([dsize], dtype=np.uint64))

    for ioid, dmap in decomp_maps.items():
        dmap = np.asarray(dmap, dtype=np.int64)
        var = io.inquire_variable(f"/__pio__/decomp/{ioid}")
        engine.put(var, dmap)

    for vname, data in variables.items():
        data = np.asarray(data)
        var = io.inquire_variable(f"/__pio__/var/{vname}")
        engine.put(var, data)

    engine.close()
