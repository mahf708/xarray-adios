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
