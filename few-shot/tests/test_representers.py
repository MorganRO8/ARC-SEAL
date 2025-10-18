import numpy as np

from arclib.representers import (
    CompositeRepresenter,
    DelimitedGridRepresenter,
    DiagonalSliceRepresenter,
    PythonListGridRepresenter,
    RotatedGridRepresenter,
    build_text_grid_representer,
)


def test_rotated_grid_representer_encodes_clockwise_rotation() -> None:
    grid = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int8)
    representer = RotatedGridRepresenter(base=DelimitedGridRepresenter())

    encoded = representer.encode(grid)

    assert encoded == "\n".join(
        [
            "Rotated 90° clockwise view:",
            "0 0 1",
            "0 1 0",
            "1 0 0",
        ]
    )


def test_diagonal_slice_representer_encodes_top_right_to_bottom_left_diagonals() -> None:
    grid = np.array([[1, 0, 0], [0, 1, 0], [2, 0, 1]], dtype=np.int8)
    representer = DiagonalSliceRepresenter(direction="top-right-to-bottom-left")

    encoded = representer.encode(grid)

    assert encoded == "\n".join(
        [
            "Top-right to bottom-left diagonals:",
            "0 1 2",
            "0 0",
            "1",
            "0 0",
            "1",
        ]
    )


def test_diagonal_slice_representer_encodes_top_left_to_bottom_right_diagonals() -> None:
    grid = np.array([[1, 0, 0], [0, 1, 0], [2, 0, 1]], dtype=np.int8)
    representer = DiagonalSliceRepresenter(direction="top-left-to-bottom-right")

    encoded = representer.encode(grid)

    assert encoded == "\n".join(
        [
            "Top-left to bottom-right diagonals:",
            "0",
            "0 0",
            "1 1 1",
            "0 0",
            "2",
        ]
    )


def test_build_text_grid_representer_with_spatial_views() -> None:
    grid = np.array([[1, 2], [3, 4]], dtype=np.int8)

    default_repr = build_text_grid_representer()
    assert isinstance(default_repr, PythonListGridRepresenter)

    composite_repr = build_text_grid_representer(include_spatial_views=True)
    assert isinstance(composite_repr, CompositeRepresenter)
    assert len(composite_repr.representers) == 4
    assert isinstance(composite_repr.representers[0], PythonListGridRepresenter)

    encoded = composite_repr.encode(grid)

    assert "Rotated 90° clockwise view:" in encoded
    assert "Top-right to bottom-left diagonals:" in encoded
    assert "Top-left to bottom-right diagonals:" in encoded
