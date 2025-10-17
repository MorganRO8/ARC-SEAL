"""Utility helpers for ARC Python solvers.

This module exposes a curated helper library intended for opt-in use by the
code-mode self-edit pipeline. The helpers focus on common ARC operations such
as connected-component extraction, rectangle and symmetry detection, and grid
transformations. They avoid external dependencies so the functions can run
inside the sandboxed execution environment.
"""

from __future__ import annotations

import inspect
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

Grid = Sequence[Sequence[int]]
Point = Tuple[int, int]
Cells = List[Point]

HELPER_NAMESPACE = "ARC_HELPERS"


def grid_shape(grid: Grid) -> Tuple[int, int]:
    """Return the ``(rows, cols)`` shape for ``grid``."""

    if not grid:
        return 0, 0
    return len(grid), len(grid[0]) if grid and grid[0] else 0


def iter_grid(grid: Grid) -> Iterator[Tuple[int, int, int]]:
    """Yield ``(row, col, value)`` tuples for every cell in ``grid``."""

    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            yield row_index, col_index, value


def _neighbors(row: int, col: int, rows: int, cols: int, connectivity: int) -> Iterator[Point]:
    if connectivity == 8:
        deltas = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        deltas = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    for d_row, d_col in deltas:
        n_row, n_col = row + d_row, col + d_col
        if 0 <= n_row < rows and 0 <= n_col < cols:
            yield n_row, n_col


def bounding_box(cells: Cells) -> Tuple[int, int, int, int]:
    """Return ``(min_row, min_col, max_row, max_col)`` for ``cells``."""

    min_row = min(r for r, _ in cells)
    max_row = max(r for r, _ in cells)
    min_col = min(c for _, c in cells)
    max_col = max(c for _, c in cells)
    return min_row, min_col, max_row, max_col


def to_relative_coordinates(cells: Cells) -> Cells:
    """Translate ``cells`` so the top-left cell is at ``(0, 0)``."""

    if not cells:
        return []
    min_row, min_col, _, _ = bounding_box(cells)
    return [(r - min_row, c - min_col) for r, c in cells]


def translate_cells(cells: Cells, offset: Point) -> Cells:
    """Translate ``cells`` by ``offset`` and return the new coordinates."""

    off_row, off_col = offset
    return [(r + off_row, c + off_col) for r, c in cells]


def get_connected_components(
    grid: Grid,
    *,
    connectivity: int = 4,
    color: Optional[int] = None,
) -> List[Mapping[str, object]]:
    """Return connected components of ``grid`` as metadata dictionaries.

    Each component dictionary contains ``color`` (int), ``cells`` (list of
    ``(row, col)`` tuples), ``size`` (number of cells), and ``bbox`` (bounding
    box coordinates).
    """

    rows, cols = grid_shape(grid)
    if rows == 0 or cols == 0:
        return []

    visited = [[False for _ in range(cols)] for _ in range(rows)]
    components: List[Mapping[str, object]] = []

    for row, col, value in iter_grid(grid):
        if color is not None and value != color:
            continue
        if visited[row][col]:
            continue
        if value is None:
            continue

        queue: deque[Point] = deque()
        queue.append((row, col))
        visited[row][col] = True
        component_cells: Cells = []

        while queue:
            cur_row, cur_col = queue.popleft()
            cur_value = grid[cur_row][cur_col]
            if color is not None and cur_value != color:
                continue
            if cur_value != value:
                continue

            component_cells.append((cur_row, cur_col))
            for n_row, n_col in _neighbors(cur_row, cur_col, rows, cols, connectivity):
                if visited[n_row][n_col]:
                    continue
                if grid[n_row][n_col] != cur_value:
                    continue
                visited[n_row][n_col] = True
                queue.append((n_row, n_col))

        if not component_cells:
            continue

        min_row, min_col, max_row, max_col = bounding_box(component_cells)
        component = {
            "color": value,
            "cells": component_cells,
            "size": len(component_cells),
            "bbox": (min_row, min_col, max_row, max_col),
        }
        components.append(component)

    return components


def get_components_by_color(
    grid: Grid,
    *,
    color: int,
    connectivity: int = 4,
) -> List[Mapping[str, object]]:
    """Return connected components for ``color`` using ``connectivity``."""

    return get_connected_components(grid, connectivity=connectivity, color=color)


def find_filled_rectangles(
    grid: Grid,
    *,
    color: Optional[int] = None,
    connectivity: int = 4,
) -> List[Mapping[str, object]]:
    """Detect filled rectangles of ``color`` (or any color when ``None``)."""

    rectangles: List[Mapping[str, object]] = []
    for component in get_connected_components(grid, connectivity=connectivity, color=color):
        cells = component["cells"]
        min_row, min_col, max_row, max_col = bounding_box(cells)
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        if len(cells) != height * width:
            continue

        rectangle = {
            "color": component["color"],
            "bbox": (min_row, min_col, max_row, max_col),
            "height": height,
            "width": width,
            "cells": cells,
        }
        rectangles.append(rectangle)
    return rectangles


def extract_subgrid(grid: Grid, top_left: Point, size: Tuple[int, int]) -> List[List[int]]:
    """Return a subgrid starting at ``top_left`` with ``size``."""

    rows, cols = size
    start_row, start_col = top_left
    return [
        [grid[start_row + r][start_col + c] for c in range(cols)]
        for r in range(rows)
    ]


def paste_subgrid(
    grid: Grid,
    subgrid: Grid,
    top_left: Point,
    *,
    in_place: bool = False,
) -> List[List[int]]:
    """Paste ``subgrid`` into ``grid`` at ``top_left`` and return the new grid."""

    base = grid if in_place else [list(row) for row in grid]
    start_row, start_col = top_left
    for r, row in enumerate(subgrid):
        for c, value in enumerate(row):
            base[start_row + r][start_col + c] = value
    return base


def rotate_grid(grid: Grid, k: int = 1) -> List[List[int]]:
    """Rotate ``grid`` by ``k`` quarter turns counter-clockwise."""

    if not grid:
        return []
    k = k % 4
    result = [list(row) for row in grid]
    for _ in range(k):
        result = [list(row) for row in zip(*result[::-1])]
    return result


def reflect_grid(grid: Grid, axis: str = "horizontal") -> List[List[int]]:
    """Reflect ``grid`` across the specified ``axis``."""

    if axis not in {"horizontal", "vertical", "main_diag", "anti_diag"}:
        raise ValueError(
            "axis must be one of 'horizontal', 'vertical', 'main_diag', or 'anti_diag'"
        )
    if not grid:
        return []
    if axis == "horizontal":
        return [list(row) for row in grid[::-1]]
    if axis == "vertical":
        return [list(reversed(row)) for row in grid]
    if axis == "main_diag":
        return [list(row) for row in zip(*grid)]
    return [list(row) for row in zip(*[reversed(row) for row in grid])]


def has_horizontal_symmetry(grid: Grid) -> bool:
    """Return ``True`` when ``grid`` is horizontally symmetric."""

    rows, _ = grid_shape(grid)
    for i in range(rows // 2):
        if list(grid[i]) != list(grid[rows - 1 - i]):
            return False
    return True


def has_vertical_symmetry(grid: Grid) -> bool:
    """Return ``True`` when ``grid`` is vertically symmetric."""

    for row in grid:
        if list(row) != list(row)[::-1]:
            return False
    return True


def has_quadrant_symmetry(grid: Grid) -> bool:
    """Return ``True`` when ``grid`` has D4 (quadrant) symmetry."""

    rotated = rotate_grid(grid)
    return has_horizontal_symmetry(grid) and has_vertical_symmetry(grid) and rotated == grid


def detect_symmetries(grid: Grid) -> Mapping[str, bool]:
    """Return a dictionary describing detected symmetries for ``grid``."""

    return {
        "horizontal": has_horizontal_symmetry(grid),
        "vertical": has_vertical_symmetry(grid),
        "d4": has_quadrant_symmetry(grid),
    }


def place_relative_object(
    grid: Grid,
    relative_cells: Cells,
    color: int,
    top_left: Point,
    *,
    in_place: bool = False,
) -> List[List[int]]:
    """Place ``relative_cells`` (origin at 0,0) onto ``grid`` at ``top_left``."""

    base = grid if in_place else [list(row) for row in grid]
    off_row, off_col = top_left
    for rel_row, rel_col in relative_cells:
        base[off_row + rel_row][off_col + rel_col] = color
    return base


def extract_object(grid: Grid, bbox: Tuple[int, int, int, int]) -> List[List[int]]:
    """Extract the rectangular slice defined by ``bbox``."""

    min_row, min_col, max_row, max_col = bbox
    return [
        [grid[r][c] for c in range(min_col, max_col + 1)]
        for r in range(min_row, max_row + 1)
    ]


def components_by_size(grid: Grid, connectivity: int = 4) -> List[Mapping[str, object]]:
    """Return connected components sorted from largest to smallest."""

    components = get_connected_components(grid, connectivity=connectivity)
    return sorted(components, key=lambda comp: comp["size"], reverse=True)


@dataclass(frozen=True)
class HelperLibrary:
    """Container describing the helper namespace for injection."""

    namespace: str
    functions: Mapping[str, object]
    prompt_overview: str
    api_reference: str
    source: str
    module_filename: str


def _format_overview(functions: Mapping[str, object]) -> str:
    lines: List[str] = []
    for name in sorted(functions):
        obj = functions[name]
        if not callable(obj):
            continue
        doc = inspect.getdoc(obj) or ""
        summary = doc.splitlines()[0] if doc else "Utility function."
        lines.append(f"- {name}: {summary}")
    return "\n".join(lines)


def _format_api_reference(functions: Mapping[str, object]) -> str:
    lines: List[str] = []
    for name in sorted(functions):
        obj = functions[name]
        if not callable(obj):
            continue
        try:
            signature = str(inspect.signature(obj))
        except (TypeError, ValueError):
            signature = "(...)"
        doc = inspect.getdoc(obj) or ""
        first_paragraph = doc.split("\n\n")[0] if doc else ""
        formatted = f"def {name}{signature}:"\
            f"\n    {first_paragraph.replace('\n', '\n    ')}"
        lines.append(formatted)
    return "\n\n".join(lines)


def get_helper_library() -> HelperLibrary:
    """Return the helper library metadata for sandbox injection."""

    exports = {
        name: globals()[name]
        for name in __all__
        if not name.startswith("_") and name in globals()
    }
    functions = {k: v for k, v in exports.items() if callable(v)}
    overview = _format_overview(functions)
    api_reference = _format_api_reference(functions)
    source = Path(__file__).read_text(encoding="utf-8")
    module_filename = str(Path(__file__).resolve())
    return HelperLibrary(
        namespace=HELPER_NAMESPACE,
        functions=functions,
        prompt_overview=overview,
        api_reference=dedent(api_reference.strip()),
        source=source,
        module_filename=module_filename,
    )


__all__ = [
    "HELPER_NAMESPACE",
    "HelperLibrary",
    "bounding_box",
    "components_by_size",
    "detect_symmetries",
    "extract_object",
    "extract_subgrid",
    "find_filled_rectangles",
    "get_components_by_color",
    "get_connected_components",
    "get_helper_library",
    "grid_shape",
    "has_horizontal_symmetry",
    "has_quadrant_symmetry",
    "has_vertical_symmetry",
    "iter_grid",
    "place_relative_object",
    "paste_subgrid",
    "reflect_grid",
    "rotate_grid",
    "to_relative_coordinates",
    "translate_cells",
]
