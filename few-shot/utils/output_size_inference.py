"""Heuristics for inferring expected ARC output grid dimensions."""

from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from arclib.arc import Example, Task

GridShape = Tuple[int, int]


@dataclass
class InferenceResult:
    shape: Optional[GridShape]
    method: str
    explanation: str
    evidence: Dict[str, object]


def _grid_shape(grid) -> Optional[GridShape]:
    if grid is None:
        return None
    try:
        array = np.asarray(grid)
    except Exception:
        return None

    if array.ndim < 2:
        return None

    rows, cols = array.shape[:2]
    try:
        return int(rows), int(cols)
    except (TypeError, ValueError):
        return None


def _all_equal(values: Sequence[GridShape]) -> bool:
    return all(value == values[0] for value in values[1:]) if values else False


def _infer_constant_shape(
    train_inputs: Sequence[GridShape],
    train_outputs: Sequence[GridShape],
    test_input: Optional[GridShape],
) -> Optional[InferenceResult]:
    if not train_inputs or not train_outputs:
        return None
    if not _all_equal(train_inputs):
        return None
    if not _all_equal(train_outputs):
        return None

    reference_input = train_inputs[0]
    reference_output = train_outputs[0]

    if reference_input != reference_output:
        return None

    if test_input is not None and test_input != reference_input:
        return None

    explanation = (
        "All training input grids share the same shape and each output grid "
        "matches that shape. The test input grid also has the same dimensions, "
        "so the expected output must be {}×{}.".format(*reference_output)
    )
    evidence = {
        "train_input_shape": reference_input,
        "train_output_shape": reference_output,
        "test_input_shape": test_input,
    }
    return InferenceResult(reference_output, "constant_match", explanation, evidence)


def _infer_arithmetic_shape(
    train_inputs: Sequence[GridShape],
    train_outputs: Sequence[GridShape],
    test_input: Optional[GridShape],
) -> Optional[InferenceResult]:
    if not train_inputs or not train_outputs or test_input is None:
        return None

    operations = (
        ("addition", lambda inp, out: out - inp, lambda dim, delta: dim + delta),
        ("subtraction", lambda inp, out: inp - out, lambda dim, delta: dim - delta),
        (
            "multiplication",
            lambda inp, out: out / inp if inp != 0 else None,
            lambda dim, factor: dim * factor,
        ),
        (
            "division",
            lambda inp, out: inp / out if out != 0 else None,
            lambda dim, factor: dim / factor if factor not in (0, None) else None,
        ),
    )

    for op_name, param_fn, apply_fn in operations:
        params_row: List[float] = []
        params_col: List[float] = []
        valid = True
        for inp, out in zip(train_inputs, train_outputs):
            if inp is None or out is None:
                valid = False
                break
            inp_r, inp_c = inp
            out_r, out_c = out
            row_param = param_fn(inp_r, out_r)
            col_param = param_fn(inp_c, out_c)
            if row_param in (None, np.nan) or col_param in (None, np.nan):
                valid = False
                break
            params_row.append(row_param)
            params_col.append(col_param)

        if not valid:
            continue

        if not params_row or not params_col:
            continue

        if not all(abs(params_row[0] - value) < 1e-9 for value in params_row[1:]):
            continue
        if not all(abs(params_col[0] - value) < 1e-9 for value in params_col[1:]):
            continue

        row_param = params_row[0]
        col_param = params_col[0]

        test_r, test_c = test_input
        inferred_r = apply_fn(test_r, row_param)
        inferred_c = apply_fn(test_c, col_param)

        if inferred_r in (None, np.nan) or inferred_c in (None, np.nan):
            continue

        if op_name in ("multiplication", "division"):
            if abs(inferred_r - round(inferred_r)) > 1e-9:
                continue
            if abs(inferred_c - round(inferred_c)) > 1e-9:
                continue
            inferred_r = round(inferred_r)
            inferred_c = round(inferred_c)
        else:
            inferred_r = int(round(inferred_r))
            inferred_c = int(round(inferred_c))

        if inferred_r <= 0 or inferred_c <= 0:
            continue

        explanation = (
            "Training grids follow {} by {} for rows and {} for columns. Applying this rule "
            "to the test grid ({}, {}) yields an expected output of {}×{}.".format(
                op_name,
                row_param,
                col_param,
                test_r,
                test_c,
                inferred_r,
                inferred_c,
            )
        )
        evidence = {
            "operation": op_name,
            "row_param": row_param,
            "col_param": col_param,
            "test_input_shape": test_input,
        }
        return InferenceResult((int(inferred_r), int(inferred_c)), f"arithmetic_{op_name}", explanation, evidence)

    return None


def _neighbors(r: int, c: int) -> Iterable[Tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _extract_components(array: np.ndarray) -> List[Dict[str, object]]:
    rows, cols = array.shape
    visited = np.zeros((rows, cols), dtype=bool)
    components: List[Dict[str, object]] = []

    for r in range(rows):
        for c in range(cols):
            if visited[r, c]:
                continue
            color = int(array[r, c])
            queue = deque([(r, c)])
            visited[r, c] = True
            coords: List[Tuple[int, int]] = []

            min_r = max_r = r
            min_c = max_c = c

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                min_r = min(min_r, cr)
                max_r = max(max_r, cr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)

                for nr, nc in _neighbors(cr, cc):
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                        if int(array[nr, nc]) == color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

            bbox_rows = max_r - min_r + 1
            bbox_cols = max_c - min_c + 1
            bbox = (bbox_rows, bbox_cols)
            area = len(coords)
            slice_view = array[min_r : max_r + 1, min_c : max_c + 1]
            is_rectangle = bool(np.all(slice_view == color) and area == bbox_rows * bbox_cols)

            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "area": area,
                    "is_rectangle": is_rectangle,
                    "min_row": min_r,
                    "max_row": max_r,
                    "min_col": min_c,
                    "max_col": max_c,
                }
            )

    return components


def _infer_rectangular_object_shape(
    train_examples: Sequence[Example],
    test_input: np.ndarray,
) -> Optional[InferenceResult]:
    rectangle_colors: List[int] = []
    bbox_shapes: List[GridShape] = []

    for example in train_examples:
        output_shape = _grid_shape(example.output)
        input_array = np.asarray(example.input)
        if output_shape is None:
            return None
        components = _extract_components(input_array)
        matches = [comp for comp in components if comp["is_rectangle"] and comp["bbox"] == output_shape]
        if len(matches) != 1:
            return None
        rectangle_colors.append(matches[0]["color"])
        bbox_shapes.append(matches[0]["bbox"])

    color_counts = Counter(rectangle_colors)
    chosen_color: Optional[int] = None
    if len(color_counts) == 1:
        chosen_color = rectangle_colors[0]

    bbox_counts = Counter(bbox_shapes)
    chosen_bbox: Optional[GridShape] = None
    if len(bbox_counts) == 1:
        chosen_bbox = bbox_shapes[0]

    test_components = _extract_components(test_input)
    inferred_bbox: Optional[GridShape] = None
    if chosen_color is not None:
        matching = [comp for comp in test_components if comp["color"] == chosen_color and comp["is_rectangle"]]
        if len(matching) == 1:
            inferred_bbox = matching[0]["bbox"]
    if inferred_bbox is None and chosen_bbox is not None:
        inferred_bbox = chosen_bbox

    if inferred_bbox is None:
        return None

    explanation = (
        "Each training example contains a single solid rectangle whose bounding box matches the "
        "output grid. Detecting the same rectangle in the test input yields an output size of {}×{}."
    ).format(*inferred_bbox)
    evidence = {
        "rectangle_color": chosen_color,
        "train_bboxes": bbox_shapes,
    }
    return InferenceResult(inferred_bbox, "rectangular_object", explanation, evidence)


def _has_horizontal_symmetry(array: np.ndarray) -> bool:
    rows = array.shape[0]
    top = array[: rows // 2]
    bottom = np.flipud(array[rows - (rows // 2) :])
    return top.shape == bottom.shape and np.array_equal(top, bottom)


def _has_vertical_symmetry(array: np.ndarray) -> bool:
    cols = array.shape[1]
    left = array[:, : cols // 2]
    right = np.fliplr(array[:, array.shape[1] - (cols // 2) :])
    return left.shape == right.shape and np.array_equal(left, right)


def _symmetry_candidates(array: np.ndarray) -> Dict[str, GridShape]:
    rows, cols = array.shape
    candidates: Dict[str, GridShape] = {}

    if _has_horizontal_symmetry(array):
        candidates["horizontal_half"] = (rows // 2, cols)
        if rows % 2 == 1:
            candidates["horizontal_half_including_mid"] = (rows // 2 + 1, cols)

    if _has_vertical_symmetry(array):
        candidates["vertical_half"] = (rows, cols // 2)
        if cols % 2 == 1:
            candidates["vertical_half_including_mid"] = (rows, cols // 2 + 1)

    if _has_horizontal_symmetry(array) and _has_vertical_symmetry(array):
        if rows % 2 == 0 and cols % 2 == 0:
            candidates["quarter"] = (rows // 2, cols // 2)

    return candidates


def _infer_symmetry_shape(
    train_examples: Sequence[Example],
    test_input: np.ndarray,
) -> Optional[InferenceResult]:
    intersection: Optional[Dict[str, GridShape]] = None
    for example in train_examples:
        output_shape = _grid_shape(example.output)
        if output_shape is None:
            return None
        candidates = _symmetry_candidates(np.asarray(example.input))
        matching = {key: value for key, value in candidates.items() if value == output_shape}
        if not matching:
            return None
        if intersection is None:
            intersection = matching
        else:
            intersection = {key: value for key, value in intersection.items() if key in matching and matching[key] == value}
        if not intersection:
            return None

    if not intersection:
        return None

    priority = [
        "horizontal_half",
        "horizontal_half_including_mid",
        "vertical_half",
        "vertical_half_including_mid",
        "quarter",
    ]

    for key in priority:
        if intersection and key in intersection:
            inferred_shape = intersection[key]
            break
    else:
        key, inferred_shape = next(iter(intersection.items()))

    rows, cols = test_input.shape
    if key == "horizontal_half":
        candidate = (rows // 2, cols)
    elif key == "horizontal_half_including_mid":
        candidate = (rows // 2 + 1, cols)
    elif key == "vertical_half":
        candidate = (rows, cols // 2)
    elif key == "vertical_half_including_mid":
        candidate = (rows, cols // 2 + 1)
    elif key == "quarter":
        candidate = (rows // 2, cols // 2)
    else:
        candidate = inferred_shape

    if candidate[0] <= 0 or candidate[1] <= 0:
        return None

    explanation = (
        "Training grids exhibit {} symmetry, and their outputs correspond to the implied sub-region. "
        "Applying the same rule to the test grid ({}, {}) yields {}×{}.".format(
            key.replace("_", " "), rows, cols, candidate[0], candidate[1]
        )
    )
    evidence = {"symmetry_rule": key}
    return InferenceResult((int(candidate[0]), int(candidate[1])), f"symmetry_{key}", explanation, evidence)


def _infer_unique_object_bbox_shape(
    train_examples: Sequence[Example],
    test_input: np.ndarray,
) -> Optional[InferenceResult]:
    colors_per_example: List[int] = []

    for example in train_examples:
        output_shape = _grid_shape(example.output)
        if output_shape is None:
            return None
        input_array = np.asarray(example.input)
        components = _extract_components(input_array)
        color_counts = Counter(comp["color"] for comp in components)
        candidates = [
            comp
            for comp in components
            if color_counts[comp["color"]] == 1 and comp["bbox"] == output_shape
        ]
        if len(candidates) != 1:
            return None
        colors_per_example.append(candidates[0]["color"])

    if not colors_per_example:
        return None

    if len(set(colors_per_example)) != 1:
        return None

    target_color = colors_per_example[0]
    test_components = _extract_components(test_input)
    color_counts_test = Counter(comp["color"] for comp in test_components)
    matching = [
        comp
        for comp in test_components
        if comp["color"] == target_color and color_counts_test[comp["color"]] == 1
    ]

    if len(matching) != 1:
        return None

    bbox = matching[0]["bbox"]
    explanation = (
        "A single uniquely colored object in each training grid has a bounding box equal to the output size. "
        "The test grid contains the same unique color, so its bounding box {}×{} defines the target dimensions."
    ).format(*bbox)
    evidence = {"unique_color": target_color}
    return InferenceResult(bbox, "unique_object_bbox", explanation, evidence)


def infer_output_shape(task: Task) -> Tuple[Optional[GridShape], Dict[str, object]]:
    train_input_shapes = [_grid_shape(example.input) for example in task.train_examples]
    train_output_shapes = [_grid_shape(example.output) for example in task.train_examples]
    test_input_shape = _grid_shape(task.test_example.input)
    test_input_array = np.asarray(task.test_example.input)

    heuristics = [
        _infer_constant_shape,
        _infer_arithmetic_shape,
        _infer_rectangular_object_shape,
        _infer_symmetry_shape,
        _infer_unique_object_bbox_shape,
    ]

    for heuristic in heuristics:
        if heuristic in (_infer_rectangular_object_shape, _infer_symmetry_shape, _infer_unique_object_bbox_shape):
            result = heuristic(task.train_examples, test_input_array)
        else:
            result = heuristic(train_input_shapes, train_output_shapes, test_input_shape)
        if result is not None and result.shape is not None:
            return result.shape, {
                "method": result.method,
                "explanation": result.explanation,
                "evidence": result.evidence,
            }

    fallback_shape = None
    fallback_reason = "No heuristic matched; output size must be inferred manually."
    if _all_equal([shape for shape in train_output_shapes if shape is not None]):
        fallback_shape = [shape for shape in train_output_shapes if shape is not None][0]
        fallback_reason = (
            "Outputs share a consistent shape {}×{}, but no automatic rule was confirmed. "
            "Use this as a hint.".format(*fallback_shape)
        )

    return fallback_shape, {
        "method": "none",
        "explanation": fallback_reason,
        "evidence": {},
    }
