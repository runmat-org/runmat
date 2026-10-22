//! Geometry and window membership remain shared with the real/integer owner.
use super::reduce::{reduce_complex_window, reduce_complex_window_with_fill};
use super::*;

pub(super) fn evaluate<T: Float>(
    name: &'static str,
    operation: MovingOp,
    input: &[ComplexElement<T>],
    input_shape: &[usize],
    parsed: &ParsedMoving,
) -> BuiltinResult<Evaluation<T>> {
    let dimension = parsed.dim.unwrap_or_else(|| default_dimension(input_shape));
    if dimension == 0 {
        return Err(invalid_argument(
            name,
            format!("{name}: dimension must be >= 1"),
        ));
    }
    let input_shape = shape_with_trailing_singletons(input_shape, dimension);
    let shape = output_shape(name, &input_shape, dimension, parsed)?;
    let count = checked_element_count(name, &shape)?;
    let axis = dimension - 1;
    let axis_len = input_shape[axis];
    validate_sample_points_len(name, axis_len, parsed)?;
    let stride_before = checked_product(name, &input_shape[..axis])?;
    let stride_after = checked_product(name, &input_shape[dimension..])?;
    let out_axis = shape[axis];
    let mut output = Output::new(operation, count);
    for after in 0..stride_after {
        for out_pos in 0..out_axis {
            let center = source_center(name, out_pos, parsed)?;
            let sample_positions = sample_window_positions(name, center, axis_len, parsed)?;
            let count_window = if sample_positions.is_none() {
                Some(count_window(name, center, axis_len, parsed)?)
            } else {
                None
            };
            let capacity = sample_positions
                .as_ref()
                .map(Vec::len)
                .or_else(|| count_window.map(|window| window.end - window.start))
                .unwrap_or(0);
            let mut values = Vec::with_capacity(capacity);
            for before in 0..stride_before {
                let out_idx = before + out_pos * stride_before + after * stride_before * out_axis;
                values.clear();
                let mut saw_nan = false;
                if let Some(positions) = sample_positions.as_ref() {
                    for &pos in positions {
                        let index = before + pos * stride_before + after * stride_before * axis_len;
                        let value = input[index];
                        if value.0.is_nan() || value.1.is_nan() {
                            if parsed.nan_mode == NanMode::Include {
                                saw_nan = true;
                                break;
                            }
                            continue;
                        }
                        values.push((value.0, value.1));
                    }
                    output.set(
                        out_idx,
                        reduce_complex_window(operation, &values, saw_nan, parsed.normalization),
                    );
                    continue;
                }
                let window = count_window.expect("count window computed");
                for pos in window.start..window.end {
                    let index = before + pos * stride_before + after * stride_before * axis_len;
                    let value = input[index];
                    if value.0.is_nan() || value.1.is_nan() {
                        if parsed.nan_mode == NanMode::Include {
                            saw_nan = true;
                            break;
                        }
                        continue;
                    }
                    values.push((value.0, value.1));
                }
                let fill =
                    fill_value(operation, parsed, window.fill_count).map(|(value, count)| {
                        (
                            (
                                T::from(value).expect("native float endpoint conversion"),
                                T::zero(),
                            ),
                            count,
                        )
                    });
                output.set(
                    out_idx,
                    reduce_complex_window_with_fill(
                        operation,
                        &values,
                        saw_nan,
                        fill,
                        parsed.nan_mode,
                        parsed.normalization,
                    ),
                );
            }
        }
    }
    Ok(Evaluation {
        values: output,
        shape,
    })
}
