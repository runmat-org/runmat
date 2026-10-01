use super::{
    internal, invalid, Distance, KmeansStart, Options, PreparedData, StartSpec, EPS,
    MAX_WORK_CELLS, NAME,
};
use crate::builtins::common::tensor;
use crate::BuiltinResult;
use runmat_value::{Tensor, Value};

/// The shape and physical class are available on a resident handle without
/// touching its provider. Reject malformed observations before conversion or
/// download, and check the structural product before allocating row buffers.
pub(super) fn preflight_observations(value: &Value) -> BuiltinResult<()> {
    let shape = match value {
        Value::Tensor(tensor) => tensor.shape.as_slice(),
        Value::LogicalArray(array) => array.shape.as_slice(),
        Value::GpuTensor(handle) => {
            if handle.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real {
                return Err(invalid(
                    "kmeans: X must have real numeric or logical storage",
                ));
            }
            handle.shape()
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => return Ok(()),
        _ => {
            return Err(invalid(
                "kmeans: X must be a real numeric vector or 2-D matrix",
            ))
        }
    };
    let (rows, columns) = match shape {
        [] => (1, 1),
        [length] => (*length, 1),
        [rows, columns] => (*rows, *columns),
        _ => return Err(invalid("kmeans: X must be a numeric vector or 2-D matrix")),
    };
    rows.checked_mul(columns)
        .ok_or_else(|| invalid("kmeans: observation shape overflow"))?;
    Ok(())
}

pub(super) fn preflight_cluster_count(value: &Value) -> BuiltinResult<()> {
    let shape = match value {
        Value::Tensor(tensor) => tensor.shape.as_slice(),
        Value::LogicalArray(array) => array.shape.as_slice(),
        Value::GpuTensor(handle) => {
            if handle.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real {
                return Err(invalid("kmeans: k must have real numeric storage"));
            }
            handle.shape()
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => return Ok(()),
        _ => return Err(invalid("kmeans: k must be a positive integer scalar or []")),
    };
    let count = shape
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .ok_or_else(|| invalid("kmeans: k shape overflow"))?;
    if count > 1 {
        return Err(invalid("kmeans: k must be a positive integer scalar or []"));
    }
    Ok(())
}

pub(super) fn prepare_data(value: Value, distance: Distance) -> BuiltinResult<PreparedData> {
    let tensor = tensor::value_into_tensor_for(NAME, value)
        .map_err(|err| invalid(format!("kmeans: {err}")))?;
    if tensor.shape.len() > 2 {
        return Err(invalid("kmeans: X must be a numeric vector or 2-D matrix"));
    }
    let values = tensor::tensor_values_f64_cow(&tensor);
    let (raw_rows, raw_cols) = tensor_rows_cols(&tensor);
    let vector_as_column = raw_rows == 1 || raw_cols == 1;
    let original_row_count = if vector_as_column {
        values.len()
    } else {
        raw_rows
    };
    let cols = if vector_as_column { 1 } else { raw_cols };
    let mut rows = Vec::new();
    let mut original_to_valid = Vec::with_capacity(original_row_count);
    for row in 0..original_row_count {
        let mut row_values = Vec::with_capacity(cols);
        let mut has_nan = false;
        for col in 0..cols {
            let value: f64 = if vector_as_column {
                values[row]
            } else {
                values[col * raw_rows + row]
            };
            if value.is_nan() {
                has_nan = true;
            } else if !value.is_finite() {
                return Err(invalid("kmeans: X cannot contain Inf values"));
            }
            row_values.push(value);
        }
        if has_nan {
            original_to_valid.push(None);
        } else {
            let transformed = transform_row(&row_values, distance)?;
            if transformed.iter().any(|value| value.is_nan()) {
                original_to_valid.push(None);
            } else {
                original_to_valid.push(Some(rows.len()));
                rows.push(transformed);
            }
        }
    }
    if rows.is_empty() {
        return Err(invalid(
            "kmeans: X must contain at least one complete observation",
        ));
    }
    Ok(PreparedData {
        rows,
        original_row_count,
        original_to_valid,
        cols,
    })
}

pub(super) fn tensor_rows_cols(tensor: &Tensor) -> (usize, usize) {
    match tensor.shape.as_slice() {
        [] => (1, 1),
        [n] => (*n, 1),
        [rows, cols, ..] => (*rows, *cols),
    }
}

fn transform_row(row: &[f64], distance: Distance) -> BuiltinResult<Vec<f64>> {
    match distance {
        Distance::Cosine => {
            let norm = row.iter().map(|value| value * value).sum::<f64>().sqrt();
            if norm <= EPS {
                Ok(vec![f64::NAN; row.len()])
            } else {
                Ok(row.iter().map(|value| value / norm).collect())
            }
        }
        Distance::Correlation => {
            if row.is_empty() {
                return Ok(Vec::new());
            }
            let mean = row.iter().sum::<f64>() / row.len() as f64;
            let centered = row.iter().map(|value| value - mean).collect::<Vec<_>>();
            let norm = centered
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            if norm <= EPS {
                Ok(vec![f64::NAN; row.len()])
            } else {
                Ok(centered.into_iter().map(|value| value / norm).collect())
            }
        }
        Distance::Hamming => {
            if row.iter().any(|value| *value != 0.0 && *value != 1.0) {
                return Err(invalid("kmeans: Hamming distance requires binary X values"));
            }
            Ok(row.to_vec())
        }
        Distance::SqEuclidean | Distance::Cityblock => Ok(row.to_vec()),
    }
}

pub(super) fn validate_options(
    options: &mut Options,
    k: usize,
    data: &PreparedData,
) -> BuiltinResult<()> {
    match &mut options.start {
        StartSpec::Numeric(centers) => {
            validate_centers(centers, k, data.cols)?;
            transform_centers(centers, options.distance)?;
            options.replicates = 1;
        }
        StartSpec::NumericPages(pages) => {
            if pages.is_empty() {
                return Err(invalid(
                    "kmeans: numeric Start array must have at least one page",
                ));
            }
            for centers in &mut *pages {
                validate_centers(centers, k, data.cols)?;
                transform_centers(centers, options.distance)?;
            }
            options.replicates = pages.len();
        }
        StartSpec::Named(KmeansStart::Uniform) if options.distance == Distance::Hamming => {
            return Err(invalid(
                "kmeans: Start 'uniform' is not valid with Hamming distance",
            ));
        }
        _ => {}
    }
    let center_work = k
        .checked_mul(data.cols)
        .and_then(|value| value.checked_mul(options.replicates))
        .ok_or_else(|| internal("kmeans: work size overflow"))?;
    let assignment_work = data
        .rows
        .len()
        .checked_mul(k)
        .ok_or_else(|| internal("kmeans: assignment work size overflow"))?;
    let d_work = data
        .original_row_count
        .checked_mul(k)
        .ok_or_else(|| internal("kmeans: distance output size overflow"))?;
    if center_work > MAX_WORK_CELLS || assignment_work > MAX_WORK_CELLS || d_work > MAX_WORK_CELLS {
        return Err(invalid("kmeans: requested clustering work is too large"));
    }
    Ok(())
}

fn transform_centers(centers: &mut [Vec<f64>], distance: Distance) -> BuiltinResult<()> {
    for center in centers {
        let transformed = transform_row(center, distance)?;
        if transformed.iter().any(|value| value.is_nan()) {
            return Err(invalid(
                "kmeans: numeric Start contains a degenerate centroid for the selected Distance",
            ));
        }
        *center = transformed;
    }
    Ok(())
}

fn validate_centers(centers: &[Vec<f64>], k: usize, cols: usize) -> BuiltinResult<()> {
    if centers.len() != k || centers.iter().any(|row| row.len() != cols) {
        return Err(invalid(
            "kmeans: numeric Start must be k-by-p or k-by-p-by-r",
        ));
    }
    if centers
        .iter()
        .flat_map(|row| row.iter())
        .any(|value| !value.is_finite())
    {
        return Err(invalid("kmeans: numeric Start values must be finite"));
    }
    Ok(())
}
