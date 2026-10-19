use super::{
    admission::{OptionInput, StartInput},
    internal, invalid, Distance, KmeansStart, Options, PreparedData, StartSpec, MAX_WORK_CELLS,
    NAME,
};
use crate::builtins::common::tensor;
use crate::BuiltinResult;
use runmat_value::{Tensor, Value};

/// The shape and physical class are available on a resident handle without
/// touching its provider. Reject malformed observations before conversion or
/// download, and check the structural product before allocating row buffers.
pub(super) fn preflight_observations(value: &Value) -> BuiltinResult<()> {
    let shape = match value {
        Value::Tensor(tensor) => tensor.shape(),
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
        Value::Tensor(tensor) => tensor.shape(),
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

/// Check the effective host Start against dimensions already carried by X
/// and k. No provider payload is needed for these structural facts. Earlier
/// Start entries may be overwritten by a later one, as in parse_options.
pub(super) fn preflight_effective_start(
    x: &Value,
    k: &Value,
    options: &[OptionInput],
) -> BuiltinResult<()> {
    let Some(start) = options.iter().rev().find_map(|option| match option {
        OptionInput::Start(start) => Some(start),
        _ => None,
    }) else {
        return Ok(());
    };
    let StartInput::Prepared(start) = start else {
        return Ok(());
    };
    let expected_cols = match x {
        Value::Tensor(tensor) => observation_feature_count(tensor.shape()),
        Value::LogicalArray(array) => observation_feature_count(&array.shape),
        Value::GpuTensor(handle) => observation_feature_count(handle.shape()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => 1,
        _ => return Ok(()), // preflight_observations owns this diagnostic.
    };
    let expected_rows = if matches!(k, Value::GpuTensor(_)) || super::admission::is_empty_numeric(k)
    {
        None
    } else {
        Some(super::options::parse_positive_usize(k, "k")?)
    };
    let (rows, cols) = match start {
        StartSpec::Numeric(centers) => (centers.len(), centers.first().map(Vec::len)),
        StartSpec::NumericPages(pages) => {
            let Some(first) = pages.first() else {
                return Err(invalid(
                    "kmeans: numeric Start array must have at least one page",
                ));
            };
            (first.len(), first.first().map(Vec::len))
        }
        StartSpec::Named(_) => return Ok(()),
    };
    if !start_matches(rows, cols, expected_rows, expected_cols) {
        return Err(invalid(
            "kmeans: numeric Start must be k-by-p or k-by-p-by-r",
        ));
    }
    Ok(())
}

fn observation_feature_count(shape: &[usize]) -> usize {
    match shape {
        [] | [_] => 1,
        [rows, cols] if *rows == 1 || *cols == 1 => 1,
        [_, cols] => *cols,
        _ => 1, // preflight_observations owns invalid-rank diagnostics.
    }
}

fn start_matches(
    rows: usize,
    cols: Option<usize>,
    expected_rows: Option<usize>,
    expected_cols: usize,
) -> bool {
    expected_rows.is_none_or(|expected| rows == expected) && cols == Some(expected_cols)
}

#[cfg(test)]
pub(super) fn prepare_data(value: Value, distance: Distance) -> BuiltinResult<PreparedData> {
    futures::executor::block_on(prepare_data_cooperative(
        value,
        distance,
        &mut super::cooperation::KmeansCooperation::new(None),
    ))
}

pub(super) async fn prepare_data_cooperative(
    value: Value,
    distance: Distance,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<PreparedData> {
    let tensor = tensor::value_into_tensor_for(NAME, value)
        .map_err(|err| invalid(format!("kmeans: {err}")))?;
    if tensor.shape().len() > 2 {
        return Err(invalid("kmeans: X must be a numeric vector or 2-D matrix"));
    }
    let (raw_rows, raw_cols) = tensor_rows_cols(&tensor);
    let vector_as_column = raw_rows == 1 || raw_cols == 1;
    let original_row_count = if vector_as_column {
        tensor::tensor_element_len(&tensor)
    } else {
        raw_rows
    };
    let cols = if vector_as_column { 1 } else { raw_cols };
    let mut rows = Vec::new();
    rows.try_reserve_exact(original_row_count)
        .map_err(|source| {
            super::internal_with_source("kmeans: observations exceed available memory", source)
        })?;
    let mut original_to_valid = Vec::new();
    original_to_valid
        .try_reserve_exact(original_row_count)
        .map_err(|source| {
            super::internal_with_source("kmeans: row indices exceed available memory", source)
        })?;
    for row in 0..original_row_count {
        let mut row_values = Vec::new();
        row_values.try_reserve_exact(cols).map_err(|source| {
            super::internal_with_source("kmeans: observations exceed available memory", source)
        })?;
        let mut has_nan = false;
        for col in 0..cols {
            let value: f64 = if vector_as_column {
                tensor::tensor_value_f64(&tensor, row)
            } else {
                tensor::tensor_value_f64(&tensor, col * raw_rows + row)
            };
            if value.is_nan() {
                has_nan = true;
            } else if !value.is_finite() {
                return Err(invalid("kmeans: X cannot contain Inf values"));
            }
            row_values.push(value);
            cooperation.completed(1).await?;
        }
        cooperation.completed(1).await?;
        if has_nan {
            original_to_valid.push(None);
        } else {
            let transformed = transform_row(&row_values, distance, cooperation).await?;
            let mut degenerate = false;
            for value in &transformed {
                degenerate |= value.is_nan();
                cooperation.completed(1).await?;
            }
            if degenerate {
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
    match tensor.shape() {
        [] => (1, 1),
        [n] => (*n, 1),
        [rows, cols, ..] => (*rows, *cols),
    }
}

async fn transform_row(
    row: &[f64],
    distance: Distance,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Vec<f64>> {
    match distance {
        Distance::Cosine | Distance::Correlation => {
            direction_or_nan(row, distance == Distance::Correlation, cooperation).await
        }
        Distance::Hamming => {
            for value in row {
                let invalid_value = *value != 0.0 && *value != 1.0;
                cooperation.completed(1).await?;
                if invalid_value {
                    return Err(invalid("kmeans: Hamming distance requires binary X values"));
                }
            }
            copy_row(row, cooperation).await
        }
        Distance::SqEuclidean | Distance::Cityblock => copy_row(row, cooperation).await,
    }
}

pub(super) async fn direction_or_nan(
    row: &[f64],
    center: bool,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Vec<f64>> {
    use super::super::pairwise_metric::{DirectionError, DirectionProgress, DirectionWork};

    let mut work = DirectionWork::new(row, center);
    loop {
        match work.advance().map_err(|error| match error {
            DirectionError::Resource(source) => {
                super::internal_with_source("kmeans: direction exceeds available memory", source)
            }
        })? {
            DirectionProgress::Pending(units) => cooperation.completed(units).await?,
            DirectionProgress::Complete(direction, units) => {
                cooperation.completed(units).await?;
                if let Some(direction) = direction {
                    return Ok(direction);
                }
                let mut nan_row = Vec::new();
                nan_row.try_reserve_exact(row.len()).map_err(|source| {
                    super::internal_with_source(
                        "kmeans: direction exceeds available memory",
                        source,
                    )
                })?;
                for _ in row {
                    nan_row.push(f64::NAN);
                    cooperation.completed(1).await?;
                }
                return Ok(nan_row);
            }
        }
    }
}

async fn copy_row(
    row: &[f64],
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(row.len()).map_err(|source| {
        super::internal_with_source("kmeans: observations exceed available memory", source)
    })?;
    for value in row {
        copy.push(*value);
        cooperation.completed(1).await?;
    }
    Ok(copy)
}

pub(super) async fn validate_options(
    options: &mut Options,
    k: usize,
    data: &PreparedData,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<()> {
    match &mut options.start {
        StartSpec::Numeric(centers) => {
            validate_centers(centers, k, data.cols, cooperation).await?;
            transform_centers(centers, options.distance, cooperation).await?;
            options.replicates = 1;
        }
        StartSpec::NumericPages(pages) => {
            if pages.is_empty() {
                return Err(invalid(
                    "kmeans: numeric Start array must have at least one page",
                ));
            }
            for centers in &mut *pages {
                validate_centers(centers, k, data.cols, cooperation).await?;
                transform_centers(centers, options.distance, cooperation).await?;
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

async fn transform_centers(
    centers: &mut [Vec<f64>],
    distance: Distance,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<()> {
    for center in centers {
        let transformed = transform_row(center, distance, cooperation).await?;
        let mut degenerate = false;
        for value in &transformed {
            degenerate |= value.is_nan();
            cooperation.completed(1).await?;
        }
        if degenerate {
            return Err(invalid(
                "kmeans: numeric Start contains a degenerate centroid for the selected Distance",
            ));
        }
        *center = transformed;
    }
    Ok(())
}

async fn validate_centers(
    centers: &[Vec<f64>],
    k: usize,
    cols: usize,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<()> {
    if !start_matches(centers.len(), centers.first().map(Vec::len), Some(k), cols) {
        return Err(invalid(
            "kmeans: numeric Start must be k-by-p or k-by-p-by-r",
        ));
    }
    for row in centers {
        if row.len() != cols {
            return Err(invalid(
                "kmeans: numeric Start must be k-by-p or k-by-p-by-r",
            ));
        }
        cooperation.completed(1).await?;
    }
    Ok(())
}
