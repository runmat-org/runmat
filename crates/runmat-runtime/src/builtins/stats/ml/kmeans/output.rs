use super::{internal, internal_with_source, invalid, PreparedData, ReplicateResult};
use crate::BuiltinResult;
use runmat_value::{Tensor, Value};

pub(super) async fn outputs_for_count_cooperative(
    result: ReplicateResult,
    data: &PreparedData,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Value> {
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![
            idx_output(&result, data, cooperation).await?,
        ])),
        Some(2) => Ok(Value::output_list(vec![
            idx_output(&result, data, cooperation).await?,
            Value::Tensor(matrix_output(&result.centers, cooperation).await?),
        ])),
        Some(3) => Ok(Value::output_list(vec![
            idx_output(&result, data, cooperation).await?,
            Value::Tensor(matrix_output(&result.centers, cooperation).await?),
            Value::Tensor(sumd_output(&result, cooperation).await?),
        ])),
        Some(4) => Ok(Value::output_list(vec![
            idx_output(&result, data, cooperation).await?,
            Value::Tensor(matrix_output(&result.centers, cooperation).await?),
            Value::Tensor(sumd_output(&result, cooperation).await?),
            Value::Tensor(distance_output(&result, data, cooperation).await?),
        ])),
        Some(_) => Err(invalid("kmeans: too many output arguments; maximum is 4")),
        None => idx_output(&result, data, cooperation).await,
    }
}

async fn idx_output(
    result: &ReplicateResult,
    data: &PreparedData,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Value> {
    let mut values = filled_f64(data.original_row_count, f64::NAN, cooperation).await?;
    for (original, valid) in data.original_to_valid.iter().enumerate() {
        if let Some(valid_idx) = valid {
            values[original] = result.idx_valid[*valid_idx] as f64 + 1.0;
        }
        cooperation.completed(1).await?;
    }
    Tensor::new(values, vec![data.original_row_count, 1])
        .map(Value::Tensor)
        .map_err(|err| internal(format!("kmeans: {err}")))
}

async fn matrix_output(
    rows: &[Vec<f64>],
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Tensor> {
    let row_count = rows.len();
    let cols = rows.first().map(|row| row.len()).unwrap_or(0);
    let len = row_count
        .checked_mul(cols)
        .ok_or_else(|| internal("kmeans: matrix output size overflow"))?;
    let mut data = filled_f64(len, 0.0, cooperation).await?;
    for row in 0..row_count {
        for col in 0..cols {
            data[col * row_count + row] = rows[row][col];
            cooperation.completed(1).await?;
        }
    }
    Tensor::new(data, vec![row_count, cols]).map_err(|err| internal(format!("kmeans: {err}")))
}

async fn sumd_output(
    result: &ReplicateResult,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Tensor> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(result.sumd.len())
        .map_err(|source| {
            internal_with_source("kmeans: sumd output exceeds available memory", source)
        })?;
    for &value in &result.sumd {
        values.push(value);
        cooperation.completed(1).await?;
    }
    Tensor::new(values, vec![result.sumd.len(), 1])
        .map_err(|err| internal(format!("kmeans: {err}")))
}

async fn distance_output(
    result: &ReplicateResult,
    data: &PreparedData,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Tensor> {
    let k = result.centers.len();
    let len = data
        .original_row_count
        .checked_mul(k)
        .ok_or_else(|| internal("kmeans: distance output size overflow"))?;
    let mut values = filled_f64(len, f64::NAN, cooperation).await?;
    for (original, valid) in data.original_to_valid.iter().enumerate() {
        if let Some(valid_idx) = valid {
            for cluster in 0..k {
                values[cluster * data.original_row_count + original] =
                    result.distances[*valid_idx][cluster];
                cooperation.completed(1).await?;
            }
        }
        cooperation.completed(1).await?;
    }
    Tensor::new(values, vec![data.original_row_count, k])
        .map_err(|err| internal(format!("kmeans: {err}")))
}

async fn filled_f64(
    len: usize,
    value: f64,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|source| {
        internal_with_source("kmeans: output exceeds available memory", source)
    })?;
    for _ in 0..len {
        values.push(value);
        cooperation.completed(1).await?;
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn output_capacity_failure_preserves_operational_cause_without_retry() {
        let mut cooperation = super::super::cooperation::KmeansCooperation::new(None);
        let error = futures::executor::block_on(filled_f64(usize::MAX, 0.0, &mut cooperation))
            .expect_err("capacity overflow must fail without attempting an allocation");
        assert_eq!(error.identifier(), Some("RunMat:kmeans:Internal"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(error
            .source()
            .expect("original reservation failure")
            .is::<std::collections::TryReserveError>());
    }
}
