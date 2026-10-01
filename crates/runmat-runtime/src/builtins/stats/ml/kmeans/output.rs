use super::{internal, invalid, PreparedData, ReplicateResult};
use crate::BuiltinResult;
use runmat_value::{Tensor, Value};

pub(super) fn outputs_for_count(
    result: ReplicateResult,
    data: &PreparedData,
) -> BuiltinResult<Value> {
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![idx_output(&result, data)?])),
        Some(2) => Ok(Value::OutputList(vec![
            idx_output(&result, data)?,
            Value::Tensor(matrix_output(&result.centers)?),
        ])),
        Some(3) => Ok(Value::OutputList(vec![
            idx_output(&result, data)?,
            Value::Tensor(matrix_output(&result.centers)?),
            Value::Tensor(sumd_output(&result)?),
        ])),
        Some(4) => Ok(Value::OutputList(vec![
            idx_output(&result, data)?,
            Value::Tensor(matrix_output(&result.centers)?),
            Value::Tensor(sumd_output(&result)?),
            Value::Tensor(distance_output(&result, data)?),
        ])),
        Some(_) => Err(invalid("kmeans: too many output arguments; maximum is 4")),
        None => idx_output(&result, data),
    }
}

fn idx_output(result: &ReplicateResult, data: &PreparedData) -> BuiltinResult<Value> {
    let mut values = vec![f64::NAN; data.original_row_count];
    for (original, valid) in data.original_to_valid.iter().enumerate() {
        if let Some(valid_idx) = valid {
            values[original] = result.idx_valid[*valid_idx] as f64 + 1.0;
        }
    }
    Tensor::new(values, vec![data.original_row_count, 1])
        .map(Value::Tensor)
        .map_err(|err| internal(format!("kmeans: {err}")))
}

fn matrix_output(rows: &[Vec<f64>]) -> BuiltinResult<Tensor> {
    let row_count = rows.len();
    let cols = rows.first().map(|row| row.len()).unwrap_or(0);
    let len = row_count
        .checked_mul(cols)
        .ok_or_else(|| internal("kmeans: matrix output size overflow"))?;
    let mut data = vec![0.0; len];
    for row in 0..row_count {
        for col in 0..cols {
            data[col * row_count + row] = rows[row][col];
        }
    }
    Tensor::new(data, vec![row_count, cols]).map_err(|err| internal(format!("kmeans: {err}")))
}

fn sumd_output(result: &ReplicateResult) -> BuiltinResult<Tensor> {
    Tensor::new(result.sumd.clone(), vec![result.sumd.len(), 1])
        .map_err(|err| internal(format!("kmeans: {err}")))
}

fn distance_output(result: &ReplicateResult, data: &PreparedData) -> BuiltinResult<Tensor> {
    let k = result.centers.len();
    let len = data
        .original_row_count
        .checked_mul(k)
        .ok_or_else(|| internal("kmeans: distance output size overflow"))?;
    let mut values = vec![f64::NAN; len];
    for (original, valid) in data.original_to_valid.iter().enumerate() {
        if let Some(valid_idx) = valid {
            for cluster in 0..k {
                values[cluster * data.original_row_count + original] =
                    result.distances[*valid_idx][cluster];
            }
        }
    }
    Tensor::new(values, vec![data.original_row_count, k])
        .map_err(|err| internal(format!("kmeans: {err}")))
}
