//! Shared polar coordinate helpers for plotting builtins.

use runmat_builtins::{ComplexTensor, Tensor, Value};

use crate::builtins::plotting::common::{gather_tensor_from_gpu_async, numeric_pair};
use crate::builtins::plotting::plotting_error;
use crate::{BuiltinResult, RuntimeError};

#[derive(Clone, Debug)]
pub(crate) struct EvaluatedPolarData {
    pub theta: Vec<f64>,
    pub rho: Vec<f64>,
}

pub(crate) async fn real_tensor_from_value(
    value: Value,
    builtin: &'static str,
) -> BuiltinResult<Tensor> {
    match value {
        Value::GpuTensor(handle) => gather_tensor_from_gpu_async(handle, builtin).await,
        Value::Num(value) => Ok(scalar_tensor(value)),
        Value::Int(value) => Ok(scalar_tensor(value.to_f64())),
        Value::Bool(value) => Ok(scalar_tensor(if value { 1.0 } else { 0.0 })),
        other => Tensor::try_from(&other)
            .map_err(|err| polar_data_err(builtin, format!("expected real numeric data: {err}"))),
    }
}

pub(crate) fn evaluate_theta_rho_tensors(
    theta: Tensor,
    rho: Tensor,
    builtin: &'static str,
) -> BuiltinResult<Vec<EvaluatedPolarData>> {
    if tensor_is_vector(&theta) && tensor_is_vector(&rho) {
        let (theta, rho) = numeric_pair(theta, rho, builtin)?;
        return Ok(vec![EvaluatedPolarData { theta, rho }]);
    }

    if tensor_is_vector(&theta) {
        let theta_vec = theta.data;
        let rho_rows = tensor_rows(&rho);
        if theta_vec.len() != rho_rows {
            return Err(polar_data_err(
                builtin,
                "theta vector length must match each rho matrix column",
            ));
        }
        return Ok(tensor_columns(&rho)
            .into_iter()
            .map(|rho| EvaluatedPolarData {
                theta: theta_vec.clone(),
                rho,
            })
            .collect());
    }

    if tensor_is_vector(&rho) {
        let rho_vec = rho.data;
        let theta_rows = tensor_rows(&theta);
        if rho_vec.len() != theta_rows {
            return Err(polar_data_err(
                builtin,
                "rho vector length must match each theta matrix column",
            ));
        }
        return Ok(tensor_columns(&theta)
            .into_iter()
            .map(|theta| EvaluatedPolarData {
                theta,
                rho: rho_vec.clone(),
            })
            .collect());
    }

    if tensor_rows(&theta) != tensor_rows(&rho) || tensor_cols(&theta) != tensor_cols(&rho) {
        return Err(polar_data_err(
            builtin,
            "theta and rho matrices must have matching row and column counts",
        ));
    }

    Ok(tensor_columns(&theta)
        .into_iter()
        .zip(tensor_columns(&rho))
        .map(|(theta, rho)| EvaluatedPolarData { theta, rho })
        .collect())
}

pub(crate) fn polar_to_cartesian(
    theta: &[f64],
    rho: &[f64],
    builtin: &'static str,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    if theta.len() != rho.len() {
        return Err(polar_data_err(
            builtin,
            "theta and rho inputs must have the same number of elements",
        ));
    }
    let x = theta
        .iter()
        .zip(rho.iter())
        .map(|(theta, rho)| rho * theta.cos())
        .collect();
    let y = theta
        .iter()
        .zip(rho.iter())
        .map(|(theta, rho)| rho * theta.sin())
        .collect();
    Ok((x, y))
}

pub(crate) fn implicit_theta(len: usize) -> Vec<f64> {
    match len {
        0 => Vec::new(),
        1 => vec![0.0],
        n => (0..n)
            .map(|idx| idx as f64 * std::f64::consts::TAU / (n - 1) as f64)
            .collect(),
    }
}

pub(crate) fn tensor_columns(tensor: &Tensor) -> Vec<Vec<f64>> {
    if tensor_is_vector(tensor) {
        return vec![tensor.data.clone()];
    }
    (0..tensor.cols)
        .map(|col| {
            let start = col * tensor.rows;
            tensor.data[start..start + tensor.rows].to_vec()
        })
        .collect()
}

pub(crate) fn complex_tensor_columns(tensor: &ComplexTensor) -> Vec<Vec<(f64, f64)>> {
    if tensor.rows <= 1 || tensor.cols <= 1 || tensor.shape.len() <= 1 {
        return vec![tensor.data.clone()];
    }
    (0..tensor.cols)
        .map(|col| {
            let start = col * tensor.rows;
            tensor.data[start..start + tensor.rows].to_vec()
        })
        .collect()
}

pub(crate) fn tensor_is_vector(tensor: &Tensor) -> bool {
    tensor.rows <= 1 || tensor.cols <= 1 || tensor.shape.len() <= 1
}

pub(crate) fn tensor_rows(tensor: &Tensor) -> usize {
    if tensor_is_vector(tensor) {
        tensor.data.len()
    } else {
        tensor.rows
    }
}

pub(crate) fn tensor_cols(tensor: &Tensor) -> usize {
    if tensor_is_vector(tensor) {
        1
    } else {
        tensor.cols
    }
}

pub(crate) fn is_real_numeric_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(_) | Value::GpuTensor(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_)
    )
}

pub(crate) fn scalar_tensor(value: f64) -> Tensor {
    Tensor {
        data: vec![value],
        integer_data: None,
        shape: vec![1],
        rows: 1,
        cols: 1,
        dtype: runmat_builtins::NumericDType::F64,
    }
}

fn polar_data_err(builtin: &'static str, msg: impl Into<String>) -> RuntimeError {
    plotting_error(builtin, format!("{builtin}: {}", msg.into()))
}
