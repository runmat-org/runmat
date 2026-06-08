use crate::interpreter::errors::mex;
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_runtime::RuntimeError;

fn map_assignment_shape_error(err: impl std::fmt::Display) -> RuntimeError {
    mex("ShapeMismatch", &format!("assignment: {err}"))
}

fn map_acceleration_error(context: &str, err: impl std::fmt::Display) -> RuntimeError {
    mex("AccelerationOperationFailed", &format!("{context}: {err}"))
}

fn is_empty_tensor(value: &Value) -> bool {
    matches!(value, Value::Tensor(t) if t.data.is_empty() || t.rows == 0 || t.cols == 0)
        || matches!(value, Value::ComplexTensor(t) if t.data.is_empty() || t.rows == 0 || t.cols == 0)
}

fn delete_tensor_linear(mut t: Tensor, idx: usize) -> Result<Value, RuntimeError> {
    let total = t.rows * t.cols;
    if idx == 0 || idx > total {
        return Err(mex("IndexOutOfBounds", "Index out of bounds"));
    }
    if !(t.rows == 1 || t.cols == 1) {
        return Err(mex(
            "UnsupportedDeletion",
            "Linear deletion is only supported for vectors",
        ));
    }
    t.data.remove(idx - 1);
    if t.data.is_empty() {
        t.rows = 0;
        t.cols = 0;
        t.shape = vec![0, 0];
    } else if t.rows == 1 {
        t.cols = t.data.len();
        t.shape = vec![1, t.cols];
    } else {
        t.rows = t.data.len();
        t.shape = vec![t.rows, 1];
    }
    Ok(Value::Tensor(t))
}

fn tensor_to_complex(t: Tensor) -> ComplexTensor {
    ComplexTensor {
        data: t.data.into_iter().map(|re| (re, 0.0)).collect(),
        shape: t.shape,
        rows: t.rows,
        cols: t.cols,
    }
}

fn delete_complex_linear(mut t: ComplexTensor, idx: usize) -> Result<Value, RuntimeError> {
    let total = t.rows * t.cols;
    if idx == 0 || idx > total {
        return Err(mex("IndexOutOfBounds", "Index out of bounds"));
    }
    if !(t.rows == 1 || t.cols == 1) {
        return Err(mex(
            "UnsupportedDeletion",
            "Linear deletion is only supported for vectors",
        ));
    }
    t.data.remove(idx - 1);
    if t.data.is_empty() {
        t.rows = 0;
        t.cols = 0;
        t.shape = vec![0, 0];
    } else if t.rows == 1 {
        t.cols = t.data.len();
        t.shape = vec![1, t.cols];
    } else {
        t.rows = t.data.len();
        t.shape = vec![t.rows, 1];
    }
    Ok(Value::ComplexTensor(t))
}

pub async fn rhs_to_real_scalar(rhs: &Value) -> Result<f64, RuntimeError> {
    match rhs {
        Value::Num(x) => Ok(*x),
        Value::Tensor(t2) => {
            if t2.data.len() == 1 {
                Ok(t2.data[0])
            } else {
                Err(mex("ScalarRequired", "RHS must be scalar"))
            }
        }
        Value::GpuTensor(h2) => {
            let total = h2.shape.iter().copied().product::<usize>();
            if total != 1 {
                return Err(mex("ScalarRequired", "RHS must be scalar"));
            }
            let provider = runmat_accelerate_api::provider().ok_or_else(|| {
                mex(
                    "AccelerationProviderUnavailable",
                    "No acceleration provider registered",
                )
            })?;
            let host = provider
                .download(h2)
                .await
                .map_err(|e| map_acceleration_error("gather rhs", e))?;
            Ok(host.data[0])
        }
        _ => rhs
            .try_into()
            .map_err(|_| mex("NumericRequired", "RHS must be numeric")),
    }
}

pub async fn rhs_to_complex_scalar(rhs: &Value) -> Result<(f64, f64), RuntimeError> {
    match rhs {
        Value::Complex(re, im) => Ok((*re, *im)),
        Value::Num(n) => Ok((*n, 0.0)),
        Value::Int(i) => Ok((i.to_f64(), 0.0)),
        Value::Bool(b) => Ok((if *b { 1.0 } else { 0.0 }, 0.0)),
        Value::Tensor(t) if t.data.len() == 1 => Ok((t.data[0], 0.0)),
        Value::ComplexTensor(t) if t.data.len() == 1 => Ok(t.data[0]),
        Value::GpuTensor(h) => {
            let total = h.shape.iter().copied().product::<usize>();
            if total != 1 {
                return Err(mex("ScalarRequired", "RHS must be scalar"));
            }
            let provider = runmat_accelerate_api::provider().ok_or_else(|| {
                mex(
                    "AccelerationProviderUnavailable",
                    "No acceleration provider registered",
                )
            })?;
            let host = provider
                .download(h)
                .await
                .map_err(|e| map_acceleration_error("gather rhs", e))?;
            Ok((host.data[0], 0.0))
        }
        _ => Err(mex("NumericRequired", "RHS must be numeric")),
    }
}

pub async fn assign_tensor_scalar(
    mut t: Tensor,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    if indices.len() == 1 {
        let total = t.rows * t.cols;
        let idx = indices[0];
        if idx == 0 {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        if delete {
            if idx > total {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            if !is_empty_tensor(rhs) {
                return Err(mex(
                    "DeletionRequiresEmptyRhs",
                    "Indexed deletion requires empty RHS",
                ));
            }
            return delete_tensor_linear(t, idx);
        }
        if matches!(rhs, Value::Complex(_, _) | Value::ComplexTensor(_)) {
            return assign_complex_scalar(tensor_to_complex(t), indices, rhs, false).await;
        }
        let val = rhs_to_real_scalar(rhs).await?;
        if idx > total {
            if !(t.rows == 1 || t.cols == 1) {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            let target_len = idx;
            if t.rows == 1 {
                t.data.resize(target_len, 0.0);
                t.cols = target_len;
                t.shape = vec![1, t.cols];
            } else {
                t.data.resize(target_len, 0.0);
                t.rows = target_len;
                t.shape = vec![t.rows, 1];
            }
        }
        t.data[idx - 1] = val;
        Ok(Value::Tensor(t))
    } else if indices.len() == 2 {
        let i = indices[0];
        let mut j = indices[1];
        let rows = t.rows;
        let cols = t.cols;
        if j == 0 {
            j = 1;
        }
        if j > cols {
            j = cols;
        }
        if i == 0 || i > rows {
            return Err(mex("SubscriptOutOfBounds", "Subscript out of bounds"));
        }
        if delete {
            return Err(mex(
                "UnsupportedDeletion",
                "Indexed deletion is only supported for linear vector indices",
            ));
        }
        if matches!(rhs, Value::Complex(_, _) | Value::ComplexTensor(_)) {
            return assign_complex_scalar(tensor_to_complex(t), indices, rhs, false).await;
        }
        let val = rhs_to_real_scalar(rhs).await?;
        let idx = (i - 1) + (j - 1) * rows;
        t.data[idx] = val;
        Ok(Value::Tensor(t))
    } else {
        Err(mex(
            "UnsupportedAssignmentRank",
            "Only 1D/2D scalar assignment supported",
        ))
    }
}

pub async fn assign_complex_scalar(
    mut t: ComplexTensor,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    if indices.len() == 1 {
        let total = t.rows * t.cols;
        let idx = indices[0];
        if idx == 0 {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        if delete {
            if idx > total {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            if !is_empty_tensor(rhs) {
                return Err(mex(
                    "DeletionRequiresEmptyRhs",
                    "Indexed deletion requires empty RHS",
                ));
            }
            return delete_complex_linear(t, idx);
        }
        let val = rhs_to_complex_scalar(rhs).await?;
        if idx > total {
            if !(t.rows == 1 || t.cols == 1) {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            let target_len = idx;
            if t.rows == 1 {
                t.data.resize(target_len, (0.0, 0.0));
                t.cols = target_len;
                t.shape = vec![1, t.cols];
            } else {
                t.data.resize(target_len, (0.0, 0.0));
                t.rows = target_len;
                t.shape = vec![t.rows, 1];
            }
        }
        t.data[idx - 1] = val;
        Ok(Value::ComplexTensor(t))
    } else if indices.len() == 2 {
        let i = indices[0];
        let mut j = indices[1];
        let rows = t.rows;
        let cols = t.cols;
        if j == 0 {
            j = 1;
        }
        if j > cols {
            j = cols;
        }
        if i == 0 || i > rows {
            return Err(mex("SubscriptOutOfBounds", "Subscript out of bounds"));
        }
        if delete {
            return Err(mex(
                "UnsupportedDeletion",
                "Indexed deletion is only supported for linear vector indices",
            ));
        }
        let val = rhs_to_complex_scalar(rhs).await?;
        let idx = (i - 1) + (j - 1) * rows;
        t.data[idx] = val;
        Ok(Value::ComplexTensor(t))
    } else {
        Err(mex(
            "UnsupportedAssignmentRank",
            "Only 1D/2D scalar assignment supported",
        ))
    }
}

pub async fn assign_gpu_scalar(
    h: &runmat_accelerate_api::GpuTensorHandle,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        mex(
            "AccelerationProviderUnavailable",
            "No acceleration provider registered",
        )
    })?;
    let host = provider
        .download(h)
        .await
        .map_err(|e| map_acceleration_error("gather for assignment", e))?;
    let t = Tensor::new(host.data, host.shape).map_err(map_assignment_shape_error)?;
    let Value::Tensor(updated) = assign_tensor_scalar(t, indices, rhs, delete).await? else {
        unreachable!()
    };
    let view = runmat_accelerate_api::HostTensorView {
        data: &updated.data,
        shape: &updated.shape,
    };
    let new_h = provider
        .upload(&view)
        .map_err(|e| map_acceleration_error("reupload after assignment", e))?;
    Ok(Value::GpuTensor(new_h))
}

#[cfg(test)]
mod tests {
    use super::{map_acceleration_error, map_assignment_shape_error};

    #[test]
    fn assignment_shape_error_mapping_reports_identifier() {
        let err = map_assignment_shape_error("invalid shape");
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn assignment_acceleration_error_mapping_reports_identifier() {
        let err = map_acceleration_error("gather rhs", "provider failed");
        assert_eq!(err.identifier(), Some("RunMat:AccelerationOperationFailed"));
    }
}
