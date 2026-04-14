use crate::interpreter::errors::mex;
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_runtime::RuntimeError;

pub async fn rhs_to_real_scalar(rhs: &Value) -> Result<f64, RuntimeError> {
    match rhs {
        Value::Num(x) => Ok(*x),
        Value::Tensor(t2) => {
            if t2.data.len() == 1 { Ok(t2.data[0]) } else { Err(mex("ScalarRequired", "RHS must be scalar")) }
        }
        Value::GpuTensor(h2) => {
            let total = h2.shape.iter().copied().product::<usize>();
            if total != 1 {
                return Err(mex("ScalarRequired", "RHS must be scalar"));
            }
            let provider = runmat_accelerate_api::provider().ok_or_else(|| {
                mex("AccelerationProviderUnavailable", "No acceleration provider registered")
            })?;
            let host = provider.download(h2).await.map_err(|e| format!("gather rhs: {e}"))?;
            Ok(host.data[0])
        }
        _ => rhs.try_into().map_err(|_| mex("NumericRequired", "RHS must be numeric")),
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
                mex("AccelerationProviderUnavailable", "No acceleration provider registered")
            })?;
            let host = provider.download(h).await.map_err(|e| format!("gather rhs: {e}"))?;
            Ok((host.data[0], 0.0))
        }
        _ => Err(mex("NumericRequired", "RHS must be numeric")),
    }
}

pub async fn assign_tensor_scalar(
    mut t: Tensor,
    indices: &[usize],
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if indices.len() == 1 {
        let total = t.rows * t.cols;
        let idx = indices[0];
        if idx == 0 || idx > total {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        let val = rhs_to_real_scalar(rhs).await?;
        t.data[idx - 1] = val;
        Ok(Value::Tensor(t))
    } else if indices.len() == 2 {
        let i = indices[0];
        let mut j = indices[1];
        let rows = t.rows;
        let cols = t.cols;
        if j == 0 { j = 1; }
        if j > cols { j = cols; }
        if i == 0 || i > rows {
            return Err(mex("SubscriptOutOfBounds", "Subscript out of bounds"));
        }
        let val = rhs_to_real_scalar(rhs).await?;
        let idx = (i - 1) + (j - 1) * rows;
        t.data[idx] = val;
        Ok(Value::Tensor(t))
    } else {
        Err(mex("UnsupportedAssignmentRank", "Only 1D/2D scalar assignment supported"))
    }
}

pub async fn assign_complex_scalar(
    mut t: ComplexTensor,
    indices: &[usize],
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if indices.len() == 1 {
        let total = t.rows * t.cols;
        let idx = indices[0];
        if idx == 0 || idx > total {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        let val = rhs_to_complex_scalar(rhs).await?;
        t.data[idx - 1] = val;
        Ok(Value::ComplexTensor(t))
    } else if indices.len() == 2 {
        let i = indices[0];
        let mut j = indices[1];
        let rows = t.rows;
        let cols = t.cols;
        if j == 0 { j = 1; }
        if j > cols { j = cols; }
        if i == 0 || i > rows {
            return Err(mex("SubscriptOutOfBounds", "Subscript out of bounds"));
        }
        let val = rhs_to_complex_scalar(rhs).await?;
        let idx = (i - 1) + (j - 1) * rows;
        t.data[idx] = val;
        Ok(Value::ComplexTensor(t))
    } else {
        Err(mex("UnsupportedAssignmentRank", "Only 1D/2D scalar assignment supported"))
    }
}

pub async fn assign_gpu_scalar(
    h: &runmat_accelerate_api::GpuTensorHandle,
    indices: &[usize],
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        mex("AccelerationProviderUnavailable", "No acceleration provider registered")
    })?;
    let host = provider.download(h).await.map_err(|e| format!("gather for assignment: {e}"))?;
    let t = Tensor::new(host.data, host.shape).map_err(|e| format!("assignment: {e}"))?;
    let Value::Tensor(updated) = assign_tensor_scalar(t, indices, rhs).await? else { unreachable!() };
    let view = runmat_accelerate_api::HostTensorView { data: &updated.data, shape: &updated.shape };
    let new_h = provider.upload(&view).map_err(|e| format!("reupload after assignment: {e}"))?;
    Ok(Value::GpuTensor(new_h))
}
