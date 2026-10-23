use super::*;

pub(super) fn value_shape(value: &Value) -> &[usize] {
    match value {
        Value::Tensor(tensor) => tensor.shape(),
        Value::LogicalArray(logical) => &logical.shape,
        Value::ComplexTensor(tensor) => tensor.shape(),
        Value::GpuTensor(handle) => handle.shape(),
        _ => &[],
    }
}

pub(super) fn value_len(value: &Value) -> usize {
    match value {
        Value::Tensor(tensor) => tensor_len(tensor),
        Value::LogicalArray(logical) => logical.data.len(),
        Value::ComplexTensor(tensor) => tensor.len(),
        Value::GpuTensor(handle) => product(handle.shape()),
        _ => 1,
    }
}

pub fn matlab_gradient_shape(shape: &[usize], len: usize) -> Vec<usize> {
    if shape.is_empty() {
        if len == 0 {
            Vec::new()
        } else {
            vec![1, 1]
        }
    } else if shape.len() == 1 {
        if shape[0] == 1 {
            vec![1, 1]
        } else {
            vec![1, shape[0]]
        }
    } else {
        shape.to_vec()
    }
}

pub(super) fn gradient_output_dims(shape: &[usize], len: usize) -> Vec<usize> {
    let normalized_shape = matlab_gradient_shape(shape, len);
    let mut ext_shape = if normalized_shape.is_empty() {
        if len == 0 {
            vec![0, 0]
        } else {
            vec![1, 1]
        }
    } else {
        normalized_shape
    };
    if ext_shape.len() == 1 {
        ext_shape.push(1);
    }

    if ext_shape.len() <= 2 {
        let rows = ext_shape.first().copied().unwrap_or(1);
        let cols = ext_shape.get(1).copied().unwrap_or(1);
        if rows == 1 && cols == 1 {
            vec![1]
        } else if rows == 1 {
            vec![2]
        } else if cols == 1 {
            vec![1]
        } else {
            vec![2, 1]
        }
    } else {
        let mut dims = vec![2, 1];
        for dim in 3..=ext_shape.len() {
            dims.push(dim);
        }
        dims
    }
}

pub(super) fn gradient_dim_lengths(shape: &[usize], len: usize, dims: &[usize]) -> Vec<usize> {
    let mut ext_shape = matlab_gradient_shape(shape, len);
    if ext_shape.is_empty() {
        ext_shape = if len == 0 { vec![0, 0] } else { vec![1, 1] };
    }

    let max_dim = dims.iter().copied().max().unwrap_or(1);
    while ext_shape.len() < max_dim {
        ext_shape.push(1);
    }

    dims.iter()
        .map(|dim| ext_shape[dim.saturating_sub(1)])
        .collect()
}

pub(super) fn tensor_len(tensor: &Tensor) -> usize {
    tensor.len()
}

pub(super) fn product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .fold(1usize, |acc, value| acc.saturating_mul(value))
}
