use crate::interpreter::errors::mex;
use runmat_builtins::Value;
use runmat_runtime::{
    builtins::common::shape::is_scalar_shape, dispatcher::gather_if_needed_async, RuntimeError,
};

pub type VmResult<T> = Result<T, RuntimeError>;

#[derive(Clone)]
pub enum SliceSelector {
    Colon,
    Scalar(usize),
    Indices(Vec<usize>),
    LinearIndices {
        values: Vec<usize>,
        output_shape: Vec<usize>,
    },
}

#[derive(Debug, Clone)]
pub struct SlicePlan {
    pub indices: Vec<u32>,
    pub output_shape: Vec<usize>,
    pub selection_lengths: Vec<usize>,
    pub dims: usize,
}

fn cartesian_product<F: FnMut(&[usize])>(lists: &[Vec<usize>], mut f: F) {
    let dims = lists.len();
    if dims == 0 {
        return;
    }
    let mut idx = vec![0usize; dims];
    loop {
        let current: Vec<usize> = (0..dims).map(|d| lists[d][idx[d]]).collect();
        f(&current);
        let mut d = 0usize;
        while d < dims {
            idx[d] += 1;
            if idx[d] < lists[d].len() {
                break;
            }
            idx[d] = 0;
            d += 1;
        }
        if d == dims {
            break;
        }
    }
}

fn total_len_from_shape(shape: &[usize]) -> usize {
    if is_scalar_shape(shape) {
        1
    } else {
        shape.iter().copied().product()
    }
}

fn index_scalar_from_host_value(value: &Value) -> Option<i64> {
    match value {
        Value::Num(n) => Some(*n as i64),
        Value::Int(int_val) => Some(int_val.to_i64()),
        Value::Tensor(t) if t.data.len() == 1 && is_scalar_shape(&t.shape) => {
            Some(t.data[0] as i64)
        }
        _ => None,
    }
}

pub async fn index_scalar_from_value(value: &Value) -> VmResult<Option<i64>> {
    if let Value::GpuTensor(handle) = value {
        let total = total_len_from_shape(&handle.shape);
        if total != 1 {
            return Ok(None);
        }
        let gathered = gather_if_needed_async(value).await?;
        return Ok(index_scalar_from_host_value(&gathered));
    }
    Ok(index_scalar_from_host_value(value))
}

pub async fn materialize_index_value(value: &Value) -> VmResult<Value> {
    if matches!(value, Value::GpuTensor(_)) {
        return gather_if_needed_async(value)
            .await
            .map_err(|e| mex("IndexGather", &format!("Failed to gather index value: {e}")));
    }
    Ok(value.clone())
}

pub async fn indices_from_value_linear(value: &Value, total_len: usize) -> VmResult<Vec<usize>> {
    if let Value::Bool(b) = value {
        return Ok(if *b { vec![1] } else { Vec::new() });
    }
    if let Value::LogicalArray(la) = value {
        if la.data.len() == 1 && is_scalar_shape(&la.shape) {
            return Ok(if la.data[0] != 0 { vec![1] } else { Vec::new() });
        }
    }
    if let Some(idx_val) = index_scalar_from_value(value).await? {
        if idx_val < 1 || (idx_val as usize) > total_len {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        return Ok(vec![idx_val as usize]);
    }
    let materialized;
    let value = if matches!(value, Value::GpuTensor(_)) {
        materialized = materialize_index_value(value).await?;
        &materialized
    } else {
        value
    };
    match value {
        Value::Tensor(idx_t) => {
            let len = idx_t.shape.iter().product::<usize>();
            let mut indices = Vec::with_capacity(len);
            for &val in &idx_t.data {
                let idx = val as isize;
                if idx < 1 || (idx as usize) > total_len {
                    return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                }
                indices.push(idx as usize);
            }
            Ok(indices)
        }
        Value::LogicalArray(la) => {
            if la.data.len() != total_len {
                return Err(mex(
                    "IndexShape",
                    "Logical mask length mismatch for linear indexing",
                ));
            }
            let mut indices = Vec::new();
            for (i, &b) in la.data.iter().enumerate() {
                if b != 0 {
                    indices.push(i + 1);
                }
            }
            Ok(indices)
        }
        _ => Err(mex(
            "UnsupportedIndexType",
            "Unsupported index type for linear indexing",
        )),
    }
}

pub async fn selector_from_value_dim(value: &Value, dim_len: usize) -> VmResult<SliceSelector> {
    if let Value::Bool(b) = value {
        if *b {
            return Ok(SliceSelector::Indices(vec![1]));
        }
        return Ok(SliceSelector::Indices(Vec::new()));
    }
    if let Value::LogicalArray(la) = value {
        if la.data.len() == 1 && is_scalar_shape(&la.shape) {
            if la.data[0] != 0 {
                return Ok(SliceSelector::Indices(vec![1]));
            }
            return Ok(SliceSelector::Indices(Vec::new()));
        }
    }
    if let Some(idx_val) = index_scalar_from_value(value).await? {
        if idx_val < 1 || (idx_val as usize) > dim_len {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        return Ok(SliceSelector::Scalar(idx_val as usize));
    }
    let materialized;
    let value = if matches!(value, Value::GpuTensor(_)) {
        materialized = materialize_index_value(value).await?;
        &materialized
    } else {
        value
    };
    match value {
        Value::Tensor(idx_t) => {
            let len = idx_t.shape.iter().product::<usize>();
            let mut indices = Vec::with_capacity(len);
            for &val in &idx_t.data {
                let idx = val as isize;
                if idx < 1 || (idx as usize) > dim_len {
                    return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                }
                indices.push(idx as usize);
            }
            Ok(SliceSelector::Indices(indices))
        }
        Value::LogicalArray(la) => {
            if la.data.len() != dim_len {
                return Err(mex(
                    "IndexShape",
                    "Logical mask length mismatch for dimension",
                ));
            }
            let mut indices = Vec::new();
            for (i, &b) in la.data.iter().enumerate() {
                if b != 0 {
                    indices.push(i + 1);
                }
            }
            Ok(SliceSelector::Indices(indices))
        }
        _ => Err(mex(
            "UnsupportedIndexType",
            "Unsupported index type for slicing",
        )),
    }
}

pub async fn build_slice_selectors(
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
) -> VmResult<Vec<SliceSelector>> {
    let mut selectors = Vec::with_capacity(dims);
    if dims == 1 {
        let total_len = total_len_from_shape(base_shape);
        if (colon_mask & 1u32) != 0 {
            selectors.push(SliceSelector::Indices((1..=total_len).collect()));
            return Ok(selectors);
        }
        if (end_mask & 1u32) != 0 {
            selectors.push(SliceSelector::Scalar(total_len.max(1)));
            return Ok(selectors);
        }
        let value = numeric.first().ok_or_else(|| {
            mex(
                "MissingNumericIndex",
                "missing numeric index for linear slice",
            )
        })?;
        let materialized = materialize_index_value(value).await?;
        if let Value::Tensor(idx_t) = &materialized {
            let len = idx_t.shape.iter().product::<usize>();
            let mut indices = Vec::with_capacity(len);
            for &val in &idx_t.data {
                let idx = val as isize;
                if idx < 1 || (idx as usize) > total_len {
                    return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                }
                indices.push(idx as usize);
            }
            selectors.push(SliceSelector::LinearIndices {
                values: indices,
                output_shape: idx_t.shape.clone(),
            });
        } else {
            let idxs = indices_from_value_linear(&materialized, total_len).await?;
            selectors.push(SliceSelector::Indices(idxs));
        }
        return Ok(selectors);
    }

    let mut numeric_iter = 0usize;
    for d in 0..dims {
        let is_colon = (colon_mask & (1u32 << d)) != 0;
        if is_colon {
            selectors.push(SliceSelector::Colon);
            continue;
        }
        let dim_len = base_shape.get(d).copied().unwrap_or(1);
        let is_end = (end_mask & (1u32 << d)) != 0;
        if is_end {
            selectors.push(SliceSelector::Scalar(dim_len));
            continue;
        }
        let value = numeric
            .get(numeric_iter)
            .ok_or_else(|| mex("MissingNumericIndex", "missing numeric index for slice"))?;
        numeric_iter += 1;
        selectors.push(selector_from_value_dim(value, dim_len).await?);
    }
    Ok(selectors)
}

fn matlab_squeezed_shape(selection_lengths: &[usize], scalar_mask: &[bool]) -> Vec<usize> {
    let mut dims: Vec<(usize, usize, bool)> = selection_lengths
        .iter()
        .enumerate()
        .map(|(d, &len)| (d, len, scalar_mask.get(d).copied().unwrap_or(false)))
        .collect();
    while dims.len() > 2
        && dims
            .last()
            .map(|&(_, len, is_scalar)| len == 1 && is_scalar)
            .unwrap_or(false)
    {
        dims.pop();
    }
    let out: Vec<usize> = dims.into_iter().map(|(_, len, _)| len).collect();
    if out.is_empty() {
        vec![1, 1]
    } else {
        out
    }
}

pub fn build_slice_plan(
    selectors: &[SliceSelector],
    dims: usize,
    base_shape: &[usize],
) -> VmResult<SlicePlan> {
    let total_len = total_len_from_shape(base_shape);
    if dims == 1 {
        let list = selectors
            .first()
            .cloned()
            .unwrap_or(SliceSelector::Indices(Vec::new()));
        let indices = match &list {
            SliceSelector::Colon => (1..=total_len).collect::<Vec<usize>>(),
            SliceSelector::Scalar(i) => vec![*i],
            SliceSelector::Indices(v) => v.clone(),
            SliceSelector::LinearIndices { values, .. } => values.clone(),
        };
        if indices.iter().any(|&i| i == 0 || i > total_len) {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        let zero_based: Vec<u32> = indices.iter().map(|&i| (i - 1) as u32).collect();
        let count = zero_based.len();
        let shape = match list {
            SliceSelector::LinearIndices { output_shape, .. } => output_shape,
            _ if count <= 1 => vec![1, 1],
            _ => vec![count, 1],
        };
        return Ok(SlicePlan {
            indices: zero_based,
            output_shape: shape,
            selection_lengths: vec![count],
            dims,
        });
    }

    let mut selection_lengths = Vec::with_capacity(dims);
    let mut per_dim_lists: Vec<Vec<usize>> = Vec::with_capacity(dims);
    let mut scalar_mask: Vec<bool> = Vec::with_capacity(dims);
    for (d, sel) in selectors.iter().enumerate().take(dims) {
        let dim_len = base_shape.get(d).copied().unwrap_or(1);
        let idxs = match sel {
            SliceSelector::Colon => (1..=dim_len).collect::<Vec<usize>>(),
            SliceSelector::Scalar(i) => vec![*i],
            SliceSelector::Indices(v) => v.clone(),
            SliceSelector::LinearIndices { values: v, .. } => v.clone(),
        };
        if idxs.iter().any(|&i| i == 0 || i > dim_len) {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        selection_lengths.push(idxs.len());
        per_dim_lists.push(idxs);
        scalar_mask.push(matches!(sel, SliceSelector::Scalar(_)));
    }

    let mut out_shape = matlab_squeezed_shape(&selection_lengths, &scalar_mask);
    if selection_lengths.contains(&0) {
        let selection_lengths = out_shape.clone();
        return Ok(SlicePlan {
            indices: Vec::new(),
            output_shape: out_shape,
            selection_lengths,
            dims,
        });
    }

    let mut base_norm = base_shape.to_vec();
    if base_norm.len() < dims {
        base_norm.resize(dims, 1);
    }
    let mut strides = vec![1usize; dims];
    for d in 1..dims {
        strides[d] = strides[d - 1] * base_norm[d - 1].max(1);
    }

    let mut indices = Vec::new();
    cartesian_product(&per_dim_lists, |multi| {
        let mut lin = 0usize;
        for d in 0..dims {
            let idx = multi[d] - 1;
            lin += idx * strides[d];
        }
        indices.push(lin as u32);
    });

    let total_out: usize = selection_lengths.iter().product();
    if total_out == 1 {
        out_shape = vec![1, 1];
    }
    let selection_lengths = out_shape.clone();
    Ok(SlicePlan {
        indices,
        output_shape: out_shape,
        selection_lengths,
        dims,
    })
}
