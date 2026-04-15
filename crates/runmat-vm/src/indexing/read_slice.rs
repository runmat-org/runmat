use crate::indexing::plan::{build_index_plan, IndexPlan};
use crate::indexing::selectors::{
    build_slice_selectors, index_scalar_from_value, materialize_index_value, SliceSelector,
};
use runmat_builtins::{CellArray, ComplexTensor, StringArray, Tensor, Value};
use runmat_runtime::RuntimeError;

pub fn build_numeric_subsref_cell(numeric: &[Value]) -> Result<Value, RuntimeError> {
    let cell = CellArray::new(numeric.to_vec(), 1, numeric.len())
        .map_err(|e| format!("subsref build error: {e}"))?;
    Ok(Value::Cell(cell))
}

pub async fn object_subsref_paren(base: Value, numeric: &[Value]) -> Result<Value, RuntimeError> {
    let cell = build_numeric_subsref_cell(numeric)?;
    match base {
        Value::Object(obj) => {
            let args = vec![
                Value::Object(obj),
                Value::String("subsref".to_string()),
                Value::String("()".to_string()),
                cell,
            ];
            runmat_runtime::call_builtin_async("call_method", &args).await
        }
        Value::HandleObject(handle) => {
            let args = vec![
                Value::HandleObject(handle),
                Value::String("subsref".to_string()),
                Value::String("()".to_string()),
                cell,
            ];
            runmat_runtime::call_builtin_async("call_method", &args).await
        }
        other => Err(format!("slice subsref requires object/handle, got {other:?}").into()),
    }
}

pub async fn read_tensor_slice_1d(
    tensor: &Tensor,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    let total = tensor.data.len();
    let mut idxs: Vec<usize> = Vec::new();
    let mut idx_shape: Option<Vec<usize>> = None;
    let is_colon = (colon_mask & 1u32) != 0;
    let is_end = (end_mask & 1u32) != 0;
    if is_colon {
        idxs = (1..=total).collect();
    } else if is_end {
        idxs = vec![total];
    } else if let Some(v) = numeric.first() {
        let materialized = materialize_index_value(v).await?;
        if let Some(i) = index_scalar_from_value(&materialized).await? {
            if i < 1 {
                return Err(crate::interpreter::errors::mex(
                    "IndexOutOfBounds",
                    "Index out of bounds",
                ));
            }
            idxs = vec![i as usize];
        } else {
            match &materialized {
                Value::Tensor(idx_t) => {
                    idx_shape = Some(idx_t.shape.clone());
                    for &val in &idx_t.data {
                        let i = val as isize;
                        if i < 1 || (i as usize) > total {
                            return Err(crate::interpreter::errors::mex(
                                "IndexOutOfBounds",
                                "Index out of bounds",
                            ));
                        }
                        idxs.push(i as usize);
                    }
                }
                Value::Bool(b) => {
                    if *b {
                        idxs = vec![1];
                    }
                }
                Value::LogicalArray(la) => {
                    if la.data.len() != total {
                        return Err(crate::interpreter::errors::mex(
                            "IndexShape",
                            "Logical mask length mismatch for linear indexing",
                        ));
                    }
                    for (i, &val) in la.data.iter().enumerate() {
                        if val != 0 {
                            idxs.push(i + 1);
                        }
                    }
                }
                _ => {
                    return Err(crate::interpreter::errors::mex(
                        "UnsupportedIndexType",
                        "Unsupported index type",
                    ))
                }
            }
        }
    } else {
        return Err(crate::interpreter::errors::mex(
            "MissingNumericIndex",
            "missing numeric index",
        ));
    }
    if idxs.iter().any(|&i| i == 0 || i > total) {
        return Err(crate::interpreter::errors::mex(
            "IndexOutOfBounds",
            "Index out of bounds",
        ));
    }
    if idxs.len() == 1 {
        Ok(Value::Num(tensor.data[idxs[0] - 1]))
    } else if idxs.is_empty() {
        let shape = idx_shape.unwrap_or_else(|| vec![0, 1]);
        let tens = Tensor::new(vec![], shape).map_err(|e| format!("Slice error: {e}"))?;
        Ok(Value::Tensor(tens))
    } else {
        let mut out = Vec::with_capacity(idxs.len());
        for &i in &idxs {
            out.push(tensor.data[i - 1]);
        }
        let shape = idx_shape.unwrap_or_else(|| vec![idxs.len(), 1]);
        let tens = Tensor::new(out, shape).map_err(|e| format!("Slice error: {e}"))?;
        Ok(Value::Tensor(tens))
    }
}

pub fn try_tensor_slice_2d_fast_path(
    tensor: &Tensor,
    dims: usize,
    selectors: &[SliceSelector],
) -> Result<Option<Value>, RuntimeError> {
    if dims != 2 {
        return Ok(None);
    }
    let rows = tensor.shape.first().copied().unwrap_or(1);
    let cols = tensor.shape.get(1).copied().unwrap_or(1);
    match (&selectors[0], &selectors[1]) {
        (SliceSelector::Colon, SliceSelector::Scalar(j)) => {
            let j0 = *j - 1;
            if j0 >= cols {
                return Err(crate::interpreter::errors::mex(
                    "IndexOutOfBounds",
                    "Index out of bounds",
                ));
            }
            let start = j0 * rows;
            let out = tensor.data[start..start + rows].to_vec();
            if out.len() == 1 {
                Ok(Some(Value::Num(out[0])))
            } else {
                let tens =
                    Tensor::new(out, vec![rows, 1]).map_err(|e| format!("Slice error: {e}"))?;
                Ok(Some(Value::Tensor(tens)))
            }
        }
        (SliceSelector::Scalar(i), SliceSelector::Colon) => {
            let i0 = *i - 1;
            if i0 >= rows {
                return Err(crate::interpreter::errors::mex(
                    "IndexOutOfBounds",
                    "Index out of bounds",
                ));
            }
            let mut out: Vec<f64> = Vec::with_capacity(cols);
            for c in 0..cols {
                out.push(tensor.data[i0 + c * rows]);
            }
            if out.len() == 1 {
                Ok(Some(Value::Num(out[0])))
            } else {
                let tens =
                    Tensor::new(out, vec![1, cols]).map_err(|e| format!("Slice error: {e}"))?;
                Ok(Some(Value::Tensor(tens)))
            }
        }
        (SliceSelector::Colon, SliceSelector::Indices(js)) => {
            if js.is_empty() {
                let tens = Tensor::new(Vec::new(), vec![rows, 0])
                    .map_err(|e| format!("Slice error: {e}"))?;
                Ok(Some(Value::Tensor(tens)))
            } else {
                let mut out: Vec<f64> = Vec::with_capacity(rows * js.len());
                for &j in js {
                    let j0 = j - 1;
                    if j0 >= cols {
                        return Err(crate::interpreter::errors::mex(
                            "IndexOutOfBounds",
                            "Index out of bounds",
                        ));
                    }
                    let start = j0 * rows;
                    out.extend_from_slice(&tensor.data[start..start + rows]);
                }
                let tens = Tensor::new(out, vec![rows, js.len()])
                    .map_err(|e| format!("Slice error: {e}"))?;
                Ok(Some(Value::Tensor(tens)))
            }
        }
        (SliceSelector::Indices(is), SliceSelector::Colon) => {
            if is.is_empty() {
                let tens = Tensor::new(Vec::new(), vec![0, cols])
                    .map_err(|e| format!("Slice error: {e}"))?;
                Ok(Some(Value::Tensor(tens)))
            } else {
                let mut out: Vec<f64> = Vec::with_capacity(is.len() * cols);
                for c in 0..cols {
                    for &i in is {
                        let i0 = i - 1;
                        if i0 >= rows {
                            return Err(crate::interpreter::errors::mex(
                                "IndexOutOfBounds",
                                "Index out of bounds",
                            ));
                        }
                        out.push(tensor.data[i0 + c * rows]);
                    }
                }
                let tens = Tensor::new(out, vec![is.len(), cols])
                    .map_err(|e| format!("Slice error: {e}"))?;
                Ok(Some(Value::Tensor(tens)))
            }
        }
        _ => Ok(None),
    }
}

pub async fn read_tensor_slice_nd(
    tensor: &Tensor,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    let selectors =
        build_slice_selectors(dims, colon_mask, end_mask, numeric, &tensor.shape).await?;
    if let Some(value) = try_tensor_slice_2d_fast_path(tensor, dims, &selectors)? {
        return Ok(value);
    }
    let plan = build_index_plan(&selectors, dims, &tensor.shape)?;
    if plan.indices.is_empty() {
        let out_tensor =
            Tensor::new(Vec::new(), plan.output_shape).map_err(|e| format!("Slice error: {e}"))?;
        return Ok(Value::Tensor(out_tensor));
    }
    let mut out_data: Vec<f64> = Vec::with_capacity(plan.indices.len());
    for &lin in &plan.indices {
        out_data.push(tensor.data[lin as usize]);
    }
    if out_data.len() == 1 {
        Ok(Value::Num(out_data[0]))
    } else {
        let out_tensor =
            Tensor::new(out_data, plan.output_shape).map_err(|e| format!("Slice error: {e}"))?;
        Ok(Value::Tensor(out_tensor))
    }
}

pub fn read_tensor_slice_from_plan(
    tensor: &Tensor,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    if plan.indices.is_empty() {
        let out_tensor = Tensor::new(Vec::new(), plan.output_shape.clone())
            .map_err(|e| format!("Slice error: {e}"))?;
        return Ok(Value::Tensor(out_tensor));
    }
    let mut out_data: Vec<f64> = Vec::with_capacity(plan.indices.len());
    for &lin in &plan.indices {
        out_data.push(tensor.data[lin as usize]);
    }
    if out_data.len() == 1 {
        Ok(Value::Num(out_data[0]))
    } else {
        let out_tensor = Tensor::new(out_data, plan.output_shape.clone())
            .map_err(|e| format!("Slice error: {e}"))?;
        Ok(Value::Tensor(out_tensor))
    }
}

pub async fn read_complex_slice(
    tensor: &ComplexTensor,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    let selectors =
        build_slice_selectors(dims, colon_mask, end_mask, numeric, &tensor.shape).await?;
    let plan = build_index_plan(&selectors, dims, &tensor.shape)?;
    read_complex_slice_from_plan(tensor, &plan)
}

pub fn read_complex_slice_from_plan(
    tensor: &ComplexTensor,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    if plan.indices.is_empty() {
        let empty = ComplexTensor::new(Vec::new(), plan.output_shape.clone())
            .map_err(|e| format!("Slice error: {e}"))?;
        return Ok(Value::ComplexTensor(empty));
    }
    if plan.indices.len() == 1 {
        let lin = plan.indices[0] as usize;
        let (re, im) = tensor.data.get(lin).copied().ok_or_else(|| {
            crate::interpreter::errors::mex(
                "IndexOutOfBounds",
                "Slice error: complex index out of bounds",
            )
        })?;
        return Ok(Value::Complex(re, im));
    }
    let mut out = Vec::with_capacity(plan.indices.len());
    for &lin in &plan.indices {
        let idx = lin as usize;
        let value = tensor.data.get(idx).copied().ok_or_else(|| {
            crate::interpreter::errors::mex(
                "IndexOutOfBounds",
                "Slice error: complex index out of bounds",
            )
        })?;
        out.push(value);
    }
    let out_ct = ComplexTensor::new(out, plan.output_shape.clone())
        .map_err(|e| format!("Slice error: {e}"))?;
    Ok(Value::ComplexTensor(out_ct))
}

pub async fn read_gpu_slice(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    let base_shape = handle.shape.clone();
    let selectors = build_slice_selectors(dims, colon_mask, end_mask, numeric, &base_shape).await?;
    let plan = build_index_plan(&selectors, dims, &base_shape)?;
    read_gpu_slice_from_plan(handle, &plan)
}

pub fn read_gpu_slice_from_plan(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        crate::interpreter::errors::mex(
            "AccelerationProviderUnavailable",
            "No acceleration provider registered",
        )
    })?;
    if plan.indices.is_empty() {
        let zeros = provider
            .zeros(&plan.output_shape)
            .map_err(|e| format!("slice: {e}"))?;
        Ok(Value::GpuTensor(zeros))
    } else {
        let result = provider
            .gather_linear(handle, &plan.indices, &plan.output_shape)
            .map_err(|e| format!("slice: {e}"))?;
        Ok(Value::GpuTensor(result))
    }
}

pub async fn read_string_slice(
    sa: &StringArray,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    let rank = sa.shape.len();
    if dims == 1 {
        let total = sa.data.len();
        let mut idxs: Vec<usize> = Vec::new();
        let is_colon = (colon_mask & 1u32) != 0;
        let is_end = (end_mask & 1u32) != 0;
        if is_colon {
            idxs = (1..=total).collect();
        } else if is_end {
            idxs = vec![total];
        } else if let Some(v) = numeric.first() {
            let materialized = materialize_index_value(v).await?;
            if let Some(i) = index_scalar_from_value(&materialized).await? {
                if i < 1 {
                    return Err(crate::interpreter::errors::mex(
                        "IndexOutOfBounds",
                        "Index out of bounds",
                    ));
                }
                idxs = vec![i as usize];
            } else {
                match &materialized {
                    Value::Tensor(idx_t) => {
                        let len = idx_t.shape.iter().product::<usize>();
                        if len == total {
                            for (i, &val) in idx_t.data.iter().enumerate() {
                                if val != 0.0 {
                                    idxs.push(i + 1);
                                }
                            }
                        } else {
                            for &val in &idx_t.data {
                                let i = val as isize;
                                if i < 1 {
                                    return Err(crate::interpreter::errors::mex(
                                        "IndexOutOfBounds",
                                        "Index out of bounds",
                                    ));
                                }
                                idxs.push(i as usize);
                            }
                        }
                    }
                    _ => {
                        return Err(crate::interpreter::errors::mex(
                            "UnsupportedIndexType",
                            "Unsupported index type",
                        ))
                    }
                }
            }
        } else {
            return Err(crate::interpreter::errors::mex(
                "MissingNumericIndex",
                "missing numeric index",
            ));
        }
        if idxs.iter().any(|&i| i == 0 || i > total) {
            return Err(crate::interpreter::errors::mex(
                "IndexOutOfBounds",
                "Index out of bounds",
            ));
        }
        if idxs.len() == 1 {
            Ok(Value::String(sa.data[idxs[0] - 1].clone()))
        } else {
            let mut out: Vec<String> = Vec::with_capacity(idxs.len());
            for &i in &idxs {
                out.push(sa.data[i - 1].clone());
            }
            let out_sa = StringArray::new(out, vec![idxs.len(), 1])
                .map_err(|e| format!("Slice error: {e}"))?;
            Ok(Value::StringArray(out_sa))
        }
    } else {
        let mut selectors: Vec<SliceSelector> = Vec::with_capacity(dims);
        let mut num_iter = 0usize;
        for d in 0..dims {
            let is_colon = (colon_mask & (1u32 << d)) != 0;
            let is_end = (end_mask & (1u32 << d)) != 0;
            if is_colon {
                selectors.push(SliceSelector::Colon);
            } else if is_end {
                let dim_len = *sa.shape.get(d).unwrap_or(&1);
                selectors.push(SliceSelector::Scalar(dim_len));
            } else {
                let v = numeric.get(num_iter).ok_or_else(|| {
                    crate::interpreter::errors::mex("MissingNumericIndex", "missing numeric index")
                })?;
                num_iter += 1;
                let materialized = materialize_index_value(v).await?;
                if let Some(idx) = index_scalar_from_value(&materialized).await? {
                    if idx < 1 {
                        return Err(crate::interpreter::errors::mex(
                            "IndexOutOfBounds",
                            "Index out of bounds",
                        ));
                    }
                    selectors.push(SliceSelector::Scalar(idx as usize));
                } else {
                    match &materialized {
                        Value::Tensor(idx_t) => {
                            let dim_len = *sa.shape.get(d).unwrap_or(&1);
                            let len = idx_t.shape.iter().product::<usize>();
                            let is_binary_mask =
                                len == dim_len && idx_t.data.iter().all(|&x| x == 0.0 || x == 1.0);
                            if is_binary_mask {
                                let mut v = Vec::new();
                                for (i, &val) in idx_t.data.iter().enumerate() {
                                    if val != 0.0 {
                                        v.push(i + 1);
                                    }
                                }
                                selectors.push(SliceSelector::Indices(v));
                            } else {
                                let mut v = Vec::with_capacity(len);
                                for &val in &idx_t.data {
                                    let idx = val as isize;
                                    if idx < 1 {
                                        return Err(crate::interpreter::errors::mex(
                                            "IndexOutOfBounds",
                                            "Index out of bounds",
                                        ));
                                    }
                                    v.push(idx as usize);
                                }
                                selectors.push(SliceSelector::Indices(v));
                            }
                        }
                        _ => {
                            return Err(crate::interpreter::errors::mex(
                                "UnsupportedIndexType",
                                "Unsupported index type",
                            ))
                        }
                    }
                }
            }
        }

        let mut out_dims: Vec<usize> = Vec::new();
        let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
        for (d, sel) in selectors.iter().enumerate().take(dims) {
            let dim_len = *sa.shape.get(d).unwrap_or(&1);
            let idxs = match sel {
                SliceSelector::Colon => (1..=dim_len).collect::<Vec<usize>>(),
                SliceSelector::Scalar(i) => vec![*i],
                SliceSelector::Indices(v) => v.clone(),
                SliceSelector::LinearIndices { values, .. } => values.clone(),
            };
            if idxs.iter().any(|&i| i == 0 || i > dim_len) {
                return Err(crate::interpreter::errors::mex(
                    "IndexOutOfBounds",
                    "Index out of bounds",
                ));
            }
            if idxs.len() > 1 {
                out_dims.push(idxs.len());
            } else {
                out_dims.push(1);
            }
            per_dim_indices.push(idxs);
        }
        if dims == 2 {
            match (
                &per_dim_indices[0].as_slice(),
                &per_dim_indices[1].as_slice(),
            ) {
                (i_list, j_list) if i_list.len() > 1 && j_list.len() == 1 => {
                    out_dims = vec![i_list.len(), 1];
                }
                (i_list, j_list) if i_list.len() == 1 && j_list.len() > 1 => {
                    out_dims = vec![1, j_list.len()];
                }
                _ => {}
            }
        }
        let full_shape: Vec<usize> = if rank < dims {
            let mut s = sa.shape.clone();
            s.resize(dims, 1);
            s
        } else {
            sa.shape.clone()
        };
        let mut strides: Vec<usize> = vec![0; dims];
        let mut acc = 1usize;
        for (d, stride) in strides.iter_mut().enumerate().take(dims) {
            *stride = acc;
            acc *= full_shape[d];
        }
        let total_out: usize = out_dims.iter().product();
        if total_out == 0 {
            return Ok(Value::StringArray(
                StringArray::new(Vec::new(), out_dims).map_err(|e| format!("Slice error: {e}"))?,
            ));
        }
        let mut out_data: Vec<String> = Vec::with_capacity(total_out);
        let mut idx = vec![0usize; dims];
        loop {
            let current: Vec<usize> = (0..dims).map(|d| per_dim_indices[d][idx[d]]).collect();
            let mut lin = 0usize;
            for d in 0..dims {
                let i0 = current[d] - 1;
                lin += i0 * strides[d];
            }
            out_data.push(sa.data[lin].clone());
            let mut d = 0usize;
            while d < dims {
                idx[d] += 1;
                if idx[d] < per_dim_indices[d].len() {
                    break;
                }
                idx[d] = 0;
                d += 1;
            }
            if d == dims {
                break;
            }
        }
        if out_data.len() == 1 {
            Ok(Value::String(out_data[0].clone()))
        } else {
            let out_sa =
                StringArray::new(out_data, out_dims).map_err(|e| format!("Slice error: {e}"))?;
            Ok(Value::StringArray(out_sa))
        }
    }
}

pub fn gather_string_slice(sa: &StringArray, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    if plan.indices.is_empty() {
        let empty = StringArray::new(Vec::new(), plan.output_shape.clone())
            .map_err(|e| format!("Slice error: {e}"))?;
        return Ok(Value::StringArray(empty));
    }
    if plan.indices.len() == 1 {
        let lin = plan.indices[0] as usize;
        let value = sa.data.get(lin).cloned().ok_or_else(|| {
            crate::interpreter::errors::mex(
                "IndexOutOfBounds",
                "Slice error: string index out of bounds",
            )
        })?;
        return Ok(Value::String(value));
    }
    let mut out = Vec::with_capacity(plan.indices.len());
    for &lin in &plan.indices {
        let idx = lin as usize;
        let value = sa.data.get(idx).cloned().ok_or_else(|| {
            crate::interpreter::errors::mex(
                "IndexOutOfBounds",
                "Slice error: string index out of bounds",
            )
        })?;
        out.push(value);
    }
    let out_sa = StringArray::new(out, plan.output_shape.clone())
        .map_err(|e| format!("Slice error: {e}"))?;
    Ok(Value::StringArray(out_sa))
}
