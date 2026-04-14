use crate::interpreter::errors::mex;
use crate::indexing::selectors::{
    build_slice_plan, build_slice_selectors, SlicePlan, SliceSelector, indices_from_value_linear,
};
use runmat_builtins::{CellArray, ComplexTensor, StringArray, Tensor, Value};
use runmat_runtime::RuntimeError;

pub fn build_subsasgn_paren_cell(numeric: &[Value]) -> Result<Value, RuntimeError> {
    let cell = CellArray::new(numeric.to_vec(), 1, numeric.len())
        .map_err(|e| format!("subsasgn build error: {e}"))?;
    Ok(Value::Cell(cell))
}

pub async fn object_subsasgn_paren(base: Value, numeric: &[Value], rhs: Value) -> Result<Value, RuntimeError> {
    let cell = build_subsasgn_paren_cell(numeric)?;
    match base {
        Value::Object(obj) => {
            let args = vec![
                Value::Object(obj),
                Value::String("subsasgn".to_string()),
                Value::String("()".to_string()),
                cell,
                rhs,
            ];
            runmat_runtime::call_builtin_async("call_method", &args).await
        }
        Value::HandleObject(handle) => {
            let args = vec![
                Value::HandleObject(handle),
                Value::String("subsasgn".to_string()),
                Value::String("()".to_string()),
                cell,
                rhs,
            ];
            runmat_runtime::call_builtin_async("call_method", &args).await
        }
        other => Err(format!("slice subsasgn requires object/handle, got {other:?}").into()),
    }
}

pub async fn assign_tensor_slice_1d(
    mut t: Tensor,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    rhs: Value,
) -> Result<Value, RuntimeError> {
    let total = t.data.len();
    let is_colon = (colon_mask & 1u32) != 0;
    let is_end = (end_mask & 1u32) != 0;
    let lin_indices: Vec<usize> = if is_colon {
        (1..=total).collect()
    } else if is_end {
        vec![total]
    } else {
        let v = numeric
            .first()
            .ok_or_else(|| mex("MissingNumericIndex", "missing numeric index"))?;
        indices_from_value_linear(v, total).await?
    };
    match rhs {
        Value::Num(v) => {
            for &li in &lin_indices {
                t.data[li - 1] = v;
            }
        }
        Value::Tensor(rt) => {
            if rt.data.len() == 1 {
                let v = rt.data[0];
                for &li in &lin_indices {
                    t.data[li - 1] = v;
                }
            } else if rt.data.len() == lin_indices.len() {
                for (k, &li) in lin_indices.iter().enumerate() {
                    t.data[li - 1] = rt.data[k];
                }
            } else {
                return Err("shape mismatch for linear slice assign".to_string().into());
            }
        }
        _ => return Err("rhs must be numeric or tensor".to_string().into()),
    }
    Ok(Value::Tensor(t))
}

pub fn assign_tensor_slice_nd(
    mut t: Tensor,
    dims: usize,
    selectors: &[SliceSelector],
    rhs: Value,
) -> Result<Value, RuntimeError> {
    let rank = t.shape.len();
    if dims == 2 {
        let rows = if rank >= 1 { t.shape[0] } else { 1 };
        let cols = if rank >= 2 { t.shape[1] } else { 1 };
        match (&selectors[0], &selectors[1]) {
            (SliceSelector::Colon, SliceSelector::Scalar(j)) => {
                let j0 = *j - 1;
                if j0 >= cols {
                    let new_cols = j0 + 1;
                    let new_rows = rows;
                    let mut new_data = vec![0.0f64; new_rows * new_cols];
                    for c in 0..cols {
                        let src_off = c * rows;
                        let dst_off = c * new_rows;
                        new_data[dst_off..dst_off + rows]
                            .copy_from_slice(&t.data[src_off..src_off + rows]);
                    }
                    t.data = new_data;
                    t.shape = vec![new_rows, new_cols];
                    t.rows = new_rows;
                    t.cols = new_cols;
                }
                let start = j0 * rows;
                match rhs {
                    Value::Num(v) => {
                        for r in 0..rows {
                            t.data[start + r] = v;
                        }
                    }
                    Value::Tensor(rt) => {
                        let len = rt.data.len();
                        if len == rows {
                            for r in 0..rows {
                                t.data[start + r] = rt.data[r];
                            }
                        } else if len == 1 {
                            for r in 0..rows {
                                t.data[start + r] = rt.data[0];
                            }
                        } else {
                            return Err("shape mismatch for slice assign".to_string().into());
                        }
                    }
                    _ => return Err("rhs must be numeric or tensor".to_string().into()),
                }
                return Ok(Value::Tensor(t));
            }
            (SliceSelector::Scalar(i), SliceSelector::Colon) => {
                let i0 = *i - 1;
                if i0 >= rows {
                    let new_rows = i0 + 1;
                    let new_cols = cols;
                    let mut new_data = vec![0.0f64; new_rows * new_cols];
                    for c in 0..cols {
                        for r in 0..rows {
                            new_data[r + c * new_rows] = t.data[r + c * rows];
                        }
                    }
                    t.data = new_data;
                    t.shape = vec![new_rows, new_cols];
                    t.rows = new_rows;
                    t.cols = new_cols;
                }
                match rhs {
                    Value::Num(v) => {
                        for c in 0..cols {
                            t.data[i0 + c * rows] = v;
                        }
                    }
                    Value::Tensor(rt) => {
                        let len = rt.data.len();
                        if len == cols {
                            for c in 0..cols {
                                t.data[i0 + c * rows] = rt.data[c];
                            }
                        } else if len == 1 {
                            for c in 0..cols {
                                t.data[i0 + c * rows] = rt.data[0];
                            }
                        } else {
                            return Err("shape mismatch for slice assign".to_string().into());
                        }
                    }
                    _ => return Err("rhs must be numeric or tensor".to_string().into()),
                }
                return Ok(Value::Tensor(t));
            }
            _ => {}
        }
    }

    let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
    let full_shape: Vec<usize> = if rank < dims {
        let mut s = t.shape.clone();
        s.resize(dims, 1);
        s
    } else {
        t.shape.clone()
    };
    for d in 0..dims {
        let dim_len = full_shape[d];
        let idxs = match &selectors[d] {
            SliceSelector::Colon => (1..=dim_len).collect(),
            SliceSelector::Scalar(i) => vec![*i],
            SliceSelector::Indices(v) => v.clone(),
            SliceSelector::LinearIndices { values: v, .. } => v.clone(),
        };
        if idxs.iter().any(|&i| i == 0 || i > dim_len) {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        per_dim_indices.push(idxs);
    }
    let mut strides: Vec<usize> = vec![0; dims];
    let mut acc = 1usize;
    for d in 0..dims {
        strides[d] = acc;
        acc *= full_shape[d];
    }
    let total_out: usize = per_dim_indices.iter().map(|v| v.len()).product();
    enum RhsView {
        Scalar(f64),
        Tensor { data: Vec<f64>, shape: Vec<usize>, strides: Vec<usize> },
    }
    let rhs_view = match rhs {
        Value::Num(n) => RhsView::Scalar(n),
        Value::Tensor(rt) => {
            let mut shape = rt.shape.clone();
            if shape.len() < dims { shape.resize(dims, 1); }
            if shape.len() > dims {
                if shape.iter().skip(dims).any(|&s| s != 1) {
                    return Err("shape mismatch for slice assign".to_string().into());
                }
                shape.truncate(dims);
            }
            for d in 0..dims {
                let out_len = per_dim_indices[d].len();
                let rhs_len = shape[d];
                if !(rhs_len == 1 || rhs_len == out_len) {
                    return Err("shape mismatch for slice assign".to_string().into());
                }
            }
            let mut rstrides = vec![0usize; dims];
            let mut racc = 1usize;
            for d in 0..dims {
                rstrides[d] = racc;
                racc *= shape[d];
            }
            RhsView::Tensor { data: rt.data, shape, strides: rstrides }
        }
        _ => return Err("rhs must be numeric or tensor".to_string().into()),
    };
    let mut idx = vec![0usize; dims];
    if total_out == 0 {
        return Ok(Value::Tensor(t));
    }
    loop {
        let mut lin = 0usize;
        for d in 0..dims {
            let i0 = per_dim_indices[d][idx[d]] - 1;
            lin += i0 * strides[d];
        }
        match &rhs_view {
            RhsView::Scalar(val) => t.data[lin] = *val,
            RhsView::Tensor { data, shape, strides } => {
                let mut rlin = 0usize;
                for d in 0..dims {
                    let rhs_len = shape[d];
                    let pos = if rhs_len == 1 { 0 } else { idx[d] };
                    rlin += pos * strides[d];
                }
                t.data[lin] = data[rlin];
            }
        }
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
    Ok(Value::Tensor(t))
}

pub enum ComplexRhsView {
    Scalar((f64, f64)),
    Tensor {
        data: Vec<(f64, f64)>,
        shape: Vec<usize>,
        strides: Vec<usize>,
    },
}

pub fn build_complex_rhs_view(rhs: &Value, selection_lengths: &[usize]) -> Result<ComplexRhsView, RuntimeError> {
    match rhs {
        Value::Complex(re, im) => Ok(ComplexRhsView::Scalar((*re, *im))),
        Value::Num(n) => Ok(ComplexRhsView::Scalar((*n, 0.0))),
        Value::ComplexTensor(rt) => {
            let dims = selection_lengths.len();
            let mut shape = rt.shape.clone();
            if shape.len() < dims { shape.resize(dims, 1); }
            if shape.len() > dims {
                if shape.iter().skip(dims).any(|&s| s != 1) {
                    return Err("shape mismatch for slice assign".to_string().into());
                }
                shape.truncate(dims);
            }
            for d in 0..dims {
                let out_len = selection_lengths[d];
                let rhs_len = shape[d];
                if !(rhs_len == 1 || rhs_len == out_len) {
                    return Err("shape mismatch for slice assign".to_string().into());
                }
            }
            let mut rstrides = vec![0usize; dims];
            let mut racc = 1usize;
            for d in 0..dims {
                rstrides[d] = racc;
                racc *= shape[d];
            }
            Ok(ComplexRhsView::Tensor { data: rt.data.clone(), shape, strides: rstrides })
        }
        _ => Err("rhs must be numeric or tensor".to_string().into()),
    }
}

pub fn scatter_complex_with_plan(
    t: &mut ComplexTensor,
    plan: &SlicePlan,
    rhs_view: &ComplexRhsView,
) -> Result<(), RuntimeError> {
    let dims = plan.dims;
    let mut idx = vec![0usize; dims];
    if plan.indices.is_empty() {
        return Ok(());
    }
    let selection_lengths = if plan.selection_lengths.is_empty() {
        plan.output_shape.clone()
    } else {
        plan.selection_lengths.clone()
    };
    loop {
        let mut rlin = 0usize;
        match rhs_view {
            ComplexRhsView::Scalar(val) => {
                let pos = plan.indices[rlin] as usize;
                t.data[pos] = *val;
            }
            ComplexRhsView::Tensor { data, shape, strides } => {
                for d in 0..dims {
                    let rhs_len = shape[d];
                    let pos = if rhs_len == 1 { 0 } else { idx[d] };
                    rlin += pos * strides[d];
                }
                let lin_pos = {
                    let mut p = 0usize;
                    let mut mul = 1usize;
                    for d in 0..dims {
                        p += idx[d] * mul;
                        mul *= selection_lengths[d].max(1);
                    }
                    p
                };
                let dst = plan.indices[lin_pos] as usize;
                t.data[dst] = data[rlin];
            }
        }
        let mut d = 0usize;
        while d < dims {
            idx[d] += 1;
            if idx[d] < selection_lengths[d].max(1) {
                break;
            }
            idx[d] = 0;
            d += 1;
        }
        if d == dims {
            break;
        }
    }
    Ok(())
}

pub enum StringRhsView {
    Scalar(String),
    Tensor {
        data: Vec<String>,
        shape: Vec<usize>,
        strides: Vec<usize>,
    },
}

pub fn build_string_rhs_view(rhs: &Value, selection_lengths: &[usize]) -> Result<StringRhsView, RuntimeError> {
    let scalar = match rhs {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(ca) => Some(ca.to_string()),
        _ => None,
    };
    if let Some(s) = scalar {
        return Ok(StringRhsView::Scalar(s));
    }
    if let Value::StringArray(rt) = rhs {
        let dims = selection_lengths.len();
        let mut shape = rt.shape.clone();
        if shape.len() < dims { shape.resize(dims, 1); }
        if shape.len() > dims {
            if shape.iter().skip(dims).any(|&s| s != 1) {
                return Err("shape mismatch for slice assign".to_string().into());
            }
            shape.truncate(dims);
        }
        for d in 0..dims {
            let out_len = selection_lengths[d];
            let rhs_len = shape[d];
            if !(rhs_len == 1 || rhs_len == out_len) {
                return Err("shape mismatch for slice assign".to_string().into());
            }
        }
        let mut rstrides = vec![0usize; dims];
        let mut racc = 1usize;
        for d in 0..dims {
            rstrides[d] = racc;
            racc *= shape[d];
        }
        return Ok(StringRhsView::Tensor { data: rt.data.clone(), shape, strides: rstrides });
    }
    Err("rhs must be string or string array".to_string().into())
}

pub fn scatter_string_with_plan(
    sa: &mut StringArray,
    plan: &SlicePlan,
    rhs_view: &StringRhsView,
) -> Result<(), RuntimeError> {
    let dims = plan.dims;
    let mut idx = vec![0usize; dims];
    if plan.indices.is_empty() {
        return Ok(());
    }
    let selection_lengths = if plan.selection_lengths.is_empty() {
        plan.output_shape.clone()
    } else {
        plan.selection_lengths.clone()
    };
    loop {
        match rhs_view {
            StringRhsView::Scalar(val) => {
                let lin_pos = {
                    let mut p = 0usize;
                    let mut mul = 1usize;
                    for d in 0..dims {
                        p += idx[d] * mul;
                        mul *= selection_lengths[d].max(1);
                    }
                    p
                };
                let dst = plan.indices[lin_pos] as usize;
                sa.data[dst] = val.clone();
            }
            StringRhsView::Tensor { data, shape, strides } => {
                let mut rlin = 0usize;
                for d in 0..dims {
                    let rhs_len = shape[d];
                    let pos = if rhs_len == 1 { 0 } else { idx[d] };
                    rlin += pos * strides[d];
                }
                let lin_pos = {
                    let mut p = 0usize;
                    let mut mul = 1usize;
                    for d in 0..dims {
                        p += idx[d] * mul;
                        mul *= selection_lengths[d].max(1);
                    }
                    p
                };
                let dst = plan.indices[lin_pos] as usize;
                sa.data[dst] = data[rlin].clone();
            }
        }
        let mut d = 0usize;
        while d < dims {
            idx[d] += 1;
            if idx[d] < selection_lengths[d].max(1) {
                break;
            }
            idx[d] = 0;
            d += 1;
        }
        if d == dims {
            break;
        }
    }
    Ok(())
}

pub async fn materialize_rhs_linear_real(rhs: &Value, count: usize) -> Result<Vec<f64>, RuntimeError> {
    let host_rhs = runmat_runtime::dispatcher::gather_if_needed_async(rhs).await?;
    match host_rhs {
        Value::Num(n) => Ok(vec![n; count]),
        Value::Int(int_val) => Ok(vec![int_val.to_f64(); count]),
        Value::Bool(b) => Ok(vec![if b { 1.0 } else { 0.0 }; count]),
        Value::Tensor(t) => {
            if t.data.len() == count {
                Ok(t.data)
            } else if t.data.len() == 1 {
                Ok(vec![t.data[0]; count])
            } else {
                Err(mex("ShapeMismatch", "shape mismatch for slice assign"))
            }
        }
        Value::LogicalArray(la) => {
            if la.data.len() == count {
                Ok(la.data.into_iter().map(|b| if b != 0 { 1.0 } else { 0.0 }).collect())
            } else if la.data.len() == 1 {
                let val = if la.data[0] != 0 { 1.0 } else { 0.0 };
                Ok(vec![val; count])
            } else {
                Err(mex("ShapeMismatch", "shape mismatch for slice assign"))
            }
        }
        other => Err(mex(
            "InvalidSliceAssignmentRhs",
            &format!("slice assign: unsupported RHS type {:?}", other),
        )),
    }
}

pub async fn materialize_rhs_nd_real(
    rhs: &Value,
    selection_lengths: &[usize],
) -> Result<Vec<f64>, RuntimeError> {
    let rhs_host = runmat_runtime::dispatcher::gather_if_needed_async(rhs).await?;
    enum RhsView {
        Scalar(f64),
        Tensor { data: Vec<f64>, shape: Vec<usize>, strides: Vec<usize> },
    }
    let view = match rhs_host {
        Value::Num(n) => RhsView::Scalar(n),
        Value::Int(iv) => RhsView::Scalar(iv.to_f64()),
        Value::Bool(b) => RhsView::Scalar(if b { 1.0 } else { 0.0 }),
        Value::Tensor(t) => {
            let mut shape = t.shape.clone();
            if shape.len() < selection_lengths.len() {
                shape.resize(selection_lengths.len(), 1);
            }
            if shape.len() > selection_lengths.len() {
                if shape.iter().skip(selection_lengths.len()).any(|&s| s != 1) {
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
                }
                shape.truncate(selection_lengths.len());
            }
            for (dim_len, &sel_len) in shape.iter().zip(selection_lengths.iter()) {
                if *dim_len != 1 && *dim_len != sel_len {
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
                }
            }
            let mut strides = vec![1usize; selection_lengths.len()];
            for d in 1..selection_lengths.len() {
                strides[d] = strides[d - 1] * shape[d - 1].max(1);
            }
            if t.data.len()
                != shape.iter().copied().fold(1usize, |acc, len| acc.saturating_mul(len.max(1)))
            {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
            RhsView::Tensor { data: t.data, shape, strides }
        }
        Value::LogicalArray(la) => {
            if la.shape.len() > selection_lengths.len()
                && la.shape.iter().skip(selection_lengths.len()).any(|&s| s != 1)
            {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
            let mut shape = la.shape.clone();
            if shape.len() < selection_lengths.len() {
                shape.resize(selection_lengths.len(), 1);
            } else {
                shape.truncate(selection_lengths.len());
            }
            for (dim_len, &sel_len) in shape.iter().zip(selection_lengths.iter()) {
                if *dim_len != 1 && *dim_len != sel_len {
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
                }
            }
            let mut strides = vec![1usize; selection_lengths.len()];
            for d in 1..selection_lengths.len() {
                strides[d] = strides[d - 1] * shape[d - 1].max(1);
            }
            if la.data.len()
                != shape.iter().copied().fold(1usize, |acc, len| acc.saturating_mul(len.max(1)))
            {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
            let data: Vec<f64> = la.data.into_iter().map(|b| if b != 0 { 1.0 } else { 0.0 }).collect();
            RhsView::Tensor { data, shape, strides }
        }
        other => {
            return Err(mex(
                "InvalidSliceAssignmentRhs",
                &format!("slice assign: unsupported RHS type {:?}", other),
            ))
        }
    };

    let total = selection_lengths
        .iter()
        .copied()
        .fold(1usize, |acc, len| acc.saturating_mul(len.max(1)));
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; selection_lengths.len()];
    if selection_lengths.is_empty() {
        return Ok(out);
    }
    loop {
        match &view {
            RhsView::Scalar(val) => out.push(*val),
            RhsView::Tensor { data, shape, strides } => {
                let mut rlin = 0usize;
                for d in 0..idx.len() {
                    let rhs_len = shape[d];
                    let pos = if rhs_len == 1 { 0 } else { idx[d] };
                    rlin += pos * strides[d];
                }
                out.push(data.get(rlin).copied().unwrap_or(0.0));
            }
        }
        let mut d = 0usize;
        while d < idx.len() {
            idx[d] += 1;
            if idx[d] < selection_lengths[d].max(1) { break; }
            idx[d] = 0;
            d += 1;
        }
        if d == idx.len() { break; }
    }
    Ok(out)
}

pub fn upload_tensor_to_gpu(t: &Tensor) -> Result<Value, RuntimeError> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        mex("AccelerationProviderUnavailable", "No acceleration provider registered")
    })?;
    let view = runmat_accelerate_api::HostTensorView { data: &t.data, shape: &t.shape };
    let new_h = provider.upload(&view).map_err(|e| format!("reupload after slice assign: {e}"))?;
    Ok(Value::GpuTensor(new_h))
}

pub async fn assign_gpu_store_slice(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    rhs: Value,
) -> Result<Value, RuntimeError> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        mex("AccelerationProviderUnavailable", "No acceleration provider registered")
    })?;
    let base_shape = handle.shape.clone();
    let selectors = build_slice_selectors(dims, colon_mask, end_mask, numeric, &base_shape).await?;

    if dims == 2 {
        if let (Some(sel0), Some(sel1)) = (selectors.first(), selectors.get(1)) {
            let rows = base_shape.first().copied().unwrap_or(1);
            let cols = base_shape.get(1).copied().unwrap_or(1);
            if let Value::GpuTensor(vh) = &rhs {
                if let (SliceSelector::Colon, SliceSelector::Scalar(j)) = (sel0, sel1) {
                    let j0 = *j - 1;
                    if j0 < cols {
                        let v_rows = match vh.shape.len() {
                            1 | 2 => vh.shape[0],
                            _ => 0,
                        };
                        if v_rows == rows {
                            if let Ok(new_h) = provider.scatter_column(handle, j0, vh) {
                                return Ok(Value::GpuTensor(new_h));
                            }
                        }
                    }
                }
                if let (SliceSelector::Scalar(i), SliceSelector::Colon) = (sel0, sel1) {
                    let i0 = *i - 1;
                    if i0 < rows {
                        let v_cols = match vh.shape.len() {
                            1 => vh.shape[0],
                            2 => vh.shape[1],
                            _ => 0,
                        };
                        if v_cols == cols {
                            if let Ok(new_h) = provider.scatter_row(handle, i0, vh) {
                                return Ok(Value::GpuTensor(new_h));
                            }
                        }
                    }
                }
            }
        }
    }

    if let Ok(plan) = build_slice_plan(&selectors, dims, &base_shape) {
        if plan.indices.is_empty() {
            return Ok(Value::GpuTensor(handle.clone()));
        }
        let values_result = if plan.dims == 1 {
            let count = plan.selection_lengths.first().copied().unwrap_or(0);
            materialize_rhs_linear_real(&rhs, count).await
        } else {
            materialize_rhs_nd_real(&rhs, &plan.selection_lengths).await
        };
        if let Ok(values) = values_result {
            if values.len() == plan.indices.len() {
                let value_shape = vec![values.len().max(1), 1];
                let upload_result = if values.is_empty() {
                    provider.zeros(&[0, 1])
                } else {
                    provider.upload(&runmat_accelerate_api::HostTensorView {
                        data: &values,
                        shape: &value_shape,
                    })
                };
                if let Ok(values_handle) = upload_result {
                    if provider
                        .scatter_linear(handle, &plan.indices, &values_handle)
                        .is_ok()
                    {
                        return Ok(Value::GpuTensor(handle.clone()));
                    }
                }
            }
        }
    }

    let host = provider
        .download(handle)
        .await
        .map_err(|e| format!("gather for slice assign: {e}"))?;
    let t = Tensor::new(host.data, host.shape).map_err(|e| format!("slice assign: {e}"))?;
    let updated = if dims == 1 {
        assign_tensor_slice_1d(t, colon_mask, end_mask, numeric, rhs).await?
    } else {
        assign_tensor_slice_nd(t, dims, &selectors, rhs)?
    };
    let Value::Tensor(updated) = updated else { unreachable!() };
    upload_tensor_to_gpu(&updated)
}
