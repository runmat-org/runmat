use crate::interpreter::errors::mex;
use crate::indexing::selectors::{SlicePlan, SliceSelector, indices_from_value_linear};
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
