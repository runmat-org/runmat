use crate::indexing::plan::IndexPlan;
use crate::indexing::selectors::SliceSelector;
use crate::interpreter::errors::mex;
use runmat_builtins::{CellArray, ComplexTensor, StringArray, Tensor, Value};
use runmat_runtime::RuntimeError;

pub fn build_subsasgn_paren_cell(numeric: &[Value]) -> Result<Value, RuntimeError> {
    let cell = CellArray::new(numeric.to_vec(), 1, numeric.len())
        .map_err(|e| format!("subsasgn build error: {e}"))?;
    Ok(Value::Cell(cell))
}

pub async fn object_subsasgn_paren(
    base: Value,
    numeric: &[Value],
    rhs: Value,
) -> Result<Value, RuntimeError> {
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

pub enum ComplexRhsView {
    Scalar((f64, f64)),
    Tensor {
        data: Vec<(f64, f64)>,
        shape: Vec<usize>,
        strides: Vec<usize>,
    },
}

pub fn build_complex_rhs_view(
    rhs: &Value,
    selection_lengths: &[usize],
) -> Result<ComplexRhsView, RuntimeError> {
    match rhs {
        Value::Complex(re, im) => Ok(ComplexRhsView::Scalar((*re, *im))),
        Value::Num(n) => Ok(ComplexRhsView::Scalar((*n, 0.0))),
        Value::ComplexTensor(rt) => {
            let dims = selection_lengths.len();
            let mut shape = rt.shape.clone();
            if shape.len() < dims {
                shape.resize(dims, 1);
            }
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
            Ok(ComplexRhsView::Tensor {
                data: rt.data.clone(),
                shape,
                strides: rstrides,
            })
        }
        _ => Err("rhs must be numeric or tensor".to_string().into()),
    }
}

pub fn scatter_complex_with_plan(
    t: &mut ComplexTensor,
    plan: &IndexPlan,
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
            ComplexRhsView::Tensor {
                data,
                shape,
                strides,
            } => {
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

pub fn build_string_rhs_view(
    rhs: &Value,
    selection_lengths: &[usize],
) -> Result<StringRhsView, RuntimeError> {
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
        if shape.len() < dims {
            shape.resize(dims, 1);
        }
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
        return Ok(StringRhsView::Tensor {
            data: rt.data.clone(),
            shape,
            strides: rstrides,
        });
    }
    Err("rhs must be string or string array".to_string().into())
}

pub fn scatter_string_with_plan(
    sa: &mut StringArray,
    plan: &IndexPlan,
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
            StringRhsView::Tensor {
                data,
                shape,
                strides,
            } => {
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

pub async fn materialize_rhs_real_for_plan(
    rhs: &Value,
    plan: &IndexPlan,
) -> Result<Vec<f64>, RuntimeError> {
    if plan.dims == 1 {
        let count = plan.selection_lengths.first().copied().unwrap_or(0);
        materialize_rhs_linear_real(rhs, count).await
    } else {
        materialize_rhs_nd_real(rhs, &plan.selection_lengths).await
    }
}

pub fn scatter_real_with_plan(
    t: &mut Tensor,
    plan: &IndexPlan,
    rhs_values: &[f64],
) -> Result<(), RuntimeError> {
    if rhs_values.len() != plan.indices.len() {
        return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
    }
    for (&dst, &value) in plan.indices.iter().zip(rhs_values.iter()) {
        t.data[dst as usize] = value;
    }
    Ok(())
}

pub async fn assign_tensor_with_plan(
    mut t: Tensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if plan.indices.is_empty() {
        return Ok(Value::Tensor(t));
    }
    let rhs_values = materialize_rhs_real_for_plan(rhs, plan).await?;
    scatter_real_with_plan(&mut t, plan, &rhs_values)?;
    Ok(Value::Tensor(t))
}

pub async fn assign_gpu_slice_with_plan(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if plan.indices.is_empty() {
        return Ok(Value::GpuTensor(handle.clone()));
    }
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        mex(
            "AccelerationProviderUnavailable",
            "No acceleration provider registered",
        )
    })?;
    if let Value::GpuTensor(vh) = rhs {
        let rows = plan.base_shape.first().copied().unwrap_or(1);
        let cols = plan.base_shape.get(1).copied().unwrap_or(1);
        if let Some(col) = plan.properties.full_column {
            if col < cols {
                let v_rows = match vh.shape.len() {
                    1 | 2 => vh.shape[0],
                    _ => 0,
                };
                if v_rows == rows {
                    if let Ok(new_h) = provider.scatter_column(handle, col, vh) {
                        return Ok(Value::GpuTensor(new_h));
                    }
                }
            }
        }
        if let Some(row) = plan.properties.full_row {
            if row < rows {
                let v_cols = match vh.shape.len() {
                    1 => vh.shape[0],
                    2 => vh.shape[1],
                    _ => 0,
                };
                if v_cols == cols {
                    if let Ok(new_h) = provider.scatter_row(handle, row, vh) {
                        return Ok(Value::GpuTensor(new_h));
                    }
                }
            }
        }
    }
    let rhs_values = materialize_rhs_real_for_plan(rhs, plan).await?;
    let value_shape = vec![rhs_values.len().max(1), 1];
    let upload_result = if rhs_values.is_empty() {
        provider.zeros(&[0, 1])
    } else {
        provider.upload(&runmat_accelerate_api::HostTensorView {
            data: &rhs_values,
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

    let host = provider
        .download(handle)
        .await
        .map_err(|e| format!("gather for slice assign: {e}"))?;
    let mut t = Tensor::new(host.data, host.shape).map_err(|e| format!("slice assign: {e}"))?;
    scatter_real_with_plan(&mut t, plan, &rhs_values)?;
    upload_tensor_to_gpu(&t)
}

pub async fn materialize_rhs_linear_real(
    rhs: &Value,
    count: usize,
) -> Result<Vec<f64>, RuntimeError> {
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
                Ok(la
                    .data
                    .into_iter()
                    .map(|b| if b != 0 { 1.0 } else { 0.0 })
                    .collect())
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
        Tensor {
            data: Vec<f64>,
            shape: Vec<usize>,
            strides: Vec<usize>,
        },
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
                != shape
                    .iter()
                    .copied()
                    .fold(1usize, |acc, len| acc.saturating_mul(len.max(1)))
            {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
            RhsView::Tensor {
                data: t.data,
                shape,
                strides,
            }
        }
        Value::LogicalArray(la) => {
            if la.shape.len() > selection_lengths.len()
                && la
                    .shape
                    .iter()
                    .skip(selection_lengths.len())
                    .any(|&s| s != 1)
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
                != shape
                    .iter()
                    .copied()
                    .fold(1usize, |acc, len| acc.saturating_mul(len.max(1)))
            {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
            let data: Vec<f64> = la
                .data
                .into_iter()
                .map(|b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            RhsView::Tensor {
                data,
                shape,
                strides,
            }
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
            RhsView::Tensor {
                data,
                shape,
                strides,
            } => {
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
            if idx[d] < selection_lengths[d].max(1) {
                break;
            }
            idx[d] = 0;
            d += 1;
        }
        if d == idx.len() {
            break;
        }
    }
    Ok(out)
}

pub fn upload_tensor_to_gpu(t: &Tensor) -> Result<Value, RuntimeError> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        mex(
            "AccelerationProviderUnavailable",
            "No acceleration provider registered",
        )
    })?;
    let view = runmat_accelerate_api::HostTensorView {
        data: &t.data,
        shape: &t.shape,
    };
    let new_h = provider
        .upload(&view)
        .map_err(|e| format!("reupload after slice assign: {e}"))?;
    Ok(Value::GpuTensor(new_h))
}

pub struct ExprSelectorSpec<'a> {
    pub dims: usize,
    pub colon_mask: u32,
    pub end_mask: u32,
    pub range_dims: &'a [usize],
    pub range_params: &'a [(f64, f64)],
    pub range_start_exprs: &'a [Option<crate::bytecode::EndExpr>],
    pub range_step_exprs: &'a [Option<crate::bytecode::EndExpr>],
    pub range_end_exprs: &'a [crate::bytecode::EndExpr],
    pub numeric: &'a [Value],
    pub shape: &'a [usize],
}

pub async fn build_expr_selectors<ResolveEnd, Fut>(
    spec: ExprSelectorSpec<'_>,
    mut resolve_end: ResolveEnd,
) -> Result<Vec<SliceSelector>, RuntimeError>
where
    ResolveEnd: FnMut(usize, &crate::bytecode::EndExpr) -> Fut,
    Fut: std::future::Future<Output = Result<i64, RuntimeError>>,
{
    let mut selectors: Vec<SliceSelector> = Vec::with_capacity(spec.dims);
    let mut num_iter = 0usize;
    let mut rp_iter = 0usize;
    for d in 0..spec.dims {
        if let Some(pos) = spec.range_dims.iter().position(|&rd| rd == d) {
            let (raw_st, raw_sp) = spec.range_params[rp_iter];
            let dim_len = *spec.shape.get(d).unwrap_or(&1);
            let st = if let Some(expr) = &spec.range_start_exprs[rp_iter] {
                resolve_end(dim_len, expr).await? as f64
            } else {
                raw_st
            };
            let sp = if let Some(expr) = &spec.range_step_exprs[rp_iter] {
                resolve_end(dim_len, expr).await? as f64
            } else {
                raw_sp
            };
            rp_iter += 1;
            let step_i = if sp >= 0.0 {
                sp as i64
            } else {
                -(sp.abs() as i64)
            };
            let end_i = resolve_end(dim_len, &spec.range_end_exprs[pos]).await?;
            if step_i == 0 {
                return Err(mex("IndexStepZero", "Index step cannot be zero"));
            }
            let mut vals = Vec::new();
            let mut cur = st as i64;
            if step_i > 0 {
                while cur <= end_i {
                    if cur < 1 || cur > dim_len as i64 {
                        break;
                    }
                    vals.push(cur as usize);
                    cur += step_i;
                }
            } else {
                while cur >= end_i {
                    if cur < 1 || cur > dim_len as i64 {
                        break;
                    }
                    vals.push(cur as usize);
                    cur += step_i;
                }
            }
            selectors.push(SliceSelector::Indices(vals));
            continue;
        }
        let is_colon = (spec.colon_mask & (1u32 << d)) != 0;
        let is_end = (spec.end_mask & (1u32 << d)) != 0;
        if is_colon {
            selectors.push(SliceSelector::Colon);
        } else if is_end {
            selectors.push(SliceSelector::Scalar(*spec.shape.get(d).unwrap_or(&1)));
        } else {
            let v = spec
                .numeric
                .get(num_iter)
                .ok_or_else(|| mex("MissingNumericIndex", "missing numeric index"))?;
            num_iter += 1;
            let dim_len = *spec.shape.get(d).unwrap_or(&1);
            selectors.push(
                match crate::indexing::selectors::selector_from_value_dim(v, dim_len).await? {
                    SliceSelector::LinearIndices { values, .. } => SliceSelector::Indices(values),
                    other => other,
                },
            );
        }
    }
    Ok(selectors)
}
