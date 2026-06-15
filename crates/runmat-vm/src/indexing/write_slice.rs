use crate::indexing::plan::IndexPlan;
use crate::interpreter::errors::mex;
use runmat_builtins::{ComplexTensor, StringArray, Tensor, Value};
use runmat_runtime::RuntimeError;

fn map_slice_shape_error(context: &str, err: impl std::fmt::Display) -> RuntimeError {
    mex("ShapeMismatch", &format!("{context}: {err}"))
}

fn map_acceleration_error(context: &str, err: impl std::fmt::Display) -> RuntimeError {
    mex("AccelerationOperationFailed", &format!("{context}: {err}"))
}

fn is_empty_delete_rhs(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(t)
            if t.data.is_empty() || t.rows == 0 || t.cols == 0
    ) || matches!(
        value,
        Value::ComplexTensor(t)
            if t.data.is_empty() || t.rows == 0 || t.cols == 0
    )
}

fn deleted_vector_shape(rows: usize, _cols: usize, len: usize) -> Vec<usize> {
    if len == 0 {
        vec![0, 0]
    } else if rows == 1 {
        vec![1, len]
    } else {
        vec![len, 1]
    }
}

fn sorted_unique_positions_desc(
    plan: &IndexPlan,
    total: usize,
) -> Result<Vec<usize>, RuntimeError> {
    let mut positions = Vec::with_capacity(plan.indices.len());
    for &idx in &plan.indices {
        let pos = idx as usize;
        if pos >= total {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        positions.push(pos);
    }
    positions.sort_unstable();
    positions.dedup();
    positions.reverse();
    Ok(positions)
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
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
                }
                shape.truncate(dims);
            }
            for d in 0..dims {
                let out_len = selection_lengths[d];
                let rhs_len = shape[d];
                if !(rhs_len == 1 || rhs_len == out_len) {
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
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
        _ => Err(mex(
            "InvalidSliceAssignmentRhs",
            "rhs must be numeric or tensor",
        )),
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
                t.data[dst] = *val;
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
        if dims == 1 && shape.iter().filter(|&&dim| dim != 1).count() <= 1 {
            shape = vec![rt.data.len()];
        } else if shape.len() < dims {
            shape.resize(dims, 1);
        }
        if shape.len() > dims {
            if shape.iter().skip(dims).any(|&s| s != 1) {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
            shape.truncate(dims);
        }
        for d in 0..dims {
            let out_len = selection_lengths[d];
            let rhs_len = shape[d];
            if !(rhs_len == 1 || rhs_len == out_len) {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
        }
        let expected = shape
            .iter()
            .try_fold(1usize, |acc, &len| acc.checked_mul(len));
        if expected != Some(rt.data.len()) {
            return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
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
    if let Value::Cell(cell) = rhs {
        let dims = selection_lengths.len();
        let mut data = Vec::with_capacity(cell.data.len());
        for handle in &cell.data {
            let value = unsafe { &*handle.as_raw() };
            match value {
                Value::String(text) => data.push(text.clone()),
                Value::CharArray(chars) => data.push(chars.to_string()),
                Value::StringArray(strings) if strings.data.len() == 1 => {
                    data.push(strings.data[0].clone())
                }
                other => {
                    return Err(mex(
                        "InvalidSliceAssignmentRhs",
                        &format!(
                            "rhs cell elements must be string scalars or character vectors, got {other:?}"
                        ),
                    ))
                }
            }
        }
        let mut shape = cell.shape.clone();
        if dims == 1 && shape.iter().filter(|&&dim| dim != 1).count() <= 1 {
            shape = vec![cell.data.len()];
        } else if shape.len() < dims {
            shape.resize(dims, 1);
        }
        if shape.len() > dims {
            if shape.iter().skip(dims).any(|&s| s != 1) {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
            shape.truncate(dims);
        }
        for d in 0..dims {
            let out_len = selection_lengths[d];
            let rhs_len = shape[d];
            if !(rhs_len == 1 || rhs_len == out_len) {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
        }
        let expected = shape
            .iter()
            .try_fold(1usize, |acc, &len| acc.checked_mul(len));
        if expected != Some(data.len()) {
            return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
        }
        let mut rstrides = vec![0usize; dims];
        let mut racc = 1usize;
        for d in 0..dims {
            rstrides[d] = racc;
            racc *= shape[d];
        }
        return Ok(StringRhsView::Tensor {
            data,
            shape,
            strides: rstrides,
        });
    }
    Err(mex(
        "InvalidSliceAssignmentRhs",
        "rhs must be string, string array, or cellstr",
    ))
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
    if matches!(rhs, Value::Complex(_, _) | Value::ComplexTensor(_)) {
        let mut ct = ComplexTensor {
            data: t.data.into_iter().map(|re| (re, 0.0)).collect(),
            shape: t.shape,
            rows: t.rows,
            cols: t.cols,
        };
        let rhs_view = build_complex_rhs_view(rhs, &plan.selection_lengths)?;
        scatter_complex_with_plan(&mut ct, plan, &rhs_view)?;
        return Ok(Value::ComplexTensor(ct));
    }
    let rhs_values = materialize_rhs_real_for_plan(rhs, plan).await?;
    scatter_real_with_plan(&mut t, plan, &rhs_values)?;
    Ok(Value::Tensor(t))
}

pub fn delete_tensor_with_plan(
    mut t: Tensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if !is_empty_delete_rhs(rhs) {
        return Err(mex(
            "DeletionRequiresEmptyRhs",
            "Indexed deletion requires empty RHS",
        ));
    }
    if plan.indices.is_empty() {
        return Ok(Value::Tensor(t));
    }
    if !(t.rows == 1 || t.cols == 1) {
        return Err(mex(
            "UnsupportedDeletion",
            "Linear deletion is only supported for vectors",
        ));
    }
    let positions = sorted_unique_positions_desc(plan, t.data.len())?;
    for pos in positions {
        t.data.remove(pos);
    }
    let shape = deleted_vector_shape(t.rows, t.cols, t.data.len());
    t.rows = shape.first().copied().unwrap_or(0);
    t.cols = shape.get(1).copied().unwrap_or(0);
    t.shape = shape;
    Ok(Value::Tensor(t))
}

pub fn delete_complex_with_plan(
    mut t: ComplexTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if !is_empty_delete_rhs(rhs) {
        return Err(mex(
            "DeletionRequiresEmptyRhs",
            "Indexed deletion requires empty RHS",
        ));
    }
    if plan.indices.is_empty() {
        return Ok(Value::ComplexTensor(t));
    }
    if !(t.rows == 1 || t.cols == 1) {
        return Err(mex(
            "UnsupportedDeletion",
            "Linear deletion is only supported for vectors",
        ));
    }
    let positions = sorted_unique_positions_desc(plan, t.data.len())?;
    for pos in positions {
        t.data.remove(pos);
    }
    let shape = deleted_vector_shape(t.rows, t.cols, t.data.len());
    t.rows = shape.first().copied().unwrap_or(0);
    t.cols = shape.get(1).copied().unwrap_or(0);
    t.shape = shape;
    Ok(Value::ComplexTensor(t))
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
        .map_err(|e| map_acceleration_error("gather for slice assign", e))?;
    let mut t =
        Tensor::new(host.data, host.shape).map_err(|e| map_slice_shape_error("slice assign", e))?;
    scatter_real_with_plan(&mut t, plan, &rhs_values)?;
    upload_tensor_to_gpu(&t)
}

pub async fn delete_gpu_slice_with_plan(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if !is_empty_delete_rhs(rhs) {
        return Err(mex(
            "DeletionRequiresEmptyRhs",
            "Indexed deletion requires empty RHS",
        ));
    }
    if plan.indices.is_empty() {
        return Ok(Value::GpuTensor(handle.clone()));
    }
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        mex(
            "AccelerationProviderUnavailable",
            "No acceleration provider registered",
        )
    })?;
    let host = provider
        .download(handle)
        .await
        .map_err(|e| map_acceleration_error("gather for slice deletion", e))?;
    let t = Tensor::new(host.data, host.shape)
        .map_err(|e| map_slice_shape_error("slice deletion", e))?;
    let Value::Tensor(updated) = delete_tensor_with_plan(t, plan, rhs)? else {
        unreachable!()
    };
    upload_tensor_to_gpu(&updated)
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
        Value::OutputList(values) => materialize_output_list_real(&values, count),
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
        Value::OutputList(values) => {
            let count = selection_lengths
                .iter()
                .copied()
                .fold(1usize, |acc, len| acc.saturating_mul(len.max(1)));
            let data = materialize_output_list_real(&values, count)?;
            let shape = if selection_lengths.is_empty() {
                vec![1]
            } else {
                selection_lengths.to_vec()
            };
            let mut strides = vec![1usize; shape.len()];
            for d in 1..shape.len() {
                strides[d] = strides[d - 1] * shape[d - 1].max(1);
            }
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

fn materialize_output_list_real(values: &[Value], count: usize) -> Result<Vec<f64>, RuntimeError> {
    if values.len() == count {
        values.iter().map(value_to_real_scalar).collect()
    } else if values.len() == 1 {
        let value = value_to_real_scalar(&values[0])?;
        Ok(vec![value; count])
    } else {
        Err(mex("ShapeMismatch", "shape mismatch for slice assign"))
    }
}

fn value_to_real_scalar(value: &Value) -> Result<f64, RuntimeError> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(int_val) => Ok(int_val.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if t.data.len() == 1 => Ok(t.data[0]),
        _ => f64::try_from(value).map_err(Into::into),
    }
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
        .map_err(|e| map_acceleration_error("reupload after slice assign", e))?;
    Ok(Value::GpuTensor(new_h))
}

#[cfg(test)]
mod tests {
    use super::{build_complex_rhs_view, build_string_rhs_view, map_acceleration_error};
    use runmat_builtins::{CellArray, ComplexTensor, StringArray, Tensor, Value};
    use runmat_gc::GcPtr;

    #[test]
    fn complex_rhs_view_shape_mismatch_reports_identifier() {
        let rhs = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)], vec![1, 3])
                .expect("complex tensor"),
        );
        let err = match build_complex_rhs_view(&rhs, &[2, 2]) {
            Ok(_) => panic!("shape mismatch should fail"),
            Err(err) => err,
        };
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn complex_rhs_view_invalid_rhs_type_reports_identifier() {
        let rhs = Value::String("x".to_string());
        let err = match build_complex_rhs_view(&rhs, &[1]) {
            Ok(_) => panic!("non-numeric rhs should be rejected"),
            Err(err) => err,
        };
        assert_eq!(err.identifier(), Some("RunMat:InvalidSliceAssignmentRhs"));
    }

    #[test]
    fn string_rhs_view_shape_mismatch_reports_identifier() {
        let rhs = Value::StringArray(
            StringArray::new(
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
                vec![1, 3],
            )
            .expect("string array"),
        );
        let err = match build_string_rhs_view(&rhs, &[2, 2]) {
            Ok(_) => panic!("shape mismatch should fail"),
            Err(err) => err,
        };
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn string_cell_rhs_view_rejects_shape_data_length_mismatch() {
        let handles = ["a", "b", "c"]
            .into_iter()
            .map(|text| unsafe {
                // The test leaks boxed values for the duration of the process,
                // which is sufficient for this short-lived GcPtr fixture.
                GcPtr::from_raw(Box::into_raw(Box::new(Value::String(text.to_string()))))
            })
            .collect();
        let rhs = Value::Cell(CellArray {
            data: handles,
            shape: vec![2, 2],
            rows: 2,
            cols: 2,
        });
        let err = match build_string_rhs_view(&rhs, &[2, 2]) {
            Ok(_) => panic!("cell shape/data mismatch should fail"),
            Err(err) => err,
        };
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn string_rhs_view_invalid_rhs_type_reports_identifier() {
        let rhs = Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).expect("tensor"));
        let err = match build_string_rhs_view(&rhs, &[1]) {
            Ok(_) => panic!("non-string rhs should be rejected"),
            Err(err) => err,
        };
        assert_eq!(err.identifier(), Some("RunMat:InvalidSliceAssignmentRhs"));
    }

    #[test]
    fn slice_acceleration_error_mapping_reports_identifier() {
        let err = map_acceleration_error("slice assign", "provider failed");
        assert_eq!(err.identifier(), Some("RunMat:AccelerationOperationFailed"));
    }
}
