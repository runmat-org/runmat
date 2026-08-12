use crate::bytecode::EndExpr;
use crate::indexing::selectors::{
    index_scalar_from_value, logical_indices_linear, materialize_index_value,
    numeric_tensor_indices, SliceSelector,
};
use crate::interpreter::errors::mex;
use runmat_runtime::{builtins::common::shape::is_scalar_shape, RuntimeError};
use runmat_value::Value;
use std::future::Future;

pub type VmResult<T> = Result<T, RuntimeError>;

#[derive(Debug, Clone, Default)]
pub struct IndexPlanProperties {
    pub full_row: Option<usize>,
    pub full_column: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct IndexPlan {
    pub indices: Vec<u32>,
    pub output_shape: Vec<usize>,
    pub selection_lengths: Vec<usize>,
    pub dims: usize,
    pub base_shape: Vec<usize>,
    pub properties: IndexPlanProperties,
}

impl IndexPlan {
    pub fn new(
        indices: Vec<u32>,
        output_shape: Vec<usize>,
        selection_lengths: Vec<usize>,
        dims: usize,
        base_shape: Vec<usize>,
    ) -> Self {
        let properties = derive_plan_properties(&indices, dims, &base_shape);
        Self {
            indices,
            output_shape,
            selection_lengths,
            dims,
            base_shape,
            properties,
        }
    }
}

fn derive_plan_properties(
    indices: &[u32],
    dims: usize,
    base_shape: &[usize],
) -> IndexPlanProperties {
    let mut properties = IndexPlanProperties {
        full_row: None,
        full_column: None,
    };
    if dims != 2 || indices.is_empty() {
        return properties;
    }
    let rows = base_shape.first().copied().unwrap_or(1);
    let cols = base_shape.get(1).copied().unwrap_or(1);
    if indices.len() == rows {
        let first = indices[0] as usize;
        if first.is_multiple_of(rows) {
            let col = first / rows;
            if col < cols
                && indices
                    .iter()
                    .enumerate()
                    .all(|(r, &idx)| idx as usize == col * rows + r)
            {
                properties.full_column = Some(col);
            }
        }
    }
    if indices.len() == cols {
        let first = indices[0] as usize;
        let row = first % rows;
        if row < rows
            && indices
                .iter()
                .enumerate()
                .all(|(c, &idx)| idx as usize == row + c * rows)
        {
            properties.full_row = Some(row);
        }
    }
    properties
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

pub fn total_len_from_shape(shape: &[usize]) -> usize {
    if is_scalar_shape(shape) {
        1
    } else {
        shape.iter().copied().product()
    }
}

fn checked_total_len_from_shape(shape: &[usize]) -> VmResult<usize> {
    if is_scalar_shape(shape) {
        return Ok(1);
    }
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))
    })
}

fn checked_u32_index(index: usize) -> VmResult<u32> {
    u32::try_from(index).map_err(|_| mex("IndexOutOfBounds", "Index exceeds supported range"))
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

fn exact_index_from_f64(value: f64) -> Option<i64> {
    if !value.is_finite() {
        return None;
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return None;
    }
    if rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
        return None;
    }
    Some(rounded as i64)
}

pub fn build_index_plan(
    selectors: &[SliceSelector],
    dims: usize,
    base_shape: &[usize],
) -> VmResult<IndexPlan> {
    let total_len = checked_total_len_from_shape(base_shape)?;
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
        let zero_based: Vec<u32> = indices
            .iter()
            .map(|&i| checked_u32_index(i - 1))
            .collect::<Result<_, _>>()?;
        let count = zero_based.len();
        let base_is_row_vector = base_shape.first().copied().unwrap_or(1) == 1
            && base_shape.get(1).copied().unwrap_or(1) > 1;
        let shape = match list {
            SliceSelector::Colon => vec![count, 1],
            SliceSelector::LinearIndices { output_shape, .. } => output_shape,
            _ if count == 0 => vec![0, 1],
            _ if count <= 1 => vec![1, 1],
            _ if base_is_row_vector => vec![1, count],
            _ => vec![count, 1],
        };
        return Ok(IndexPlan::new(
            zero_based,
            shape,
            vec![count],
            dims,
            base_shape.to_vec(),
        ));
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
        return Ok(IndexPlan::new(
            Vec::new(),
            out_shape,
            selection_lengths,
            dims,
            base_shape.to_vec(),
        ));
    }

    let mut base_norm = base_shape.to_vec();
    if base_norm.len() < dims {
        base_norm.resize(dims, 1);
    }
    let mut strides = vec![1usize; dims];
    for d in 1..dims {
        strides[d] = strides[d - 1]
            .checked_mul(base_norm[d - 1].max(1))
            .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))?;
    }

    let mut indices = Vec::new();
    let mut index_error: Option<RuntimeError> = None;
    cartesian_product(&per_dim_lists, |multi| {
        if index_error.is_some() {
            return;
        }
        let mut lin = 0usize;
        for d in 0..dims {
            let idx = multi[d] - 1;
            let offset = match idx.checked_mul(strides[d]) {
                Some(offset) => offset,
                None => {
                    index_error = Some(mex("IndexOutOfBounds", "Index dimensions overflow"));
                    return;
                }
            };
            lin = match lin.checked_add(offset) {
                Some(sum) => sum,
                None => {
                    index_error = Some(mex("IndexOutOfBounds", "Index dimensions overflow"));
                    return;
                }
            };
        }
        match checked_u32_index(lin) {
            Ok(index) => indices.push(index),
            Err(err) => index_error = Some(err),
        }
    });
    if let Some(err) = index_error {
        return Err(err);
    }

    let total_out: usize = selection_lengths.iter().product();
    if total_out == 1 {
        out_shape = vec![1, 1];
    }
    let selection_lengths = out_shape.clone();
    Ok(IndexPlan::new(
        indices,
        out_shape,
        selection_lengths,
        dims,
        base_shape.to_vec(),
    ))
}

/// Builds a write plan for sparse two-subscript indexed assignment. The plan's
/// base shape is the target shape so CSC updates use the expanded stride, while
/// colon selectors are materialized against the original shape before planning.
pub fn build_sparse_assignment_plan(
    selectors: &[SliceSelector],
    dims: usize,
    base_shape: &[usize],
) -> VmResult<IndexPlan> {
    if dims != 2 {
        return build_index_plan(selectors, dims, base_shape);
    }

    let mut target_shape = base_shape.to_vec();
    target_shape.resize(dims, 1);
    let mut planned_selectors = Vec::with_capacity(dims);
    for (d, target_len) in target_shape.iter_mut().enumerate().take(dims) {
        let original_len = base_shape.get(d).copied().unwrap_or(1);
        let selector = selectors
            .get(d)
            .cloned()
            .unwrap_or(SliceSelector::Indices(Vec::new()));
        let values = match &selector {
            SliceSelector::Colon => (1..=original_len).collect::<Vec<_>>(),
            SliceSelector::Scalar(value) => vec![*value],
            SliceSelector::Indices(values) => values.clone(),
            SliceSelector::LinearIndices { values, .. } => values.clone(),
        };
        if values.contains(&0) {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        if let Some(&max_value) = values.iter().max() {
            *target_len = (*target_len).max(max_value);
        }
        planned_selectors.push(match selector {
            SliceSelector::Colon => SliceSelector::Indices(values),
            other => other,
        });
    }
    build_index_plan(&planned_selectors, dims, &target_shape)
}

#[derive(Clone)]
enum ExprSel {
    Colon,
    Scalar(usize),
    Indices(Vec<usize>),
    Range {
        start: i64,
        step: i64,
        end_off: EndExpr,
    },
}

pub struct ExprPlanSpec<'a> {
    pub dims: usize,
    pub colon_mask: u32,
    pub end_mask: u32,
    pub range_dims: &'a [usize],
    pub range_params: &'a [(f64, f64)],
    pub range_start_exprs: &'a [Option<EndExpr>],
    pub range_step_exprs: &'a [Option<EndExpr>],
    pub range_end_exprs: &'a [EndExpr],
    pub numeric: &'a [Value],
    pub shape: &'a [usize],
}

fn selector_mask_has_dim(mask: u32, dim: usize) -> bool {
    dim < u32::BITS as usize && (mask & (1u32 << dim)) != 0
}

fn validate_expr_range_selector_plan(
    spec: &ExprPlanSpec<'_>,
) -> Result<Vec<Option<usize>>, RuntimeError> {
    let range_len = spec.range_dims.len();
    if spec.range_params.len() != range_len
        || spec.range_start_exprs.len() != range_len
        || spec.range_step_exprs.len() != range_len
        || spec.range_end_exprs.len() != range_len
    {
        return Err(mex(
            "InvalidRangeSelectorPlan",
            "inconsistent range selector metadata",
        ));
    }

    let mut by_dim = vec![None; spec.dims];
    for (pos, &dim) in spec.range_dims.iter().enumerate() {
        if dim >= spec.dims {
            return Err(mex(
                "InvalidRangeSelectorDim",
                "range selector dimension is out of bounds",
            ));
        }
        let conflicts_with_colon = selector_mask_has_dim(spec.colon_mask, dim);
        let conflicts_with_end = selector_mask_has_dim(spec.end_mask, dim);
        if conflicts_with_colon || conflicts_with_end {
            return Err(mex(
                "InvalidRangeSelectorPlan",
                "range selector conflicts with colon/end selector masks",
            ));
        }
        if by_dim[dim].replace(pos).is_some() {
            return Err(mex(
                "InvalidRangeSelectorPlan",
                "range selector dimension appears more than once",
            ));
        }
    }
    Ok(by_dim)
}

pub async fn build_expr_index_plan<ResolveEnd, Fut>(
    spec: ExprPlanSpec<'_>,
    resolve_end: ResolveEnd,
) -> Result<IndexPlan, RuntimeError>
where
    ResolveEnd: FnMut(usize, &EndExpr) -> Fut,
    Fut: Future<Output = Result<f64, RuntimeError>>,
{
    build_expr_index_plan_with_growth(spec, resolve_end, false).await
}

pub async fn build_expr_sparse_assignment_plan<ResolveEnd, Fut>(
    spec: ExprPlanSpec<'_>,
    resolve_end: ResolveEnd,
) -> Result<IndexPlan, RuntimeError>
where
    ResolveEnd: FnMut(usize, &EndExpr) -> Fut,
    Fut: Future<Output = Result<f64, RuntimeError>>,
{
    build_expr_index_plan_with_growth(spec, resolve_end, true).await
}

async fn build_expr_index_plan_with_growth<ResolveEnd, Fut>(
    spec: ExprPlanSpec<'_>,
    mut resolve_end: ResolveEnd,
    allow_sparse_growth: bool,
) -> Result<IndexPlan, RuntimeError>
where
    ResolveEnd: FnMut(usize, &EndExpr) -> Fut,
    Fut: Future<Output = Result<f64, RuntimeError>>,
{
    let allow_sparse_growth = allow_sparse_growth && spec.dims == 2;
    let rank = spec.shape.len();
    let full_shape: Vec<usize> = if spec.dims == 1 {
        vec![checked_total_len_from_shape(spec.shape)?]
    } else if rank < spec.dims {
        let mut s = spec.shape.to_vec();
        s.resize(spec.dims, 1);
        s
    } else {
        spec.shape.to_vec()
    };

    let range_pos_by_dim = validate_expr_range_selector_plan(&spec)?;
    let mut selectors: Vec<ExprSel> = Vec::with_capacity(spec.dims);
    let mut linear_output_shape: Option<Vec<usize>> = None;
    let mut num_iter = 0usize;
    for (d, range_pos) in range_pos_by_dim.iter().enumerate().take(spec.dims) {
        let is_colon = selector_mask_has_dim(spec.colon_mask, d);
        let is_end = selector_mask_has_dim(spec.end_mask, d);
        if is_colon {
            selectors.push(ExprSel::Colon);
        } else if is_end {
            selectors.push(ExprSel::Scalar(*full_shape.get(d).unwrap_or(&1)));
        } else if let Some(pos) = *range_pos {
            let (raw_st, raw_sp) = spec.range_params[pos];
            let dim_len = *full_shape.get(d).unwrap_or(&1);
            let st = if let Some(expr) = &spec.range_start_exprs[pos] {
                resolve_end(dim_len, expr).await? as f64
            } else {
                raw_st
            };
            let sp = if let Some(expr) = &spec.range_step_exprs[pos] {
                resolve_end(dim_len, expr).await? as f64
            } else {
                raw_sp
            };
            let start = exact_index_from_f64(st).ok_or_else(|| {
                mex(
                    "UnsupportedIndexType",
                    "Index values must be positive integers or logical values",
                )
            })?;
            let step = exact_index_from_f64(sp).ok_or_else(|| {
                mex(
                    "UnsupportedIndexType",
                    "Index values must be positive integers or logical values",
                )
            })?;
            let off = spec.range_end_exprs[pos].clone();
            selectors.push(ExprSel::Range {
                start,
                step,
                end_off: off,
            });
        } else {
            let v = spec
                .numeric
                .get(num_iter)
                .ok_or_else(|| mex("MissingNumericIndex", "missing numeric index"))?;
            num_iter += 1;
            if let Some(idx) = index_scalar_from_value(v).await? {
                if idx.is_below_one() {
                    return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                }
                let index = idx
                    .positive_usize()
                    .ok_or_else(|| mex("IndexOutOfBounds", "Index out of bounds"))?;
                selectors.push(ExprSel::Scalar(index));
            } else {
                // Vector selectors may be GPU-backed. Materialize them before
                // inspecting their class so typed integer storage remains the
                // source of truth instead of a floating-point compatibility view.
                let materialized = materialize_index_value(v).await?;
                let v = &materialized;
                match v {
                    Value::Bool(b) => {
                        selectors.push(if *b {
                            ExprSel::Indices(vec![1])
                        } else {
                            ExprSel::Indices(Vec::new())
                        });
                    }
                    Value::LogicalArray(la) => {
                        if la.data.len() == 1 && is_scalar_shape(&la.shape) {
                            selectors.push(if la.data[0] != 0 {
                                ExprSel::Indices(vec![1])
                            } else {
                                ExprSel::Indices(Vec::new())
                            });
                        } else {
                            let dim_len = *full_shape.get(d).unwrap_or(&1);
                            if spec.dims == 1 {
                                let vv = logical_indices_linear(la, dim_len)?;
                                // MATLAB-style linear logical indexing returns a column vector.
                                linear_output_shape = Some(vec![vv.len(), 1]);
                                selectors.push(ExprSel::Indices(vv));
                            } else {
                                if la.data.len() != dim_len {
                                    return Err(mex(
                                        "IndexShape",
                                        "Logical mask length mismatch for dimension",
                                    ));
                                }
                                let vv = la
                                    .data
                                    .iter()
                                    .enumerate()
                                    .filter_map(|(index, &selected)| {
                                        (selected != 0).then_some(index + 1)
                                    })
                                    .collect();
                                selectors.push(ExprSel::Indices(vv));
                            }
                        }
                    }
                    Value::Tensor(idx_t) => {
                        if spec.dims == 1 {
                            linear_output_shape = Some(idx_t.shape.clone());
                        }
                        let vv = numeric_tensor_indices(idx_t, None)?;
                        selectors.push(ExprSel::Indices(vv));
                    }
                    _ => return Err(mex("UnsupportedIndexType", "Unsupported index type")),
                }
            }
        }
    }

    let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(spec.dims);
    let mut selection_lengths: Vec<usize> = Vec::with_capacity(spec.dims);
    let mut scalar_mask: Vec<bool> = Vec::with_capacity(spec.dims);
    let base_is_row_vector = spec.dims == 1
        && spec.shape.first().copied().unwrap_or(1) == 1
        && spec.shape.get(1).copied().unwrap_or(1) > 1;
    let linear_selector_is_colon = matches!(selectors.first(), Some(ExprSel::Colon));
    let linear_selector_is_range = matches!(selectors.first(), Some(ExprSel::Range { .. }));
    for (d, sel) in selectors.iter().enumerate().take(spec.dims) {
        let dim_len = full_shape[d] as i64;
        let idxs: Vec<usize> = match sel {
            ExprSel::Colon => (1..=full_shape[d]).collect(),
            ExprSel::Scalar(i) => vec![*i],
            ExprSel::Indices(v) => v.clone(),
            ExprSel::Range {
                start,
                step,
                end_off,
            } => {
                let mut v = Vec::new();
                let mut cur = *start;
                let stp = *step;
                let end_bound = resolve_end(dim_len as usize, end_off).await?;
                if stp == 0 {
                    return Err(mex("IndexStepZero", "Index step cannot be zero"));
                }
                if !end_bound.is_finite() {
                    return Err(mex(
                        "UnsupportedIndexType",
                        "Index values must be positive integers or logical values",
                    ));
                }
                let end_i = if stp > 0 {
                    end_bound.floor()
                } else {
                    end_bound.ceil()
                };
                if end_i < i64::MIN as f64 || end_i > i64::MAX as f64 {
                    return Err(mex(
                        "UnsupportedIndexType",
                        "Index values must be positive integers or logical values",
                    ));
                }
                let end_i = end_i as i64;
                if stp > 0 {
                    while cur <= end_i {
                        if cur < 1 || (!allow_sparse_growth && cur > dim_len) {
                            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                        }
                        v.push(cur as usize);
                        cur += stp;
                    }
                } else {
                    while cur >= end_i {
                        if cur < 1 || (!allow_sparse_growth && cur > dim_len) {
                            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                        }
                        v.push(cur as usize);
                        cur += stp;
                    }
                }
                v
            }
        };
        if idxs
            .iter()
            .any(|&i| i == 0 || (!allow_sparse_growth && i > full_shape[d]))
        {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        selection_lengths.push(idxs.len());
        per_dim_indices.push(idxs);
        scalar_mask.push(matches!(sel, ExprSel::Scalar(_)));
    }

    let mut planned_shape = full_shape.clone();
    if allow_sparse_growth && spec.dims > 1 {
        for (d, indices) in per_dim_indices.iter().enumerate().take(spec.dims) {
            if let Some(&max_index) = indices.iter().max() {
                planned_shape[d] = planned_shape[d].max(max_index);
            }
        }
    }
    let mut strides: Vec<usize> = vec![0; spec.dims];
    let mut acc = 1usize;
    for (d, stride) in strides.iter_mut().enumerate().take(spec.dims) {
        *stride = acc;
        acc = acc
            .checked_mul(planned_shape[d])
            .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))?;
    }
    let total_out: usize = per_dim_indices.iter().try_fold(1usize, |acc, values| {
        acc.checked_mul(values.len())
            .ok_or_else(|| mex("IndexOutOfBounds", "Index result dimensions overflow"))
    })?;
    if total_out == 0 {
        let output_shape = if spec.dims == 1 {
            if let Some(shape) = linear_output_shape.clone() {
                shape
            } else if linear_selector_is_colon {
                vec![0, 1]
            } else if linear_selector_is_range || base_is_row_vector {
                vec![1, 0]
            } else {
                vec![0, 1]
            }
        } else {
            let mut dims_out: Vec<(usize, usize, bool)> = selection_lengths
                .iter()
                .enumerate()
                .map(|(d, &len)| (d, len, scalar_mask.get(d).copied().unwrap_or(false)))
                .collect();
            while dims_out.len() > 2
                && dims_out
                    .last()
                    .map(|&(_, len, is_scalar)| len == 1 && is_scalar)
                    .unwrap_or(false)
            {
                dims_out.pop();
            }
            if dims_out.is_empty() {
                vec![1, 1]
            } else if dims_out.len() == 1 {
                let (dim, len, _) = dims_out[0];
                if dim == 1 {
                    vec![1, len]
                } else {
                    vec![len, 1]
                }
            } else {
                dims_out.into_iter().map(|(_, len, _)| len).collect()
            }
        };
        return Ok(IndexPlan::new(
            Vec::new(),
            output_shape,
            selection_lengths,
            spec.dims,
            planned_shape,
        ));
    }

    let mut indices: Vec<u32> = Vec::with_capacity(total_out);
    let mut idx = vec![0usize; spec.dims];
    loop {
        let mut lin = 0usize;
        for d in 0..spec.dims {
            let i0 = per_dim_indices[d][idx[d]] - 1;
            let offset = i0
                .checked_mul(strides[d])
                .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))?;
            lin = lin
                .checked_add(offset)
                .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))?;
        }
        indices.push(checked_u32_index(lin)?);
        let mut d = 0usize;
        while d < spec.dims {
            idx[d] += 1;
            if idx[d] < per_dim_indices[d].len() {
                break;
            }
            idx[d] = 0;
            d += 1;
        }
        if d == spec.dims {
            break;
        }
    }

    let output_shape = if spec.dims == 1 {
        if let Some(shape) = linear_output_shape {
            shape
        } else if total_out <= 1 {
            vec![1, 1]
        } else if linear_selector_is_colon {
            vec![total_out, 1]
        } else if linear_selector_is_range || base_is_row_vector {
            vec![1, total_out]
        } else {
            vec![total_out, 1]
        }
    } else {
        let mut dims_out: Vec<(usize, usize, bool)> = selection_lengths
            .iter()
            .enumerate()
            .map(|(d, &len)| (d, len, scalar_mask.get(d).copied().unwrap_or(false)))
            .collect();
        while dims_out.len() > 2
            && dims_out
                .last()
                .map(|&(_, len, is_scalar)| len == 1 && is_scalar)
                .unwrap_or(false)
        {
            dims_out.pop();
        }
        if dims_out.is_empty() {
            vec![1, 1]
        } else if dims_out.len() == 1 {
            let (dim, len, _) = dims_out[0];
            if dim == 1 {
                vec![1, len]
            } else {
                vec![len, 1]
            }
        } else {
            dims_out.into_iter().map(|(_, len, _)| len).collect()
        }
    };
    Ok(IndexPlan::new(
        indices,
        output_shape,
        selection_lengths,
        spec.dims,
        planned_shape,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        build_expr_index_plan, build_index_plan, build_sparse_assignment_plan, ExprPlanSpec,
    };
    use crate::bytecode::EndExpr;
    use crate::indexing::selectors::{build_slice_selectors, SliceSelector};
    use runmat_value::{IntegerStorage, LogicalArray, Tensor, Value};

    #[test]
    fn sparse_assignment_plan_expands_numeric_dimensions_but_keeps_colon_at_old_extent() {
        let selectors = vec![SliceSelector::Indices(vec![3, 4]), SliceSelector::Colon];
        let plan = build_sparse_assignment_plan(&selectors, 2, &[2, 2])
            .expect("sparse assignment plan should grow rows");
        assert_eq!(plan.base_shape, vec![4, 2]);
        assert_eq!(plan.selection_lengths, vec![2, 2]);
        assert_eq!(plan.indices, vec![2, 3, 6, 7]);
    }

    #[test]
    fn plain_and_expr_linear_range_plans_match() {
        futures::executor::block_on(async {
            let shape = vec![1, 10];
            let numeric = vec![Value::Tensor(
                Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![1, 4]).unwrap(),
            )];
            let plain_selectors = build_slice_selectors(1, 0, 0, &numeric, &shape)
                .await
                .unwrap();
            let plain = build_index_plan(&plain_selectors, 1, &shape).unwrap();
            let expr = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 1,
                    colon_mask: 0,
                    end_mask: 0,
                    range_dims: &[0],
                    range_params: &[(2.0, 2.0)],
                    range_start_exprs: &[None],
                    range_step_exprs: &[None],
                    range_end_exprs: &[EndExpr::Sub(
                        Box::new(EndExpr::End),
                        Box::new(EndExpr::Const(1.0)),
                    )],
                    numeric: &[],
                    shape: &shape,
                },
                |dim_len, expr| {
                    let expr = expr.clone();
                    async move {
                        Ok(match &expr {
                            EndExpr::End => dim_len as f64,
                            EndExpr::Const(value) => *value,
                            EndExpr::Sub(lhs, rhs) => {
                                let lhs_val = match lhs.as_ref() {
                                    EndExpr::End => dim_len as f64,
                                    EndExpr::Const(value) => *value,
                                    other => panic!("unsupported lhs expr: {other:?}"),
                                };
                                let rhs_val = match rhs.as_ref() {
                                    EndExpr::Const(value) => *value,
                                    other => panic!("unsupported rhs expr: {other:?}"),
                                };
                                lhs_val - rhs_val
                            }
                            other => panic!("unsupported expr: {other:?}"),
                        })
                    }
                },
            )
            .await
            .unwrap();
            assert_eq!(plain.indices, expr.indices);
            assert_eq!(plain.output_shape, expr.output_shape);
            assert_eq!(plain.selection_lengths, expr.selection_lengths);
            assert_eq!(plain.properties.full_row, expr.properties.full_row);
            assert_eq!(plain.properties.full_column, expr.properties.full_column);
        })
    }

    #[test]
    fn expr_integer_index_vectors_use_exact_storage_for_all_classes() {
        macro_rules! assert_indices {
            ($storage:expr) => {{
                let indices =
                    Tensor::new_integer($storage, vec![1, 2]).expect("typed integer index tensor");
                let numeric = vec![Value::Tensor(indices)];
                let plan = futures::executor::block_on(build_expr_index_plan(
                    ExprPlanSpec {
                        dims: 1,
                        colon_mask: 0,
                        end_mask: 0,
                        range_dims: &[],
                        range_params: &[],
                        range_start_exprs: &[],
                        range_step_exprs: &[],
                        range_end_exprs: &[],
                        numeric: &numeric,
                        shape: &[1, 2],
                    },
                    |_, _| async { Ok(0.0) },
                ))
                .expect("exact integer vector index plan");
                assert_eq!(plan.indices, vec![0, 1]);
            }};
        }

        assert_indices!(IntegerStorage::I8(vec![1, 2]));
        assert_indices!(IntegerStorage::I16(vec![1, 2]));
        assert_indices!(IntegerStorage::I32(vec![1, 2]));
        assert_indices!(IntegerStorage::I64(vec![1, 2]));
        assert_indices!(IntegerStorage::U8(vec![1, 2]));
        assert_indices!(IntegerStorage::U16(vec![1, 2]));
        assert_indices!(IntegerStorage::U32(vec![1, 2]));
        assert_indices!(IntegerStorage::U64(vec![1, 2]));
    }

    #[test]
    fn expr_integer_index_vectors_reject_wide_values() {
        let indices = Tensor::new_integer(IntegerStorage::U64(vec![1, u64::MAX]), vec![1, 2])
            .expect("typed integer index tensor");
        let numeric = vec![Value::Tensor(indices)];

        let err = futures::executor::block_on(build_expr_index_plan(
            ExprPlanSpec {
                dims: 1,
                colon_mask: 0,
                end_mask: 0,
                range_dims: &[],
                range_params: &[],
                range_start_exprs: &[],
                range_step_exprs: &[],
                range_end_exprs: &[],
                numeric: &numeric,
                shape: &[1, 2],
            },
            |_, _| async { Ok(0.0) },
        ))
        .expect_err("wide exact integer index must not use its float mirror");
        assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
    }

    #[test]
    fn plain_and_expr_column_plans_match_properties() {
        futures::executor::block_on(async {
            let shape = vec![3, 4];
            let numeric = vec![Value::Num(3.0)];
            let plain_selectors = build_slice_selectors(2, 1, 0, &numeric, &shape)
                .await
                .unwrap();
            let plain = build_index_plan(&plain_selectors, 2, &shape).unwrap();
            let expr = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 2,
                    colon_mask: 1,
                    end_mask: 0,
                    range_dims: &[],
                    range_params: &[],
                    range_start_exprs: &[],
                    range_step_exprs: &[],
                    range_end_exprs: &[],
                    numeric: &numeric,
                    shape: &shape,
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .unwrap();
            assert_eq!(plain.indices, expr.indices);
            assert_eq!(plain.properties.full_column, Some(2));
            assert_eq!(plain.properties.full_column, expr.properties.full_column);
            assert_eq!(plain.properties.full_row, expr.properties.full_row);
        })
    }

    #[test]
    fn expr_linear_range_on_column_vector_uses_range_shape() {
        futures::executor::block_on(async {
            let plan = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 1,
                    colon_mask: 0,
                    end_mask: 0,
                    range_dims: &[0],
                    range_params: &[(1.0, 1.0)],
                    range_start_exprs: &[None],
                    range_step_exprs: &[None],
                    range_end_exprs: &[EndExpr::Var(0)],
                    numeric: &[],
                    shape: &[10, 1],
                },
                |_dim_len, expr| {
                    let expr = expr.clone();
                    async move {
                        match expr {
                            EndExpr::Var(_) => Ok(6.0),
                            other => panic!("unsupported expr: {other:?}"),
                        }
                    }
                },
            )
            .await
            .unwrap();
            assert_eq!(plan.indices, vec![0, 1, 2, 3, 4, 5]);
            assert_eq!(plan.output_shape, vec![1, 6]);
            assert_eq!(plan.selection_lengths, vec![6]);
        })
    }

    #[test]
    fn expr_empty_linear_range_uses_range_shape() {
        futures::executor::block_on(async {
            let plan = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 1,
                    colon_mask: 0,
                    end_mask: 0,
                    range_dims: &[0],
                    range_params: &[(1.0, 1.0)],
                    range_start_exprs: &[None],
                    range_step_exprs: &[None],
                    range_end_exprs: &[EndExpr::Var(0)],
                    numeric: &[],
                    shape: &[10, 1],
                },
                |_dim_len, expr| {
                    let expr = expr.clone();
                    async move {
                        match expr {
                            EndExpr::Var(_) => Ok(0.0),
                            other => panic!("unsupported expr: {other:?}"),
                        }
                    }
                },
            )
            .await
            .unwrap();
            assert!(plan.indices.is_empty());
            assert_eq!(plan.output_shape, vec![1, 0]);
            assert_eq!(plan.selection_lengths, vec![0]);
        })
    }

    #[test]
    fn expr_plan_rejects_range_dim_conflicting_with_colon_mask() {
        futures::executor::block_on(async {
            let err = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 2,
                    colon_mask: 0b01,
                    end_mask: 0,
                    range_dims: &[0],
                    range_params: &[(1.0, 1.0)],
                    range_start_exprs: &[None],
                    range_step_exprs: &[None],
                    range_end_exprs: &[EndExpr::End],
                    numeric: &[Value::Num(1.0)],
                    shape: &[3, 3],
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .expect_err("range/colon conflict should fail");
            assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorPlan"));
        })
    }

    #[test]
    fn expr_plan_rejects_range_dim_conflicting_with_end_mask() {
        futures::executor::block_on(async {
            let err = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 2,
                    colon_mask: 0,
                    end_mask: 0b10,
                    range_dims: &[1],
                    range_params: &[(1.0, 1.0)],
                    range_start_exprs: &[None],
                    range_step_exprs: &[None],
                    range_end_exprs: &[EndExpr::End],
                    numeric: &[Value::Num(1.0)],
                    shape: &[3, 3],
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .expect_err("range/end conflict should fail");
            assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorPlan"));
        })
    }

    #[test]
    fn expr_plan_rejects_duplicate_range_dims() {
        futures::executor::block_on(async {
            let err = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 2,
                    colon_mask: 0,
                    end_mask: 0,
                    range_dims: &[1, 1],
                    range_params: &[(1.0, 1.0), (1.0, 1.0)],
                    range_start_exprs: &[None, None],
                    range_step_exprs: &[None, None],
                    range_end_exprs: &[EndExpr::End, EndExpr::End],
                    numeric: &[Value::Num(1.0)],
                    shape: &[3, 3],
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .expect_err("duplicate range dims should fail");
            assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorPlan"));
        })
    }

    #[test]
    fn expr_plan_rejects_out_of_bounds_range_dim() {
        futures::executor::block_on(async {
            let err = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 2,
                    colon_mask: 0,
                    end_mask: 0,
                    range_dims: &[2],
                    range_params: &[(1.0, 1.0)],
                    range_start_exprs: &[None],
                    range_step_exprs: &[None],
                    range_end_exprs: &[EndExpr::End],
                    numeric: &[Value::Num(1.0), Value::Num(1.0)],
                    shape: &[3, 3],
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .expect_err("out-of-bounds range dim should fail");
            assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorDim"));
        })
    }

    #[test]
    fn expr_plan_rejects_inconsistent_range_metadata_lengths() {
        futures::executor::block_on(async {
            let err = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 2,
                    colon_mask: 0,
                    end_mask: 0,
                    range_dims: &[1],
                    range_params: &[],
                    range_start_exprs: &[None],
                    range_step_exprs: &[None],
                    range_end_exprs: &[EndExpr::End],
                    numeric: &[Value::Num(1.0)],
                    shape: &[3, 3],
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .expect_err("inconsistent range metadata should fail");
            assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorPlan"));
        })
    }

    #[test]
    fn index_plan_rejects_sparse_sized_indices_beyond_u32_storage() {
        let selectors = vec![SliceSelector::Scalar((u32::MAX as usize) + 2)];
        let err = build_index_plan(&selectors, 1, &[u32::MAX as usize + 2, 1])
            .expect_err("linear index should exceed u32 plan storage");
        assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
    }

    #[test]
    fn index_plan_rejects_dimension_product_overflow() {
        let selectors = vec![SliceSelector::Colon];
        let err = build_index_plan(&selectors, 1, &[usize::MAX, 2])
            .expect_err("linearized sparse shape should overflow");
        assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
    }

    #[test]
    fn expr_plan_supports_dims_beyond_mask_width() {
        futures::executor::block_on(async {
            let numeric = vec![Value::Num(1.0); 31];
            let shape = vec![1usize; 33];
            let plan = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 33,
                    colon_mask: 0b1,
                    end_mask: 0b10,
                    range_dims: &[],
                    range_params: &[],
                    range_start_exprs: &[],
                    range_step_exprs: &[],
                    range_end_exprs: &[],
                    numeric: &numeric,
                    shape: &shape,
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .expect("expr plan for dims beyond mask width");
            assert_eq!(plan.dims, 33);
            assert!(!plan.indices.is_empty());
        })
    }

    #[test]
    fn expr_plan_tensor_selector_length_match_uses_numeric_indices() {
        futures::executor::block_on(async {
            let shape = vec![3, 2];
            let numeric = vec![Value::Tensor(
                Tensor::new(vec![2.0, 1.0, 3.0], vec![1, 3]).expect("selector tensor"),
            )];
            let plain_selectors = build_slice_selectors(2, 0b10, 0, &numeric, &shape)
                .await
                .unwrap();
            let plain = build_index_plan(&plain_selectors, 2, &shape).unwrap();
            let expr = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 2,
                    colon_mask: 0b10,
                    end_mask: 0,
                    range_dims: &[],
                    range_params: &[],
                    range_start_exprs: &[],
                    range_step_exprs: &[],
                    range_end_exprs: &[],
                    numeric: &numeric,
                    shape: &shape,
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .unwrap();
            assert_eq!(plain.indices, expr.indices);
            assert_eq!(plain.output_shape, expr.output_shape);
            assert_eq!(plain.selection_lengths, expr.selection_lengths);
        })
    }

    #[test]
    fn expr_plan_logical_selector_remains_logical_mask() {
        futures::executor::block_on(async {
            let shape = vec![3, 2];
            let numeric = vec![Value::LogicalArray(
                LogicalArray::new(vec![0, 1, 1], vec![1, 3]).expect("logical selector"),
            )];
            let plain_selectors = build_slice_selectors(2, 0b10, 0, &numeric, &shape)
                .await
                .unwrap();
            let plain = build_index_plan(&plain_selectors, 2, &shape).unwrap();
            let expr = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 2,
                    colon_mask: 0b10,
                    end_mask: 0,
                    range_dims: &[],
                    range_params: &[],
                    range_start_exprs: &[],
                    range_step_exprs: &[],
                    range_end_exprs: &[],
                    numeric: &numeric,
                    shape: &shape,
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .unwrap();
            assert_eq!(plain.indices, expr.indices);
            assert_eq!(plain.output_shape, expr.output_shape);
            assert_eq!(plain.selection_lengths, expr.selection_lengths);
        })
    }

    #[test]
    fn expr_plan_linear_tensor_selector_preserves_tensor_shape() {
        futures::executor::block_on(async {
            let shape = vec![1, 10];
            let numeric = vec![Value::Tensor(
                Tensor::new(vec![2.0, 4.0], vec![2, 1]).expect("selector tensor"),
            )];
            let plain_selectors = build_slice_selectors(1, 0, 0, &numeric, &shape)
                .await
                .unwrap();
            let plain = build_index_plan(&plain_selectors, 1, &shape).unwrap();
            let expr = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 1,
                    colon_mask: 0,
                    end_mask: 0,
                    range_dims: &[],
                    range_params: &[],
                    range_start_exprs: &[],
                    range_step_exprs: &[],
                    range_end_exprs: &[],
                    numeric: &numeric,
                    shape: &shape,
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .unwrap();
            assert_eq!(plain.indices, expr.indices);
            assert_eq!(plain.output_shape, expr.output_shape);
            assert_eq!(plain.selection_lengths, expr.selection_lengths);
        })
    }

    #[test]
    fn expr_plan_linear_colon_selector_matches_plain_shape() {
        futures::executor::block_on(async {
            let shape = vec![1, 5];
            let plain = build_index_plan(&[SliceSelector::Colon], 1, &shape).unwrap();
            let expr = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 1,
                    colon_mask: 0b1,
                    end_mask: 0,
                    range_dims: &[],
                    range_params: &[],
                    range_start_exprs: &[],
                    range_step_exprs: &[],
                    range_end_exprs: &[],
                    numeric: &[],
                    shape: &shape,
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .unwrap();
            assert_eq!(plain.indices, expr.indices);
            assert_eq!(plain.output_shape, expr.output_shape);
            assert_eq!(plain.selection_lengths, expr.selection_lengths);
        })
    }

    #[test]
    fn expr_plan_linear_logical_mask_matches_plain_shape() {
        futures::executor::block_on(async {
            let shape = vec![1, 5];
            let numeric = vec![Value::LogicalArray(
                LogicalArray::new(vec![1, 0, 1, 0, 1], vec![1, 5]).expect("logical selector"),
            )];
            let plain_selectors = build_slice_selectors(1, 0, 0, &numeric, &shape)
                .await
                .unwrap();
            let plain = build_index_plan(&plain_selectors, 1, &shape).unwrap();
            let expr = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 1,
                    colon_mask: 0,
                    end_mask: 0,
                    range_dims: &[],
                    range_params: &[],
                    range_start_exprs: &[],
                    range_step_exprs: &[],
                    range_end_exprs: &[],
                    numeric: &numeric,
                    shape: &shape,
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .unwrap();
            assert_eq!(plain.indices, expr.indices);
            assert_eq!(plain.output_shape, expr.output_shape);
            assert_eq!(plain.selection_lengths, expr.selection_lengths);
        })
    }

    #[test]
    fn expr_plan_short_linear_logical_mask_matches_plain_column_shape() {
        futures::executor::block_on(async {
            let shape = vec![2, 3];
            let numeric = vec![Value::LogicalArray(
                LogicalArray::new(vec![0, 1, 1], vec![1, 3]).expect("logical selector"),
            )];
            let plain_selectors = build_slice_selectors(1, 0, 0, &numeric, &shape)
                .await
                .unwrap();
            let plain = build_index_plan(&plain_selectors, 1, &shape).unwrap();
            let expr = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 1,
                    colon_mask: 0,
                    end_mask: 0,
                    range_dims: &[],
                    range_params: &[],
                    range_start_exprs: &[],
                    range_step_exprs: &[],
                    range_end_exprs: &[],
                    numeric: &numeric,
                    shape: &shape,
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .unwrap();
            assert_eq!(plain.indices, vec![1, 2]);
            assert_eq!(plain.indices, expr.indices);
            assert_eq!(plain.output_shape, vec![2, 1]);
            assert_eq!(plain.output_shape, expr.output_shape);
            assert_eq!(plain.selection_lengths, expr.selection_lengths);
        })
    }

    #[test]
    fn linear_empty_selector_uses_empty_column_shape() {
        let plan = build_index_plan(&[SliceSelector::Indices(Vec::new())], 1, &[1, 5])
            .expect("empty linear selector should build");
        assert!(plan.indices.is_empty());
        assert_eq!(plan.output_shape, vec![0, 1]);
    }

    #[test]
    fn expr_plan_linear_empty_logical_mask_matches_plain_shape() {
        futures::executor::block_on(async {
            let shape = vec![1, 5];
            let numeric = vec![Value::LogicalArray(
                LogicalArray::new(vec![0, 0, 0, 0, 0], vec![1, 5]).expect("logical selector"),
            )];
            let plain_selectors = build_slice_selectors(1, 0, 0, &numeric, &shape)
                .await
                .unwrap();
            let plain = build_index_plan(&plain_selectors, 1, &shape).unwrap();
            let expr = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 1,
                    colon_mask: 0,
                    end_mask: 0,
                    range_dims: &[],
                    range_params: &[],
                    range_start_exprs: &[],
                    range_step_exprs: &[],
                    range_end_exprs: &[],
                    numeric: &numeric,
                    shape: &shape,
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .unwrap();
            assert!(plain.indices.is_empty());
            assert!(expr.indices.is_empty());
            assert_eq!(plain.output_shape, vec![0, 1]);
            assert_eq!(plain.output_shape, expr.output_shape);
            assert_eq!(plain.selection_lengths, expr.selection_lengths);
        })
    }
}
