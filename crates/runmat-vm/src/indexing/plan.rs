use crate::bytecode::EndExpr;
use crate::indexing::selectors::{index_scalar_from_value, SliceSelector};
use crate::interpreter::errors::mex;
use runmat_builtins::Value;
use runmat_runtime::{builtins::common::shape::is_scalar_shape, RuntimeError};
use std::future::Future;

pub type VmResult<T> = Result<T, RuntimeError>;

#[derive(Debug, Clone, Default)]
pub struct IndexPlanProperties {
    pub is_empty: bool,
    pub is_scalar: bool,
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
    let is_empty = indices.is_empty();
    let is_scalar = !is_empty && indices.len() == 1;
    let mut properties = IndexPlanProperties {
        is_empty,
        is_scalar,
        full_row: None,
        full_column: None,
    };
    if dims != 2 || is_empty {
        return properties;
    }
    let rows = base_shape.first().copied().unwrap_or(1);
    let cols = base_shape.get(1).copied().unwrap_or(1);
    if indices.len() == rows {
        let first = indices[0] as usize;
        if first % rows == 0 {
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

pub fn build_index_plan(
    selectors: &[SliceSelector],
    dims: usize,
    base_shape: &[usize],
) -> VmResult<IndexPlan> {
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
    Ok(IndexPlan::new(
        indices,
        out_shape,
        selection_lengths,
        dims,
        base_shape.to_vec(),
    ))
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

pub async fn build_expr_index_plan<ResolveEnd, Fut>(
    spec: ExprPlanSpec<'_>,
    mut resolve_end: ResolveEnd,
) -> Result<IndexPlan, RuntimeError>
where
    ResolveEnd: FnMut(usize, &EndExpr) -> Fut,
    Fut: Future<Output = Result<i64, RuntimeError>>,
{
    let rank = spec.shape.len();
    let full_shape: Vec<usize> = if spec.dims == 1 {
        vec![total_len_from_shape(spec.shape)]
    } else if rank < spec.dims {
        let mut s = spec.shape.to_vec();
        s.resize(spec.dims, 1);
        s
    } else {
        spec.shape.to_vec()
    };

    let mut selectors: Vec<ExprSel> = Vec::with_capacity(spec.dims);
    let mut num_iter = 0usize;
    let mut rp_iter = 0usize;
    for d in 0..spec.dims {
        let is_colon = (spec.colon_mask & (1u32 << d)) != 0;
        let is_end = (spec.end_mask & (1u32 << d)) != 0;
        if is_colon {
            selectors.push(ExprSel::Colon);
        } else if is_end {
            selectors.push(ExprSel::Scalar(*full_shape.get(d).unwrap_or(&1)));
        } else if let Some(pos) = spec.range_dims.iter().position(|&rd| rd == d) {
            let (raw_st, raw_sp) = spec.range_params[rp_iter];
            let dim_len = *full_shape.get(d).unwrap_or(&1);
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
            let off = spec.range_end_exprs[pos].clone();
            selectors.push(ExprSel::Range {
                start: st as i64,
                step: if sp >= 0.0 {
                    sp as i64
                } else {
                    -(sp.abs() as i64)
                },
                end_off: off,
            });
        } else {
            let v = spec
                .numeric
                .get(num_iter)
                .ok_or_else(|| mex("MissingNumericIndex", "missing numeric index"))?;
            num_iter += 1;
            if let Some(idx) = index_scalar_from_value(v).await? {
                if idx < 1 {
                    return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                }
                selectors.push(ExprSel::Scalar(idx as usize));
            } else {
                match v {
                    Value::Tensor(idx_t) => {
                        let dim_len = *full_shape.get(d).unwrap_or(&1);
                        let len = idx_t.shape.iter().product::<usize>();
                        if len == dim_len {
                            let mut vv = Vec::new();
                            for (i, &val) in idx_t.data.iter().enumerate() {
                                if val != 0.0 {
                                    vv.push(i + 1);
                                }
                            }
                            selectors.push(ExprSel::Indices(vv));
                        } else {
                            let mut vv = Vec::with_capacity(len);
                            for &val in &idx_t.data {
                                let idx = val as isize;
                                if idx < 1 {
                                    return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                                }
                                vv.push(idx as usize);
                            }
                            selectors.push(ExprSel::Indices(vv));
                        }
                    }
                    _ => return Err(mex("UnsupportedIndexType", "Unsupported index type")),
                }
            }
        }
    }

    let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(spec.dims);
    let mut selection_lengths: Vec<usize> = Vec::with_capacity(spec.dims);
    let mut scalar_mask: Vec<bool> = Vec::with_capacity(spec.dims);
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
                let end_i = resolve_end(dim_len as usize, end_off).await?;
                if stp == 0 {
                    return Err(mex("IndexStepZero", "Index step cannot be zero"));
                }
                if stp > 0 {
                    while cur <= end_i {
                        if cur < 1 || cur > dim_len {
                            break;
                        }
                        v.push(cur as usize);
                        cur += stp;
                    }
                } else {
                    while cur >= end_i {
                        if cur < 1 || cur > dim_len {
                            break;
                        }
                        v.push(cur as usize);
                        cur += stp;
                    }
                }
                v
            }
        };
        if idxs.iter().any(|&i| i == 0 || i > full_shape[d]) {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        selection_lengths.push(idxs.len());
        per_dim_indices.push(idxs);
        scalar_mask.push(matches!(sel, ExprSel::Scalar(_)));
    }

    let mut strides: Vec<usize> = vec![0; spec.dims];
    let mut acc = 1usize;
    for (d, stride) in strides.iter_mut().enumerate().take(spec.dims) {
        *stride = acc;
        acc *= full_shape[d];
    }
    let total_out: usize = per_dim_indices.iter().map(|v| v.len()).product();
    if total_out == 0 {
        let output_shape = if spec.dims == 1 {
            vec![1, 0]
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
            spec.shape.to_vec(),
        ));
    }

    let mut indices: Vec<u32> = Vec::with_capacity(total_out);
    let mut idx = vec![0usize; spec.dims];
    loop {
        let mut lin = 0usize;
        for d in 0..spec.dims {
            let i0 = per_dim_indices[d][idx[d]] - 1;
            lin += i0 * strides[d];
        }
        indices.push(lin as u32);
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
        if total_out <= 1 {
            vec![1, 1]
        } else {
            vec![1, total_out]
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
        spec.shape.to_vec(),
    ))
}

#[cfg(test)]
mod tests {
    use super::{build_expr_index_plan, build_index_plan, ExprPlanSpec};
    use crate::bytecode::EndExpr;
    use crate::indexing::selectors::build_slice_selectors;
    use runmat_builtins::{Tensor, Value};

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
                            EndExpr::End => dim_len as i64,
                            EndExpr::Const(value) => *value as i64,
                            EndExpr::Sub(lhs, rhs) => {
                                let lhs_val = match lhs.as_ref() {
                                    EndExpr::End => dim_len as i64,
                                    EndExpr::Const(value) => *value as i64,
                                    other => panic!("unsupported lhs expr: {other:?}"),
                                };
                                let rhs_val = match rhs.as_ref() {
                                    EndExpr::Const(value) => *value as i64,
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
}
