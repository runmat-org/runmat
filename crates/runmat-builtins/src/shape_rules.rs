use crate::Type;

pub fn tensor_shape(ty: &Type) -> Option<&[Option<usize>]> {
    match ty {
        Type::Tensor { shape: Some(shape) } => Some(shape.as_slice()),
        _ => None,
    }
}

pub fn scalar_tensor_shape() -> Vec<Option<usize>> {
    vec![Some(1), Some(1)]
}

pub fn unknown_shape(rank: usize) -> Vec<Option<usize>> {
    vec![None; rank]
}

pub fn element_count_if_known(shape: &[Option<usize>]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, dim| match dim {
        Some(value) => acc.checked_mul(*value),
        None => None,
    })
}

pub fn broadcast_shapes(lhs: &[Option<usize>], rhs: &[Option<usize>]) -> Vec<Option<usize>> {
    let max_rank = lhs.len().max(rhs.len());
    let mut out: Vec<Option<usize>> = Vec::with_capacity(max_rank);
    for i in 0..max_rank {
        let lhs_idx = lhs.len().checked_sub(1 + i);
        let rhs_idx = rhs.len().checked_sub(1 + i);
        let da = lhs_idx
            .and_then(|idx| lhs.get(idx))
            .cloned()
            .unwrap_or(Some(1));
        let db = rhs_idx
            .and_then(|idx| rhs.get(idx))
            .cloned()
            .unwrap_or(Some(1));
        let dim = match (da, db) {
            (Some(a), Some(b)) => {
                if a == b {
                    Some(a)
                } else if a == 1 {
                    Some(b)
                } else if b == 1 {
                    Some(a)
                } else {
                    None
                }
            }
            (Some(a), None) => {
                if a == 1 {
                    None
                } else {
                    Some(a)
                }
            }
            (None, Some(b)) => {
                if b == 1 {
                    None
                } else {
                    Some(b)
                }
            }
            (None, None) => None,
        };
        out.push(dim);
    }
    out.reverse();
    out
}

pub fn broadcast_compatible(lhs: &[Option<usize>], rhs: &[Option<usize>]) -> Option<bool> {
    let max_rank = lhs.len().max(rhs.len());
    let mut unknown = false;
    for i in 0..max_rank {
        let lhs_idx = lhs.len().checked_sub(1 + i);
        let rhs_idx = rhs.len().checked_sub(1 + i);
        let da = lhs_idx
            .and_then(|idx| lhs.get(idx))
            .cloned()
            .unwrap_or(Some(1));
        let db = rhs_idx
            .and_then(|idx| rhs.get(idx))
            .cloned()
            .unwrap_or(Some(1));
        match (da, db) {
            (Some(a), Some(b)) => {
                if a != b && a != 1 && b != 1 {
                    return Some(false);
                }
            }
            _ => unknown = true,
        }
    }
    if unknown {
        None
    } else {
        Some(true)
    }
}

pub fn is_scalar_shape(shape: &[Option<usize>]) -> bool {
    let mut product: usize = 1;
    for dim in shape {
        let Some(value) = dim else {
            return false;
        };
        if *value == 0 {
            return false;
        }
        product = product.saturating_mul(*value);
        if product > 1 {
            return false;
        }
    }
    product == 1
}

pub fn infer_range_shape(
    start: Option<f64>,
    step: Option<f64>,
    end: Option<f64>,
) -> Option<Vec<Option<usize>>> {
    fn range_len(start: f64, step: f64, end: f64) -> Option<usize> {
        if step == 0.0 {
            return None;
        }
        let raw = (end - start) / step;
        if !raw.is_finite() {
            return None;
        }
        let rounded = raw.round();
        let n = if (raw - rounded).abs() <= 1e-9 {
            rounded
        } else {
            raw.floor()
        };
        if n < 0.0 {
            return Some(0);
        }
        Some(n as usize + 1)
    }

    match (start, step, end) {
        (Some(s), Some(step), Some(e)) => {
            let len = range_len(s, step, e)?;
            Some(vec![Some(1), Some(len)])
        }
        (Some(s), None, Some(e)) => {
            // Implicit two-argument colon always uses a +1 increment. If the end
            // lies "behind" the start, MATLAB returns an empty row vector.
            let len = range_len(s, 1.0, e)?;
            Some(vec![Some(1), Some(len)])
        }
        _ => None,
    }
}

pub fn constructor_shape_from_dims(dims: &[Option<usize>]) -> Option<Vec<Option<usize>>> {
    if dims.is_empty() {
        return None;
    }
    if dims.len() == 1 {
        return Some(vec![dims[0], dims[0]]);
    }
    Some(dims.to_vec())
}

pub fn concat_shape(rows: &[Vec<Type>]) -> Option<Vec<Option<usize>>> {
    fn dims_for_value(ty: &Type) -> Option<(Option<usize>, Option<usize>)> {
        match ty {
            Type::Num | Type::Int | Type::Bool => Some((Some(1), Some(1))),
            Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
                Some(matrix_dims(shape))
            }
            _ => None,
        }
    }

    let mut total_rows: Option<usize> = Some(0);
    let mut total_cols: Option<usize> = None;

    for row in rows {
        let mut row_dim: Option<usize> = None;
        let mut row_cols: Option<usize> = Some(0);
        for entry in row {
            let (rows_dim, cols_dim) = dims_for_value(entry)?;
            if let (Some(prev), Some(curr)) = (row_dim, rows_dim) {
                if prev != curr {
                    return None;
                }
            } else if row_dim.is_none() {
                row_dim = rows_dim;
            }

            row_cols = match (row_cols, cols_dim) {
                (Some(total), Some(value)) => Some(total + value),
                _ => None,
            };
        }

        total_rows = match (total_rows, row_dim) {
            (Some(total), Some(value)) => Some(total + value),
            _ => None,
        };

        total_cols = match (total_cols, row_cols) {
            (None, value) => value,
            (Some(prev), Some(curr)) if prev == curr => Some(prev),
            (Some(_), Some(_)) => return None,
            _ => None,
        };
    }

    Some(vec![total_rows, total_cols])
}

pub fn repmat_shape(input: &[Option<usize>], reps: &[Option<usize>]) -> Option<Vec<Option<usize>>> {
    if reps.is_empty() {
        return Some(input.to_vec());
    }
    let rank = input.len().max(reps.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let dim = input.get(i).cloned().unwrap_or(Some(1));
        let rep = reps.get(i).cloned().unwrap_or(Some(1));
        let value = match (dim, rep) {
            (Some(d), Some(r)) => Some(d * r),
            _ => None,
        };
        out.push(value);
    }
    Some(out)
}

pub fn index_output_type(base: &Type, indices: &[Type]) -> Type {
    fn scalar_from_base(base: &Type) -> Type {
        match base {
            Type::Logical { .. } | Type::Bool => Type::Bool,
            Type::Unknown => Type::Unknown,
            _ => Type::Num,
        }
    }

    fn array_from_base(base: &Type, shape: Vec<Option<usize>>) -> Type {
        match base {
            Type::Logical { .. } | Type::Bool => Type::Logical { shape: Some(shape) },
            _ => Type::Tensor { shape: Some(shape) },
        }
    }

    fn shape_from_index(ty: &Type) -> Option<Vec<Option<usize>>> {
        match ty {
            Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
                Some(shape.clone())
            }
            _ => None,
        }
    }

    fn is_scalar_index(ty: &Type) -> bool {
        match ty {
            Type::Int | Type::Num | Type::Bool => true,
            Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
                is_scalar_shape(shape)
            }
            _ => false,
        }
    }

    let base_shape = match base {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => shape.clone(),
        Type::Tensor { shape: None } | Type::Logical { shape: None } => {
            if indices.iter().all(is_scalar_index) {
                return scalar_from_base(base);
            }
            return match base {
                Type::Logical { .. } | Type::Bool => Type::logical(),
                Type::Unknown => Type::Unknown,
                _ => Type::tensor(),
            };
        }
        Type::Unknown => return Type::Unknown,
        _ => return Type::Unknown,
    };

    if indices.is_empty() {
        return array_from_base(base, base_shape);
    }

    if indices.len() == 1 {
        let idx = &indices[0];
        if is_scalar_index(idx) {
            return scalar_from_base(base);
        }
        if matches!(idx, Type::Logical { .. }) {
            return array_from_base(base, vec![Some(1), None]);
        }
        if let Some(shape) = shape_from_index(idx) {
            return array_from_base(base, shape);
        }
        return array_from_base(base, unknown_shape(2));
    }

    let mut out: Vec<Option<usize>> = Vec::new();
    for idx in indices {
        if is_scalar_index(idx) {
            continue;
        }
        if matches!(idx, Type::Logical { .. }) {
            out.push(None);
            continue;
        }
        if let Some(shape) = shape_from_index(idx) {
            out.extend(shape);
        } else {
            out.push(None);
        }
    }
    if indices.len() < base_shape.len() {
        out.extend(base_shape.iter().skip(indices.len()).copied());
    }
    if out.is_empty() {
        scalar_from_base(base)
    } else {
        array_from_base(base, out)
    }
}

pub fn matmul_output_type(lhs: &Type, rhs: &Type) -> Type {
    if is_numeric_scalar(lhs) && is_numeric_scalar(rhs) {
        return Type::Num;
    }
    if is_numeric_scalar(lhs) {
        return numeric_like(rhs);
    }
    if is_numeric_scalar(rhs) {
        return numeric_like(lhs);
    }

    let lhs_shape = numeric_array_shape(lhs);
    let rhs_shape = numeric_array_shape(rhs);
    match (lhs_shape, rhs_shape) {
        (Some(a), Some(b)) if is_effective_matrix(&a) && is_effective_matrix(&b) => {
            let (rows, _) = matrix_dims(&a);
            let (_, cols) = matrix_dims(&b);
            numeric_tensor_from_shape(vec![rows, cols])
        }
        (Some(_), Some(_)) => Type::tensor(),
        (Some(_), None) | (None, Some(_)) => Type::tensor(),
        (None, None) => {
            if is_numeric_array_type(lhs) || is_numeric_array_type(rhs) {
                Type::tensor()
            } else if matches!(lhs, Type::Unknown) || matches!(rhs, Type::Unknown) {
                Type::Unknown
            } else {
                Type::Num
            }
        }
    }
}

pub fn matmul_compatible(lhs: &Type, rhs: &Type) -> Option<bool> {
    let lhs_shape = numeric_array_shape(lhs)?;
    let rhs_shape = numeric_array_shape(rhs)?;
    if !is_effective_matrix(&lhs_shape) || !is_effective_matrix(&rhs_shape) {
        return None;
    }
    let (_, lhs_cols) = matrix_dims(&lhs_shape);
    let (rhs_rows, _) = matrix_dims(&rhs_shape);
    match (lhs_cols, rhs_rows) {
        (Some(a), Some(b)) => Some(a == b),
        _ => None,
    }
}

pub fn left_divide_compatible(lhs: &Type, rhs: &Type) -> Option<bool> {
    let lhs_shape = numeric_array_shape(lhs)?;
    let rhs_shape = numeric_array_shape(rhs)?;
    if !is_effective_matrix(&lhs_shape) || !is_effective_matrix(&rhs_shape) {
        return None;
    }
    let (lhs_rows, _) = matrix_dims(&lhs_shape);
    let (rhs_rows, _) = matrix_dims(&rhs_shape);
    match (lhs_rows, rhs_rows) {
        (Some(a), Some(b)) => Some(a == b),
        _ => None,
    }
}

pub fn right_divide_compatible(lhs: &Type, rhs: &Type) -> Option<bool> {
    let lhs_shape = numeric_array_shape(lhs)?;
    let rhs_shape = numeric_array_shape(rhs)?;
    if !is_effective_matrix(&lhs_shape) || !is_effective_matrix(&rhs_shape) {
        return None;
    }
    let (_, lhs_cols) = matrix_dims(&lhs_shape);
    let (_, rhs_cols) = matrix_dims(&rhs_shape);
    match (lhs_cols, rhs_cols) {
        (Some(a), Some(b)) => Some(a == b),
        _ => None,
    }
}

pub fn left_divide_output_type(lhs: &Type, rhs: &Type) -> Type {
    if is_numeric_scalar(lhs) {
        return numeric_like(rhs);
    }
    if is_numeric_scalar(rhs) {
        if let Some(shape) = numeric_array_shape(lhs) {
            let (_, cols) = matrix_dims(&shape);
            return numeric_tensor_from_shape(vec![cols, Some(1)]);
        }
    }

    let lhs_shape = numeric_array_shape(lhs);
    let rhs_shape = numeric_array_shape(rhs);
    match (lhs_shape, rhs_shape) {
        (Some(a), Some(b)) if is_effective_matrix(&a) && is_effective_matrix(&b) => {
            let (_, lhs_cols) = matrix_dims(&a);
            let (_, rhs_cols) = matrix_dims(&b);
            numeric_tensor_from_shape(vec![lhs_cols, rhs_cols])
        }
        (Some(_), Some(_)) => Type::tensor(),
        (Some(_), None) | (None, Some(_)) => Type::tensor(),
        (None, None) => Type::Unknown,
    }
}

pub fn right_divide_output_type(lhs: &Type, rhs: &Type) -> Type {
    if is_numeric_scalar(lhs) {
        return numeric_like(rhs);
    }
    if is_numeric_scalar(rhs) {
        return numeric_like(lhs);
    }

    let lhs_shape = numeric_array_shape(lhs);
    let rhs_shape = numeric_array_shape(rhs);
    match (lhs_shape, rhs_shape) {
        (Some(a), Some(b)) if is_effective_matrix(&a) && is_effective_matrix(&b) => {
            let (lhs_rows, _) = matrix_dims(&a);
            let (rhs_rows, _) = matrix_dims(&b);
            numeric_tensor_from_shape(vec![lhs_rows, rhs_rows])
        }
        (Some(_), Some(_)) => Type::tensor(),
        (Some(_), None) | (None, Some(_)) => Type::tensor(),
        (None, None) => Type::Unknown,
    }
}

pub fn matrix_dims(shape: &[Option<usize>]) -> (Option<usize>, Option<usize>) {
    match shape.len() {
        0 => (Some(1), Some(1)),
        1 => (shape[0], Some(1)),
        _ => (shape[0], shape[1]),
    }
}

pub fn numeric_tensor_from_shape(shape: Vec<Option<usize>>) -> Type {
    if element_count_if_known(&shape) == Some(1) {
        Type::Num
    } else {
        Type::Tensor { shape: Some(shape) }
    }
}

fn numeric_like(input: &Type) -> Type {
    match input {
        Type::Tensor { shape: Some(shape) } => numeric_tensor_from_shape(shape.clone()),
        Type::Logical { shape: Some(shape) } => numeric_tensor_from_shape(shape.clone()),
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
        Type::Num | Type::Int | Type::Bool => Type::Num,
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

fn numeric_array_shape(ty: &Type) -> Option<Vec<Option<usize>>> {
    match ty {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            if element_count_if_known(shape.as_slice()) == Some(1) {
                None
            } else {
                Some(shape.clone())
            }
        }
        _ => None,
    }
}

fn is_numeric_scalar(ty: &Type) -> bool {
    match ty {
        Type::Num | Type::Int | Type::Bool => true,
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            element_count_if_known(shape.as_slice()) == Some(1)
        }
        _ => false,
    }
}

fn is_numeric_array_type(ty: &Type) -> bool {
    matches!(ty, Type::Tensor { .. } | Type::Logical { .. })
}

fn is_effective_matrix(shape: &[Option<usize>]) -> bool {
    if shape.len() <= 2 {
        return true;
    }
    shape.iter().skip(2).all(|dim| matches!(dim, Some(1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_shapes_aligns_trailing_dims() {
        let out = broadcast_shapes(&[Some(1), Some(3)], &[Some(2), Some(1)]);
        assert_eq!(out, vec![Some(2), Some(3)]);
    }

    #[test]
    fn infer_range_shape_two_arg_descending_is_empty() {
        let shape = infer_range_shape(Some(5.0), None, Some(1.0)).expect("shape");
        assert_eq!(shape, vec![Some(1), Some(0)]);
    }

    #[test]
    fn infer_range_shape_three_arg_descending_respects_step() {
        let shape = infer_range_shape(Some(5.0), Some(-1.0), Some(1.0)).expect("shape");
        assert_eq!(shape, vec![Some(1), Some(5)]);
    }

    #[test]
    fn index_scalar_returns_num() {
        let base = Type::Tensor {
            shape: Some(vec![Some(1), Some(10)]),
        };
        let out = index_output_type(&base, &[Type::Num]);
        assert_eq!(out, Type::Num);
    }

    #[test]
    fn index_range_returns_index_shape() {
        let base = Type::Tensor {
            shape: Some(vec![Some(1), Some(10)]),
        };
        let idx = Type::Tensor {
            shape: Some(vec![Some(1), Some(4)]),
        };
        let out = index_output_type(&base, &[idx]);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(4)])
            }
        );
    }

    #[test]
    fn matmul_output_infers_dims() {
        let lhs = Type::Tensor {
            shape: Some(vec![Some(2), Some(3)]),
        };
        let rhs = Type::Tensor {
            shape: Some(vec![Some(3), Some(4)]),
        };
        let out = matmul_output_type(&lhs, &rhs);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(4)])
            }
        );
    }

    #[test]
    fn constructor_shape_from_scalar_dims() {
        assert_eq!(
            constructor_shape_from_dims(&[Some(3)]),
            Some(vec![Some(3), Some(3)])
        );
        assert_eq!(
            constructor_shape_from_dims(&[Some(2), Some(4)]),
            Some(vec![Some(2), Some(4)])
        );
    }

    #[test]
    fn broadcast_compatible_detects_mismatch() {
        assert_eq!(
            broadcast_compatible(&[Some(2), Some(3)], &[Some(4), Some(3)]),
            Some(false)
        );
    }

    #[test]
    fn matmul_compatible_detects_inner_mismatch() {
        let lhs = Type::Tensor {
            shape: Some(vec![Some(2), Some(3)]),
        };
        let rhs = Type::Tensor {
            shape: Some(vec![Some(4), Some(5)]),
        };
        assert_eq!(matmul_compatible(&lhs, &rhs), Some(false));
    }

    #[test]
    fn concat_shape_combines_blocks() {
        let rows = vec![
            vec![
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(4)]),
                },
            ],
            vec![Type::Tensor {
                shape: Some(vec![Some(4), Some(7)]),
            }],
        ];
        assert_eq!(concat_shape(&rows), Some(vec![Some(6), Some(7)]));
    }

    #[test]
    fn repmat_shape_scales_dims() {
        let input = vec![Some(2), Some(3)];
        let reps = vec![Some(2), Some(4)];
        assert_eq!(repmat_shape(&input, &reps), Some(vec![Some(4), Some(12)]));
    }
}
