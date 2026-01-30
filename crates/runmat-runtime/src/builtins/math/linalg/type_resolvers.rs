use runmat_builtins::Type;

use crate::builtins::common::type_shapes::{element_count_if_known, unknown_shape};
use crate::builtins::math::reduction::type_resolvers::reduce_first_nonsingleton;
use crate::builtins::math::type_resolvers::numeric_unary_type;

pub fn numeric_scalar_type(args: &[Type]) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    match input {
        Type::Tensor { .. } | Type::Logical { .. } | Type::Num | Type::Int | Type::Bool => {
            Type::Num
        }
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

pub fn logical_scalar_type(args: &[Type]) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    match input {
        Type::Tensor { .. } | Type::Logical { .. } | Type::Num | Type::Int | Type::Bool => {
            Type::Bool
        }
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

pub fn matrix_unary_type(args: &[Type]) -> Type {
    match args.first() {
        Some(first) => numeric_unary_type(std::slice::from_ref(first)),
        None => Type::Unknown,
    }
}

pub fn transpose_type(args: &[Type]) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    match input {
        Type::Tensor { shape: Some(shape) } => {
            if element_count_if_known(shape.as_slice()) == Some(1) {
                Type::Num
            } else {
                Type::Tensor {
                    shape: Some(transpose_shape(shape)),
                }
            }
        }
        Type::Tensor { shape: None } => Type::tensor(),
        Type::Logical { shape: Some(shape) } => Type::Logical {
            shape: Some(transpose_shape(shape)),
        },
        Type::Logical { shape: None } => Type::logical(),
        Type::Num | Type::Int | Type::Bool | Type::String => input.clone(),
        Type::Cell { .. } => input.clone(),
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

pub fn matmul_type(args: &[Type]) -> Type {
    let lhs = args.get(0);
    let rhs = args.get(1);
    match (lhs, rhs) {
        (Some(left), Some(right)) => matmul_binary_type(left, right),
        (Some(single), None) | (None, Some(single)) => {
            numeric_unary_type(std::slice::from_ref(single))
        }
        (None, None) => Type::Unknown,
    }
}

pub fn left_divide_type(args: &[Type]) -> Type {
    let lhs = args.get(0);
    let rhs = args.get(1);
    match (lhs, rhs) {
        (Some(left), Some(right)) => left_divide_binary_type(left, right),
        (Some(single), None) | (None, Some(single)) => {
            numeric_unary_type(std::slice::from_ref(single))
        }
        (None, None) => Type::Unknown,
    }
}

pub fn right_divide_type(args: &[Type]) -> Type {
    let lhs = args.get(0);
    let rhs = args.get(1);
    match (lhs, rhs) {
        (Some(left), Some(right)) => right_divide_binary_type(left, right),
        (Some(single), None) | (None, Some(single)) => {
            numeric_unary_type(std::slice::from_ref(single))
        }
        (None, None) => Type::Unknown,
    }
}

pub fn dot_type(args: &[Type]) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    if is_numeric_scalar(input) {
        return Type::Num;
    }
    let shape = match input {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => shape,
        Type::Tensor { shape: None } | Type::Logical { shape: None } => return Type::tensor(),
        Type::Num | Type::Int | Type::Bool => return Type::Num,
        Type::Unknown => return Type::Unknown,
        _ => return Type::Unknown,
    };
    let out_shape = if args.len() > 2 {
        unknown_shape(shape.len())
    } else {
        reduce_first_nonsingleton(shape)
    };
    numeric_tensor_from_shape(out_shape)
}

pub fn pinv_type(args: &[Type]) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    match input {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            if element_count_if_known(shape.as_slice()) == Some(1) {
                return Type::Num;
            }
            let (rows, cols) = matrix_dims(shape);
            let out_shape = vec![cols, rows];
            numeric_tensor_from_shape(out_shape)
        }
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
        Type::Num | Type::Int | Type::Bool => Type::Num,
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

pub fn eig_type(args: &[Type]) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    match input {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            let (rows, _cols) = matrix_dims(shape);
            let out_shape = vec![rows, Some(1)];
            numeric_tensor_from_shape(out_shape)
        }
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
        Type::Num | Type::Int | Type::Bool => Type::Num,
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

pub fn svd_type(args: &[Type]) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    match input {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            let (rows, cols) = matrix_dims(shape);
            let diag = min_dim(rows, cols);
            let out_shape = vec![diag, Some(1)];
            numeric_tensor_from_shape(out_shape)
        }
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
        Type::Num | Type::Int | Type::Bool => Type::Num,
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

pub fn symrcm_type(args: &[Type]) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    let rows = match input {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            matrix_dims(shape).0
        }
        Type::Tensor { shape: None } | Type::Logical { shape: None } => return Type::tensor(),
        Type::Num | Type::Int | Type::Bool => Some(1),
        Type::Unknown => return Type::Unknown,
        _ => return Type::Unknown,
    };
    Type::Tensor {
        shape: Some(vec![Some(1), rows]),
    }
}

pub fn bandwidth_type(args: &[Type]) -> Type {
    if args.len() > 1 {
        return Type::Num;
    }
    Type::Tensor {
        shape: Some(vec![Some(1), Some(2)]),
    }
}

pub fn qr_type(args: &[Type]) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    match input {
        Type::Tensor { .. } | Type::Logical { .. } => Type::tensor(),
        Type::Num | Type::Int | Type::Bool => Type::Num,
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

fn matmul_binary_type(lhs: &Type, rhs: &Type) -> Type {
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

fn left_divide_binary_type(lhs: &Type, rhs: &Type) -> Type {
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

fn right_divide_binary_type(lhs: &Type, rhs: &Type) -> Type {
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

fn numeric_tensor_from_shape(shape: Vec<Option<usize>>) -> Type {
    if element_count_if_known(&shape) == Some(1) {
        Type::Num
    } else {
        Type::Tensor { shape: Some(shape) }
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

fn transpose_shape(shape: &[Option<usize>]) -> Vec<Option<usize>> {
    match shape.len() {
        0 => vec![Some(1), Some(1)],
        1 => vec![Some(1), shape[0]],
        _ => {
            let mut out = shape.to_vec();
            if out.len() >= 2 {
                out.swap(0, 1);
            }
            out
        }
    }
}

fn matrix_dims(shape: &[Option<usize>]) -> (Option<usize>, Option<usize>) {
    match shape.len() {
        0 => (Some(1), Some(1)),
        1 => (shape[0], Some(1)),
        _ => (shape[0], shape[1]),
    }
}

fn is_effective_matrix(shape: &[Option<usize>]) -> bool {
    if shape.len() <= 2 {
        return true;
    }
    shape.iter().skip(2).all(|dim| matches!(dim, Some(1)))
}

fn min_dim(lhs: Option<usize>, rhs: Option<usize>) -> Option<usize> {
    match (lhs, rhs) {
        (Some(a), Some(b)) => Some(a.min(b)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_swaps_first_two_dims() {
        let ty = Type::Tensor {
            shape: Some(vec![Some(2), Some(3), Some(4)]),
        };
        let out = transpose_type(&[ty]);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(2), Some(4)])
            }
        );
    }

    #[test]
    fn matmul_shape_infers_rows_and_cols() {
        let lhs = Type::Tensor {
            shape: Some(vec![Some(2), Some(3)]),
        };
        let rhs = Type::Tensor {
            shape: Some(vec![Some(3), Some(4)]),
        };
        let out = matmul_type(&[lhs, rhs]);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(4)])
            }
        );
    }
}
