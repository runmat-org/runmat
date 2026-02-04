use runmat_builtins::Type;

use crate::builtins::math::reduction::type_resolvers::reduce_first_nonsingleton;
use crate::builtins::math::type_resolvers::numeric_unary_type;
use runmat_builtins::shape_rules::{element_count_if_known, unknown_shape};
use runmat_builtins::shape_rules::{
    left_divide_output_type, matmul_output_type, right_divide_output_type,
};
use runmat_builtins::ResolveContext;

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
        (Some(left), Some(right)) => matmul_output_type(left, right),
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
        (Some(left), Some(right)) => left_divide_output_type(left, right),
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
        (Some(left), Some(right)) => right_divide_output_type(left, right),
        (Some(single), None) | (None, Some(single)) => {
            numeric_unary_type(std::slice::from_ref(single))
        }
        (None, None) => Type::Unknown,
    }
}

pub fn dot_type(args: &[Type], ctx: &ResolveContext) -> Type {
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
    if args.len() > 2 {
        if let Some(dim) = ctx.numeric_dims_from(2).first().and_then(|value| *value) {
            if is_vector_shape(shape) {
                return Type::Num;
            }
            let out_shape = unknown_shape(shape.len().max(dim));
            return numeric_tensor_from_shape(out_shape);
        }
    }
    let out_shape = if args.len() > 2 {
        unknown_shape(shape.len().max(2))
    } else {
        reduce_first_nonsingleton(shape)
    };
    numeric_tensor_from_shape(out_shape)
}

pub fn dot_type_legacy(args: &[Type]) -> Type {
    dot_type(args, &ResolveContext::empty())
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

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::LiteralValue;

    #[test]
    fn dot_type_uses_literal_dim_for_vector() {
        let vec_type = Type::Tensor {
            shape: Some(vec![Some(3), Some(1)]),
        };
        let ctx = ResolveContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Number(1.0),
        ]);
        let out = dot_type(&[vec_type.clone(), vec_type, Type::Num], &ctx);
        assert_eq!(out, Type::Num);
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

pub fn numeric_tensor_from_shape(shape: Vec<Option<usize>>) -> Type {
    if element_count_if_known(&shape) == Some(1) {
        Type::Num
    } else {
        Type::Tensor { shape: Some(shape) }
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

pub fn matrix_dims(shape: &[Option<usize>]) -> (Option<usize>, Option<usize>) {
    match shape.len() {
        0 => (Some(1), Some(1)),
        1 => (shape[0], Some(1)),
        _ => (shape[0], shape[1]),
    }
}

fn is_vector_shape(shape: &[Option<usize>]) -> bool {
    match shape.len() {
        0 => false,
        1 => true,
        _ => shape.iter().take(2).any(|dim| matches!(dim, Some(1))),
    }
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

    #[test]
    fn right_divide_scalar_by_matrix_returns_matrix_shape() {
        let lhs = Type::Num;
        let rhs = Type::Tensor {
            shape: Some(vec![Some(2), Some(2)]),
        };
        let out = right_divide_type(&[lhs, rhs]);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(2)])
            }
        );
    }
}
