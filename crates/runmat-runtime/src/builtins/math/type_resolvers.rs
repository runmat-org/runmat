use runmat_builtins::Type;

use crate::builtins::common::type_shapes::{broadcast_shapes, element_count_if_known};

pub fn numeric_unary_type(args: &[Type]) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    match input {
        Type::Tensor { shape } => tensor_like_result(shape),
        Type::Logical { shape } => tensor_like_result(shape),
        Type::Num | Type::Int | Type::Bool => Type::Num,
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

pub fn numeric_binary_type(args: &[Type]) -> Type {
    let lhs = args.get(0);
    let rhs = args.get(1);
    match (lhs, rhs) {
        (Some(left), Some(right)) => binary_result_type(left, right),
        (Some(single), None) | (None, Some(single)) => {
            numeric_unary_type(std::slice::from_ref(single))
        }
        (None, None) => Type::Unknown,
    }
}

fn binary_result_type(lhs: &Type, rhs: &Type) -> Type {
    if is_numeric_scalar(lhs) && is_numeric_scalar(rhs) {
        return Type::Num;
    }

    let lhs_shape = numeric_array_shape(lhs);
    let rhs_shape = numeric_array_shape(rhs);
    if let (Some(a), Some(b)) = (&lhs_shape, &rhs_shape) {
        return Type::Tensor {
            shape: Some(broadcast_shapes(a, b)),
        };
    }
    if let Some(shape) = lhs_shape.or(rhs_shape) {
        return Type::Tensor { shape: Some(shape) };
    }
    if is_numeric_array_type(lhs) || is_numeric_array_type(rhs) {
        return Type::tensor();
    }
    if matches!(lhs, Type::Unknown) || matches!(rhs, Type::Unknown) {
        return Type::Unknown;
    }
    Type::Num
}

fn tensor_like_result(shape: &Option<Vec<Option<usize>>>) -> Type {
    match shape {
        Some(dims) => match element_count_if_known(dims) {
            Some(1) => Type::Num,
            _ => Type::Tensor {
                shape: Some(dims.clone()),
            },
        },
        None => Type::tensor(),
    }
}

fn is_numeric_scalar(ty: &Type) -> bool {
    match ty {
        Type::Num | Type::Int | Type::Bool => true,
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            element_count_if_known(shape) == Some(1)
        }
        _ => false,
    }
}

fn numeric_array_shape(ty: &Type) -> Option<Vec<Option<usize>>> {
    match ty {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            if element_count_if_known(shape) == Some(1) {
                None
            } else {
                Some(shape.clone())
            }
        }
        _ => None,
    }
}

fn is_numeric_array_type(ty: &Type) -> bool {
    matches!(ty, Type::Tensor { .. } | Type::Logical { .. })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_unary_preserves_shape() {
        let out = numeric_unary_type(&[Type::Tensor {
            shape: Some(vec![Some(2), Some(3)]),
        }]);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn numeric_unary_scalar_tensor_returns_num() {
        let out = numeric_unary_type(&[Type::Tensor {
            shape: Some(vec![Some(1), Some(1)]),
        }]);
        assert_eq!(out, Type::Num);
    }

    #[test]
    fn numeric_binary_scalar_returns_num() {
        let out = numeric_binary_type(&[Type::Num, Type::Int]);
        assert_eq!(out, Type::Num);
    }

    #[test]
    fn numeric_binary_prefers_tensor_shape() {
        let out = numeric_binary_type(&[
            Type::Num,
            Type::Tensor {
                shape: Some(vec![Some(4), Some(1)]),
            },
        ]);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(4), Some(1)])
            }
        );
    }

    #[test]
    fn numeric_binary_broadcasts_shapes() {
        let out = numeric_binary_type(&[
            Type::Tensor {
                shape: Some(vec![Some(1), Some(3)]),
            },
            Type::Tensor {
                shape: Some(vec![Some(2), Some(1)]),
            },
        ]);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }
}
