use runmat_builtins::Type;

use runmat_builtins::shape_rules::broadcast_shapes;
use runmat_builtins::{ResolveContext, Type};

pub fn logical_like(input: &Type) -> Type {
    match input {
        Type::Tensor { shape: Some(shape) } => Type::Logical {
            shape: Some(shape.clone()),
        },
        Type::Tensor { shape: None } => Type::logical(),
        Type::Logical { shape } => Type::Logical {
            shape: shape.clone(),
        },
        Type::Unknown => Type::logical(),
        _ => Type::Bool,
    }
}

pub fn logical_result_for_binary(lhs: &Type, rhs: &Type) -> Type {
    let lhs_shape = match lhs {
        Type::Tensor { shape: Some(shape) } => Some(shape.clone()),
        Type::Logical { shape: Some(shape) } => Some(shape.clone()),
        _ => None,
    };
    let rhs_shape = match rhs {
        Type::Tensor { shape: Some(shape) } => Some(shape.clone()),
        Type::Logical { shape: Some(shape) } => Some(shape.clone()),
        _ => None,
    };
    if let (Some(a), Some(b)) = (&lhs_shape, &rhs_shape) {
        return Type::Logical {
            shape: Some(broadcast_shapes(a, b)),
        };
    }
    if let Some(shape) = lhs_shape {
        return Type::Logical { shape: Some(shape) };
    }
    if let Some(shape) = rhs_shape {
        return Type::Logical { shape: Some(shape) };
    }
    if matches!(lhs, Type::Tensor { .. } | Type::Logical { .. })
        || matches!(rhs, Type::Tensor { .. } | Type::Logical { .. })
    {
        Type::logical()
    } else if matches!(lhs, Type::Unknown) || matches!(rhs, Type::Unknown) {
        Type::Unknown
    } else {
        Type::Bool
    }
}

pub fn logical_binary_type(args: &[Type], _context: &ResolveContext) -> Type {
    if args.len() >= 2 {
        logical_result_for_binary(&args[0], &args[1])
    } else if let Some(first) = args.first() {
        logical_like(first)
    } else {
        Type::Unknown
    }
}

pub fn logical_unary_type(args: &[Type], _context: &ResolveContext) -> Type {
    args.first().map(logical_like).unwrap_or(Type::logical())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_like_preserves_shape() {
        let ty = Type::Tensor {
            shape: Some(vec![Some(2), Some(3)]),
        };
        assert_eq!(
            logical_like(&ty),
            Type::Logical {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn logical_binary_prefers_matching_shape() {
        let lhs = Type::Tensor {
            shape: Some(vec![Some(2), Some(2)]),
        };
        let rhs = Type::Logical {
            shape: Some(vec![Some(2), Some(2)]),
        };
        let out = logical_binary_type(&[lhs, rhs], &ResolveContext::new(Vec::new()));
        assert_eq!(
            out,
            Type::Logical {
                shape: Some(vec![Some(2), Some(2)])
            }
        );
    }

    #[test]
    fn logical_binary_scalar_defaults_bool() {
        let out = logical_binary_type(&[Type::Num, Type::Bool], &ResolveContext::new(Vec::new()));
        assert_eq!(out, Type::Bool);
    }

    #[test]
    fn logical_binary_broadcasts_shapes() {
        let lhs = Type::Tensor {
            shape: Some(vec![Some(1), Some(4)]),
        };
        let rhs = Type::Logical {
            shape: Some(vec![Some(3), Some(1)]),
        };
        let out = logical_binary_type(&[lhs, rhs], &ResolveContext::new(Vec::new()));
        assert_eq!(
            out,
            Type::Logical {
                shape: Some(vec![Some(3), Some(4)])
            }
        );
    }
}
