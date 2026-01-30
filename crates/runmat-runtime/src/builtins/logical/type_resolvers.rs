use runmat_builtins::Type;

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
        if a == b {
            return Type::Logical {
                shape: Some(a.clone()),
            };
        }
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

pub fn logical_binary_type(args: &[Type]) -> Type {
    if args.len() >= 2 {
        logical_result_for_binary(&args[0], &args[1])
    } else if let Some(first) = args.first() {
        logical_like(first)
    } else {
        Type::Unknown
    }
}

pub fn logical_unary_type(args: &[Type]) -> Type {
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
        assert_eq!(
            logical_binary_type(&[lhs, rhs]),
            Type::Logical {
                shape: Some(vec![Some(2), Some(2)])
            }
        );
    }

    #[test]
    fn logical_binary_scalar_defaults_bool() {
        assert_eq!(logical_binary_type(&[Type::Num, Type::Bool]), Type::Bool);
    }
}
