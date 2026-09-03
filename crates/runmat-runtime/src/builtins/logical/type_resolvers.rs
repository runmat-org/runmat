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
}
