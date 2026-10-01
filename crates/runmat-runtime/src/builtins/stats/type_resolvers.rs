use crate::builtins::math::reduction::type_resolvers::reduce_numeric_type;
use runmat_builtins::shape_rules::unknown_shape;
use runmat_builtins::{ResolveContext, Type};

/// Type resolver for `mode`. Mirrors the reduction-style shape inference used by
/// `mean` / `median` / `var`: the leading output (`M`) keeps the input's element
/// type and reduces the requested dimension (or the first non-singleton dim by
/// default). The frequency (`F`) output mirrors `M`'s shape, while the cell
/// array (`C`) is described separately at the type level.
pub fn mode_type(args: &[Type], context: &ResolveContext) -> Type {
    let reduced = reduce_numeric_type(args, context);
    match args.first() {
        Some(Type::Logical { .. }) => match reduced {
            Type::Tensor { shape } | Type::Logical { shape } => Type::Logical { shape },
            Type::Num | Type::Bool => Type::Bool,
            other => other,
        },
        Some(Type::Bool) => Type::Bool,
        Some(Type::Int) => Type::Int,
        _ => reduced,
    }
}

pub fn rng_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Struct {
        known_fields: Some(vec![
            "Seed".to_string(),
            "State".to_string(),
            "Type".to_string(),
        ]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_type_preserves_logical_shape() {
        let ty = Type::Logical {
            shape: Some(vec![Some(3), Some(4)]),
        };
        let out = mode_type(&[ty], &ResolveContext::new(Vec::new()));
        assert_eq!(
            out,
            Type::Logical {
                shape: Some(vec![Some(1), Some(4)])
            }
        );
    }

    #[test]
    fn mode_type_preserves_scalar_bool_and_int() {
        assert_eq!(
            mode_type(&[Type::Bool], &ResolveContext::new(Vec::new())),
            Type::Bool
        );
        assert_eq!(
            mode_type(&[Type::Int], &ResolveContext::new(Vec::new())),
            Type::Int
        );
    }

    #[test]
    fn rng_type_is_struct() {
        assert!(matches!(
            rng_type(&[], &ResolveContext::new(Vec::new())),
            Type::Struct { .. }
        ));
    }
}
