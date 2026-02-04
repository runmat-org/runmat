use runmat_builtins::Type;

use runmat_builtins::shape_rules::{
    constructor_shape_from_dims, element_count_if_known, unknown_shape,
};
use runmat_builtins::ResolveContext;

pub fn rank_from_dims_args(args: &[Type], ctx: &ResolveContext) -> Option<usize> {
    if args.is_empty() {
        return None;
    }
    let dims = ctx.numeric_dims();
    if !dims.is_empty() {
        return Some(dims.len());
    }
    rank_from_dims_args_legacy(args)
}

pub fn rank_from_dims_args_legacy(args: &[Type]) -> Option<usize> {
    if args.is_empty() {
        return None;
    }
    if args.len() >= 2 {
        return Some(args.len());
    }
    let arg = &args[0];
    match arg {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            element_count_if_known(shape).map(|count| if count == 1 { 2 } else { count })
        }
        Type::Num | Type::Int | Type::Bool => Some(2),
        _ => None,
    }
}

pub fn tensor_type_from_rank(args: &[Type], ctx: &ResolveContext) -> Type {
    if let Some(ty) = tensor_type_from_literal_dims(args, ctx) {
        return ty;
    }
    let rank = rank_from_dims_args(args, ctx);
    tensor_type_from_rank_legacy(rank)
}

pub fn tensor_type_from_rank_legacy(rank: Option<usize>) -> Type {
    match rank {
        Some(rank) => Type::Tensor {
            shape: Some(unknown_shape(rank)),
        },
        None => Type::tensor(),
    }
}

pub fn tensor_type_from_literal_dims(args: &[Type], ctx: &ResolveContext) -> Option<Type> {
    if args.is_empty() {
        return None;
    }
    if args.iter().any(|arg| matches!(arg, Type::String)) {
        return Some(Type::Unknown);
    }
    let dims = ctx.numeric_dims();
    if dims.is_empty() {
        return None;
    }
    constructor_shape_from_dims(&dims).map(|shape| Type::Tensor { shape: Some(shape) })
}

pub fn logical_type_from_rank(rank: Option<usize>) -> Type {
    match rank {
        Some(rank) => Type::Logical {
            shape: Some(unknown_shape(rank)),
        },
        None => Type::logical(),
    }
}

pub fn row_vector_type(ctx: &ResolveContext) -> Type {
    if let Some(Some(len)) = ctx.numeric_dims().get(0) {
        return Type::Tensor {
            shape: Some(vec![Some(1), Some(*len)]),
        };
    }
    row_vector_type_legacy()
}

pub fn row_vector_type_legacy() -> Type {
    Type::Tensor {
        shape: Some(vec![Some(1), None]),
    }
}

pub fn column_vector_type() -> Type {
    Type::Tensor {
        shape: Some(vec![None, Some(1)]),
    }
}

pub fn size_vector_len(ty: &Type) -> Option<usize> {
    match ty {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            element_count_if_known(shape)
        }
        Type::Num | Type::Int | Type::Bool => Some(1),
        _ => None,
    }
}

pub fn is_scalar_type(ty: &Type) -> bool {
    match ty {
        Type::Num | Type::Int | Type::Bool => true,
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            element_count_if_known(shape) == Some(1)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::LiteralValue;

    #[test]
    fn rank_from_dims_args_uses_literal_vector() {
        let ctx = ResolveContext::new(vec![LiteralValue::Vector(vec![
            LiteralValue::Number(2.0),
            LiteralValue::Number(3.0),
        ])]);
        let rank = rank_from_dims_args(&[Type::Num], &ctx);
        assert_eq!(rank, Some(2));
    }

    #[test]
    fn tensor_type_from_rank_uses_literal_dims() {
        let ctx = ResolveContext::new(vec![LiteralValue::Vector(vec![
            LiteralValue::Number(2.0),
            LiteralValue::Number(3.0),
        ])]);
        let out = tensor_type_from_rank(&[Type::Num], &ctx);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn row_vector_type_uses_literal_length() {
        let ctx = ResolveContext::new(vec![LiteralValue::Number(4.0)]);
        let out = row_vector_type(&ctx);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(4)])
            }
        );
    }
}
