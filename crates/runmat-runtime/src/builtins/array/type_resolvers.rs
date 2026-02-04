use runmat_builtins::Type;

use runmat_builtins::shape_rules::{
    constructor_shape_from_dims, element_count_if_known, unknown_shape,
};
use runmat_builtins::ResolveContext;

pub fn rank_from_dims_args(args: &[Type]) -> Option<usize> {
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

pub fn tensor_type_from_rank(rank: Option<usize>) -> Type {
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

pub fn row_vector_type() -> Type {
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
