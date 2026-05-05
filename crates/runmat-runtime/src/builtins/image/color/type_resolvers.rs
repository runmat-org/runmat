use runmat_builtins::{ResolveContext, Type};

use super::common;

pub fn same_shape_type(args: &[Type], _context: &ResolveContext) -> Type {
    common::same_shape_type(args)
}

pub fn rgb2gray_type(args: &[Type], _context: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Tensor { shape: Some(shape) }) if shape.len() == 3 => Type::Tensor {
            shape: Some(vec![shape[0], shape[1]]),
        },
        Some(Type::Tensor { .. }) => Type::tensor(),
        _ => Type::tensor(),
    }
}

pub fn gray2rgb_type(args: &[Type], _context: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Tensor { shape: Some(shape) }) if shape.len() == 2 => Type::Tensor {
            shape: Some(vec![shape[0], shape[1], Some(3)]),
        },
        Some(Type::Logical { shape: Some(shape) }) if shape.len() == 2 => Type::Tensor {
            shape: Some(vec![shape[0], shape[1], Some(3)]),
        },
        _ => Type::tensor(),
    }
}

pub fn ind2rgb_type(args: &[Type], _context: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Tensor { shape: Some(shape) }) | Some(Type::Logical { shape: Some(shape) }) => {
            let mut out = shape.clone();
            out.push(Some(3));
            Type::Tensor { shape: Some(out) }
        }
        Some(Type::Num) | Some(Type::Int) | Some(Type::Bool) => Type::Tensor {
            shape: Some(vec![Some(1), Some(1), Some(3)]),
        },
        _ => Type::tensor(),
    }
}
