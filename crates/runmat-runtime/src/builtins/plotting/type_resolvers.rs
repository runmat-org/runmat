use runmat_builtins::{ResolveContext, Type};

use runmat_builtins::shape_rules::element_count_if_known;

use crate::builtins::array::type_resolvers::{row_vector_type, size_vector_len};

pub fn handle_scalar_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Num
}

pub fn bool_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Bool
}

pub fn axis_type(args: &[Type], _context: &ResolveContext) -> Type {
    match args {
        [] => Type::tensor(),
        [Type::Num | Type::Int | Type::Unknown] => Type::Unknown,
        _ => Type::Void,
    }
}

pub fn handle_logical_type(args: &[Type], _context: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Logical {
            shape: shape.clone(),
        },
        _ => Type::Bool,
    }
}

pub fn handle_array_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::tensor()
}

pub fn gca_type(args: &[Type], _context: &ResolveContext) -> Type {
    match args {
        [] => Type::Num,
        [Type::Num | Type::Int] => Type::Num,
        [Type::Tensor { shape: Some(shape) }] if element_count_if_known(shape) == Some(1) => {
            Type::Num
        }
        [Type::String] => Type::Struct {
            known_fields: Some(vec![
                "handle".to_string(),
                "figure".to_string(),
                "rows".to_string(),
                "cols".to_string(),
                "index".to_string(),
            ]),
        },
        _ => Type::Unknown,
    }
}

pub fn hist_type(args: &[Type], ctx: &ResolveContext) -> Type {
    let bins_len = args.get(1).and_then(size_vector_len).filter(|len| *len > 1);
    match bins_len {
        Some(len) => Type::Tensor {
            shape: Some(vec![Some(1), Some(len)]),
        },
        None => row_vector_type(ctx),
    }
}

pub fn get_type(args: &[Type], _context: &ResolveContext) -> Type {
    if args.len() <= 1 {
        return Type::Struct { known_fields: None };
    }
    Type::Unknown
}

pub fn daspect_type(args: &[Type], _context: &ResolveContext) -> Type {
    let ratio = Type::Tensor {
        shape: Some(vec![Some(1), Some(3)]),
    };
    match args {
        [] => ratio,
        [Type::String] => Type::String,
        [Type::Num | Type::Int] => ratio,
        [Type::Tensor { .. } | Type::Logical { .. }] => ratio,
        [Type::Num | Type::Int, Type::String] => Type::String,
        [Type::Num | Type::Int, Type::Tensor { .. } | Type::Logical { .. }] => ratio,
        _ => Type::Unknown,
    }
}

pub fn set_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Void
}
