use runmat_builtins::{ResolveContext, Type};

use crate::builtins::array::type_resolvers::{row_vector_type, size_vector_len};

pub fn handle_scalar_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Num
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

pub fn set_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Void
}
