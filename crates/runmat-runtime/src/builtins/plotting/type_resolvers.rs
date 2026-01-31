use runmat_builtins::Type;

use crate::builtins::array::type_resolvers::{row_vector_type, size_vector_len};

pub fn string_type(_args: &[Type]) -> Type {
    Type::String
}

pub fn handle_scalar_type(_args: &[Type]) -> Type {
    Type::Num
}

pub fn gca_type(args: &[Type]) -> Type {
    if args.is_empty() {
        return Type::Num;
    }
    Type::Struct {
        known_fields: Some(vec![
            "handle".to_string(),
            "figure".to_string(),
            "rows".to_string(),
            "cols".to_string(),
            "index".to_string(),
        ]),
    }
}

pub fn hist_type(args: &[Type]) -> Type {
    let bins_len = args.get(1).and_then(size_vector_len).filter(|len| *len > 1);
    match bins_len {
        Some(len) => Type::Tensor {
            shape: Some(vec![Some(1), Some(len)]),
        },
        None => row_vector_type(),
    }
}
