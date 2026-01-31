use runmat_builtins::Type;

pub fn string_array_type(_args: &[Type]) -> Type {
    Type::cell_of(Type::String)
}

pub fn string_scalar_type(_args: &[Type]) -> Type {
    Type::String
}

pub fn numeric_text_scalar_or_tensor_type(args: &[Type]) -> Type {
    match args.first() {
        Some(Type::String) => Type::Num,
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::tensor(),
    }
}

pub fn logical_text_match_type(args: &[Type]) -> Type {
    if args.len() >= 2 && args[0] == Type::String && args[1] == Type::String {
        Type::Bool
    } else if args.iter().any(|ty| matches!(ty, Type::Unknown)) {
        Type::Unknown
    } else {
        Type::logical()
    }
}

pub fn unknown_type(_args: &[Type]) -> Type {
    Type::Unknown
}
