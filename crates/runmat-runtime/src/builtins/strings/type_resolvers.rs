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

pub fn text_preserve_type(args: &[Type]) -> Type {
    match args.first() {
        Some(Type::String) => Type::String,
        Some(Type::Cell {
            element_type: Some(element_type),
            ..
        }) if **element_type == Type::String => Type::cell_of(Type::String),
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::Unknown,
    }
}

pub fn text_concat_type(args: &[Type]) -> Type {
    if args.iter().any(|ty| matches!(ty, Type::Unknown)) {
        return Type::Unknown;
    }

    if args.iter().any(|ty| {
        matches!(
            ty,
            Type::Cell {
                element_type: Some(element_type),
                ..
            } if **element_type == Type::String
        )
    }) {
        return Type::cell_of(Type::String);
    }

    if args.iter().any(|ty| *ty == Type::String) {
        return Type::String;
    }

    Type::Unknown
}

pub fn text_search_indices_type(args: &[Type]) -> Type {
    if args.iter().any(|ty| matches!(ty, Type::Unknown)) {
        return Type::Unknown;
    }

    match args.first() {
        Some(Type::String) => Type::tensor(),
        Some(Type::Cell {
            element_type: Some(element_type),
            ..
        }) if **element_type == Type::String => Type::cell(),
        _ => Type::Unknown,
    }
}

pub fn unknown_type(_args: &[Type]) -> Type {
    Type::Unknown
}
