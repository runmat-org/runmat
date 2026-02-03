use runmat_builtins::Type;

pub fn index_output_type(_args: &[Type]) -> Type {
    Type::tensor()
}

pub fn logical_output_type(_args: &[Type]) -> Type {
    Type::logical()
}

pub fn bool_output_type(_args: &[Type]) -> Type {
    Type::Bool
}

pub fn tensor_output_type(args: &[Type]) -> Type {
    match args.first() {
        Some(Type::Tensor { .. })
        | Some(Type::Logical { .. })
        | Some(Type::Num)
        | Some(Type::Int)
        | Some(Type::Bool) => Type::tensor(),
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::Unknown,
    }
}

pub fn unknown_output_type(_args: &[Type]) -> Type {
    Type::Unknown
}
