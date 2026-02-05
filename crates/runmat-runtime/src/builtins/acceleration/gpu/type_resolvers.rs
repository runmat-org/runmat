use runmat_builtins::{ResolveContext, Type};

pub fn arrayfun_type(args: &[Type], _context: &ResolveContext) -> Type {
    if args.len() < 2 {
        return Type::Unknown;
    }

    if args.iter().skip(1).any(|ty| matches!(ty, Type::String)) {
        return Type::Unknown;
    }

    let Type::Function { returns, .. } = &args[0] else {
        return Type::Unknown;
    };

    arrayfun_output_type(returns)
}

pub fn gather_type(args: &[Type], _context: &ResolveContext) -> Type {
    match args.len() {
        0 => Type::Unknown,
        1 => args[0].clone(),
        _ => Type::cell(),
    }
}

pub fn gpuarray_type(args: &[Type], _context: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Logical { shape }) => logical_type_from_shape(shape.as_ref()),
        Some(Type::Bool) => Type::logical_with_shape(vec![1, 1]),
        Some(Type::Unknown) | None => Type::Unknown,
        Some(Type::Cell { .. }) => Type::Unknown,
        Some(Type::Tensor { shape }) => tensor_type_from_shape(shape.as_ref()),
        Some(Type::Num) | Some(Type::Int) => Type::tensor_with_shape(vec![1, 1]),
        _ => Type::tensor(),
    }
}

pub fn gpudevice_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Struct {
        known_fields: Some(vec![
            "backend".to_string(),
            "device_id".to_string(),
            "index".to_string(),
            "memory_bytes".to_string(),
            "name".to_string(),
            "precision".to_string(),
            "supports_double".to_string(),
            "vendor".to_string(),
        ]),
    }
}

pub fn gpuinfo_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::String
}

pub fn pagefun_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::tensor()
}

fn arrayfun_output_type(returns: &Type) -> Type {
    match returns {
        Type::Bool | Type::Logical { .. } => Type::logical(),
        Type::Num | Type::Int | Type::Tensor { .. } => Type::tensor(),
        Type::Unknown | Type::Cell { .. } | Type::String | Type::Struct { .. } => Type::Unknown,
        Type::Function { .. } | Type::Void | Type::Union(_) => Type::Unknown,
    }
}

fn tensor_type_from_shape(shape: Option<&Vec<Option<usize>>>) -> Type {
    match shape {
        Some(shape) => Type::Tensor {
            shape: Some(shape.clone()),
        },
        None => Type::tensor(),
    }
}

fn logical_type_from_shape(shape: Option<&Vec<Option<usize>>>) -> Type {
    match shape {
        Some(shape) => Type::Logical {
            shape: Some(shape.clone()),
        },
        None => Type::logical(),
    }
}
