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
        Type::Unknown
        | Type::Cell { .. }
        | Type::String
        | Type::Struct { .. }
        | Type::Object { .. }
        | Type::Symbolic
        | Type::SymbolicArray { .. } => Type::Unknown,
        Type::Function { .. } | Type::Void | Type::Union(_) | Type::OutputList(_) => Type::Unknown,
    }
}
