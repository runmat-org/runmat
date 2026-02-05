use runmat_builtins::Type;

use runmat_builtins::{ResolveContext, Type};

pub fn index_output_type(_args: &[Type], _context: &ResolveContext) -> Type {
    match _args.first() {
        Some(Type::Tensor { shape: Some(shape) }) => Type::Tensor {
            shape: Some(shape.clone()),
        },
        Some(Type::Logical { shape: Some(shape) }) => Type::Tensor {
            shape: Some(shape.clone()),
        },
        Some(Type::Tensor { .. }) | Some(Type::Logical { .. }) => Type::tensor(),
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::tensor(),
    }
}

pub fn logical_output_type(_args: &[Type], _context: &ResolveContext) -> Type {
    match _args.first() {
        Some(Type::Tensor { shape: Some(shape) }) => Type::Logical {
            shape: Some(shape.clone()),
        },
        Some(Type::Logical { shape }) => Type::Logical {
            shape: shape.clone(),
        },
        Some(Type::Tensor { .. }) => Type::logical(),
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::Bool,
    }
}

pub fn bool_output_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Bool
}

pub fn tensor_output_type(args: &[Type], _context: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Tensor { shape: Some(shape) }) => Type::Tensor {
            shape: Some(shape.clone()),
        },
        Some(Type::Logical { shape: Some(shape) }) => Type::Tensor {
            shape: Some(shape.clone()),
        },
        Some(Type::Tensor { .. })
        | Some(Type::Logical { .. })
        | Some(Type::Num)
        | Some(Type::Int)
        | Some(Type::Bool) => Type::tensor(),
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::Unknown,
    }
}

pub fn set_values_output_type(args: &[Type], _context: &ResolveContext) -> Type {
    if args.is_empty() {
        return Type::Unknown;
    }

    if args.iter().any(|ty| matches!(ty, Type::Unknown)) {
        return Type::Unknown;
    }

    let mut has_string = false;
    let mut has_numeric = false;

    for arg in args {
        match arg {
            Type::String => has_string = true,
            Type::Cell {
                element_type: Some(element_type),
                ..
            } if **element_type == Type::String => has_string = true,
            Type::Tensor { .. } | Type::Logical { .. } | Type::Num | Type::Int | Type::Bool => {
                has_numeric = true
            }
            _ => {}
        }
    }

    if has_string && has_numeric {
        return Type::Unknown;
    }

    if has_string {
        return Type::cell_of(Type::String);
    }

    if has_numeric {
        return Type::tensor();
    }

    Type::Unknown
}
