use runmat_builtins::{ResolveContext, Type};

use crate::builtins::common::arg_tokens::{tokens_from_context, ArgToken};

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

pub fn unique_values_output_type(args: &[Type], context: &ResolveContext) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    if matches!(input, Type::Unknown) {
        return Type::Unknown;
    }

    let rows = tokens_from_context(context)
        .iter()
        .skip(1)
        .any(|token| matches!(token, ArgToken::String(option) if option == "rows"));

    match input {
        Type::Tensor { shape } | Type::Logical { shape } => Type::Tensor {
            shape: unique_numeric_output_shape(shape.as_deref(), rows),
        },
        Type::Num | Type::Int | Type::Bool => Type::Tensor {
            shape: Some(vec![Some(1), Some(1)]),
        },
        Type::String => Type::cell_of(Type::String),
        Type::Cell {
            element_type: Some(element_type),
            ..
        } if **element_type == Type::String => Type::cell_of(Type::String),
        _ => Type::Unknown,
    }
}

fn unique_numeric_output_shape(
    input_shape: Option<&[Option<usize>]>,
    rows: bool,
) -> Option<Vec<Option<usize>>> {
    let shape = input_shape?;
    if rows {
        if shape.len() != 2 {
            return None;
        }
        let output_rows = if shape.first() == Some(&Some(0)) {
            Some(0)
        } else {
            None
        };
        return Some(vec![output_rows, shape[1]]);
    }

    let output_len = if shape.iter().any(|dimension| *dimension == Some(0)) {
        Some(0)
    } else if shape.iter().all(|dimension| *dimension == Some(1)) {
        Some(1)
    } else {
        None
    };
    match known_row_vector_orientation(shape) {
        Some(true) => Some(vec![Some(1), output_len]),
        Some(false) => Some(vec![output_len, Some(1)]),
        None => None,
    }
}

fn known_row_vector_orientation(shape: &[Option<usize>]) -> Option<bool> {
    match shape {
        [] => Some(false),
        [_] => Some(true),
        [Some(rows), ..] if *rows != 1 => Some(false),
        [Some(1), _, rest @ ..] => {
            if rest.iter().all(|dimension| *dimension == Some(1)) {
                Some(true)
            } else if rest
                .iter()
                .any(|dimension| dimension.is_some_and(|value| value != 1))
            {
                Some(false)
            } else {
                None
            }
        }
        [Some(_), ..] => Some(false),
        [None, ..] => None,
    }
}
