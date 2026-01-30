use crate::builtins::common::type_shapes::{scalar_tensor_shape, unknown_shape};
use runmat_builtins::Type;

fn reduction_shape_from_args(args: &[Type]) -> Option<Vec<Option<usize>>> {
    let input = args.first()?;
    let shape = match input {
        Type::Tensor { shape } => shape.clone(),
        Type::Logical { shape } => shape.clone(),
        _ => None,
    };
    let shape = shape?;
    if args.len() == 1 {
        Some(reduce_first_nonsingleton(&shape))
    } else {
        Some(unknown_shape(shape.len()))
    }
}

pub fn reduce_first_nonsingleton(shape: &[Option<usize>]) -> Vec<Option<usize>> {
    if shape.is_empty() {
        return scalar_tensor_shape();
    }
    let mut dim = 0usize;
    for (i, entry) in shape.iter().enumerate() {
        if let Some(value) = entry {
            if *value != 1 {
                dim = i;
                break;
            }
        } else {
            dim = 0;
            break;
        }
    }
    let mut out = shape.to_vec();
    if let Some(entry) = out.get_mut(dim) {
        *entry = Some(1);
    }
    out
}

pub fn reduce_numeric_type(args: &[Type]) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    match input {
        Type::Tensor { shape: Some(_) } | Type::Logical { shape: Some(_) } => Type::Tensor {
            shape: reduction_shape_from_args(args),
        },
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
        Type::Bool | Type::Num | Type::Int => Type::Num,
        _ => Type::Unknown,
    }
}

pub fn reduce_logical_type(args: &[Type]) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    match input {
        Type::Tensor { shape: Some(_) } | Type::Logical { shape: Some(_) } => Type::Logical {
            shape: reduction_shape_from_args(args),
        },
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::logical(),
        Type::Bool | Type::Num | Type::Int => Type::Bool,
        _ => Type::Unknown,
    }
}

pub fn cumulative_numeric_type(args: &[Type]) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    match input {
        Type::Tensor { shape: Some(shape) } => Type::Tensor {
            shape: Some(shape.clone()),
        },
        Type::Logical { shape: Some(shape) } => Type::Tensor {
            shape: Some(shape.clone()),
        },
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
        Type::Bool | Type::Num | Type::Int => Type::Num,
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

pub fn diff_numeric_type(args: &[Type]) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    let only_input = args.len() <= 1;
    match input {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            if only_input {
                Type::Tensor {
                    shape: Some(unknown_shape(shape.len())),
                }
            } else {
                Type::tensor()
            }
        }
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
        Type::Bool | Type::Num | Type::Int => {
            if only_input {
                Type::tensor()
            } else {
                Type::Unknown
            }
        }
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

pub fn count_nonzero_type(args: &[Type]) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    if args.len() <= 1 {
        match input {
            Type::Tensor { .. } | Type::Logical { .. } | Type::Bool | Type::Num | Type::Int => {
                Type::Num
            }
            Type::Unknown => Type::Unknown,
            _ => Type::Unknown,
        }
    } else {
        reduce_numeric_type(args)
    }
}

pub fn min_max_type(args: &[Type]) -> Type {
    if args.len() <= 1 {
        return reduce_numeric_type(args);
    }
    let mut has_array = false;
    let mut has_unknown = false;
    for arg in args {
        match arg {
            Type::Tensor { .. } | Type::Logical { .. } => has_array = true,
            Type::Unknown => has_unknown = true,
            _ => {}
        }
    }
    if has_array {
        Type::tensor()
    } else if has_unknown {
        Type::Unknown
    } else {
        Type::Num
    }
}
