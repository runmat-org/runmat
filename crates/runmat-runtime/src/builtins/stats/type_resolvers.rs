use runmat_builtins::Type;

use crate::builtins::array::type_resolvers::{row_vector_type, size_vector_len};
use crate::builtins::common::type_shapes::{element_count_if_known, unknown_shape};

pub fn cov_type(args: &[Type]) -> Type {
    square_summary_type(args)
}

pub fn corrcoef_type(args: &[Type]) -> Type {
    square_summary_type(args)
}

pub fn histcounts_type(args: &[Type]) -> Type {
    let mut edges_len: Option<usize> = None;
    for arg in args.iter().skip(1) {
        if let Some(len) = size_vector_len(arg) {
            if len > 1 {
                edges_len = Some(len);
                break;
            }
        }
    }
    match edges_len {
        Some(len) if len > 1 => Type::Tensor {
            shape: Some(vec![Some(1), Some(len - 1)]),
        },
        _ => row_vector_type(),
    }
}

pub fn histcounts2_type(args: &[Type]) -> Type {
    let mut edges_lens: Vec<usize> = Vec::new();
    for arg in args.iter().skip(2) {
        if let Some(len) = size_vector_len(arg) {
            if len > 1 {
                edges_lens.push(len);
            }
            if edges_lens.len() == 2 {
                break;
            }
        }
    }
    let bins_x = edges_lens.get(0).and_then(|len| len.checked_sub(1));
    let bins_y = edges_lens.get(1).and_then(|len| len.checked_sub(1));
    Type::Tensor {
        shape: Some(vec![bins_x, bins_y]),
    }
}

pub fn rng_type(_args: &[Type]) -> Type {
    Type::Struct {
        known_fields: Some(vec![
            "Seed".to_string(),
            "State".to_string(),
            "Type".to_string(),
        ]),
    }
}

fn square_summary_type(args: &[Type]) -> Type {
    let Some(first) = args.first() else {
        return Type::Unknown;
    };
    let mut data: Vec<&Type> = args
        .iter()
        .filter(|arg| matches!(arg, Type::Tensor { .. } | Type::Logical { .. }))
        .take(2)
        .collect();
    if data.is_empty() {
        if is_numeric_scalar(first) {
            data.push(first);
        }
    }
    if data.is_empty() {
        return Type::Unknown;
    }
    let total = match data.len() {
        1 => variable_count(data[0]),
        2 => match (variable_count(data[0]), variable_count(data[1])) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        },
        _ => None,
    };
    match total {
        Some(count) => numeric_tensor_from_shape(vec![Some(count), Some(count)]),
        None => Type::Tensor {
            shape: Some(unknown_shape(2)),
        },
    }
}

fn variable_count(ty: &Type) -> Option<usize> {
    match ty {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            variable_count_from_shape(shape)
        }
        Type::Tensor { shape: None } | Type::Logical { shape: None } => None,
        Type::Num | Type::Int | Type::Bool => Some(1),
        Type::Unknown => None,
        _ => None,
    }
}

fn variable_count_from_shape(shape: &[Option<usize>]) -> Option<usize> {
    match shape.len() {
        0 => Some(1),
        1 => Some(1),
        _ => {
            let rows = shape.get(0).and_then(|v| *v);
            let cols = shape.get(1).and_then(|v| *v);
            if rows == Some(1) || cols == Some(1) {
                Some(1)
            } else {
                cols
            }
        }
    }
}

fn numeric_tensor_from_shape(shape: Vec<Option<usize>>) -> Type {
    if element_count_if_known(&shape) == Some(1) {
        Type::Num
    } else {
        Type::Tensor { shape: Some(shape) }
    }
}

fn is_numeric_scalar(ty: &Type) -> bool {
    match ty {
        Type::Num | Type::Int | Type::Bool => true,
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            element_count_if_known(shape.as_slice()) == Some(1)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_type_is_struct() {
        assert!(matches!(rng_type(&[]), Type::Struct { .. }));
    }
}
