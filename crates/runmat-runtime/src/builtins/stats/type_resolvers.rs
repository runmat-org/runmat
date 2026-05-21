use crate::builtins::array::type_resolvers::{row_vector_type, size_vector_len};
use crate::builtins::math::reduction::type_resolvers::reduce_numeric_type;
use runmat_builtins::shape_rules::{element_count_if_known, unknown_shape};
use runmat_builtins::{ResolveContext, Type};

pub fn cov_type(args: &[Type], _context: &ResolveContext) -> Type {
    square_summary_type(args)
}

pub fn corrcoef_type(args: &[Type], _context: &ResolveContext) -> Type {
    square_summary_type(args)
}

/// Type resolver for `mode`. Mirrors the reduction-style shape inference used by
/// `mean` / `median` / `var`: the leading output (`M`) keeps the input's element
/// type and reduces the requested dimension (or the first non-singleton dim by
/// default). The frequency (`F`) output mirrors `M`'s shape, while the cell
/// array (`C`) is described separately at the type level.
pub fn mode_type(args: &[Type], context: &ResolveContext) -> Type {
    let reduced = reduce_numeric_type(args, context);
    match args.first() {
        Some(Type::Logical { .. }) => match reduced {
            Type::Tensor { shape } | Type::Logical { shape } => Type::Logical { shape },
            Type::Num | Type::Bool => Type::Bool,
            other => other,
        },
        Some(Type::Bool) => Type::Bool,
        Some(Type::Int) => Type::Int,
        _ => reduced,
    }
}

pub fn histcounts_type(args: &[Type], ctx: &ResolveContext) -> Type {
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
        _ => row_vector_type(ctx),
    }
}

pub fn histcounts2_type(args: &[Type], _context: &ResolveContext) -> Type {
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

pub fn rng_type(_args: &[Type], _context: &ResolveContext) -> Type {
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
    if data.is_empty() && is_numeric_scalar(first) {
        data.push(first);
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
    fn mode_type_preserves_logical_shape() {
        let ty = Type::Logical {
            shape: Some(vec![Some(3), Some(4)]),
        };
        let out = mode_type(&[ty], &ResolveContext::new(Vec::new()));
        assert_eq!(
            out,
            Type::Logical {
                shape: Some(vec![Some(1), Some(4)])
            }
        );
    }

    #[test]
    fn mode_type_preserves_scalar_bool_and_int() {
        assert_eq!(
            mode_type(&[Type::Bool], &ResolveContext::new(Vec::new())),
            Type::Bool
        );
        assert_eq!(
            mode_type(&[Type::Int], &ResolveContext::new(Vec::new())),
            Type::Int
        );
    }

    #[test]
    fn rng_type_is_struct() {
        assert!(matches!(
            rng_type(&[], &ResolveContext::new(Vec::new())),
            Type::Struct { .. }
        ));
    }
}
