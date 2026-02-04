use runmat_builtins::shape_rules::{scalar_tensor_shape, unknown_shape};
use runmat_builtins::{ResolveContext, Type};

fn reduction_shape_from_args(args: &[Type], ctx: &ResolveContext) -> Option<Vec<Option<usize>>> {
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
        let dims = ctx.numeric_dims_from(1);
        let literal_dim = dims.first().and_then(|value| *value);
        if let Some(dim_1based) = literal_dim {
            if dim_1based >= 1 {
                let mut out = shape.clone();
                let index = dim_1based - 1;
                if index < out.len() {
                    out[index] = Some(1);
                } else {
                    out = unknown_shape(out.len());
                }
                return Some(out);
            }
        }
        Some(unknown_shape(shape.len()))
    }
}

fn reduction_shape_from_args_legacy(args: &[Type]) -> Option<Vec<Option<usize>>> {
    reduction_shape_from_args(args, &ResolveContext::empty())
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

pub fn reduce_numeric_type(args: &[Type], ctx: &ResolveContext) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    match input {
        Type::Tensor { shape: Some(_) } | Type::Logical { shape: Some(_) } => Type::Tensor {
            shape: reduction_shape_from_args(args, ctx),
        },
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
        Type::Bool | Type::Num | Type::Int => Type::Num,
        _ => Type::Unknown,
    }
}

pub fn reduce_numeric_type_legacy(args: &[Type]) -> Type {
    reduce_numeric_type(args, &ResolveContext::empty())
}

pub fn reduce_logical_type(args: &[Type], ctx: &ResolveContext) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    match input {
        Type::Tensor { shape: Some(_) } | Type::Logical { shape: Some(_) } => Type::Logical {
            shape: reduction_shape_from_args(args, ctx),
        },
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::logical(),
        Type::Bool | Type::Num | Type::Int => Type::Bool,
        _ => Type::Unknown,
    }
}

pub fn reduce_logical_type_legacy(args: &[Type]) -> Type {
    reduce_logical_type(args, &ResolveContext::empty())
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
        reduce_numeric_type_legacy(args)
    }
}

pub fn min_max_type(args: &[Type]) -> Type {
    if args.len() <= 1 {
        return reduce_numeric_type_legacy(args);
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

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::LiteralValue;

    #[test]
    fn reduce_numeric_preserves_rank() {
        let ty = Type::Tensor {
            shape: Some(vec![Some(3), Some(4)]),
        };
        let out = reduce_numeric_type(&[ty], &ResolveContext::empty());
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(4)])
            }
        );
    }

    #[test]
    fn reduce_numeric_uses_literal_dim() {
        let ty = Type::Tensor {
            shape: Some(vec![Some(3), Some(4)]),
        };
        let ctx = ResolveContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(2.0)]);
        let out = reduce_numeric_type(&[ty], &ctx);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(1)])
            }
        );
    }

    #[test]
    fn reduce_logical_returns_logical() {
        let ty = Type::Logical {
            shape: Some(vec![Some(2), Some(2)]),
        };
        let out = reduce_logical_type(&[ty], &ResolveContext::empty());
        assert_eq!(
            out,
            Type::Logical {
                shape: Some(vec![Some(1), Some(2)])
            }
        );
    }

    #[test]
    fn cumulative_numeric_keeps_shape() {
        let ty = Type::Tensor {
            shape: Some(vec![Some(5), Some(1)]),
        };
        let out = cumulative_numeric_type(&[ty]);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(5), Some(1)])
            }
        );
    }

    #[test]
    fn reduction_shape_legacy_matches_first_nonsingleton() {
        let ty = Type::Tensor {
            shape: Some(vec![Some(3), Some(4)]),
        };
        let out = reduction_shape_from_args_legacy(&[ty]);
        assert_eq!(out, Some(vec![Some(1), Some(4)]));
    }

    #[test]
    fn min_max_two_args_scalar() {
        assert_eq!(min_max_type(&[Type::Num, Type::Num]), Type::Num);
    }
}
