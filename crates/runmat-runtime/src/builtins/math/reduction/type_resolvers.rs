use crate::builtins::common::arg_tokens::{tokens_from_context, ArgToken};
use crate::builtins::common::spec::ReductionNaN;
use runmat_builtins::shape_rules::{scalar_tensor_shape, unknown_shape};
use runmat_builtins::{ResolveContext, Type};

#[derive(Default)]
struct ReductionTokenInfo {
    all: bool,
    dims: Option<Vec<usize>>,
    nan_mode: Option<ReductionNaN>,
}

fn parse_reduction_tokens(ctx: &ResolveContext) -> ReductionTokenInfo {
    let tokens = tokens_from_context(ctx);
    let mut info = ReductionTokenInfo::default();
    for token in tokens.iter().skip(1) {
        match token {
            ArgToken::String(text) if text == "all" => info.all = true,
            ArgToken::String(text) if text == "omitnan" => info.nan_mode = Some(ReductionNaN::Omit),
            ArgToken::String(text) if text == "includenan" => {
                info.nan_mode = Some(ReductionNaN::Include)
            }
            ArgToken::Vector(values) => {
                if info.dims.is_none() {
                    let mut dims = Vec::with_capacity(values.len());
                    for value in values {
                        let dim = match value {
                            ArgToken::Number(num) => coerce_dim(*num),
                            _ => None,
                        };
                        match dim {
                            Some(value) => dims.push(value),
                            None => {
                                dims.clear();
                                break;
                            }
                        }
                    }
                    if !dims.is_empty() {
                        info.dims = Some(dims);
                    }
                }
            }
            ArgToken::Number(num) => {
                if info.dims.is_none() {
                    if let Some(dim) = coerce_dim(*num) {
                        info.dims = Some(vec![dim]);
                    }
                }
            }
            _ => {}
        }
    }
    info
}

fn coerce_dim(value: f64) -> Option<usize> {
    if !value.is_finite() {
        return None;
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return None;
    }
    if rounded < 1.0 {
        return None;
    }
    Some(rounded as usize)
}

fn reduction_shape_from_args(args: &[Type], ctx: &ResolveContext) -> Option<Vec<Option<usize>>> {
    let input = args.first()?;
    let shape = match input {
        Type::Tensor { shape } => shape.clone(),
        Type::Logical { shape } => shape.clone(),
        _ => None,
    };
    let shape = shape?;
    let parsed = parse_reduction_tokens(ctx);
    if parsed.all {
        return Some(scalar_tensor_shape());
    }
    if args.len() == 1 || (args.len() > 1 && parsed.dims.is_none() && parsed.nan_mode.is_some()) {
        Some(reduce_first_nonsingleton(&shape))
    } else {
        if let Some(dims) = parsed.dims {
            let mut out = shape.clone();
            for dim_1based in dims {
                let index = dim_1based.saturating_sub(1);
                if index < out.len() {
                    out[index] = Some(1);
                } else {
                    out = unknown_shape(out.len());
                    break;
                }
            }
            return Some(out);
        }
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

pub fn cumulative_numeric_type(args: &[Type], _context: &ResolveContext) -> Type {
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

pub fn diff_numeric_type(args: &[Type], _context: &ResolveContext) -> Type {
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

pub fn count_nonzero_type(args: &[Type], ctx: &ResolveContext) -> Type {
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
        reduce_numeric_type(args, ctx)
    }
}

pub fn min_max_type(args: &[Type], ctx: &ResolveContext) -> Type {
    if args.len() <= 1 {
        return reduce_numeric_type(args, ctx);
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
        let out = reduce_numeric_type(&[ty], &ResolveContext::new(Vec::new()));
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
        let out = reduce_numeric_type(&[ty, Type::Num], &ctx);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(1)])
            }
        );
    }

    #[test]
    fn reduce_numeric_all_returns_scalar_shape() {
        let ty = Type::Tensor {
            shape: Some(vec![Some(3), Some(4)]),
        };
        let ctx = ResolveContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::String("all".to_string()),
        ]);
        let out = reduce_numeric_type(&[ty, Type::String], &ctx);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(1)])
            }
        );
    }

    #[test]
    fn reduce_numeric_omitnan_is_ignored_for_shape() {
        let ty = Type::Tensor {
            shape: Some(vec![Some(3), Some(4)]),
        };
        let ctx = ResolveContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::String("omitnan".to_string()),
        ]);
        let out = reduce_numeric_type(&[ty, Type::String], &ctx);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(4)])
            }
        );
    }

    #[test]
    fn reduce_logical_returns_logical() {
        let ty = Type::Logical {
            shape: Some(vec![Some(2), Some(2)]),
        };
        let out = reduce_logical_type(&[ty], &ResolveContext::new(Vec::new()));
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
        let out = cumulative_numeric_type(&[ty], &ResolveContext::new(Vec::new()));
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(5), Some(1)])
            }
        );
    }

    #[test]
    fn min_max_two_args_scalar() {
        assert_eq!(
            min_max_type(&[Type::Num, Type::Num], &ResolveContext::new(Vec::new())),
            Type::Num
        );
    }
}
