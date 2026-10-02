use runmat_builtins::{LiteralValue, ResolveContext, Type};

use runmat_builtins::shape_rules::element_count_if_known;

pub fn envelope_type(args: &[Type], _context: &ResolveContext) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    numeric_like(input)
}

pub fn fir1_type(args: &[Type], context: &ResolveContext) -> Type {
    if args.len() < 2 {
        return Type::Unknown;
    }
    if let Some(order) = literal_nonnegative_integer_at(context, 0) {
        let Some(width) = fir1_width_for_literal_order(order, args, context) else {
            return Type::Tensor {
                shape: Some(vec![Some(1), None]),
            };
        };
        return Type::Tensor {
            shape: Some(vec![Some(1), Some(width)]),
        };
    }
    Type::Tensor {
        shape: Some(vec![Some(1), None]),
    }
}

pub fn freqz_type(args: &[Type], context: &ResolveContext) -> Type {
    if args.len() < 2 {
        return Type::Unknown;
    }
    let n = if args.len() >= 3 {
        literal_positive_integer_at(context, 2)
    } else {
        Some(512)
    };
    Type::Tensor {
        shape: Some(vec![n, Some(1)]),
    }
}

#[cfg(feature = "plot-core")]
pub fn zplane_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Tensor {
        shape: Some(vec![Some(1), None]),
    }
}

pub fn butter_type(args: &[Type], _context: &ResolveContext) -> Type {
    if args.len() < 2 {
        return Type::Unknown;
    }
    Type::Tensor {
        shape: Some(vec![Some(1), None]),
    }
}

pub fn buttord_type(args: &[Type], _context: &ResolveContext) -> Type {
    if args.len() < 4 {
        return Type::Unknown;
    }
    Type::Num
}

pub fn cheb2ord_type(args: &[Type], _context: &ResolveContext) -> Type {
    if args.len() < 4 {
        return Type::Unknown;
    }
    Type::Num
}

pub fn pulse_train_type(args: &[Type], context: &ResolveContext) -> Type {
    numeric_unary_shape_type(args, context)
}

pub fn upsample_type(args: &[Type], context: &ResolveContext) -> Type {
    sample_rate_type(args, context, SampleRateOp::Up)
}

pub fn downsample_type(args: &[Type], context: &ResolveContext) -> Type {
    sample_rate_type(args, context, SampleRateOp::Down)
}

pub fn resample_type(args: &[Type], context: &ResolveContext) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    let Some(shape) = numeric_array_shape(input) else {
        return match input {
            Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
            Type::Unknown => Type::Unknown,
            _ => Type::Unknown,
        };
    };
    let mut output_shape = canonical_sample_shape(shape);
    let dim = literal_dimension_name_value(context)
        .unwrap_or_else(|| first_non_singleton_shape_dim(&output_shape));
    if dim >= output_shape.len() {
        output_shape.resize(dim + 1, Some(1));
    }
    let p = literal_positive_integer_at(context, 1);
    let q = literal_positive_integer_at(context, 2);
    output_shape[dim] = match (output_shape[dim], p, q) {
        (Some(len), Some(p), Some(q)) => len
            .checked_mul(p)
            .and_then(|scaled| scaled.checked_add(q.saturating_sub(1)))
            .map(|scaled| scaled / q),
        _ => None,
    };
    if element_count_if_known(&output_shape) == Some(1) {
        return Type::Num;
    }
    Type::Tensor {
        shape: Some(output_shape),
    }
}

pub fn numeric_unary_shape_type(args: &[Type], _context: &ResolveContext) -> Type {
    let Some(arg) = args.first() else {
        return Type::Unknown;
    };
    match arg {
        Type::Num | Type::Int | Type::Bool => Type::Num,
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            if element_count_if_known(shape) == Some(1) {
                Type::Num
            } else {
                Type::Tensor {
                    shape: Some(shape.clone()),
                }
            }
        }
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampleRateOp {
    Up,
    Down,
}

fn sample_rate_type(args: &[Type], context: &ResolveContext, op: SampleRateOp) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    let Some(shape) = numeric_array_shape(input) else {
        return match input {
            Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
            Type::Unknown => Type::Unknown,
            _ => Type::Unknown,
        };
    };
    let mut output_shape = canonical_sample_shape(shape);
    let dim = first_non_singleton_shape_dim(&output_shape);
    let factor = literal_positive_integer_at(context, 1);
    let phase = if context.literal_args.len() <= 2 {
        Some(0)
    } else {
        literal_nonnegative_integer_at(context, 2)
    };
    output_shape[dim] = match op {
        SampleRateOp::Up => match (output_shape[dim], factor) {
            (Some(len), Some(n)) => len.checked_mul(n),
            _ => None,
        },
        SampleRateOp::Down => match (output_shape[dim], factor, phase) {
            (Some(len), Some(n), Some(offset)) => {
                if len <= offset {
                    Some(0)
                } else {
                    Some(((len - 1 - offset) / n) + 1)
                }
            }
            _ => None,
        },
    };
    if element_count_if_known(&output_shape) == Some(1) {
        return Type::Num;
    }
    Type::Tensor {
        shape: Some(output_shape),
    }
}

fn numeric_array_shape(input: &Type) -> Option<Vec<Option<usize>>> {
    match input {
        Type::Num | Type::Int | Type::Bool => Some(vec![Some(1), Some(1)]),
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            Some(shape.clone())
        }
        _ => None,
    }
}

fn canonical_sample_shape(mut shape: Vec<Option<usize>>) -> Vec<Option<usize>> {
    match shape.len() {
        0 => vec![Some(1), Some(1)],
        1 => {
            shape.insert(0, Some(1));
            shape
        }
        _ => shape,
    }
}

fn first_non_singleton_shape_dim(shape: &[Option<usize>]) -> usize {
    shape.iter().position(|dim| *dim != Some(1)).unwrap_or(0)
}

fn literal_positive_integer_at(ctx: &ResolveContext, index: usize) -> Option<usize> {
    literal_nonnegative_integer_at(ctx, index).filter(|value| *value > 0)
}

fn literal_dimension_name_value(ctx: &ResolveContext) -> Option<usize> {
    let mut idx = 3usize;
    while idx + 1 < ctx.literal_args.len() {
        if ctx
            .literal_string_at(idx)
            .as_deref()
            .is_some_and(|name| name.eq_ignore_ascii_case("dimension"))
        {
            return literal_positive_integer_at(ctx, idx + 1).map(|dim| dim.saturating_sub(1));
        }
        idx += 1;
    }
    None
}

fn literal_nonnegative_integer_at(ctx: &ResolveContext, index: usize) -> Option<usize> {
    match ctx.literal_args.get(index) {
        Some(LiteralValue::Number(value)) if value.is_finite() => {
            let rounded = value.round();
            if rounded < 0.0 || (rounded - value).abs() > 1e-9 {
                None
            } else {
                Some(rounded as usize)
            }
        }
        Some(LiteralValue::Bool(value)) => Some(usize::from(*value)),
        _ => None,
    }
}

fn fir1_width_for_literal_order(
    order: usize,
    args: &[Type],
    context: &ResolveContext,
) -> Option<usize> {
    let mut needs_even_order = false;
    for idx in 2..args.len() {
        if let Some(word) = context.literal_string_at(idx) {
            needs_even_order |= matches!(
                word.trim().to_ascii_lowercase().as_str(),
                "high" | "highpass" | "stop" | "bandstop" | "bandreject"
            );
        } else if order % 2 == 1 && could_be_fir1_option_text(&args[idx]) {
            return None;
        }
    }
    if needs_even_order && order % 2 == 1 {
        order.checked_add(2)
    } else {
        order.checked_add(1)
    }
}

fn could_be_fir1_option_text(ty: &Type) -> bool {
    match ty {
        Type::String | Type::Unknown => true,
        Type::Cell {
            element_type: Some(element_type),
            ..
        } if **element_type == Type::String => true,
        Type::Union(types) => types.iter().any(could_be_fir1_option_text),
        _ => false,
    }
}

fn numeric_like(input: &Type) -> Type {
    match input {
        Type::Tensor { shape: Some(shape) } => Type::Tensor {
            shape: Some(shape.clone()),
        },
        Type::Logical { shape: Some(shape) } => Type::Tensor {
            shape: Some(shape.clone()),
        },
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
        Type::Num | Type::Int | Type::Bool => Type::Num,
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::LiteralValue;

    #[test]
    fn fir1_type_requires_order_and_cutoff_arguments() {
        let ctx = ResolveContext::default();

        assert_eq!(fir1_type(&[], &ctx), Type::Unknown);
        assert_eq!(fir1_type(&[Type::Num], &ctx), Type::Unknown);
    }

    #[test]
    fn freqz_type_requires_numerator_and_denominator_arguments() {
        let ctx = ResolveContext::default();

        assert_eq!(freqz_type(&[], &ctx), Type::Unknown);
        assert_eq!(
            freqz_type(&[Type::Tensor { shape: None }], &ctx),
            Type::Unknown
        );
    }

    #[test]
    fn upsample_type_scales_row_vector_literal_factor() {
        let ctx = ResolveContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(2.0)]);
        assert_eq!(
            upsample_type(
                &[Type::Tensor {
                    shape: Some(vec![Some(1), Some(3)])
                }],
                &ctx
            ),
            Type::Tensor {
                shape: Some(vec![Some(1), Some(6)])
            }
        );
    }

    #[test]
    fn downsample_type_shrinks_column_vector_literal_factor_and_phase() {
        let ctx = ResolveContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
            LiteralValue::Number(1.0),
        ]);
        assert_eq!(
            downsample_type(
                &[Type::Tensor {
                    shape: Some(vec![Some(5), Some(1)])
                }],
                &ctx
            ),
            Type::Tensor {
                shape: Some(vec![Some(2), Some(1)])
            }
        );
    }

    #[test]
    fn resample_type_honors_literal_dimension_name_value() {
        let ctx = ResolveContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
            LiteralValue::Number(1.0),
            LiteralValue::String("Dimension".to_string()),
            LiteralValue::Number(2.0),
        ]);
        assert_eq!(
            resample_type(
                &[
                    Type::Tensor {
                        shape: Some(vec![Some(2), Some(2)])
                    },
                    Type::Num,
                    Type::Num,
                    Type::String,
                    Type::Num,
                ],
                &ctx
            ),
            Type::Tensor {
                shape: Some(vec![Some(2), Some(4)])
            }
        );
    }

    #[test]
    fn downsample_type_reports_scalar_when_output_has_one_element() {
        let ctx = ResolveContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(2.0)]);
        assert_eq!(downsample_type(&[Type::Num], &ctx), Type::Num);
    }

    #[test]
    fn upsample_type_reports_scalar_for_scalar_identity_factor() {
        let ctx = ResolveContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(1.0)]);
        assert_eq!(upsample_type(&[Type::Num], &ctx), Type::Num);
    }
}
