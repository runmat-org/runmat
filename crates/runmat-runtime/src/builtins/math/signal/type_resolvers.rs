use runmat_builtins::{LiteralValue, ResolveContext, Type};

use runmat_builtins::shape_rules::element_count_if_known;

pub fn conv_type(args: &[Type], _context: &ResolveContext) -> Type {
    conv_like_type(args)
}

pub fn conv2_type(args: &[Type], _context: &ResolveContext) -> Type {
    let lhs = args.get(0);
    let rhs = args.get(1);
    match (lhs, rhs) {
        (Some(left), Some(right)) => conv2_binary_type(left, right, args.len() == 2),
        _ => Type::Unknown,
    }
}

pub fn deconv_type(args: &[Type], _context: &ResolveContext) -> Type {
    let numerator = args.get(0);
    let denominator = args.get(1);
    match (numerator, denominator) {
        (Some(num), Some(den)) => deconv_binary_type(num, den),
        _ => Type::Unknown,
    }
}

pub fn filter_type(args: &[Type], _context: &ResolveContext) -> Type {
    let signal = args.get(2);
    match signal {
        Some(ty) => numeric_like(ty),
        None => Type::Unknown,
    }
}

pub fn filtfilt_type(args: &[Type], _context: &ResolveContext) -> Type {
    let signal = args.get(2);
    match signal {
        Some(ty) => numeric_like(ty),
        None => Type::Unknown,
    }
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

pub fn pulse_train_type(args: &[Type], context: &ResolveContext) -> Type {
    numeric_unary_shape_type(args, context)
}

pub fn upsample_type(args: &[Type], context: &ResolveContext) -> Type {
    sample_rate_type(args, context, SampleRateOp::Up)
}

pub fn downsample_type(args: &[Type], context: &ResolveContext) -> Type {
    sample_rate_type(args, context, SampleRateOp::Down)
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

pub fn window_vector_type(args: &[Type], ctx: &ResolveContext) -> Type {
    let Some(arg) = args.first() else {
        return Type::Unknown;
    };
    if let Some(n) = literal_nonnegative_scalar(ctx) {
        return Type::Tensor {
            shape: Some(vec![Some(n), Some(1)]),
        };
    }
    match arg {
        Type::Num | Type::Int | Type::Bool => Type::Tensor {
            shape: Some(vec![None, Some(1)]),
        },
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            if element_count_if_known(shape).unwrap_or(1) == 1 {
                Type::Tensor {
                    shape: Some(vec![None, Some(1)]),
                }
            } else {
                Type::Unknown
            }
        }
        Type::Tensor { shape: None } | Type::Logical { shape: None } | Type::Unknown => {
            Type::Tensor {
                shape: Some(vec![None, Some(1)]),
            }
        }
        _ => Type::Unknown,
    }
}

fn literal_nonnegative_scalar(ctx: &ResolveContext) -> Option<usize> {
    match ctx.literal_args.first() {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrientationHint {
    Row,
    Column,
    Scalar,
    General,
    Empty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Orientation {
    Row,
    Column,
}

fn conv_like_type(args: &[Type]) -> Type {
    let lhs = args.get(0);
    let rhs = args.get(1);
    match (lhs, rhs) {
        (Some(left), Some(right)) => conv_binary_type(left, right, args.len() == 2),
        _ => Type::Unknown,
    }
}

fn conv_binary_type(lhs: &Type, rhs: &Type, is_default_mode: bool) -> Type {
    if is_numeric_scalar(lhs) && is_numeric_scalar(rhs) {
        return Type::Num;
    }

    let lhs_len = vector_len(lhs);
    let rhs_len = vector_len(rhs);
    let lhs_hint = orientation_hint(lhs);
    let rhs_hint = orientation_hint(rhs);
    let orientation = output_orientation(lhs_hint, rhs_hint);

    let out_len = if is_default_mode {
        match (lhs_len, rhs_len) {
            (Some(a), Some(b)) => Some(a + b - 1),
            _ => None,
        }
    } else {
        None
    };
    vector_output_type(out_len, orientation)
}

fn conv2_binary_type(lhs: &Type, rhs: &Type, is_default_mode: bool) -> Type {
    let (lhs_rows, lhs_cols) = matrix_dims(lhs);
    let (rhs_rows, rhs_cols) = matrix_dims(rhs);
    let (rows, cols) = if is_default_mode {
        match (lhs_rows, lhs_cols, rhs_rows, rhs_cols) {
            (Some(a), Some(b), Some(c), Some(d)) => (Some(a + c - 1), Some(b + d - 1)),
            _ => (None, None),
        }
    } else {
        (None, None)
    };
    Type::Tensor {
        shape: Some(vec![rows, cols]),
    }
}

fn deconv_binary_type(numerator: &Type, denominator: &Type) -> Type {
    if is_numeric_scalar(numerator) && is_numeric_scalar(denominator) {
        return Type::Num;
    }
    let num_len = vector_len(numerator);
    let den_len = vector_len(denominator);
    let hint = orientation_hint(numerator);
    let out_len = match (num_len, den_len) {
        (Some(a), Some(b)) if a >= b => Some(a - b + 1),
        (Some(_), Some(_)) => Some(0),
        _ => None,
    };
    vector_output_type(out_len, orientation_from_hint(hint))
}

fn vector_output_type(len: Option<usize>, orientation: Orientation) -> Type {
    let shape = match (orientation, len) {
        (Orientation::Row, Some(0)) => vec![Some(1), Some(0)],
        (Orientation::Column, Some(0)) => vec![Some(0), Some(1)],
        (Orientation::Row, Some(n)) => vec![Some(1), Some(n)],
        (Orientation::Column, Some(n)) => vec![Some(n), Some(1)],
        (Orientation::Row, None) => vec![Some(1), None],
        (Orientation::Column, None) => vec![None, Some(1)],
    };
    Type::Tensor { shape: Some(shape) }
}

fn output_orientation(lhs: OrientationHint, rhs: OrientationHint) -> Orientation {
    match lhs {
        OrientationHint::Row => Orientation::Row,
        OrientationHint::Column => Orientation::Column,
        OrientationHint::General => Orientation::Column,
        OrientationHint::Scalar | OrientationHint::Empty => orientation_from_hint(rhs),
    }
}

fn orientation_from_hint(hint: OrientationHint) -> Orientation {
    match hint {
        OrientationHint::Column | OrientationHint::General => Orientation::Column,
        OrientationHint::Row => Orientation::Row,
        OrientationHint::Scalar | OrientationHint::Empty => Orientation::Row,
    }
}

fn orientation_hint(ty: &Type) -> OrientationHint {
    match ty {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            let len = element_count_if_known(shape);
            let (rows, cols) = shape_rows_cols(shape);
            classify_orientation(rows, cols, len)
        }
        Type::Tensor { shape: None } | Type::Logical { shape: None } => OrientationHint::General,
        Type::Num | Type::Int | Type::Bool => OrientationHint::Scalar,
        _ => OrientationHint::General,
    }
}

fn classify_orientation(
    rows: Option<usize>,
    cols: Option<usize>,
    len: Option<usize>,
) -> OrientationHint {
    if len == Some(0) {
        OrientationHint::Empty
    } else if rows == Some(1) && cols.unwrap_or(0) > 1 {
        OrientationHint::Row
    } else if cols == Some(1) && rows.unwrap_or(0) > 1 {
        OrientationHint::Column
    } else if rows == Some(1) && cols == Some(1) {
        OrientationHint::Scalar
    } else {
        OrientationHint::General
    }
}

fn shape_rows_cols(shape: &[Option<usize>]) -> (Option<usize>, Option<usize>) {
    match shape.len() {
        0 => (Some(1), Some(1)),
        1 => (Some(1), shape.get(0).copied().flatten()),
        _ => (
            shape.get(0).copied().flatten(),
            shape.get(1).copied().flatten(),
        ),
    }
}

fn matrix_dims(ty: &Type) -> (Option<usize>, Option<usize>) {
    match ty {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            if shape.len() > 2 && shape.iter().skip(2).any(|dim| dim != &Some(1)) {
                return (None, None);
            }
            match shape.len() {
                0 => (Some(1), Some(1)),
                1 => (shape.get(0).copied().flatten(), Some(1)),
                _ => (
                    shape.get(0).copied().flatten(),
                    shape.get(1).copied().flatten(),
                ),
            }
        }
        Type::Num | Type::Int | Type::Bool => (Some(1), Some(1)),
        Type::Tensor { shape: None } | Type::Logical { shape: None } => (None, None),
        _ => (None, None),
    }
}

fn vector_len(ty: &Type) -> Option<usize> {
    match ty {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            element_count_if_known(shape)
        }
        Type::Num | Type::Int | Type::Bool => Some(1),
        _ => None,
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
    fn conv_full_uses_length_sum() {
        let out = conv_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(3)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(2)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(4)])
            }
        );
    }

    #[test]
    fn filter_preserves_signal_shape() {
        let out = filter_type(
            &[
                Type::Num,
                Type::Num,
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(2)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(2)])
            }
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
