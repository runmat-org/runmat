use runmat_builtins::{ResolveContext, Type};

use crate::builtins::math::type_resolvers::{numeric_binary_type, numeric_unary_type};

pub fn db_type(args: &[Type], context: &ResolveContext) -> Type {
    match args {
        [input] => numeric_unary_type(std::slice::from_ref(input), context),
        [input, mode] if is_text_mode_type(mode) => {
            numeric_unary_type(std::slice::from_ref(input), context)
        }
        [input, reference] => numeric_binary_type(&[input.clone(), reference.clone()], context),
        _ => Type::Unknown,
    }
}

fn is_text_mode_type(ty: &Type) -> bool {
    matches!(ty, Type::String)
        || matches!(
            ty,
            Type::Cell {
                element_type: Some(element_type),
                ..
            } if **element_type == Type::String
        )
}

pub fn step_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::tensor()
}

pub fn stepinfo_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Struct {
        known_fields: Some(vec![
            "Overshoot".to_string(),
            "Peak".to_string(),
            "PeakTime".to_string(),
            "RiseTime".to_string(),
            "SettlingMax".to_string(),
            "SettlingMin".to_string(),
            "SettlingTime".to_string(),
            "SteadyStateValue".to_string(),
            "TransientTime".to_string(),
            "Undershoot".to_string(),
        ]),
    }
}

pub fn feedback_type(args: &[Type], _context: &ResolveContext) -> Type {
    match args {
        [sys] if is_tf_or_scalar_type(sys) => tf_object_type(),
        [sys1, sys2] if is_tf_or_scalar_type(sys1) && is_tf_or_scalar_type(sys2) => {
            tf_object_type()
        }
        [sys1, sys2, sign]
            if is_tf_or_scalar_type(sys1)
                && is_tf_or_scalar_type(sys2)
                && is_sample_time_type(sign) =>
        {
            tf_object_type()
        }
        _ => Type::Unknown,
    }
}

pub fn dcgain_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Num
}

pub fn pole_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::tensor()
}

pub fn rlocus_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::tensor()
}

pub fn isstable_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Bool
}

pub fn ss_type(args: &[Type], context: &ResolveContext) -> Type {
    let rest = match args {
        [a, b, c, d, rest @ ..] if valid_state_space_matrix_types(a, b, c, d) => rest,
        _ => return Type::Unknown,
    };

    if !valid_state_space_shapes(&args[0], &args[1], &args[2], &args[3]) {
        return Type::Unknown;
    }

    match rest {
        [] => ss_object_type(),
        [sample_time] if is_sample_time_type(sample_time) => ss_object_type(),
        _ if rest.len().is_multiple_of(2) && valid_ss_name_value_options(rest, context) => {
            ss_object_type()
        }
        _ => Type::Unknown,
    }
}

pub fn tf_type(args: &[Type], _context: &ResolveContext) -> Type {
    match args {
        [variable] if is_text_mode_type(variable) => tf_object_type(),
        [variable, sample_time]
            if is_text_mode_type(variable) && is_sample_time_type(sample_time) =>
        {
            tf_object_type()
        }
        [numerator, denominator, rest @ ..]
            if is_tf_coefficient_type(numerator)
                && is_tf_coefficient_type(denominator)
                && valid_tf_options(rest) =>
        {
            tf_object_type()
        }
        _ => Type::Unknown,
    }
}

pub fn impulse_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::tensor()
}

pub fn nyquist_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::tensor()
}

fn ss_object_type() -> Type {
    Type::Struct {
        known_fields: Some(vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
            "InputDelay".to_string(),
            "InputName".to_string(),
            "OutputDelay".to_string(),
            "OutputName".to_string(),
            "StateName".to_string(),
            "Ts".to_string(),
        ]),
    }
}

fn tf_object_type() -> Type {
    Type::Struct {
        known_fields: Some(vec![
            "Denominator".to_string(),
            "InputDelay".to_string(),
            "Numerator".to_string(),
            "OutputDelay".to_string(),
            "Ts".to_string(),
            "Variable".to_string(),
        ]),
    }
}

fn valid_state_space_matrix_types(a: &Type, b: &Type, c: &Type, d: &Type) -> bool {
    [a, b, c, d].iter().all(|ty| is_real_matrix_type(ty))
}

fn is_real_matrix_type(ty: &Type) -> bool {
    matches!(ty, Type::Num | Type::Int | Type::Tensor { .. })
}

fn is_sample_time_type(ty: &Type) -> bool {
    matches!(ty, Type::Num | Type::Int | Type::Unknown)
}

fn is_tf_coefficient_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Num
            | Type::Int
            | Type::Bool
            | Type::Tensor { .. }
            | Type::Logical { .. }
            | Type::Unknown
    )
}

fn is_tf_or_scalar_type(ty: &Type) -> bool {
    matches!(ty, Type::Struct { .. } | Type::Unknown) || is_tf_coefficient_type(ty)
}

fn valid_tf_options(rest: &[Type]) -> bool {
    match rest {
        [] => true,
        [sample_time] if is_sample_time_type(sample_time) => true,
        _ => {
            rest.len().is_multiple_of(2)
                && rest.chunks_exact(2).all(|pair| {
                    let [name, value] = pair else {
                        return false;
                    };
                    is_text_mode_type(name)
                        && (is_sample_time_type(value)
                            || is_text_mode_type(value)
                            || matches!(value, Type::Unknown))
                })
        }
    }
}

fn valid_ss_name_value_options(rest: &[Type], context: &ResolveContext) -> bool {
    rest.chunks_exact(2)
        .enumerate()
        .all(|(idx, pair)| valid_ss_option_pair(pair, 4 + idx * 2, context))
}

fn valid_ss_option_pair(pair: &[Type], arg_index: usize, context: &ResolveContext) -> bool {
    let [name, value] = pair else {
        return false;
    };
    if !is_text_mode_type(name) || !is_sample_time_type(value) {
        return false;
    }
    match context.literal_string_at(arg_index) {
        Some(name) => matches!(name.trim(), "ts" | "sampletime"),
        None => true,
    }
}

#[derive(Clone, Copy)]
struct MatrixShape {
    rows: Option<usize>,
    cols: Option<usize>,
}

fn valid_state_space_shapes(a: &Type, b: &Type, c: &Type, d: &Type) -> bool {
    let Some(a_shape) = matrix_shape(a) else {
        return false;
    };
    let Some(b_shape) = matrix_shape(b) else {
        return false;
    };
    let Some(c_shape) = matrix_shape(c) else {
        return false;
    };
    let Some(d_shape) = matrix_shape(d) else {
        return false;
    };

    known_equal_or_unknown(a_shape.rows, a_shape.cols)
        && known_equal_or_unknown(b_shape.rows, a_shape.rows)
        && known_equal_or_unknown(c_shape.cols, a_shape.rows)
        && known_equal_or_unknown(d_shape.rows, c_shape.rows)
        && known_equal_or_unknown(d_shape.cols, b_shape.cols)
}

fn matrix_shape(ty: &Type) -> Option<MatrixShape> {
    match ty {
        Type::Num | Type::Int => Some(MatrixShape {
            rows: Some(1),
            cols: Some(1),
        }),
        Type::Tensor { shape: None } => Some(MatrixShape {
            rows: None,
            cols: None,
        }),
        Type::Tensor { shape: Some(shape) } => match shape.as_slice() {
            [] => Some(MatrixShape {
                rows: Some(1),
                cols: Some(1),
            }),
            [cols] => Some(MatrixShape {
                rows: Some(1),
                cols: *cols,
            }),
            [rows, cols] => Some(MatrixShape {
                rows: *rows,
                cols: *cols,
            }),
            _ => None,
        },
        _ => None,
    }
}

fn known_equal_or_unknown(lhs: Option<usize>, rhs: Option<usize>) -> bool {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => lhs == rhs,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> ResolveContext {
        ResolveContext::new(Vec::new())
    }

    fn tensor(rows: usize, cols: usize) -> Type {
        Type::Tensor {
            shape: Some(vec![Some(rows), Some(cols)]),
        }
    }

    fn assert_ss_struct(ty: Type) {
        assert_eq!(ty, ss_object_type());
    }

    #[test]
    fn ss_type_returns_known_fields_for_valid_core_form() {
        assert_ss_struct(ss_type(
            &[tensor(2, 2), tensor(2, 1), tensor(1, 2), tensor(1, 1)],
            &ctx(),
        ));
    }

    #[test]
    fn ss_type_accepts_positional_sample_time() {
        assert_ss_struct(ss_type(
            &[
                tensor(1, 1),
                tensor(1, 2),
                tensor(2, 1),
                tensor(2, 2),
                Type::Num,
            ],
            &ctx(),
        ));
    }

    #[test]
    fn ss_type_accepts_supported_name_value_sample_time() {
        let ctx = ResolveContext::new(vec![
            runmat_builtins::LiteralValue::Unknown,
            runmat_builtins::LiteralValue::Unknown,
            runmat_builtins::LiteralValue::Unknown,
            runmat_builtins::LiteralValue::Unknown,
            runmat_builtins::LiteralValue::String("Ts".to_string()),
        ]);
        assert_ss_struct(ss_type(
            &[
                tensor(1, 1),
                tensor(1, 1),
                tensor(1, 1),
                tensor(1, 1),
                Type::String,
                Type::Int,
            ],
            &ctx,
        ));
    }

    #[test]
    fn ss_type_rejects_invalid_argument_count() {
        assert_eq!(
            ss_type(&[tensor(1, 1), tensor(1, 1), tensor(1, 1)], &ctx()),
            Type::Unknown
        );
    }

    #[test]
    fn ss_type_rejects_incompatible_known_shapes() {
        assert_eq!(
            ss_type(
                &[tensor(2, 2), tensor(3, 1), tensor(1, 2), tensor(1, 1)],
                &ctx(),
            ),
            Type::Unknown
        );
    }

    #[test]
    fn ss_type_rejects_unsupported_literal_option_name() {
        let ctx = ResolveContext::new(vec![
            runmat_builtins::LiteralValue::Unknown,
            runmat_builtins::LiteralValue::Unknown,
            runmat_builtins::LiteralValue::Unknown,
            runmat_builtins::LiteralValue::Unknown,
            runmat_builtins::LiteralValue::String("BadOption".to_string()),
        ]);
        assert_eq!(
            ss_type(
                &[
                    tensor(1, 1),
                    tensor(1, 1),
                    tensor(1, 1),
                    tensor(1, 1),
                    Type::String,
                    Type::Num,
                ],
                &ctx,
            ),
            Type::Unknown
        );
    }
}
