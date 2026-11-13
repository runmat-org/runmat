//! MATLAB-compatible scalar `vpa` builtin for variable-precision symbolic decimals.

use crate::{call::BuiltinArguments, context::RuntimeContext};
use num_bigint::BigInt;
use num_traits::{FromPrimitive, One};
use runmat_builtins::{
    BuiltinErrorDescriptor, SymbolicPrecisionKind, SymbolicScalarOperation, VPA_DESCRIPTOR,
    VPA_ERRORS, VPA_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, SymbolicExpr, Value};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::{
    digits::{current_digits, MAX_DIGITS},
    symbolic_expr_to_value, text_scalar, value_to_symbolic_scalar,
};
use crate::builtins::common::tensor;

mod decimal;
use decimal::{significant_decimal_from_f64, significant_decimal_from_rational};
const BUILTIN_NAME: &str = "vpa";

#[runtime_builtin(
    name = "vpa",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::symbolic::vpa",
    context = true
)]
async fn vpa_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
    rest: BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &VPA_DESCRIPTOR,
        &VPA_ERROR_TOO_MANY_OUTPUTS,
        BUILTIN_NAME,
    )
    .map_err(super::invocation::terminal)?;
    if rest.len() > 1 {
        return Err(vpa_error(&VPA_ERRORS[0]));
    }
    let digits = match rest.first() {
        Some(value) => parse_precision(value)?,
        None => current_digits(),
    };
    super::invocation::validate(SymbolicScalarOperation::Decimal, 0, rest.len() + 1, &value)?;
    super::invocation::admit(SymbolicScalarOperation::Decimal, runtime.as_ref(), true)?;
    let expr = symbolic_input(&value)?;
    Ok(symbolic_expr_to_value(vpa_expr(&expr, digits)))
}

fn parse_precision(value: &Value) -> BuiltinResult<usize> {
    if let Some(value) = tensor::scalar_integer_value(value) {
        return validate_precision_integer(&value);
    }
    let parsed = match value {
        Value::Num(value) => *value,
        Value::Bool(value) => {
            if *value {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        _ => return Err(vpa_error(&VPA_ERRORS[2])),
    };
    validate_precision(parsed)
}

fn validate_precision_integer(value: &IntValue) -> BuiltinResult<usize> {
    let value = value
        .try_to_usize()
        .ok_or_else(|| vpa_error(&VPA_ERRORS[2]))?;
    if !SymbolicPrecisionKind::Decimal.accepts(value) {
        return Err(vpa_error_with_message(
            &VPA_ERRORS[2],
            format!(
                "{}: supported range is 1..={MAX_DIGITS}",
                VPA_ERRORS[2].message
            ),
        ));
    }
    Ok(value)
}

fn validate_precision(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(vpa_error(&VPA_ERRORS[2]));
    }
    let Some(rounded) = SymbolicPrecisionKind::Decimal.rounded(value) else {
        return Err(vpa_error_with_message(
            &VPA_ERRORS[2],
            format!(
                "{}: supported range is 1..={MAX_DIGITS}",
                VPA_ERRORS[2].message
            ),
        ));
    };
    Ok(rounded)
}

fn symbolic_input(value: &Value) -> BuiltinResult<SymbolicExpr> {
    if let Some(expr) = value_to_symbolic_scalar(value) {
        return Ok(expr);
    }
    if let Some(text) = text_scalar(value).map_err(|source| {
        let mut error = vpa_error(&VPA_ERRORS[1]);
        error.source = Some(Box::new(source));
        error
    })? {
        let trimmed = text.trim();
        if let Some(expr) = parse_rational_literal(trimmed) {
            return Ok(expr);
        }
        if let Ok(value) = trimmed.parse::<f64>() {
            if value.is_finite() {
                return Ok(SymbolicExpr::constant(value));
            }
        }
    }
    Err(vpa_error(&VPA_ERRORS[1]))
}

fn parse_rational_literal(text: &str) -> Option<SymbolicExpr> {
    let (lhs, rhs) = text.split_once('/')?;
    if rhs.contains('/') {
        return None;
    }
    let numerator = lhs.trim().parse::<BigInt>().ok()?;
    let denominator = rhs.trim().parse::<BigInt>().ok()?;
    SymbolicExpr::rational(numerator, denominator)
}

fn vpa_expr(expr: &SymbolicExpr, digits: usize) -> SymbolicExpr {
    match expr {
        SymbolicExpr::Constant(value) => decimal_from_f64(*value, digits),
        SymbolicExpr::Rational {
            numerator,
            denominator,
        } => decimal_from_rational(numerator, denominator, digits),
        SymbolicExpr::DecimalLiteral { value, .. } => decimal_from_f64(*value, digits),
        SymbolicExpr::Variable(_) | SymbolicExpr::FunctionReference(_, _) => expr.clone(),
        SymbolicExpr::FunctionCall(name, args) => SymbolicExpr::FunctionCall(
            name.clone(),
            runmat_value::ExactArrayStorage::shared_unbounded(
                args.iter().map(|arg| vpa_expr(arg, digits)).collect(),
            )
            .into(),
        ),
        SymbolicExpr::Equation(lhs, rhs) => SymbolicExpr::Equation(
            runmat_value::symbolic::SymbolicOwner::new_unbounded(vpa_expr(lhs, digits)),
            runmat_value::symbolic::SymbolicOwner::new_unbounded(vpa_expr(rhs, digits)),
        ),
        SymbolicExpr::Derivative {
            expr,
            variable,
            order,
        } => SymbolicExpr::Derivative {
            expr: runmat_value::symbolic::SymbolicOwner::new_unbounded(vpa_expr(expr, digits)),
            variable: variable.clone(),
            order: *order,
        },
        SymbolicExpr::Neg(inner) => SymbolicExpr::Neg(
            runmat_value::symbolic::SymbolicOwner::new_unbounded(vpa_expr(inner, digits)),
        ),
        SymbolicExpr::Add(lhs, rhs) => SymbolicExpr::Add(
            runmat_value::symbolic::SymbolicOwner::new_unbounded(vpa_expr(lhs, digits)),
            runmat_value::symbolic::SymbolicOwner::new_unbounded(vpa_expr(rhs, digits)),
        ),
        SymbolicExpr::Sub(lhs, rhs) => SymbolicExpr::Sub(
            runmat_value::symbolic::SymbolicOwner::new_unbounded(vpa_expr(lhs, digits)),
            runmat_value::symbolic::SymbolicOwner::new_unbounded(vpa_expr(rhs, digits)),
        ),
        SymbolicExpr::Mul(lhs, rhs) => SymbolicExpr::Mul(
            runmat_value::symbolic::SymbolicOwner::new_unbounded(vpa_expr(lhs, digits)),
            runmat_value::symbolic::SymbolicOwner::new_unbounded(vpa_expr(rhs, digits)),
        ),
        SymbolicExpr::Div(lhs, rhs) => {
            if let Some(SymbolicExpr::Rational {
                numerator,
                denominator,
            }) = expr_as_exact_rational(lhs, rhs)
            {
                decimal_from_rational(&numerator, &denominator, digits)
            } else {
                SymbolicExpr::Div(
                    runmat_value::symbolic::SymbolicOwner::new_unbounded(vpa_expr(lhs, digits)),
                    runmat_value::symbolic::SymbolicOwner::new_unbounded(vpa_expr(rhs, digits)),
                )
            }
        }
        SymbolicExpr::Pow(lhs, rhs) => {
            if let Some(value) = expr.numeric_constant_value() {
                decimal_from_f64(value, digits)
            } else {
                SymbolicExpr::Pow(
                    runmat_value::symbolic::SymbolicOwner::new_unbounded(vpa_expr(lhs, digits)),
                    runmat_value::symbolic::SymbolicOwner::new_unbounded(vpa_expr(rhs, digits)),
                )
            }
        }
        SymbolicExpr::Function(function, inner) => {
            if let Some(value) = expr.numeric_constant_value() {
                decimal_from_f64(value, digits)
            } else {
                SymbolicExpr::Function(
                    *function,
                    runmat_value::symbolic::SymbolicOwner::new_unbounded(vpa_expr(inner, digits)),
                )
            }
        }
    }
}

fn expr_as_exact_rational(lhs: &SymbolicExpr, rhs: &SymbolicExpr) -> Option<SymbolicExpr> {
    let numerator = exact_integer(lhs)?;
    let denominator = exact_integer(rhs)?;
    SymbolicExpr::rational(numerator, denominator)
}

fn exact_integer(expr: &SymbolicExpr) -> Option<BigInt> {
    match expr {
        SymbolicExpr::Constant(value) if value.is_finite() && value.fract() == 0.0 => {
            BigInt::from_f64(*value)
        }
        SymbolicExpr::Rational {
            numerator,
            denominator,
        } if denominator.is_one() => Some(numerator.as_ref().clone()),
        _ => None,
    }
}

fn decimal_from_f64(value: f64, digits: usize) -> SymbolicExpr {
    let text = significant_decimal_from_f64(value, digits);
    SymbolicExpr::decimal_literal(text, value, digits)
}

fn decimal_from_rational(numerator: &BigInt, denominator: &BigInt, digits: usize) -> SymbolicExpr {
    let text = significant_decimal_from_rational(numerator, denominator, digits);
    let value = runmat_value::symbolic::rational_to_f64(numerator, denominator).unwrap_or(f64::NAN);
    SymbolicExpr::decimal_literal(text, value, digits)
}

fn vpa_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    vpa_error_with_message(error, error.message)
}
fn vpa_error_with_message(
    error: &'static BuiltinErrorDescriptor,
    message: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder = build_runtime_error(message.to_string())
        .with_builtin(BUILTIN_NAME)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{IntegerStorage, Tensor};
    use std::sync::Mutex;

    static DIGITS_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vpa_formats_numeric_input_to_requested_digits() {
        let value = block_on(vpa_builtin(
            None,
            Value::Num(std::f64::consts::PI),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(50.0)]),
        ))
        .expect("vpa");
        let text = value.to_string();
        assert!(text.starts_with("3.141592653589793"));
        assert_eq!(text.chars().filter(|ch| ch.is_ascii_digit()).count(), 50);
    }

    #[test]
    fn vpa_precision_reads_typed_integer_tensor_storage_exactly() {
        for storage in [
            IntegerStorage::I8(vec![12]),
            IntegerStorage::I16(vec![12]),
            IntegerStorage::I32(vec![12]),
            IntegerStorage::I64(vec![12]),
            IntegerStorage::U8(vec![12]),
            IntegerStorage::U16(vec![12]),
            IntegerStorage::U32(vec![12]),
            IntegerStorage::U64(vec![12]),
        ] {
            let precision = Tensor::new_integer(storage, vec![1, 1]).expect("precision");

            let value = block_on(vpa_builtin(
                None,
                Value::Num(std::f64::consts::PI),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
                    precision,
                )]),
            ))
            .expect("vpa");
            assert_eq!(value.to_string(), "3.14159265359");
        }
    }

    #[test]
    fn vpa_preserves_all_integer_scalar_classes_exactly() {
        for value in [
            Value::Int(runmat_value::IntValue::I8(i8::MIN)),
            Value::Int(runmat_value::IntValue::I16(i16::MIN)),
            Value::Int(runmat_value::IntValue::I32(i32::MIN)),
            Value::Int(runmat_value::IntValue::I64(i64::MIN)),
            Value::Int(runmat_value::IntValue::U8(u8::MAX)),
            Value::Int(runmat_value::IntValue::U16(u16::MAX)),
            Value::Int(runmat_value::IntValue::U32(u32::MAX)),
            Value::Int(runmat_value::IntValue::U64(u64::MAX)),
        ] {
            let expected = match &value {
                Value::Int(runmat_value::IntValue::I8(value)) => value.to_string(),
                Value::Int(runmat_value::IntValue::I16(value)) => value.to_string(),
                Value::Int(runmat_value::IntValue::I32(value)) => value.to_string(),
                Value::Int(runmat_value::IntValue::I64(value)) => value.to_string(),
                Value::Int(runmat_value::IntValue::U8(value)) => value.to_string(),
                Value::Int(runmat_value::IntValue::U16(value)) => value.to_string(),
                Value::Int(runmat_value::IntValue::U32(value)) => value.to_string(),
                Value::Int(runmat_value::IntValue::U64(value)) => value.to_string(),
                _ => unreachable!(),
            };
            let result = block_on(vpa_builtin(
                None,
                value,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(32.0)]),
            ))
            .expect("vpa");
            assert_eq!(result.to_string(), expected);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vpa_uses_current_digits_when_precision_is_omitted() {
        let _guard = DIGITS_TEST_LOCK.lock().expect("digits lock");
        struct DigitsReset;
        impl Drop for DigitsReset {
            fn drop(&mut self) {
                super::super::digits::set_current_digits_for_test(
                    super::super::digits::DEFAULT_DIGITS,
                );
            }
        }

        let _reset = DigitsReset;
        super::super::digits::set_current_digits_for_test(12);
        let value = block_on(vpa_builtin(
            None,
            Value::Num(std::f64::consts::PI),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("vpa");
        assert_eq!(value.to_string(), "3.14159265359");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vpa_formats_exact_rational_text() {
        let rational = SymbolicExpr::rational(1, 3).expect("rational");
        let value = block_on(vpa_builtin(
            None,
            Value::Symbolic(rational),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(20.0)]),
        ))
        .expect("vpa");
        assert_eq!(value.to_string(), "0.33333333333333333333");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vpa_rounds_exact_rationals_to_requested_digits() {
        let large = BigInt::from(10).pow(400);
        let ratio = SymbolicExpr::rational(large.clone(), large * 2 + 1).unwrap();
        let finite = block_on(vpa_builtin(
            None,
            Value::Symbolic(ratio),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(20.)]),
        ))
        .unwrap();
        let Value::Symbolic(finite) = finite else {
            panic!("symbolic decimal")
        };
        assert_eq!(finite.numeric_constant_value(), Some(0.5));
        assert_eq!(finite.to_string(), "0.5");
        for (numerator, denominator, digits, expected) in [
            (4, 3, 2, "1.3"),
            (1234, 1, 2, "1200"),
            (9999, 1, 2, "10000"),
            (-1499, 1, 2, "-1500"),
        ] {
            let result = block_on(vpa_builtin(
                None,
                Value::Symbolic(SymbolicExpr::rational(numerator, denominator).unwrap()),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(
                    digits as f64,
                )]),
            ))
            .unwrap();
            assert_eq!(result.to_string(), expected);
        }
        let tiny = SymbolicExpr::rational(BigInt::from(1), BigInt::from(10).pow(80)).unwrap();
        let tiny = block_on(vpa_builtin(
            None,
            Value::Symbolic(tiny),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(3.)]),
        ))
        .unwrap();
        assert_eq!(tiny.to_string(), format!("0.{}1", "0".repeat(79)));
        let rational = SymbolicExpr::rational(2, 3).expect("rational");
        let value = block_on(vpa_builtin(
            None,
            Value::Symbolic(rational),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
        ))
        .expect("vpa");
        assert_eq!(value.to_string(), "0.7");

        let rational = SymbolicExpr::rational(999, 1000).expect("rational");
        let value = block_on(vpa_builtin(
            None,
            Value::Symbolic(rational),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
        ))
        .expect("vpa");
        assert_eq!(value.to_string(), "1");

        let rational = SymbolicExpr::rational(999, 10000).expect("rational");
        let value = block_on(vpa_builtin(
            None,
            Value::Symbolic(rational),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
        ))
        .expect("vpa");
        assert_eq!(value.to_string(), "0.1");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vpa_preserves_symbolic_variables() {
        let value = block_on(vpa_builtin(
            None,
            Value::Symbolic(SymbolicExpr::variable("x")),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(20.0)]),
        ))
        .expect("vpa");
        assert_eq!(value.to_string(), "x");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vpa_rejects_invalid_precision() {
        let err = block_on(vpa_builtin(
            None,
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.0)]),
        ))
        .unwrap_err();
        assert_eq!(err.identifier.as_deref(), Some("RunMat:vpa:InvalidDigits"));
    }

    #[test]
    fn vpa_rejects_negative_typed_integer_tensor_precision() {
        let precision =
            Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("precision");

        let err = block_on(vpa_builtin(
            None,
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
                precision,
            )]),
        ))
        .unwrap_err();
        assert_eq!(err.identifier.as_deref(), Some("RunMat:vpa:InvalidDigits"));
    }
}
