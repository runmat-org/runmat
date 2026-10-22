//! MATLAB-compatible scalar `sym` builtin for symbolic variables and constants.

use runmat_builtins::{
    BuiltinErrorDescriptor, SYM_DESCRIPTOR, SYM_ERRORS, SYM_ERROR_TOO_MANY_OUTPUTS,
};

use num_bigint::BigInt;
use runmat_value::{SymbolicExpr, Value};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::{is_valid_identifier, symbolic_expr_to_value, text_scalar, value_to_symbolic_scalar};

const BUILTIN_NAME: &str = "sym";

#[runmat_macros::runtime_builtin(
    name = "sym",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::symbolic::sym"
)]
async fn sym_builtin(value: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &SYM_DESCRIPTOR,
        &SYM_ERROR_TOO_MANY_OUTPUTS,
        "sym",
    )?;
    if !rest.is_empty() {
        return Err(sym_error(&SYM_ERRORS[0]));
    }
    if matches!(&value, Value::CharArray(chars) if chars.shape.iter().skip(2).any(|dimension| *dimension != 1))
    {
        return Err(sym_error(&SYM_ERRORS[2]));
    }
    if let Some(expr) = value_to_symbolic_scalar(&value) {
        return Ok(symbolic_expr_to_value(expr));
    }
    if let Some(text) = text_scalar(&value).map_err(|source| {
        let mut error = sym_error(&SYM_ERRORS[2]);
        error.source = Some(Box::new(source));
        error
    })? {
        let trimmed = text.trim();
        if is_valid_identifier(trimmed) {
            return Ok(Value::Symbolic(SymbolicExpr::variable(trimmed)));
        }
        if let Some(expr) = parse_rational_literal(trimmed) {
            return Ok(symbolic_expr_to_value(expr));
        }
        if let Ok(integer) = trimmed.parse::<BigInt>() {
            return Ok(symbolic_expr_to_value(
                SymbolicExpr::rational(integer, BigInt::from(1_u8)).expect("nonzero denominator"),
            ));
        }
        if let Ok(value) = trimmed.parse::<f64>() {
            if value.is_finite() {
                return Ok(symbolic_expr_to_value(SymbolicExpr::constant(value)));
            }
        }
        return Err(sym_error(&SYM_ERRORS[1]));
    }
    Err(sym_error(&SYM_ERRORS[2]))
}

fn sym_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
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

#[cfg(test)]
pub(crate) async fn sym_builtin_source(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    sym_builtin(
        value,
        crate::call::BuiltinArguments::from_borrowed(&rest, None)?,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authored_wide_integers_and_fractions_do_not_round_through_double() {
        for (text, expected) in [
            ("9007199254740993", "9007199254740993"),
            ("18446744073709551615", "18446744073709551615"),
            ("-9007199254740993", "-9007199254740993"),
            ("9007199254740993/2", "9007199254740993/2"),
        ] {
            let result =
                futures::executor::block_on(sym_builtin_source(Value::from(text), vec![])).unwrap();
            assert_eq!(result.to_string(), expected);
        }
        let decimal =
            futures::executor::block_on(sym_builtin_source(Value::from("3.25"), vec![])).unwrap();
        let Value::Symbolic(decimal) = decimal else {
            panic!("expected symbolic decimal");
        };
        assert_eq!(decimal.numeric_constant_value(), Some(3.25));
        let result = futures::executor::block_on(sym_builtin_source(
            Value::Int(runmat_value::IntValue::U64(u64::MAX)),
            vec![],
        ))
        .unwrap();
        assert_eq!(result.to_string(), u64::MAX.to_string());
        let tensor = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::I64(vec![i64::MIN]),
            vec![1, 1],
        )
        .unwrap();
        let result =
            futures::executor::block_on(sym_builtin_source(Value::Tensor(tensor), vec![])).unwrap();
        assert_eq!(result.to_string(), i64::MIN.to_string());
    }

    #[test]
    fn invalid_rank_options_and_output_arity_have_stable_errors() {
        let chars =
            runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['x', 'y'], vec![1, 1, 2])
                .unwrap();
        let error =
            futures::executor::block_on(sym_builtin_source(Value::CharArray(chars), vec![]))
                .unwrap_err();
        assert_eq!(error.identifier(), SYM_ERRORS[2].identifier);
        let error = futures::executor::block_on(sym_builtin_source(
            Value::from("x"),
            vec![Value::from("real")],
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), SYM_ERRORS[0].identifier);
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error =
            futures::executor::block_on(sym_builtin_source(Value::from("x"), vec![])).unwrap_err();
        assert_eq!(error.identifier(), SYM_ERROR_TOO_MANY_OUTPUTS.identifier);
    }

    #[test]
    fn creates_symbolic_variable_from_text() {
        let value = futures::executor::block_on(sym_builtin_source(Value::from("x"), Vec::new()))
            .expect("sym");
        assert_eq!(value.to_string(), "x");
    }

    #[test]
    fn creates_symbolic_constant_from_numeric_text() {
        let value = futures::executor::block_on(sym_builtin_source(Value::from("3.5"), Vec::new()))
            .expect("sym");
        match value {
            Value::Symbolic(expr) => assert_eq!(expr.constant_value(), Some(3.5)),
            other => panic!("expected symbolic constant, got {other:?}"),
        }
    }

    #[test]
    fn creates_symbolic_rational_from_text() {
        let value =
            futures::executor::block_on(sym_builtin_source(Value::from(" 2 / 6 "), Vec::new()))
                .expect("sym");
        assert_eq!(value.to_string(), "1/3");
        match value {
            Value::Symbolic(expr) => assert_eq!(expr.numeric_constant_value(), Some(1.0 / 3.0)),
            other => panic!("expected symbolic rational, got {other:?}"),
        }
    }

    #[test]
    fn creates_symbolic_rational_from_extreme_signed_text() {
        let value = futures::executor::block_on(sym_builtin_source(
            Value::from("-170141183460469231731687303715884105728 / -1"),
            Vec::new(),
        ))
        .expect("sym");
        assert_eq!(value.to_string(), "170141183460469231731687303715884105728");
    }

    #[test]
    fn rejects_identifier_with_leading_underscore() {
        let err = futures::executor::block_on(sym_builtin_source(Value::from("_x"), Vec::new()))
            .expect_err("invalid identifier should error");

        assert_eq!(err.identifier.as_deref(), Some("RunMat:sym:InvalidText"));
    }
}
