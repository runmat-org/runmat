//! MATLAB-compatible scalar `sym` builtin for symbolic variables and constants.

use num_bigint::BigInt;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    SymbolicExpr, Value,
};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::{is_valid_identifier, symbolic_expr_to_value, text_scalar, value_to_symbolic_scalar};

const BUILTIN_NAME: &str = "sym";

const SYM_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar symbolic variable or constant.",
}];

const SYM_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Variable name, numeric scalar, or existing symbolic scalar.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Reserved symbolic options.",
    },
];

const SYM_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "S = sym(value)",
    inputs: &SYM_INPUTS,
    outputs: &SYM_OUTPUT,
}];

const SYM_ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.SYM.ARG_COUNT",
        identifier: Some("RunMat:sym:ArgCount"),
        when: "Options beyond the scalar input are supplied.",
        message: "sym: unsupported options",
    },
    BuiltinErrorDescriptor {
        code: "RM.SYM.INVALID_TEXT",
        identifier: Some("RunMat:sym:InvalidText"),
        when: "Text input is neither a valid identifier nor a numeric scalar literal.",
        message: "sym: invalid symbolic text",
    },
    BuiltinErrorDescriptor {
        code: "RM.SYM.INVALID_INPUT",
        identifier: Some("RunMat:sym:InvalidInput"),
        when: "Input cannot be represented as a scalar symbolic value.",
        message: "sym: expected a scalar text, numeric, logical, or symbolic input",
    },
];

pub const SYM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SYM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SYM_ERRORS,
};

#[runtime_builtin(
    name = "sym",
    category = "math/symbolic",
    summary = "Create a scalar symbolic variable or constant.",
    keywords = "sym,symbolic,variable,algebra",
    descriptor(crate::builtins::math::symbolic::sym::SYM_DESCRIPTOR),
    builtin_path = "crate::builtins::math::symbolic::sym"
)]
async fn sym_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(sym_error(&SYM_ERRORS[0]));
    }
    if let Some(expr) = value_to_symbolic_scalar(&value) {
        return Ok(symbolic_expr_to_value(expr));
    }
    if let Some(text) = text_scalar(&value) {
        let trimmed = text.trim();
        if is_valid_identifier(trimmed) {
            return Ok(Value::Symbolic(SymbolicExpr::variable(trimmed)));
        }
        if let Some(expr) = parse_rational_literal(trimmed) {
            return Ok(symbolic_expr_to_value(expr));
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
mod tests {
    use super::*;

    #[test]
    fn creates_symbolic_variable_from_text() {
        let value =
            futures::executor::block_on(sym_builtin(Value::from("x"), Vec::new())).expect("sym");
        assert_eq!(value.to_string(), "x");
    }

    #[test]
    fn creates_symbolic_constant_from_numeric_text() {
        let value =
            futures::executor::block_on(sym_builtin(Value::from("3.5"), Vec::new())).expect("sym");
        match value {
            Value::Symbolic(expr) => assert_eq!(expr.constant_value(), Some(3.5)),
            other => panic!("expected symbolic constant, got {other:?}"),
        }
    }

    #[test]
    fn creates_symbolic_rational_from_text() {
        let value = futures::executor::block_on(sym_builtin(Value::from(" 2 / 6 "), Vec::new()))
            .expect("sym");
        assert_eq!(value.to_string(), "1/3");
        match value {
            Value::Symbolic(expr) => assert_eq!(expr.numeric_constant_value(), Some(1.0 / 3.0)),
            other => panic!("expected symbolic rational, got {other:?}"),
        }
    }

    #[test]
    fn creates_symbolic_rational_from_extreme_signed_text() {
        let value = futures::executor::block_on(sym_builtin(
            Value::from("-170141183460469231731687303715884105728 / -1"),
            Vec::new(),
        ))
        .expect("sym");
        assert_eq!(value.to_string(), "170141183460469231731687303715884105728");
    }

    #[test]
    fn rejects_identifier_with_leading_underscore() {
        let err = futures::executor::block_on(sym_builtin(Value::from("_x"), Vec::new()))
            .expect_err("invalid identifier should error");

        assert_eq!(err.identifier.as_deref(), Some("RunMat:sym:InvalidText"));
    }
}
