//! Minimal symbolic `piecewise` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    SymbolicExpr, Value,
};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::{symbolic_expr_to_value, value_to_symbolic_scalar};

const BUILTIN_NAME: &str = "piecewise";

const PIECEWISE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "f",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Symbolic piecewise expression.",
}];

const PIECEWISE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "arg",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Condition/value pairs and optional default value.",
}];

const PIECEWISE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "f = piecewise(cond1, value1, ...)",
    inputs: &PIECEWISE_INPUTS,
    outputs: &PIECEWISE_OUTPUT,
}];

const PIECEWISE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PIECEWISE.INVALID_INPUT",
    identifier: Some("RunMat:piecewise:InvalidInput"),
    when: "Arguments cannot be represented by the scalar symbolic core.",
    message: "piecewise: expected symbolic or scalar numeric arguments",
};

const PIECEWISE_ERRORS: [BuiltinErrorDescriptor; 1] = [PIECEWISE_ERROR_INVALID_INPUT];

pub const PIECEWISE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PIECEWISE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PIECEWISE_ERRORS,
};

fn piecewise_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "piecewise",
    category = "math/symbolic",
    summary = "Construct a scalar symbolic piecewise expression.",
    keywords = "piecewise,symbolic,condition",
    descriptor(crate::builtins::math::symbolic::piecewise::PIECEWISE_DESCRIPTOR),
    builtin_path = "crate::builtins::math::symbolic::piecewise"
)]
async fn piecewise_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(piecewise_error(&PIECEWISE_ERROR_INVALID_INPUT));
    }

    let mut symbolic_args = Vec::with_capacity(args.len());
    for arg in &args {
        let Some(expr) = value_to_symbolic_scalar(arg) else {
            return Err(piecewise_error(&PIECEWISE_ERROR_INVALID_INPUT));
        };
        symbolic_args.push(expr);
    }

    Ok(symbolic_expr_to_value(SymbolicExpr::function_call(
        BUILTIN_NAME,
        symbolic_args,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn builds_symbolic_piecewise_expression() {
        let result = block_on(piecewise_builtin(vec![
            Value::Symbolic(SymbolicExpr::function_call(
                "lt",
                vec![SymbolicExpr::variable("x"), SymbolicExpr::constant(0.0)],
            )),
            Value::Num(-1.0),
            Value::Symbolic(SymbolicExpr::function_call(
                "gt",
                vec![SymbolicExpr::variable("x"), SymbolicExpr::constant(0.0)],
            )),
            Value::Num(1.0),
        ]))
        .expect("piecewise");

        assert_eq!(result.to_string(), "piecewise(lt(x, 0), -1, gt(x, 0), 1)");
    }

    #[test]
    fn rejects_too_few_arguments() {
        let err = block_on(piecewise_builtin(vec![Value::Symbolic(
            SymbolicExpr::variable("x"),
        )]))
        .expect_err("piecewise arity should fail");

        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:piecewise:InvalidInput")
        );
    }

    #[test]
    fn rejects_nonconvertible_arguments() {
        let err = block_on(piecewise_builtin(vec![
            Value::Symbolic(SymbolicExpr::variable("x")),
            Value::String("unsupported".to_string()),
        ]))
        .expect_err("piecewise unsupported argument should fail");

        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:piecewise:InvalidInput")
        );
    }
}
