//! MATLAB-compatible `digits` helper for symbolic variable-precision defaults.

use std::cell::Cell;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::text_scalar;

const BUILTIN_NAME: &str = "digits";
pub(crate) const DEFAULT_DIGITS: usize = 32;
pub(crate) const MIN_DIGITS: usize = 1;
pub(crate) const MAX_DIGITS: usize = 4096;

thread_local! {
    static CURRENT_DIGITS: Cell<usize> = const { Cell::new(DEFAULT_DIGITS) };
}

const DIGITS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "d",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current or previous variable-precision digit count.",
}];

const DIGITS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "d",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "New digit count, or 'default' to reset to 32.",
}];

const DIGITS_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "d = digits",
        inputs: &[],
        outputs: &DIGITS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "old = digits(d)",
        inputs: &DIGITS_INPUTS,
        outputs: &DIGITS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "old = digits('default')",
        inputs: &DIGITS_INPUTS,
        outputs: &DIGITS_OUTPUT,
    },
];

const DIGITS_ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.DIGITS.ARG_COUNT",
        identifier: Some("RunMat:digits:ArgCount"),
        when: "More than one input argument is supplied.",
        message: "digits: too many input arguments",
    },
    BuiltinErrorDescriptor {
        code: "RM.DIGITS.INVALID_DIGITS",
        identifier: Some("RunMat:digits:InvalidDigits"),
        when: "The requested precision is not an integer in the supported range.",
        message: "digits: expected a positive integer digit count",
    },
];

pub const DIGITS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DIGITS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DIGITS_ERRORS,
};

#[runtime_builtin(
    name = "digits",
    category = "math/symbolic",
    summary = "Get or set the default variable-precision digit count.",
    keywords = "digits,vpa,symbolic,precision",
    descriptor(crate::builtins::math::symbolic::digits::DIGITS_DESCRIPTOR),
    builtin_path = "crate::builtins::math::symbolic::digits"
)]
async fn digits_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(digits_error(&DIGITS_ERRORS[0]));
    }
    let old = current_digits();
    if let Some(value) = rest.first() {
        let new_digits = parse_digits(value)?;
        CURRENT_DIGITS.with(|digits| digits.set(new_digits));
    }
    Ok(Value::Num(old as f64))
}

pub(crate) fn current_digits() -> usize {
    CURRENT_DIGITS.with(Cell::get)
}

#[cfg(test)]
pub(crate) fn set_current_digits_for_test(digits: usize) {
    CURRENT_DIGITS.with(|current| current.set(digits));
}

fn parse_digits(value: &Value) -> BuiltinResult<usize> {
    if let Some(text) = text_scalar(value) {
        if text.trim().eq_ignore_ascii_case("default") {
            return Ok(DEFAULT_DIGITS);
        }
        if let Ok(parsed) = text.trim().parse::<f64>() {
            return validate_digits(parsed);
        }
        return Err(digits_error(&DIGITS_ERRORS[1]));
    }
    let parsed = match value {
        Value::Num(value) => *value,
        Value::Int(value) => value.to_f64(),
        Value::Bool(value) => {
            if *value {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data[0],
        other => {
            return Err(digits_error_with_message(
                &DIGITS_ERRORS[1],
                format!("{}: got {other:?}", DIGITS_ERRORS[1].message),
            ))
        }
    };
    validate_digits(parsed)
}

pub(crate) fn validate_digits(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(digits_error(&DIGITS_ERRORS[1]));
    }
    if value < MIN_DIGITS as f64 || value > MAX_DIGITS as f64 {
        return Err(digits_error_with_message(
            &DIGITS_ERRORS[1],
            format!(
                "{}: supported range is {MIN_DIGITS}..={MAX_DIGITS}",
                DIGITS_ERRORS[1].message
            ),
        ));
    }
    Ok(value as usize)
}

fn digits_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    digits_error_with_message(error, error.message)
}

fn digits_error_with_message(
    error: &'static BuiltinErrorDescriptor,
    message: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder = build_runtime_error(message.to_string()).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use std::sync::{Mutex, MutexGuard};

    static DIGITS_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn lock_digits() -> MutexGuard<'static, ()> {
        let guard = DIGITS_TEST_LOCK.lock().expect("digits lock");
        set_current_digits_for_test(DEFAULT_DIGITS);
        guard
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn digits_gets_and_sets_default_precision() {
        let _guard = lock_digits();
        assert_eq!(
            block_on(digits_builtin(Vec::new())).expect("digits"),
            Value::Num(32.0)
        );
        assert_eq!(
            block_on(digits_builtin(vec![Value::Num(50.0)])).expect("digits"),
            Value::Num(32.0)
        );
        assert_eq!(
            block_on(digits_builtin(Vec::new())).expect("digits"),
            Value::Num(50.0)
        );
        assert_eq!(
            block_on(digits_builtin(vec![Value::from("default")])).expect("digits"),
            Value::Num(50.0)
        );
        assert_eq!(current_digits(), DEFAULT_DIGITS);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn digits_rejects_invalid_precision() {
        let _guard = lock_digits();
        let err = block_on(digits_builtin(vec![Value::Num(2.5)])).unwrap_err();
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:digits:InvalidDigits")
        );
    }
}
