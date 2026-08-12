//! MATLAB-compatible `digits` helper for symbolic variable-precision defaults.

use std::cell::Cell;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, Value};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::text_scalar;
use crate::builtins::common::tensor;

const BUILTIN_NAME: &str = "digits";
pub(crate) const DEFAULT_DIGITS: usize = 32;
pub(crate) const MIN_DIGITS: usize = 2;
pub(crate) const MAX_DIGITS: usize = 100_000_000;

const DIGITS_DEFAULT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "digits-default-reset",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "digits('default') is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DigitsDefaultResetExtension"),
};
const DIGITS_NUMERIC_TEXT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "digits-numeric-text",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "digits with a numeric text scalar is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DigitsNumericTextExtension"),
};
pub const DIGITS_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [DIGITS_DEFAULT_EXTENSION, DIGITS_NUMERIC_TEXT_EXTENSION];

const DIGITS_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "d",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The positive integer precision control is read exactly from every host integer class; host double is rounded to the nearest integer before enforcing the documented 2..=100000000 range.",
    }];
pub const DIGITS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "old = digits(integer_d)",
        inputs: &DIGITS_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "d changes host symbolic precision state only; the returned previous precision is always a double scalar, and invalid controls leave state unchanged.",
    }];

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
    description: "New numeric digit count.",
}];

const DIGITS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
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
    extensions(crate::builtins::math::symbolic::digits::DIGITS_EXTENSIONS),
    integer_capabilities(crate::builtins::math::symbolic::digits::DIGITS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::symbolic::digits"
)]
async fn digits_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(digits_error(&DIGITS_ERRORS[0]));
    }
    let old = current_digits();
    if let Some(value) = rest.first() {
        reject_nonnumeric_digits_control(value)?;
        ensure_digits_text_extension(value)?;
        let new_digits = parse_digits(value)?;
        CURRENT_DIGITS.with(|digits| digits.set(new_digits));
    }
    Ok(Value::Num(old as f64))
}

fn reject_nonnumeric_digits_control(value: &Value) -> BuiltinResult<()> {
    if matches!(
        value,
        Value::Bool(_) | Value::LogicalArray(_) | Value::GpuTensor(_)
    ) {
        return Err(digits_error(&DIGITS_ERRORS[1]));
    }
    Ok(())
}

fn ensure_digits_text_extension(value: &Value) -> BuiltinResult<()> {
    let Some(text) = text_scalar(value) else {
        return Ok(());
    };
    if text.trim().eq_ignore_ascii_case("default") {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DIGITS_DEFAULT_EXTENSION,
            BUILTIN_NAME,
        )
    } else {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DIGITS_NUMERIC_TEXT_EXTENSION,
            BUILTIN_NAME,
        )
    }
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
    if let Some(value) = tensor::scalar_integer_value(value) {
        return validate_integer_digits(&value);
    }
    let parsed = match value {
        Value::Num(value) => *value,
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        other => {
            return Err(digits_error_with_message(
                &DIGITS_ERRORS[1],
                format!("{}: got {other:?}", DIGITS_ERRORS[1].message),
            ))
        }
    };
    validate_digits(parsed)
}

pub(crate) fn validate_integer_digits(value: &IntValue) -> BuiltinResult<usize> {
    let value = value
        .try_to_usize()
        .ok_or_else(|| digits_error(&DIGITS_ERRORS[1]))?;
    if !(MIN_DIGITS..=MAX_DIGITS).contains(&value) {
        return Err(digits_error_with_message(
            &DIGITS_ERRORS[1],
            format!(
                "{}: supported range is {MIN_DIGITS}..={MAX_DIGITS}",
                DIGITS_ERRORS[1].message
            ),
        ));
    }
    Ok(value)
}

pub(crate) fn validate_digits(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(digits_error(&DIGITS_ERRORS[1]));
    }
    let rounded = value.round();
    if rounded < MIN_DIGITS as f64 || rounded > MAX_DIGITS as f64 {
        return Err(digits_error_with_message(
            &DIGITS_ERRORS[1],
            format!(
                "{}: supported range is {MIN_DIGITS}..={MAX_DIGITS}",
                DIGITS_ERRORS[1].message
            ),
        ));
    }
    Ok(rounded as usize)
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
    use runmat_value::{IntegerStorage, LogicalArray, Tensor};
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
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
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
    fn digits_rounds_fractional_precision_and_enforces_public_bounds() {
        let _guard = lock_digits();
        assert_eq!(
            block_on(digits_builtin(vec![Value::Num(2.5)])).expect("rounded precision"),
            Value::Num(DEFAULT_DIGITS as f64)
        );
        assert_eq!(current_digits(), 3);
        assert_eq!(
            block_on(digits_builtin(vec![Value::Num(MAX_DIGITS as f64)])).expect("maximum"),
            Value::Num(3.0)
        );
        assert_eq!(current_digits(), MAX_DIGITS);
        let err = block_on(digits_builtin(vec![Value::Num(1.0)])).unwrap_err();
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:digits:InvalidDigits")
        );
        assert_eq!(current_digits(), MAX_DIGITS);
        assert!(block_on(digits_builtin(vec![Value::Num(MAX_DIGITS as f64 + 1.0)])).is_err());
        assert_eq!(current_digits(), MAX_DIGITS);
        set_current_digits_for_test(DEFAULT_DIGITS);
    }

    #[test]
    fn digits_reads_typed_integer_tensor_storage_exactly() {
        let _guard = lock_digits();
        for storage in [
            IntegerStorage::I8(vec![40]),
            IntegerStorage::I16(vec![40]),
            IntegerStorage::I32(vec![40]),
            IntegerStorage::I64(vec![40]),
            IntegerStorage::U8(vec![40]),
            IntegerStorage::U16(vec![40]),
            IntegerStorage::U32(vec![40]),
            IntegerStorage::U64(vec![40]),
        ] {
            set_current_digits_for_test(DEFAULT_DIGITS);
            let precision = Tensor::new_integer(storage, vec![1, 1]).expect("precision");

            assert_eq!(
                block_on(digits_builtin(vec![Value::Tensor(precision)])).expect("digits"),
                Value::Num(DEFAULT_DIGITS as f64)
            );
            assert_eq!(current_digits(), 40);
        }
    }

    #[test]
    fn digits_rejects_negative_typed_integer_tensor_storage() {
        let _guard = lock_digits();
        let precision =
            Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("precision");

        let err = block_on(digits_builtin(vec![Value::Tensor(precision)])).unwrap_err();
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:digits:InvalidDigits")
        );
        assert_eq!(current_digits(), DEFAULT_DIGITS);
    }

    #[test]
    fn digits_rejects_logical_and_resident_controls_without_provider_access() {
        let _guard = lock_digits();
        for value in [
            Value::Bool(true),
            Value::LogicalArray(LogicalArray::new(vec![1], vec![1, 1]).unwrap()),
            Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
                shape: vec![1, 1],
                device_id: u32::MAX,
                buffer_id: u64::MAX,
            }),
        ] {
            assert!(block_on(digits_builtin(vec![value])).is_err());
            assert_eq!(current_digits(), DEFAULT_DIGITS);
        }
    }

    #[test]
    fn digits_text_extensions_are_gated_in_matlab_mode() {
        let _guard = lock_digits();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let default_err = block_on(digits_builtin(vec![Value::from("default")])).unwrap_err();
        assert_eq!(
            default_err.identifier.as_deref(),
            Some("RunMat:compatibility:DigitsDefaultResetExtension")
        );
        let numeric_err = block_on(digits_builtin(vec![Value::from("50")])).unwrap_err();
        assert_eq!(
            numeric_err.identifier.as_deref(),
            Some("RunMat:compatibility:DigitsNumericTextExtension")
        );
        assert_eq!(current_digits(), DEFAULT_DIGITS);
    }

    #[test]
    fn digits_descriptor_declares_public_integer_contract() {
        assert_eq!(DIGITS_DESCRIPTOR.signatures.len(), 2);
        assert_eq!(DIGITS_EXTENSIONS.len(), 2);
        assert_eq!(DIGITS_INTEGER_CAPABILITIES.len(), 1);
        assert_eq!(
            DIGITS_INTEGER_CAPABILITIES[0].output_class,
            BuiltinIntegerOutputClassRule::Double
        );
    }
}
