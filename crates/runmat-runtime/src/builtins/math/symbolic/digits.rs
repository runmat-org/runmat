//! MATLAB-compatible `digits` helper for symbolic variable-precision defaults.

use std::cell::Cell;

use crate::{call::BuiltinArguments, context::RuntimeContext};
use runmat_builtins::{
    BuiltinErrorDescriptor, SymbolicPrecisionKind, SymbolicScalarOperation,
    DIGITS_DEFAULT_EXTENSION, DIGITS_DESCRIPTOR, DIGITS_ERRORS, DIGITS_ERROR_TOO_MANY_OUTPUTS,
    DIGITS_NUMERIC_TEXT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, Value};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::text_scalar;
use crate::builtins::common::tensor;

const BUILTIN_NAME: &str = "digits";
pub(crate) use runmat_builtins::{
    SYMBOLIC_DEFAULT_DIGITS as DEFAULT_DIGITS, SYMBOLIC_MAX_DIGITS as MAX_DIGITS,
};
pub(crate) const MIN_DIGITS: usize = SymbolicPrecisionKind::Default.minimum();

thread_local! {
    static CURRENT_DIGITS: Cell<usize> = const { Cell::new(DEFAULT_DIGITS) };
}

#[runtime_builtin(
    name = "digits",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::symbolic::digits",
    context = true
)]
async fn digits_builtin(
    runtime: Option<RuntimeContext>,
    rest: BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &DIGITS_DESCRIPTOR,
        &DIGITS_ERROR_TOO_MANY_OUTPUTS,
        BUILTIN_NAME,
    )
    .map_err(super::invocation::terminal)?;
    if rest.len() > 1 {
        return Err(digits_error(&DIGITS_ERRORS[0]));
    }
    if let Some(value) = rest.first() {
        super::invocation::validate(SymbolicScalarOperation::Precision, 0, 1, value)?;
    }
    super::invocation::admit(
        SymbolicScalarOperation::Precision,
        runtime.as_ref(),
        rest.first().is_some_and(super::invocation::is_text),
    )?;
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
    let Some(text) = text_scalar(value).map_err(|source| {
        let mut error = digits_error(&DIGITS_ERRORS[1]);
        error.source = Some(Box::new(source));
        error
    })?
    else {
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
    if let Some(text) = text_scalar(value).map_err(|source| {
        let mut error = digits_error(&DIGITS_ERRORS[1]);
        error.source = Some(Box::new(source));
        error
    })? {
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
    if !SymbolicPrecisionKind::Default.accepts(value) {
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
    let Some(rounded) = SymbolicPrecisionKind::Default.rounded(value) else {
        return Err(digits_error_with_message(
            &DIGITS_ERRORS[1],
            format!(
                "{}: supported range is {MIN_DIGITS}..={MAX_DIGITS}",
                DIGITS_ERRORS[1].message
            ),
        ));
    };
    Ok(rounded)
}

fn digits_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    digits_error_with_message(error, error.message)
}

fn digits_error_with_message(
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
#[path = "digits/admission_tests.rs"]
mod admission_tests;
#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{
        BuiltinIntegerOutputClassRule, DIGITS_EXTENSIONS, DIGITS_INTEGER_CAPABILITIES,
    };
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
            block_on(digits_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(Vec::new())
            ))
            .expect("digits"),
            Value::Num(32.0)
        );
        assert_eq!(
            block_on(digits_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(50.0)])
            ))
            .expect("digits"),
            Value::Num(32.0)
        );
        assert_eq!(
            block_on(digits_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(Vec::new())
            ))
            .expect("digits"),
            Value::Num(50.0)
        );
        assert_eq!(
            block_on(digits_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
                    "default"
                )])
            ))
            .expect("digits"),
            Value::Num(50.0)
        );
        assert_eq!(current_digits(), DEFAULT_DIGITS);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn digits_rounds_fractional_precision_and_enforces_public_bounds() {
        let _guard = lock_digits();
        assert_eq!(
            block_on(digits_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.5)])
            ))
            .expect("rounded precision"),
            Value::Num(DEFAULT_DIGITS as f64)
        );
        assert_eq!(current_digits(), 3);
        assert_eq!(
            block_on(digits_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(
                    MAX_DIGITS as f64
                )])
            ))
            .expect("maximum"),
            Value::Num(3.0)
        );
        assert_eq!(current_digits(), MAX_DIGITS);
        let err = block_on(digits_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
        ))
        .unwrap_err();
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:digits:InvalidDigits")
        );
        assert_eq!(current_digits(), MAX_DIGITS);
        assert!(block_on(digits_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(
                MAX_DIGITS as f64 + 1.0
            )])
        ))
        .is_err());
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
                block_on(digits_builtin(
                    None,
                    crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
                        precision
                    )])
                ))
                .expect("digits"),
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

        let err = block_on(digits_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
                precision,
            )]),
        ))
        .unwrap_err();
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:digits:InvalidDigits")
        );
        assert_eq!(current_digits(), DEFAULT_DIGITS);
    }

    #[test]
    fn digits_rejects_logical_and_resident_controls_without_provider_access() {
        let _guard = lock_digits();
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("resident numeric digits control");
        let caller_alias = allocation.handle.clone();
        for value in [
            Value::Bool(true),
            Value::LogicalArray(LogicalArray::new(vec![1], vec![1, 1]).unwrap()),
            Value::GpuTensor(allocation.handle),
        ] {
            assert!(block_on(digits_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![value])
            ))
            .is_err());
            assert_eq!(current_digits(), DEFAULT_DIGITS);
        }
        assert_eq!(allocation.probe.retirements(), 0);
        caller_alias.allocation_lease().validate_live().unwrap();
        drop(caller_alias);
        assert_eq!(allocation.probe.retirements(), 1);
    }

    #[test]
    fn digits_text_extensions_are_gated_in_matlab_mode() {
        let _guard = lock_digits();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let default_err = block_on(digits_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("default")]),
        ))
        .unwrap_err();
        assert_eq!(
            default_err.identifier.as_deref(),
            Some("RunMat:compatibility:DigitsDefaultResetExtension")
        );
        let numeric_err = block_on(digits_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("50")]),
        ))
        .unwrap_err();
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
