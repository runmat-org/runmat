//! Admission of consumed host model properties before provider materialization.
use super::{
    all_zero, coefficients_from_property, control_error, invalid_model_identifier, scalar_f64,
    scalar_property, scalar_text, unsupported_model_identifier, validate_coefficients,
    validate_delay, validate_sample_time, validate_variable, validate_variable_domain,
};
use crate::builtins::common::tensor;
use crate::builtins::common::validation::{
    native_integer_value_is_exact_f64, value_has_logical_class, value_has_native_integer_class,
};
use crate::BuiltinResult;
use runmat_builtins::{
    BuiltinExtensionDescriptor, INTEGER_COEFFICIENT_EXTENSION, INTEGER_SAMPLE_TIME_EXTENSION,
    LOGICAL_COEFFICIENT_EXTENSION, SS_INTEGER_MATRIX_EXTENSION, SS_INTEGER_SAMPLE_TIME_EXTENSION,
};
use runmat_value::{ObjectInstance, Tensor, Value};

/// Host state matrices retain the model domain's row-by-column projection.
/// Rank-zero tensors have no matrix extent; rank-one tensors form one row.
pub(in super::super) fn state_matrix_dimensions(tensor: &Tensor) -> (usize, usize) {
    match tensor.shape() {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

pub(super) fn preflight_host_model_properties(
    object: &ObjectInstance,
    builtin: &'static str,
) -> BuiltinResult<()> {
    for (name, label) in [("Numerator", "numerator"), ("Denominator", "denominator")] {
        let value = object.properties().get(name).ok_or_else(|| {
            control_error(
                builtin,
                invalid_model_identifier(builtin),
                format!("{builtin}: tf object is missing {name}"),
            )
        })?;
        admit_coefficient_property(value, label, builtin)?;
        if !matches!(value, Value::GpuTensor(_)) {
            let coefficients = coefficients_from_property(object, name, builtin)?;
            validate_coefficients(label, &coefficients, builtin)?;
            if name == "Denominator" && all_zero(&coefficients) {
                return Err(control_error(
                    builtin,
                    invalid_model_identifier(builtin),
                    format!("{builtin}: denominator coefficients must not all be zero"),
                ));
            }
        }
    }
    for name in ["Ts", "InputDelay", "OutputDelay"] {
        if let Some(value) = object.properties().get(name) {
            if name == "Ts" {
                admit_sample_time_property(value, builtin)?;
            } else {
                admit_delay_property(value, name, builtin)?;
            }
        }
        if !matches!(object.properties().get(name), Some(Value::GpuTensor(_))) {
            let value = scalar_property(object, name, builtin)?;
            if name == "Ts" {
                validate_sample_time(value, builtin)?;
            } else {
                validate_delay(value, name, builtin)?;
            }
        }
    }
    let variable = object
        .properties()
        .get("Variable")
        .filter(|value| !matches!(value, Value::GpuTensor(_)))
        .map(|value| {
            scalar_text(value, "Variable", builtin)
                .and_then(|text| validate_variable(&text, builtin))
        })
        .transpose()?;
    if let (Some(variable), Some(sample_time)) = (variable, object.properties().get("Ts")) {
        if !matches!(sample_time, Value::GpuTensor(_)) {
            validate_variable_domain(&variable, scalar_property(object, "Ts", builtin)?, builtin)?;
        }
    }
    Ok(())
}

/// A tf property's numeric class can change after construction through public
/// property assignment. Check the authoritative class and integer payload each
/// time the model is admitted, before converting coefficients to binary64.
/// Resident 64-bit payloads are checked after their single owning gather.
pub(super) fn admit_coefficient_property(
    value: &Value,
    role: &str,
    builtin: &'static str,
) -> BuiltinResult<()> {
    admit_integer_f64_boundary(
        value,
        &INTEGER_COEFFICIENT_EXTENSION,
        super::invalid_coefficients_identifier(builtin),
        role,
        builtin,
    )?;
    if value_has_logical_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOGICAL_COEFFICIENT_EXTENSION,
            builtin,
        )?;
    }
    Ok(())
}

pub(super) fn admit_sample_time_property(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    admit_integer_f64_boundary(
        value,
        &INTEGER_SAMPLE_TIME_EXTENSION,
        super::invalid_sample_time_identifier(builtin),
        "sample time",
        builtin,
    )
}

/// Delays are consumed as binary64 model metadata. Unlike coefficients and
/// sample time, there is no separate typed-integer delay language extension;
/// only the lossless conversion requirement applies.
pub(super) fn admit_delay_property(
    value: &Value,
    name: &str,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if value_has_native_integer_class(value)
        && !matches!(value, Value::GpuTensor(_))
        && !native_integer_value_is_exact_f64(value)
    {
        return Err(control_error(
            builtin,
            invalid_model_identifier(builtin),
            format!("{builtin}: {name} integer value must be exactly representable as double"),
        ));
    }
    Ok(())
}

fn admit_integer_f64_boundary(
    value: &Value,
    extension: &'static BuiltinExtensionDescriptor,
    identifier: &'static str,
    role: &str,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)?;
        if !matches!(value, Value::GpuTensor(_)) && !native_integer_value_is_exact_f64(value) {
            return Err(control_error(
                builtin,
                identifier,
                format!("{builtin}: integer {role} values must be exactly representable as double"),
            ));
        }
    }
    Ok(())
}

/// Check only the state-space fields consumed by an operation; unused model
/// properties must not affect admission or cause provider access.
pub fn preflight_ss_host_properties(
    object: &ObjectInstance,
    required: &[&str],
    builtin: &'static str,
) -> BuiltinResult<()> {
    for &name in required {
        let value = object.properties().get(name).ok_or_else(|| {
            control_error(
                builtin,
                invalid_model_identifier(builtin),
                format!("{builtin}: ss object is missing {name}"),
            )
        })?;
        preflight_ss_host_value(name, value, builtin)?;
    }
    Ok(())
}

pub(super) fn preflight_ss_host_value(
    name: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    admit_ss_integer_property(value, name, builtin)?;
    if matches!(value, Value::GpuTensor(_)) {
        return Ok(());
    }
    match name {
        "Ts" => validate_sample_time(scalar_f64(value, name, builtin)?, builtin)?,
        "InputDelay" | "OutputDelay" => preflight_ss_delay(value, name, builtin)?,
        "A" => preflight_ss_state_matrix(value, builtin)?,
        "B" | "C" | "D" => preflight_ss_matrix_value(value, name, builtin)?,
        _ => {}
    }
    Ok(())
}

fn admit_ss_integer_property(
    value: &Value,
    name: &str,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if !value_has_native_integer_class(value) {
        return Ok(());
    }
    let extension = match name {
        "A" | "B" | "C" | "D" => Some(&SS_INTEGER_MATRIX_EXTENSION),
        "Ts" => Some(&SS_INTEGER_SAMPLE_TIME_EXTENSION),
        // The constructor has no delay argument. Mutated delays still cannot
        // lose integer precision when an analysis operation consumes them.
        "InputDelay" | "OutputDelay" => None,
        _ => None,
    };
    if let Some(extension) = extension {
        crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)?;
    }
    if !matches!(value, Value::GpuTensor(_)) && !native_integer_value_is_exact_f64(value) {
        return Err(control_error(
            builtin,
            invalid_model_identifier(builtin),
            format!("{builtin}: ss {name} integer values must be exactly representable as double"),
        ));
    }
    Ok(())
}

fn preflight_ss_state_matrix(value: &Value, builtin: &'static str) -> BuiltinResult<()> {
    preflight_ss_matrix_value(value, "A", builtin)?;
    match value {
        Value::Tensor(tensor)
            if state_matrix_dimensions(tensor).0 != state_matrix_dimensions(tensor).1 =>
        {
            return Err(control_error(
                builtin,
                invalid_model_identifier(builtin),
                format!("{builtin}: ss A must be square, got {:?}", tensor.shape()),
            ));
        }
        Value::ComplexTensor(tensor) if tensor.rows() != tensor.cols() => {
            return Err(control_error(
                builtin,
                invalid_model_identifier(builtin),
                format!("{builtin}: ss A must be square, got {:?}", tensor.shape()),
            ));
        }
        Value::LogicalArray(logical)
            if match logical.shape.as_slice() {
                [size] => *size != 1,
                [rows, columns] => rows != columns,
                _ => false,
            } =>
        {
            return Err(control_error(
                builtin,
                invalid_model_identifier(builtin),
                format!("{builtin}: ss A must be square, got {:?}", logical.shape),
            ));
        }
        _ => {}
    }
    Ok(())
}

fn preflight_ss_delay(value: &Value, name: &str, builtin: &'static str) -> BuiltinResult<()> {
    let valid = |re: f64, im: f64| re.is_finite() && im.is_finite() && re >= 0.0 && im == 0.0;
    let values_valid = match value {
        Value::Tensor(tensor) => tensor::tensor_values_f64_cow(tensor)
            .iter()
            .all(|&re| valid(re, 0.0)),
        Value::ComplexTensor(tensor) => tensor::complex_tensor_values_complex64(tensor)
            .iter()
            .all(|number| valid(number.re, number.im)),
        Value::LogicalArray(_) | Value::Bool(_) | Value::Int(_) => true,
        Value::Num(re) => valid(*re, 0.0),
        Value::Complex(re, im) => valid(*re, *im),
        other => {
            return Err(control_error(
                builtin,
                unsupported_model_identifier(builtin),
                format!("{builtin}: ss {name} must be a finite numeric delay, got {other:?}"),
            ))
        }
    };
    if !values_valid {
        return Err(control_error(
            builtin,
            invalid_model_identifier(builtin),
            format!("{builtin}: ss {name} must contain finite nonnegative real values"),
        ));
    }
    Ok(())
}

fn preflight_ss_matrix_value(
    value: &Value,
    name: &str,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let invalid_shape = |shape: &[usize]| {
        control_error(
            builtin,
            invalid_model_identifier(builtin),
            format!("{builtin}: ss {name} must be a 2-D matrix, got {shape:?}"),
        )
    };
    let finite = match value {
        Value::Tensor(tensor) => {
            if tensor.shape().len() > 2 {
                return Err(invalid_shape(tensor.shape()));
            }
            tensor::tensor_values_f64_cow(tensor)
                .iter()
                .all(|value| value.is_finite())
        }
        Value::ComplexTensor(tensor) => {
            if tensor.shape().len() > 2 {
                return Err(invalid_shape(tensor.shape()));
            }
            tensor::complex_tensor_values_complex64(tensor)
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        }
        Value::LogicalArray(logical) => {
            if logical.shape.len() > 2 {
                return Err(invalid_shape(&logical.shape));
            }
            true
        }
        Value::Num(value) => value.is_finite(),
        Value::Int(_) | Value::Bool(_) => true,
        Value::Complex(re, im) => re.is_finite() && im.is_finite(),
        other => {
            return Err(control_error(
                builtin,
                unsupported_model_identifier(builtin),
                format!("{builtin}: ss {name} must be a finite numeric matrix, got {other:?}"),
            ));
        }
    };
    if !finite {
        return Err(control_error(
            builtin,
            unsupported_model_identifier(builtin),
            format!("{builtin}: ss {name} must contain finite numeric values"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};
    use runmat_value::{ComplexTensor, IntegerStorage, LogicalArray, Tensor};

    fn ss_object(properties: impl IntoIterator<Item = (&'static str, Value)>) -> ObjectInstance {
        let mut object = runmat_value::UnboundedObjectBuilder::new("ss");
        for (name, value) in properties {
            object.insert(name.to_owned(), value);
        }
        object.finish()
    }

    #[test]
    fn generic_ss_admission_accepts_complex_logical_and_zero_delay_arrays() {
        for a in [
            Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).unwrap()),
            Value::LogicalArray(LogicalArray::new(vec![1], vec![1, 1]).unwrap()),
        ] {
            let object = ss_object([
                ("A", a),
                ("Ts", Value::Num(0.0)),
                (
                    "InputDelay",
                    Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()),
                ),
                (
                    "OutputDelay",
                    Value::LogicalArray(LogicalArray::new(vec![0, 0], vec![2, 1]).unwrap()),
                ),
            ]);
            preflight_ss_host_properties(
                &object,
                &["A", "Ts", "InputDelay", "OutputDelay"],
                "pzmap",
            )
            .expect("consumed values admitted");
        }
    }

    #[test]
    fn consumed_ss_integer_properties_require_exact_binary64_values() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let wide = || {
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                    .unwrap(),
            )
        };
        for name in ["A", "B", "C", "D", "Ts", "InputDelay", "OutputDelay"] {
            let object = ss_object([(name, wide())]);
            let error = preflight_ss_host_properties(&object, &[name], "pole")
                .expect_err("inexact consumed model property");
            assert_eq!(error.identifier(), Some("RunMat:pole:InvalidModel"));
        }
        let exact = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_992]), vec![1, 1])
                .unwrap(),
        );
        let object = ss_object([("A", exact)]);
        preflight_ss_host_properties(&object, &["A"], "pole")
            .expect("aligned wide integer is exact");
    }

    #[test]
    fn consumed_ss_integer_class_is_gated_before_resident_download() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let value = [1_u64];
            let resident = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&value),
                    shape: &[1, 1],
                })
                .expect("resident integer state matrix");
            let object = ss_object([("A", Value::GpuTensor(resident)), ("Ts", Value::Num(0.0))]);
            provider.reset_telemetry();
            let error = block_on(super::super::ss_poles_from_object_async(&object, "pole"))
                .expect_err("integer state matrix extension disabled");
            assert_eq!(
                error.identifier(),
                SS_INTEGER_MATRIX_EXTENSION.error_identifier
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn consumed_ss_resident_integer_is_checked_after_one_owning_gather() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let value = [9_007_199_254_740_993_u64];
            let resident = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&value),
                    shape: &[1, 1],
                })
                .expect("resident integer state matrix");
            let object = ss_object([("A", Value::GpuTensor(resident)), ("Ts", Value::Num(0.0))]);
            provider.reset_telemetry();
            let error = block_on(super::super::ss_poles_from_object_async(&object, "pole"))
                .expect_err("inexact resident state matrix");
            assert_eq!(error.identifier(), Some("RunMat:pole:InvalidModel"));
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 8);

            // Exactness is significant-bit based, not an abs(value)<=2^53
            // cutoff. This wider integer must still pass one owning gather.
            let exact = [1_u64 << 54];
            let resident = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&exact),
                    shape: &[1, 1],
                })
                .expect("exact wide resident state matrix");
            let exact_object =
                ss_object([("A", Value::GpuTensor(resident)), ("Ts", Value::Num(0.0))]);
            provider.reset_telemetry();
            let (poles, sample) = block_on(super::super::ss_poles_from_object_async(
                &exact_object,
                "pole",
            ))
            .expect("exact wide resident state matrix is supported");
            assert_eq!(poles.len(), 1);
            assert_eq!(poles[0].re, exact[0] as f64);
            assert_eq!(poles[0].im, 0.0);
            assert_eq!(sample, 0.0);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 8);

            let value = [1_u64];
            let resident = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&value),
                    shape: &[1, 1],
                })
                .expect("resident integer state matrix");
            let object = ss_object([
                ("A", Value::GpuTensor(resident)),
                (
                    "B",
                    Value::Tensor(
                        Tensor::new_integer(
                            IntegerStorage::U64(vec![9_007_199_254_740_993]),
                            vec![1, 1],
                        )
                        .unwrap(),
                    ),
                ),
                ("Ts", Value::Num(0.0)),
            ]);
            provider.reset_telemetry();
            let error = block_on(super::super::gather_ss_property_values(
                &object,
                ["A", "B", "Ts"],
                "pzmap",
            ))
            .expect_err("host B invalid before resident A gather");
            assert_eq!(error.identifier(), Some("RunMat:pzmap:InvalidModel"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }
}
