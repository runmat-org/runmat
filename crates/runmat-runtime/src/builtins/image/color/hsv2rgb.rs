//! MATLAB-compatible `hsv2rgb` conversion.

use runmat_builtins::{
    BuiltinBindingIdentity, BuiltinCatalogIdentity, BuiltinErrorDescriptor,
    BuiltinPreparationFormId, HSV2RGB_DESCRIPTOR, HSV2RGB_ERROR_INTERNAL,
    HSV2RGB_ERROR_INVALID_INPUT, HSV2RGB_ERROR_TOO_MANY_INPUTS, HSV2RGB_ERROR_TOO_MANY_OUTPUTS,
};
#[cfg(test)]
use runmat_builtins::{BuiltinIntegerInputAvailability, HSV2RGB_INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Value};

use crate::builtins::image::color::color_presentation::{
    self, ColorHostRoute, ColorHostUpload, SelectedColorPresentation,
};
use crate::builtins::image::color::common;
use crate::context::preparation::{
    execute_request, AdmittedBuiltinPreparation, BuiltinPreparationAdmission,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "hsv2rgb";

fn hsv2rgb_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    hsv2rgb_error_with_message(error.message, error)
}

fn hsv2rgb_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn hsv2rgb_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        build_runtime_error(err.message().to_string())
            .with_builtin(NAME)
            .with_identifier(fallback.identifier.expect("closed HSV conversion error"))
            .with_source(err)
            .build()
    }
}

#[runtime_builtin(
    name = "hsv2rgb",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::image::color::hsv2rgb"
)]
async fn hsv2rgb_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    hsv: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    if !rest.is_empty() {
        return Err(hsv2rgb_error(&HSV2RGB_ERROR_TOO_MANY_INPUTS));
    }
    crate::output_count::validate_fixed_outputs(
        &HSV2RGB_DESCRIPTOR,
        &HSV2RGB_ERROR_TOO_MANY_OUTPUTS,
        NAME,
    )?;
    let admitted = admit_hsv(&hsv)?;
    let resident_sources = match &hsv {
        Value::GpuTensor(handle) => vec![handle.clone()],
        _ => Vec::new(),
    };
    let selected = match &hsv {
        Value::GpuTensor(handle) => {
            let output_dtype = if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
                && !runmat_accelerate_api::handle_is_logical(handle)
            {
                NumericDType::F32
            } else {
                NumericDType::F64
            };
            color_presentation::select(
                runtime.as_ref(),
                ColorHostUpload {
                    route: ColorHostRoute::HsvToRgb,
                    sources: &resident_sources,
                    output_shape: handle.shape().to_vec(),
                    output_dtype,
                    host_compute_scratch: color_presentation::checked_scratch(
                        color_presentation::checked_elements(handle.shape()),
                        16,
                    ),
                    control_facts: runmat_execution::Digest::sha256(b"hsv2rgb.default"),
                },
            )?
        }
        _ => None,
    };
    if !admitted {
        return hsv2rgb_execute(hsv, resident_sources, selected).await;
    }
    let request = AdmittedBuiltinPreparation::new(
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: BuiltinCatalogIdentity { name: NAME },
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature: 0,
            input_count: 1,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![hsv],
    )?;
    let execute = |mut operands: Vec<Value>| async move {
        hsv2rgb_execute(
            operands.pop().expect("one admitted HSV operand"),
            resident_sources,
            selected,
        )
        .await
    };
    execute_request(runtime.as_ref(), request, execute).await
}

fn admit_hsv(hsv: &Value) -> BuiltinResult<bool> {
    match hsv {
        Value::Tensor(tensor) => {
            common::color_layout(tensor, NAME)
                .map_err(|err| hsv2rgb_map_error(err, &HSV2RGB_ERROR_INVALID_INPUT))?;
            if !matches!(
                tensor.numeric_dtype(),
                NumericDType::F32 | NumericDType::F64
            ) {
                return Err(hsv2rgb_error_with_message(
                    format!(
                        "hsv2rgb: {} input is not supported; expected single, double, or logical",
                        tensor.numeric_dtype().class_name()
                    ),
                    &HSV2RGB_ERROR_INVALID_INPUT,
                ));
            }
            Ok(true)
        }
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                return Err(hsv2rgb_error_with_message(
                    "hsv2rgb: integer gpuArray input is not supported; expected single, double, or logical",
                    &HSV2RGB_ERROR_INVALID_INPUT,
                ));
            }
            if runmat_accelerate_api::handle_is_logical(handle) {
                super::resident_owner::validate_live(handle, NAME)
                    .map_err(|err| hsv2rgb_map_error(err, &HSV2RGB_ERROR_INVALID_INPUT))?;
                let shape = handle.shape();
                if shape.len() != 3 || shape[2] != 3 || shape[0] == 0 || shape[1] == 0 {
                    return Err(hsv2rgb_error_with_message(
                        "hsv2rgb: logical input is supported only for nonempty MxNx3 HSV images",
                        &HSV2RGB_ERROR_INVALID_INPUT,
                    ));
                }
                return Ok(false);
            }
            let dtype = common::resident_numeric_dtype(handle, NAME)
                .map_err(|err| hsv2rgb_map_error(err, &HSV2RGB_ERROR_INVALID_INPUT))?;
            let shape = handle.shape();
            let valid = (shape.len() == 3 && shape[2] == 3 && shape[0] > 0 && shape[1] > 0)
                || (shape.len() == 2 && shape[1] == 3 && shape[0] > 0);
            if !valid || !matches!(dtype, NumericDType::F32 | NumericDType::F64) {
                return Err(hsv2rgb_error_with_message(
                    "hsv2rgb: invalid resident HSV class or shape",
                    &HSV2RGB_ERROR_INVALID_INPUT,
                ));
            }
            Ok(true)
        }
        Value::LogicalArray(array) => {
            let shape = &array.shape;
            if shape.len() != 3 || shape[2] != 3 || shape[0] == 0 || shape[1] == 0 {
                return Err(hsv2rgb_error_with_message(
                    "hsv2rgb: logical input is supported only for nonempty MxNx3 HSV images",
                    &HSV2RGB_ERROR_INVALID_INPUT,
                ));
            }
            Ok(false)
        }
        Value::Bool(_) => Ok(false),
        _ => Ok(false),
    }
}

async fn hsv2rgb_execute(
    hsv: Value,
    resident_sources: Vec<runmat_accelerate_api::GpuTensorHandle>,
    mut selected: Option<SelectedColorPresentation>,
) -> BuiltinResult<crate::execution::RootedValue> {
    let resident_guard = common::protect_resident_inputs(&resident_sources);
    if matches!(&hsv, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
    {
        return Err(hsv2rgb_error_with_message(
            "hsv2rgb: integer gpuArray input is not supported; expected single, double, or logical",
            &HSV2RGB_ERROR_INVALID_INPUT,
        ));
    }
    let logical_input = match &hsv {
        Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_logical(handle),
        _ => false,
    };
    let tensor = match hsv {
        Value::Tensor(tensor) => tensor,
        other => common::gather_tensor(NAME, other)
            .await
            .map_err(|err| hsv2rgb_map_error(err, &HSV2RGB_ERROR_INVALID_INPUT))?,
    };
    resident_guard.restore();
    let layout = common::color_layout(&tensor, NAME)
        .map_err(|err| hsv2rgb_map_error(err, &HSV2RGB_ERROR_INVALID_INPUT))?;
    if logical_input && matches!(layout, common::ColorLayout::Colormap { .. }) {
        return Err(hsv2rgb_error_with_message(
            "hsv2rgb: logical input is supported only for MxNx3 HSV images",
            &HSV2RGB_ERROR_INVALID_INPUT,
        ));
    }
    let input_dtype = tensor.numeric_dtype();
    if !logical_input && !matches!(input_dtype, NumericDType::F32 | NumericDType::F64) {
        return Err(hsv2rgb_error_with_message(
            format!(
                "hsv2rgb: {} input is not supported; expected single, double, or logical",
                input_dtype.class_name()
            ),
            &HSV2RGB_ERROR_INVALID_INPUT,
        ));
    }
    let dtype = if input_dtype == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    let values = common::tensor_values_f64(&tensor);
    let mut data = vec![0.0; values.len()];
    for pixel in 0..layout.pixels() {
        let h = values[layout.index(pixel, 0)].rem_euclid(1.0);
        let s = common::clamp01(common::unit_value(
            values[layout.index(pixel, 1)],
            input_dtype,
        ));
        let v = common::clamp01(common::unit_value(
            values[layout.index(pixel, 2)],
            input_dtype,
        ));
        let (r, g, b) = hsv_to_rgb_unit(h, s, v);
        data[layout.index(pixel, 0)] = cast_float(r, dtype);
        data[layout.index(pixel, 1)] = cast_float(g, dtype);
        data[layout.index(pixel, 2)] = cast_float(b, dtype);
    }
    let out = common::tensor_with_dtype(data, layout.output_shape(), dtype, NAME)
        .map_err(|err| hsv2rgb_map_error(err, &HSV2RGB_ERROR_INTERNAL))?;
    let output = color_presentation::publish(&mut selected, || {
        common::restore_resident_numeric_result_for_sources(
            &resident_sources,
            common::image_value_from_tensor(out),
            NAME,
        )
    })?;
    crate::execution::RootedValue::new(output)
        .map_err(|err| hsv2rgb_map_error(err.into_runtime_error(), &HSV2RGB_ERROR_INTERNAL))
}

pub(crate) fn hsv_to_rgb_unit(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    if s == 0.0 {
        return (v, v, v);
    }
    let h6 = h * 6.0;
    let i = h6.floor();
    let f = h6 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match (i as i32).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

fn cast_float(value: f64, dtype: NumericDType) -> f64 {
    if matches!(dtype, NumericDType::F32) {
        (value as f32) as f64
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Terminal numerical test projection; these color outputs have no
    // GC-managed children and do not escape into asynchronous work.
    async fn hsv2rgb_builtin(
        runtime: Option<crate::context::RuntimeContext>,
        hsv: Value,
        rest: Vec<Value>,
    ) -> BuiltinResult<Value> {
        super::hsv2rgb_builtin(
            runtime,
            hsv,
            crate::builtins::common::test_support::builtin_arguments(rest),
        )
        .await
        .map(|owned| owned.with_owned(|value| value))
    }
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{IntegerStorage, LogicalArray, Tensor};

    fn call(tensor: Tensor) -> BuiltinResult<Tensor> {
        let Value::Tensor(out) =
            block_on(hsv2rgb_builtin(None, Value::Tensor(tensor), Vec::new())).expect("hsv2rgb")
        else {
            panic!("expected tensor");
        };
        Ok(out)
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    fn values(tensor: &Tensor) -> Vec<f64> {
        tensor.materialize_f64()
    }

    #[test]
    fn converts_red_hsv_to_rgb() {
        let hsv = Tensor::new(vec![0.0, 1.0, 1.0], vec![1, 1, 3]).unwrap();
        let out = call(hsv).unwrap();
        let values = values(&out);
        assert_close(values[0], 1.0);
        assert_close(values[1], 0.0);
        assert_close(values[2], 0.0);
    }

    #[test]
    fn converts_colormap_with_wrapped_hue_and_gray() {
        let hsv = Tensor::new(
            vec![1.0 / 3.0, -1.0 / 6.0, 0.2, 1.0, 1.0, 0.0, 1.0, 1.0, 0.25],
            vec![3, 3],
        )
        .unwrap();
        let out = call(hsv).unwrap();
        assert_eq!(out.shape(), &[3, 3]);
        let expected = vec![0.0, 1.0, 0.25, 1.0, 0.0, 0.25, 0.0, 1.0, 0.25];
        for (actual, expected) in values(&out).iter().zip(expected) {
            assert_close(*actual, expected);
        }
    }

    #[test]
    fn clamps_saturation_and_value() {
        let hsv = Tensor::new(vec![0.0, 2.0, 1.5], vec![1, 1, 3]).unwrap();
        let out = call(hsv).unwrap();
        let values = values(&out);
        assert_close(values[0], 1.0);
        assert_close(values[1], 0.0);
        assert_close(values[2], 0.0);
    }

    #[test]
    fn hsv2rgb_preserves_single_and_accepts_logical_images() {
        let single = Tensor::from_f32(vec![0.0, 1.0, 1.0], vec![1, 1, 3]).unwrap();
        let single_out = call(single).expect("single HSV");
        assert_eq!(single_out.numeric_dtype(), NumericDType::F32);
        assert_eq!(values(&single_out), vec![1.0, 0.0, 0.0]);

        let logical = LogicalArray::new(vec![0, 1, 1], vec![1, 1, 3]).unwrap();
        let Value::Tensor(logical_out) = block_on(hsv2rgb_builtin(
            None,
            Value::LogicalArray(logical),
            Vec::new(),
        ))
        .expect("logical HSV") else {
            panic!("expected logical HSV tensor output");
        };
        assert_eq!(logical_out.numeric_dtype(), NumericDType::F64);
        assert_eq!(values(&logical_out), vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn hsv2rgb_rejects_every_integer_class() {
        for storage in [
            IntegerStorage::I8(vec![0; 3]),
            IntegerStorage::I16(vec![0; 3]),
            IntegerStorage::I32(vec![0; 3]),
            IntegerStorage::I64(vec![0; 3]),
            IntegerStorage::U8(vec![0; 3]),
            IntegerStorage::U16(vec![0; 3]),
            IntegerStorage::U32(vec![0; 3]),
            IntegerStorage::U64(vec![0; 3]),
        ] {
            let input = Tensor::new_integer(storage, vec![1, 1, 3]).unwrap();
            let err = block_on(hsv2rgb_builtin(None, Value::Tensor(input), Vec::new()))
                .expect_err("integer HSV input");
            assert_eq!(err.identifier(), HSV2RGB_ERROR_INVALID_INPUT.identifier);
        }
    }

    #[test]
    fn hsv2rgb_rejects_logical_colormap_shape() {
        let logical = LogicalArray::new(vec![0, 1, 1], vec![1, 3]).unwrap();
        let err = block_on(hsv2rgb_builtin(
            None,
            Value::LogicalArray(logical),
            Vec::new(),
        ))
        .expect_err("logical colormap input");
        assert_eq!(err.identifier(), HSV2RGB_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn hsv2rgb_rejects_unsupported_resident_class_before_download() {
        let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1, 3],
                runmat_accelerate_api::NumericElementType::U32,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("admit owned fixture")
            .handle;
        let err = block_on(hsv2rgb_builtin(None, Value::GpuTensor(handle), Vec::new()))
            .expect_err("unsupported resident integer");
        assert_eq!(err.identifier(), HSV2RGB_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn hsv2rgb_rejects_logical_resident_colormap_before_download() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &[0, 1, 1],
                    shape: &[1, 3],
                })
                .expect("upload logical fixture");
            provider.reset_telemetry();
            let err = block_on(hsv2rgb_builtin(
                None,
                Value::GpuTensor(handle.clone()),
                Vec::new(),
            ))
            .expect_err("logical colormap geometry");
            assert_eq!(err.identifier(), HSV2RGB_ERROR_INVALID_INPUT.identifier);
            assert!(err.message().contains("nonempty MxNx3"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            assert!(handle.allocation_lease().validate_live().is_ok());
        });
    }

    #[test]
    fn hsv2rgb_integer_capability_records_all_classes_as_rejected() {
        assert_eq!(HSV2RGB_INTEGER_CAPABILITIES.len(), 1);
        let input = &HSV2RGB_INTEGER_CAPABILITIES[0].inputs[0];
        assert_eq!(
            input.availability,
            BuiltinIntegerInputAvailability::Rejected
        );
        assert_eq!(input.classes.len(), 8);
    }

    #[test]
    fn hsv2rgb_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = HSV2RGB_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(labels, vec!["RGB = hsv2rgb(HSV)"]);
    }

    #[test]
    fn hsv2rgb_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = HSV2RGB_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.HSV2RGB.TOO_MANY_INPUTS"));
        assert!(codes.contains(&"RM.HSV2RGB.TOO_MANY_OUTPUTS"));
        assert!(codes.contains(&"RM.HSV2RGB.INVALID_INPUT"));
        assert!(codes.contains(&"RM.HSV2RGB.INTERNAL"));
    }

    #[test]
    fn hsv2rgb_too_many_args_uses_stable_identifier() {
        let err = block_on(hsv2rgb_builtin(
            None,
            Value::Num(1.0),
            vec![Value::Num(2.0)],
        ))
        .expect_err("expected argument error");
        assert_eq!(err.identifier(), HSV2RGB_ERROR_TOO_MANY_INPUTS.identifier);
    }
}
