//! MATLAB-compatible `rgb2hsv` conversion.

use runmat_builtins::{
    BuiltinBindingIdentity, BuiltinCatalogIdentity, BuiltinErrorDescriptor,
    BuiltinPreparationFormId, RgbConversion, RGB2HSV_DESCRIPTOR, RGB2HSV_ERROR_INTERNAL,
    RGB2HSV_ERROR_INVALID_INPUT, RGB2HSV_ERROR_TOO_MANY_INPUTS, RGB2HSV_ERROR_TOO_MANY_OUTPUTS,
};
#[cfg(test)]
use runmat_builtins::{BuiltinIntegerInputAvailability, RGB2HSV_INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Value};

use crate::builtins::image::color::common;
use crate::builtins::image::color::rgb_input;
use crate::builtins::image::color::rgb_presentation::{self, SelectedRgbPresentation};
use crate::context::preparation::{
    execute_request, AdmittedBuiltinPreparation, BuiltinPreparationAdmission,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "rgb2hsv";

fn rgb2hsv_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    rgb2hsv_error_with_message(error.message, error)
}

fn rgb2hsv_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn rgb2hsv_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        build_runtime_error(err.message().to_string())
            .with_builtin(NAME)
            .with_identifier(fallback.identifier.expect("closed RGB conversion error"))
            .with_source(err)
            .build()
    }
}

#[runtime_builtin(
    name = "rgb2hsv",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::image::color::rgb2hsv"
)]
async fn rgb2hsv_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    rgb: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    if !rest.is_empty() {
        return Err(rgb2hsv_error(&RGB2HSV_ERROR_TOO_MANY_INPUTS));
    }
    crate::output_count::validate_fixed_outputs(
        &RGB2HSV_DESCRIPTOR,
        &RGB2HSV_ERROR_TOO_MANY_OUTPUTS,
        NAME,
    )?;
    let admitted = rgb_input::admitted(&rgb, NAME, RgbConversion::Hsv, |err| {
        rgb2hsv_map_error(err, &RGB2HSV_ERROR_INVALID_INPUT)
    })?;
    let resident_sources = match &rgb {
        Value::GpuTensor(handle) => vec![handle.clone()],
        _ => Vec::new(),
    };
    let selected = match (&rgb, &admitted) {
        (Value::GpuTensor(owner), rgb_input::RgbAdmission::PreparedResident(input)) => {
            rgb_presentation::select(runtime.as_ref(), NAME, owner, *input, RgbConversion::Hsv)?
        }
        _ => None,
    };
    if !admitted.prepares() {
        return rgb2hsv_execute(rgb, resident_sources, selected).await;
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
        vec![rgb],
    )?;
    let execute = |mut operands: Vec<Value>| async move {
        rgb2hsv_execute(
            operands.pop().expect("one admitted RGB operand"),
            resident_sources,
            selected,
        )
        .await
    };
    execute_request(runtime.as_ref(), request, execute).await
}

async fn rgb2hsv_execute(
    rgb: Value,
    resident_sources: Vec<runmat_accelerate_api::GpuTensorHandle>,
    mut selected: Option<SelectedRgbPresentation>,
) -> BuiltinResult<crate::execution::RootedValue> {
    let resident_guard = common::protect_resident_inputs(&resident_sources);
    let tensor = match rgb {
        Value::Tensor(tensor) => tensor,
        other => common::gather_tensor(NAME, other)
            .await
            .map_err(|err| rgb2hsv_map_error(err, &RGB2HSV_ERROR_INVALID_INPUT))?,
    };
    resident_guard.restore();
    let layout = rgb_input::host_layout(&tensor, NAME, RgbConversion::Hsv)
        .map_err(|err| rgb2hsv_map_error(err, &RGB2HSV_ERROR_INVALID_INPUT))?;
    let input_dtype = tensor.numeric_dtype();
    let dtype = common::image_output_dtype(input_dtype);
    let values = common::tensor_values_f64(&tensor);
    let mut data = vec![0.0; values.len()];
    for pixel in 0..layout.pixels() {
        let r = common::clamp01(common::unit_value(
            values[layout.index(pixel, 0)],
            input_dtype,
        ));
        let g = common::clamp01(common::unit_value(
            values[layout.index(pixel, 1)],
            input_dtype,
        ));
        let b = common::clamp01(common::unit_value(
            values[layout.index(pixel, 2)],
            input_dtype,
        ));
        let (h, s, v) = rgb_to_hsv_unit(r, g, b);
        data[layout.index(pixel, 0)] = cast_float(h, dtype);
        data[layout.index(pixel, 1)] = cast_float(s, dtype);
        data[layout.index(pixel, 2)] = cast_float(v, dtype);
    }
    let out = common::tensor_with_dtype(data, layout.output_shape(), dtype, NAME)
        .map_err(|err| rgb2hsv_map_error(err, &RGB2HSV_ERROR_INTERNAL))?;
    let output = rgb_presentation::publish(&mut selected, || {
        common::restore_resident_numeric_result_for_sources(
            &resident_sources,
            common::image_value_from_tensor(out),
            NAME,
        )
    })?;
    crate::execution::RootedValue::new(output)
        .map_err(|err| rgb2hsv_map_error(err.into_runtime_error(), &RGB2HSV_ERROR_INTERNAL))
}

pub(crate) fn rgb_to_hsv_unit(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        ((g - b) / delta).rem_euclid(6.0) / 6.0
    } else if max == g {
        ((b - r) / delta + 2.0) / 6.0
    } else {
        ((r - g) / delta + 4.0) / 6.0
    };
    let s = if max == 0.0 { 0.0 } else { delta / max };
    (h, s, max)
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
    async fn rgb2hsv_builtin(
        runtime: Option<crate::context::RuntimeContext>,
        rgb: Value,
        rest: Vec<Value>,
    ) -> BuiltinResult<Value> {
        super::rgb2hsv_builtin(
            runtime,
            rgb,
            crate::builtins::common::test_support::builtin_arguments(rest),
        )
        .await
        .map(|owned| owned.with_owned(|value| value))
    }
    use crate::builtins::common::{gpu_helpers, test_support};
    use futures::executor::block_on;
    use runmat_value::{IntegerStorage, Tensor};

    fn call(tensor: Tensor) -> BuiltinResult<Tensor> {
        let Value::Tensor(out) =
            block_on(rgb2hsv_builtin(None, Value::Tensor(tensor), Vec::new())).expect("rgb2hsv")
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
    fn converts_red_to_hsv() {
        let rgb = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 1, 3]).unwrap();
        let out = call(rgb).unwrap();
        assert_eq!(out.shape(), &[1, 1, 3]);
        let values = values(&out);
        assert_close(values[0], 0.0);
        assert_close(values[1], 1.0);
        assert_close(values[2], 1.0);
    }

    #[test]
    fn converts_colormap_secondary_and_gray_values() {
        let rgb = Tensor::new(
            vec![0.0, 1.0, 1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 1.0, 0.0, 0.5],
            vec![4, 3],
        )
        .unwrap();
        let out = call(rgb).unwrap();
        assert_eq!(out.shape(), &[4, 3]);
        let expected = vec![
            0.5,
            5.0 / 6.0,
            1.0 / 6.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.5,
        ];
        for (actual, expected) in values(&out).iter().zip(expected) {
            assert_close(*actual, expected);
        }
    }

    #[test]
    fn scales_uint8_rgb_before_conversion() {
        let rgb = Tensor::new_with_dtype(vec![128.0, 64.0, 32.0], vec![1, 1, 3], NumericDType::U8)
            .unwrap();
        let out = call(rgb).unwrap();
        assert_eq!(out.numeric_dtype(), NumericDType::F64);
        let values = values(&out);
        assert_close(values[0], 1.0 / 18.0);
        assert_close(values[1], 0.75);
        assert_close(values[2], 128.0 / 255.0);
    }

    #[test]
    fn rgb2hsv_reads_typed_integer_rgb_storage_exactly() {
        let rgb = Tensor::new_integer(IntegerStorage::U8(vec![255, 0, 0]), vec![1, 1, 3]).unwrap();

        let out = call(rgb).unwrap();

        assert_eq!(out.numeric_dtype(), NumericDType::F64);
        let values = values(&out);
        assert_close(values[0], 0.0);
        assert_close(values[1], 1.0);
        assert_close(values[2], 1.0);
    }

    #[test]
    fn rgb2hsv_preserves_single_and_accepts_uint16() {
        let single = Tensor::from_f32(vec![1.0, 0.0, 0.0], vec![1, 1, 3]).unwrap();
        let single_out = call(single).expect("single RGB");
        assert_eq!(single_out.numeric_dtype(), NumericDType::F32);
        assert_eq!(values(&single_out), vec![0.0, 1.0, 1.0]);

        let uint16 =
            Tensor::new_integer(IntegerStorage::U16(vec![65535, 0, 0]), vec![1, 1, 3]).unwrap();
        let uint16_out = call(uint16).expect("uint16 RGB");
        assert_eq!(uint16_out.numeric_dtype(), NumericDType::F64);
        assert_eq!(values(&uint16_out), vec![0.0, 1.0, 1.0]);
    }

    #[test]
    fn rgb2hsv_rejects_unsupported_integer_classes() {
        for storage in [
            IntegerStorage::I8(vec![0; 3]),
            IntegerStorage::I16(vec![0; 3]),
            IntegerStorage::I32(vec![0; 3]),
            IntegerStorage::I64(vec![0; 3]),
            IntegerStorage::U32(vec![0; 3]),
            IntegerStorage::U64(vec![0; 3]),
        ] {
            let input = Tensor::new_integer(storage, vec![1, 1, 3]).unwrap();
            let err = block_on(rgb2hsv_builtin(None, Value::Tensor(input), Vec::new()))
                .expect_err("unsupported integer class");
            assert_eq!(err.identifier(), RGB2HSV_ERROR_INVALID_INPUT.identifier);
        }
    }

    #[test]
    fn rejects_grayscale_shape() {
        let gray = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = block_on(rgb2hsv_builtin(None, Value::Tensor(gray), Vec::new())).unwrap_err();
        assert!(err
            .message()
            .contains("expected an MxNx3 RGB image or an Nx3 colormap"));
    }

    #[test]
    fn rgb2hsv_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = RGB2HSV_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(labels, vec!["HSV = rgb2hsv(RGB)"]);
    }

    #[test]
    fn rgb2hsv_integer_capabilities_record_supported_and_rejected_classes() {
        use runmat_builtins::BuiltinIntegerClass;
        assert_eq!(RGB2HSV_INTEGER_CAPABILITIES.len(), 2);
        assert_eq!(
            RGB2HSV_INTEGER_CAPABILITIES[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::Documented
        );
        assert_eq!(
            RGB2HSV_INTEGER_CAPABILITIES[1].inputs[0].classes,
            &[
                BuiltinIntegerClass::Int8,
                BuiltinIntegerClass::Int16,
                BuiltinIntegerClass::Int32,
                BuiltinIntegerClass::Int64,
                BuiltinIntegerClass::Uint32,
                BuiltinIntegerClass::Uint64,
            ]
        );
    }

    #[test]
    fn rgb2hsv_gpu_fallback_restores_double_output_and_preserves_source() {
        test_support::with_test_provider(|provider| {
            let rgb =
                Tensor::new_integer(IntegerStorage::U8(vec![255, 0, 0]), vec![1, 1, 3]).unwrap();
            let source = gpu_helpers::upload_tensor(provider, &rgb).expect("upload rgb");
            let source = source
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("publish provenance view");
            let result = block_on(rgb2hsv_builtin(
                None,
                Value::GpuTensor(source.clone()),
                Vec::new(),
            ))
            .expect("rgb2hsv");
            let Value::GpuTensor(output) = result else {
                panic!("expected restored gpu output");
            };
            assert_eq!(runmat_accelerate_api::handle_integer_type(&output), None);
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(runmat_accelerate_api::ProviderPrecision::F64)
            );
            assert_eq!(
                test_support::gather(Value::GpuTensor(output))
                    .unwrap()
                    .materialize_f64(),
                vec![0.0, 1.0, 1.0]
            );
            assert_eq!(
                test_support::gather(Value::GpuTensor(source))
                    .unwrap()
                    .integer_storage(),
                Some(&IntegerStorage::U8(vec![255, 0, 0]))
            );
        });
    }

    #[test]
    fn rgb2hsv_rejects_unsupported_resident_class_before_download() {
        let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1, 3],
                runmat_accelerate_api::NumericElementType::U32,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("admit owned fixture")
            .handle;
        let err = block_on(rgb2hsv_builtin(
            None,
            Value::GpuTensor(handle.clone()),
            Vec::new(),
        ))
        .expect_err("unsupported resident integer");
        assert_eq!(err.identifier(), RGB2HSV_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn rgb2hsv_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = RGB2HSV_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.RGB2HSV.TOO_MANY_INPUTS"));
        assert!(codes.contains(&"RM.RGB2HSV.INVALID_INPUT"));
        assert!(codes.contains(&"RM.RGB2HSV.INTERNAL"));
    }

    #[test]
    fn rgb2hsv_too_many_args_uses_stable_identifier() {
        let err = block_on(rgb2hsv_builtin(
            None,
            Value::Num(1.0),
            vec![Value::Num(2.0)],
        ))
        .expect_err("expected argument error");
        assert_eq!(err.identifier(), RGB2HSV_ERROR_TOO_MANY_INPUTS.identifier);
    }
}
