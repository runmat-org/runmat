//! MATLAB-compatible `rgb2gray` conversion.

use runmat_builtins::{
    BuiltinBindingIdentity, BuiltinCatalogIdentity, BuiltinErrorDescriptor,
    BuiltinPreparationFormId, RgbConversion, RGB2GRAY_DESCRIPTOR, RGB2GRAY_ERROR_INTERNAL,
    RGB2GRAY_ERROR_INVALID_INPUT, RGB2GRAY_ERROR_TOO_MANY_INPUTS, RGB2GRAY_ERROR_TOO_MANY_OUTPUTS,
};
#[cfg(test)]
use runmat_builtins::{
    BuiltinIntegerInputAvailability, BuiltinIntegerOverflowRule, RGB2GRAY_INTEGER_CAPABILITIES,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, NumericScalar, NumericStorage, Tensor, Value};

use crate::builtins::image::color::common;
use crate::builtins::image::color::rgb_input;
use crate::builtins::image::color::rgb_presentation::{self, SelectedRgbPresentation};
use crate::context::preparation::{
    execute_request, AdmittedBuiltinPreparation, BuiltinPreparationAdmission,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "rgb2gray";

fn rgb2gray_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    rgb2gray_error_with_message(error.message, error)
}

fn rgb2gray_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn rgb2gray_map_error(
    err: RuntimeError,
    fallback: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
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
    name = "rgb2gray",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::image::color::rgb2gray"
)]
async fn rgb2gray_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    rgb: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    if !rest.is_empty() {
        return Err(rgb2gray_error(&RGB2GRAY_ERROR_TOO_MANY_INPUTS));
    }
    crate::output_count::validate_fixed_outputs(
        &RGB2GRAY_DESCRIPTOR,
        &RGB2GRAY_ERROR_TOO_MANY_OUTPUTS,
        NAME,
    )?;
    let admitted = rgb_input::admitted(&rgb, NAME, RgbConversion::Gray, |err| {
        rgb2gray_map_error(err, &RGB2GRAY_ERROR_INVALID_INPUT)
    })?;
    let resident_sources = match &rgb {
        Value::GpuTensor(handle) => vec![handle.clone()],
        _ => Vec::new(),
    };
    let selected = match (&rgb, &admitted) {
        (Value::GpuTensor(owner), rgb_input::RgbAdmission::PreparedResident(input)) => {
            rgb_presentation::select(runtime.as_ref(), NAME, owner, *input, RgbConversion::Gray)?
        }
        _ => None,
    };
    if !admitted.prepares() {
        return rgb2gray_execute(rgb, resident_sources, selected).await;
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
        rgb2gray_execute(
            operands.pop().expect("one admitted RGB operand"),
            resident_sources,
            selected,
        )
        .await
    };
    execute_request(runtime.as_ref(), request, execute).await
}

async fn rgb2gray_execute(
    rgb: Value,
    resident_sources: Vec<runmat_accelerate_api::GpuTensorHandle>,
    mut selected: Option<SelectedRgbPresentation>,
) -> BuiltinResult<crate::execution::RootedValue> {
    let resident_guard = common::protect_resident_inputs(&resident_sources);
    let tensor = match rgb {
        Value::Tensor(tensor) => tensor,
        other => common::gather_tensor(NAME, other)
            .await
            .map_err(|err| rgb2gray_map_error(err, &RGB2GRAY_ERROR_INVALID_INPUT))?,
    };
    resident_guard.restore();
    let out = rgb2gray_tensor(&tensor)
        .map_err(|err| rgb2gray_map_error(err, &RGB2GRAY_ERROR_INTERNAL))?;
    let output = rgb_presentation::publish(&mut selected, || {
        common::restore_resident_numeric_result_for_sources(
            &resident_sources,
            common::image_value_from_tensor(out),
            NAME,
        )
    })?;
    crate::execution::RootedValue::new(output)
        .map_err(|err| rgb2gray_map_error(err.into_runtime_error(), &RGB2GRAY_ERROR_INTERNAL))
}

fn rgb2gray_tensor(rgb: &Tensor) -> BuiltinResult<Tensor> {
    const RED: f64 = 0.298_936_021_293_775;
    const GREEN: f64 = 0.587_043_074_451_121;
    const BLUE: f64 = 0.114_020_904_255_103;

    let dtype = rgb.numeric_dtype();
    let layout = rgb_input::host_layout(rgb, NAME, RgbConversion::Gray)?;

    let pixels = layout.pixels();
    if rgb.len() != pixels * 3 {
        return Err(rgb2gray_error_with_message(
            "rgb2gray: image data length does not match shape",
            &RGB2GRAY_ERROR_INVALID_INPUT,
        ));
    }
    let mut grayscale = Vec::with_capacity(pixels);
    for pixel in 0..pixels {
        let r = rgb2gray_value(rgb, layout.index(pixel, 0))?;
        let g = rgb2gray_value(rgb, layout.index(pixel, 1))?;
        let b = rgb2gray_value(rgb, layout.index(pixel, 2))?;
        grayscale.push(RED * r + GREEN * g + BLUE * b);
    }

    let (storage, shape) = match layout {
        common::ColorLayout::Truecolor { rows, cols } => {
            (grayscale_storage(grayscale, dtype), vec![rows, cols])
        }
        common::ColorLayout::Colormap { rows } => {
            let mut repeated = vec![0.0; rows * 3];
            for channel in 0..3 {
                repeated[channel * rows..(channel + 1) * rows].copy_from_slice(&grayscale);
            }
            (NumericStorage::F64(repeated), vec![rows, 3])
        }
    };
    Tensor::from_numeric_storage(storage, shape)
        .map_err(|err| rgb2gray_error_with_message(err, &RGB2GRAY_ERROR_INTERNAL))
}

fn rgb2gray_value(rgb: &Tensor, index: usize) -> BuiltinResult<f64> {
    match rgb.numeric_value_at(index) {
        Some(NumericScalar::F64(value)) => Ok(value),
        Some(NumericScalar::F32(value)) => Ok(f64::from(value)),
        Some(NumericScalar::U8(value)) => Ok(f64::from(value)),
        Some(NumericScalar::U16(value)) => Ok(f64::from(value)),
        Some(value) => Err(rgb2gray_error_with_message(
            format!(
                "rgb2gray: unsupported numeric sample class {}",
                value.class_name()
            ),
            &RGB2GRAY_ERROR_INVALID_INPUT,
        )),
        None => Err(rgb2gray_error_with_message(
            format!(
                "rgb2gray: {} storage is unavailable at element {index}",
                rgb.numeric_dtype().class_name()
            ),
            &RGB2GRAY_ERROR_INVALID_INPUT,
        )),
    }
}

fn grayscale_storage(values: Vec<f64>, dtype: NumericDType) -> NumericStorage {
    match dtype {
        NumericDType::F64 => NumericStorage::F64(values),
        NumericDType::F32 => {
            NumericStorage::F32(values.into_iter().map(|value| value as f32).collect())
        }
        NumericDType::U8 => NumericStorage::U8(
            values
                .into_iter()
                .map(|value| value.round().clamp(0.0, f64::from(u8::MAX)) as u8)
                .collect(),
        ),
        NumericDType::U16 => NumericStorage::U16(
            values
                .into_iter()
                .map(|value| value.round().clamp(0.0, f64::from(u16::MAX)) as u16)
                .collect(),
        ),
        NumericDType::I8
        | NumericDType::I16
        | NumericDType::I32
        | NumericDType::I64
        | NumericDType::U32
        | NumericDType::U64 => unreachable!("unsupported rgb2gray class rejected before output"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Terminal numerical test projection; these color outputs have no
    // GC-managed children and do not escape into asynchronous work.
    async fn rgb2gray_builtin(
        runtime: Option<crate::context::RuntimeContext>,
        rgb: Value,
        rest: Vec<Value>,
    ) -> BuiltinResult<Value> {
        super::rgb2gray_builtin(
            runtime,
            rgb,
            crate::builtins::common::test_support::builtin_arguments(rest),
        )
        .await
        .map(|owned| owned.with_owned(|value| value))
    }
    use crate::builtins::common::{gpu_helpers, test_support};
    use futures::executor::block_on;
    use runmat_value::IntegerStorage;

    fn call(value: Value) -> Value {
        block_on(rgb2gray_builtin(None, value, Vec::new())).expect("rgb2gray")
    }

    fn typed_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
        Tensor::new_integer(storage, shape).expect("integer tensor")
    }

    #[test]
    fn converts_uint8_rgb_to_grayscale_uint8() {
        let rgb =
            Tensor::new_with_dtype(vec![255.0, 0.0, 0.0], vec![1, 1, 3], NumericDType::U8).unwrap();
        let Value::Int(value) = call(Value::Tensor(rgb)) else {
            panic!("expected scalar int");
        };
        assert_eq!(value.to_i64(), 76);
    }

    #[test]
    fn rgb2gray_reads_typed_integer_rgb_storage_exactly() {
        let rgb = typed_tensor(IntegerStorage::U8(vec![255, 0, 0]), vec![1, 1, 3]);

        let Value::Int(value) = call(Value::Tensor(rgb)) else {
            panic!("expected scalar int");
        };

        assert_eq!(value.to_i64(), 76);
    }

    #[test]
    fn preserves_native_single_output_storage() {
        let rgb = Tensor::from_f32(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![1, 2, 3]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(rgb)) else {
            panic!("expected single tensor");
        };
        assert_eq!(
            out.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![
                0.298_936_021_293_775_f64 as f32,
                0.587_043_074_451_121_f64 as f32,
            ])
        );
    }

    #[test]
    fn one_pixel_single_grayscale_publishes_double_scalar() {
        let rgb = Tensor::from_f32(vec![1.0, 0.0, 0.0], vec![1, 1, 3]).unwrap();
        let Value::Num(value) = call(Value::Tensor(rgb)) else {
            panic!("one-pixel single result is a double scalar");
        };
        assert_eq!(value, 0.298_936_021_293_775_f64 as f32 as f64);
    }

    #[test]
    fn preserves_2d_shape() {
        let rgb = Tensor::new(vec![1.0; 12], vec![2, 2, 3]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(rgb)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape(), &[2, 2]);
        assert!(out
            .as_f64_slice()
            .expect("double output")
            .iter()
            .all(|v| (*v - 1.0).abs() < 1e-12));
    }

    #[test]
    fn converts_column_major_uint16_planes() {
        let rgb = Tensor::new_with_dtype(
            vec![65535.0, 0.0, 0.0, 65535.0, 0.0, 0.0],
            vec![2, 1, 3],
            NumericDType::U16,
        )
        .unwrap();
        let Value::Tensor(out) = call(Value::Tensor(rgb)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.numeric_dtype(), NumericDType::U16);
        assert_eq!(out.shape(), &[2, 1]);
        assert_eq!(
            out.integer_storage(),
            Some(&IntegerStorage::U16(vec![19591, 38472]))
        );
    }

    #[test]
    fn converts_double_colormap_to_three_equal_gray_columns() {
        let map = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(map)) else {
            panic!("expected colormap tensor");
        };
        assert_eq!(out.shape(), &[2, 3]);
        let values = out.as_f64_slice().expect("double colormap");
        for channel in 0..3 {
            assert!((values[channel * 2] - 0.298_936_021_293_775).abs() < 1e-15);
            assert!((values[channel * 2 + 1] - 0.587_043_074_451_121).abs() < 1e-15);
        }
    }

    #[test]
    fn rejects_int16_truecolor_images() {
        let rgb =
            Tensor::new_with_dtype(vec![0.0, 0.0, 0.0], vec![1, 1, 3], NumericDType::I16).unwrap();
        let err = block_on(rgb2gray_builtin(None, Value::Tensor(rgb), Vec::new()))
            .expect_err("int16 truecolor is not a supported MATLAB rgb2gray input class");
        assert_eq!(err.identifier(), RGB2GRAY_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("int16"));
    }

    #[test]
    fn rejects_non_rgb_matrix_shape() {
        let map = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let err = block_on(rgb2gray_builtin(None, Value::Tensor(map), Vec::new())).unwrap_err();
        assert!(err
            .message()
            .contains("expected an MxNx3 RGB image or an Nx3 colormap"));
    }

    #[test]
    fn rgb2gray_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = RGB2GRAY_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(labels, vec!["I = rgb2gray(RGB)"]);
    }

    #[test]
    fn rgb2gray_integer_capabilities_record_supported_and_rejected_classes() {
        use runmat_builtins::BuiltinIntegerClass;
        assert_eq!(RGB2GRAY_INTEGER_CAPABILITIES.len(), 2);
        assert_eq!(
            RGB2GRAY_INTEGER_CAPABILITIES[0].inputs[0].classes,
            &[BuiltinIntegerClass::Uint8, BuiltinIntegerClass::Uint16]
        );
        assert_eq!(
            RGB2GRAY_INTEGER_CAPABILITIES[1].inputs[0].availability,
            BuiltinIntegerInputAvailability::Rejected
        );
        assert_eq!(
            RGB2GRAY_INTEGER_CAPABILITIES[0].overflow,
            BuiltinIntegerOverflowRule::Saturate
        );
    }

    #[test]
    fn rgb2gray_gpu_fallback_restores_exact_integer_class_and_preserves_source() {
        test_support::with_test_provider(|provider| {
            let rgb =
                Tensor::new_integer(IntegerStorage::U8(vec![255, 0, 0]), vec![1, 1, 3]).unwrap();
            let source = gpu_helpers::upload_tensor(provider, &rgb).expect("upload rgb");
            let source = source
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("publish provenance view");
            let result = block_on(rgb2gray_builtin(
                None,
                Value::GpuTensor(source.clone()),
                Vec::new(),
            ))
            .expect("rgb2gray");
            let Value::GpuTensor(output) = result else {
                panic!("expected restored gpu output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&output),
                Some(runmat_accelerate_api::IntegerElementType::U8)
            );
            assert_eq!(
                test_support::gather(Value::GpuTensor(output))
                    .unwrap()
                    .integer_storage(),
                Some(&IntegerStorage::U8(vec![76]))
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
    fn rgb2gray_rejects_unsupported_resident_class_before_download() {
        let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1, 3],
                runmat_accelerate_api::NumericElementType::I16,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("admit owned fixture")
            .handle;
        let err = block_on(rgb2gray_builtin(
            None,
            Value::GpuTensor(handle.clone()),
            Vec::new(),
        ))
        .expect_err("unsupported resident integer");
        assert_eq!(err.identifier(), RGB2GRAY_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn rgb2gray_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = RGB2GRAY_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.RGB2GRAY.TOO_MANY_INPUTS"));
        assert!(codes.contains(&"RM.RGB2GRAY.INVALID_INPUT"));
        assert!(codes.contains(&"RM.RGB2GRAY.INTERNAL"));
    }

    #[test]
    fn rgb2gray_too_many_args_uses_stable_identifier() {
        let err = block_on(rgb2gray_builtin(
            None,
            Value::Num(1.0),
            vec![Value::Num(2.0)],
        ))
        .expect_err("expected argument error");
        assert_eq!(err.identifier(), RGB2GRAY_ERROR_TOO_MANY_INPUTS.identifier);
    }

    #[test]
    fn rgb2gray_is_registered_with_dispatcher() {
        let rgb = Tensor::new(vec![1.0, 1.0, 1.0], vec![1, 1, 3]).unwrap();
        let result = block_on(crate::call_builtin_async(NAME, &[Value::Tensor(rgb)]))
            .expect("rgb2gray registered");
        assert!(matches!(result.value(), Value::Num(value) if (value - 1.0).abs() < 1e-12));
    }
}
