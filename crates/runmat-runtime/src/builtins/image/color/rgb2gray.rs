//! MATLAB-compatible `rgb2gray` conversion.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, NumericDType, NumericScalar, NumericStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::image::color::common;
use crate::builtins::image::color::type_resolvers::rgb2gray_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "rgb2gray";

const RGB2GRAY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "I",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Grayscale image converted from RGB input.",
}];

const RGB2GRAY_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "RGB",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RGB truecolor image or double colormap.",
}];

const RGB2GRAY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "I = rgb2gray(RGB)",
    inputs: &RGB2GRAY_INPUTS,
    outputs: &RGB2GRAY_OUTPUT,
}];

const RGB2GRAY_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2GRAY.TOO_MANY_INPUTS",
    identifier: Some("RunMat:rgb2gray:TooManyInputs"),
    when: "More than one input argument is supplied.",
    message: "rgb2gray: too many input arguments",
};

const RGB2GRAY_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2GRAY.INVALID_INPUT",
    identifier: Some("RunMat:rgb2gray:InvalidInput"),
    when: "Input cannot be interpreted as an MxNx3 RGB image or an Nx3 double colormap.",
    message: "rgb2gray: invalid input",
};

const RGB2GRAY_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2GRAY.INTERNAL",
    identifier: Some("RunMat:rgb2gray:Internal"),
    when: "Grayscale output tensor construction fails internally.",
    message: "rgb2gray: internal conversion failure",
};

const RGB2GRAY_ERRORS: [BuiltinErrorDescriptor; 3] = [
    RGB2GRAY_ERROR_TOO_MANY_INPUTS,
    RGB2GRAY_ERROR_INVALID_INPUT,
    RGB2GRAY_ERROR_INTERNAL,
];

pub const RGB2GRAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RGB2GRAY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RGB2GRAY_ERRORS,
};

const RGB2GRAY_DOCUMENTED_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "RGB",
        classes: &common::DOCUMENTED_IMAGE_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "uint8 and uint16 truecolor images are documented and produce grayscale output in the same native integer class.",
    }];
const RGB2GRAY_REJECTED_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "RGB",
        classes: &common::REJECTED_IMAGE_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Signed integer and uint32/uint64 RGB images are outside the documented surface and reject before resident download.",
    }];
pub const RGB2GRAY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "I = rgb2gray(integer_RGB)",
        inputs: &RGB2GRAY_DOCUMENTED_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "RunMat reads authoritative uint8/uint16 samples, applies the documented luminance coefficients, and returns the same integer image class using the public integer-conversion rule of nearest rounding with ties away from zero and saturation; resident output is restored through the input owner.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "rgb2gray(unsupported_integer_RGB)",
        inputs: &RGB2GRAY_REJECTED_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Unsupported integer classes reject consistently on host and from resident dtype metadata.",
    },
];

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
        rgb2gray_error_with_message(err.message().to_string(), fallback)
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::color::rgb2gray")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("rgb2gray"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Owner-aware host fallback preserves image dtype semantics and restores documented GPU results to the source provider.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::color::rgb2gray")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not fused yet; channel-aware image shape handling is custom.",
};

#[runtime_builtin(
    name = "rgb2gray",
    category = "image/color",
    summary = "Convert RGB image data to grayscale using luminance weighting.",
    keywords = "rgb2gray,rgb,gray,grayscale,luminance,image",
    accel = "sink",
    type_resolver(rgb2gray_type),
    descriptor(crate::builtins::image::color::rgb2gray::RGB2GRAY_DESCRIPTOR),
    integer_capabilities(crate::builtins::image::color::rgb2gray::RGB2GRAY_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::image::color::rgb2gray"
)]
async fn rgb2gray_builtin(rgb: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(rgb2gray_error(&RGB2GRAY_ERROR_TOO_MANY_INPUTS));
    }
    let resident_sources = match &rgb {
        Value::GpuTensor(handle) => {
            validate_resident_rgb(handle)?;
            vec![handle.clone()]
        }
        _ => Vec::new(),
    };
    let resident_guard = common::protect_resident_inputs(&resident_sources);
    let tensor = common::gather_tensor(NAME, rgb)
        .await
        .map_err(|err| rgb2gray_map_error(err, &RGB2GRAY_ERROR_INVALID_INPUT))?;
    resident_guard.restore();
    let out = rgb2gray_tensor(&tensor)
        .map_err(|err| rgb2gray_map_error(err, &RGB2GRAY_ERROR_INTERNAL))?;
    common::restore_resident_numeric_result_for_sources(
        &resident_sources,
        common::image_value_from_tensor(out),
        NAME,
    )
}

fn validate_resident_rgb(handle: &runmat_accelerate_api::GpuTensorHandle) -> BuiltinResult<()> {
    let dtype = common::resident_numeric_dtype(handle, NAME)
        .map_err(|err| rgb2gray_map_error(err, &RGB2GRAY_ERROR_INVALID_INPUT))?;
    let supported = if handle.shape.len() == 3 && handle.shape.get(2) == Some(&3) {
        matches!(
            dtype,
            NumericDType::F32 | NumericDType::F64 | NumericDType::U8 | NumericDType::U16
        )
    } else if handle.shape.len() == 2 && handle.shape.get(1) == Some(&3) {
        dtype == NumericDType::F64
    } else {
        false
    };
    if !supported {
        return Err(rgb2gray_error_with_message(
            format!(
                "rgb2gray: unsupported resident class {} or shape {:?}",
                dtype.class_name(),
                handle.shape
            ),
            &RGB2GRAY_ERROR_INVALID_INPUT,
        ));
    }
    Ok(())
}

fn rgb2gray_tensor(rgb: &Tensor) -> BuiltinResult<Tensor> {
    const RED: f64 = 0.298_936_021_293_775;
    const GREEN: f64 = 0.587_043_074_451_121;
    const BLUE: f64 = 0.114_020_904_255_103;

    let dtype = rgb.numeric_dtype();
    let layout = common::color_layout(rgb, NAME)?;
    let supported = match layout {
        common::ColorLayout::Truecolor { .. } => matches!(
            dtype,
            NumericDType::F32 | NumericDType::F64 | NumericDType::U8 | NumericDType::U16
        ),
        common::ColorLayout::Colormap { .. } => dtype == NumericDType::F64,
    };
    if !supported {
        return Err(rgb2gray_map_error(
            common::builtin_error(
                NAME,
                format!(
                    "rgb2gray: unsupported {} class {}",
                    match layout {
                        common::ColorLayout::Truecolor { .. } => "truecolor",
                        common::ColorLayout::Colormap { .. } => "colormap",
                    },
                    dtype.class_name()
                ),
            ),
            &RGB2GRAY_ERROR_INVALID_INPUT,
        ));
    }

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
    use crate::builtins::common::{gpu_helpers, test_support};
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

    fn call(value: Value) -> Value {
        block_on(rgb2gray_builtin(value, Vec::new())).expect("rgb2gray")
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
    fn preserves_2d_shape() {
        let rgb = Tensor::new(vec![1.0; 12], vec![2, 2, 3]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(rgb)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape, vec![2, 2]);
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
        assert_eq!(out.shape, vec![2, 1]);
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
        assert_eq!(out.shape, vec![2, 3]);
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
        let err = block_on(rgb2gray_builtin(Value::Tensor(rgb), Vec::new()))
            .expect_err("int16 truecolor is not a supported MATLAB rgb2gray input class");
        assert_eq!(err.identifier(), RGB2GRAY_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("int16"));
    }

    #[test]
    fn rejects_non_rgb_matrix_shape() {
        let map = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let err = block_on(rgb2gray_builtin(Value::Tensor(map), Vec::new())).unwrap_err();
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
        assert_eq!(RGB2GRAY_INTEGER_CAPABILITIES.len(), 2);
        assert_eq!(
            RGB2GRAY_INTEGER_CAPABILITIES[0].inputs[0].classes,
            &common::DOCUMENTED_IMAGE_INTEGER_CLASSES
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
            let source =
                source.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let result = block_on(rgb2gray_builtin(
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
        let handle = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1, 3],
            device_id: u32::MAX - 11,
            buffer_id: 1,
            descriptor: Default::default(),
        };
        runmat_accelerate_api::set_handle_integer_type(
            &handle,
            runmat_accelerate_api::IntegerElementType::I16,
        );
        let err = block_on(rgb2gray_builtin(
            Value::GpuTensor(handle.clone()),
            Vec::new(),
        ))
        .expect_err("unsupported resident integer");
        runmat_accelerate_api::clear_handle_metadata(&handle);
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
        let err = block_on(rgb2gray_builtin(Value::Num(1.0), vec![Value::Num(2.0)]))
            .expect_err("expected argument error");
        assert_eq!(err.identifier(), RGB2GRAY_ERROR_TOO_MANY_INPUTS.identifier);
    }

    #[test]
    fn rgb2gray_is_registered_with_dispatcher() {
        let rgb = Tensor::new(vec![1.0, 1.0, 1.0], vec![1, 1, 3]).unwrap();
        let result = block_on(crate::call_builtin_async(NAME, &[Value::Tensor(rgb)]))
            .expect("rgb2gray registered");
        assert!(matches!(result, Value::Num(value) if (value - 1.0).abs() < 1e-12));
    }
}
