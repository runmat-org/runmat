//! MATLAB-compatible `rgb2hsv` conversion.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::image::color::common;
use crate::builtins::image::color::type_resolvers::same_shape_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "rgb2hsv";

const RGB2HSV_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "HSV",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "HSV image or colormap converted from RGB input.",
}];

const RGB2HSV_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "RGB",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RGB image or Nx3 RGB colormap values.",
}];

const RGB2HSV_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "HSV = rgb2hsv(RGB)",
    inputs: &RGB2HSV_INPUTS,
    outputs: &RGB2HSV_OUTPUT,
}];

const RGB2HSV_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2HSV.TOO_MANY_INPUTS",
    identifier: Some("RunMat:rgb2hsv:TooManyInputs"),
    when: "More than one input argument is supplied.",
    message: "rgb2hsv: too many input arguments",
};

const RGB2HSV_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2HSV.INVALID_INPUT",
    identifier: Some("RunMat:rgb2hsv:InvalidInput"),
    when: "Input cannot be interpreted as an MxNx3 RGB image or Nx3 RGB colormap.",
    message: "rgb2hsv: invalid input",
};

const RGB2HSV_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2HSV.INTERNAL",
    identifier: Some("RunMat:rgb2hsv:Internal"),
    when: "HSV output tensor construction fails internally.",
    message: "rgb2hsv: internal conversion failure",
};

const RGB2HSV_ERRORS: [BuiltinErrorDescriptor; 3] = [
    RGB2HSV_ERROR_TOO_MANY_INPUTS,
    RGB2HSV_ERROR_INVALID_INPUT,
    RGB2HSV_ERROR_INTERNAL,
];

pub const RGB2HSV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RGB2HSV_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RGB2HSV_ERRORS,
};

const RGB2HSV_DOCUMENTED_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "RGB",
        classes: &common::DOCUMENTED_IMAGE_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "uint8 and uint16 MxNx3 truecolor images are documented and scale to double HSV output.",
    },
];
const RGB2HSV_REJECTED_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "RGB",
        classes: &common::REJECTED_IMAGE_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Signed integer and uint32/uint64 RGB values are outside the documented surface and reject before resident download.",
    }];
pub const RGB2HSV_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "HSV = rgb2hsv(integer_RGB)",
        inputs: &RGB2HSV_DOCUMENTED_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Authoritative uint8/uint16 samples are normalized before color conversion; documented GPU input is downloaded non-destructively and restored through its owner.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "rgb2hsv(unsupported_integer_RGB)",
        inputs: &RGB2HSV_REJECTED_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Unsupported integer classes reject consistently on host and from resident dtype metadata.",
    },
];

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
        rgb2hsv_error_with_message(err.message().to_string(), fallback)
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::color::rgb2hsv")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("rgb2hsv"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Owner-aware host fallback restores documented GPU results to the source provider.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::color::rgb2hsv")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not fused yet; conversion operates across RGB channels.",
};

#[runtime_builtin(
    name = "rgb2hsv",
    category = "image/color",
    summary = "Convert RGB image or colormap values to HSV color space.",
    keywords = "rgb2hsv,rgb,hsv,color,image,colormap",
    accel = "sink",
    type_resolver(same_shape_type),
    descriptor(crate::builtins::image::color::rgb2hsv::RGB2HSV_DESCRIPTOR),
    integer_capabilities(crate::builtins::image::color::rgb2hsv::RGB2HSV_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::image::color::rgb2hsv"
)]
async fn rgb2hsv_builtin(rgb: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(rgb2hsv_error(&RGB2HSV_ERROR_TOO_MANY_INPUTS));
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
        .map_err(|err| rgb2hsv_map_error(err, &RGB2HSV_ERROR_INVALID_INPUT))?;
    resident_guard.restore();
    let layout = common::color_layout(&tensor, NAME)
        .map_err(|err| rgb2hsv_map_error(err, &RGB2HSV_ERROR_INVALID_INPUT))?;
    let input_dtype = tensor.numeric_dtype();
    let supported = match layout {
        common::ColorLayout::Truecolor { .. } => matches!(
            input_dtype,
            NumericDType::F32 | NumericDType::F64 | NumericDType::U8 | NumericDType::U16
        ),
        common::ColorLayout::Colormap { .. } => input_dtype == NumericDType::F64,
    };
    if !supported {
        return Err(rgb2hsv_error_with_message(
            format!(
                "rgb2hsv: {} input is not supported; expected single, double, uint8, or uint16",
                input_dtype.class_name()
            ),
            &RGB2HSV_ERROR_INVALID_INPUT,
        ));
    }
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
    common::restore_resident_numeric_result_for_sources(
        &resident_sources,
        common::image_value_from_tensor(out),
        NAME,
    )
}

fn validate_resident_rgb(handle: &runmat_accelerate_api::GpuTensorHandle) -> BuiltinResult<()> {
    let dtype = common::resident_numeric_dtype(handle, NAME)
        .map_err(|err| rgb2hsv_map_error(err, &RGB2HSV_ERROR_INVALID_INPUT))?;
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
        return Err(rgb2hsv_error_with_message(
            format!(
                "rgb2hsv: unsupported resident class {} or shape {:?}",
                dtype.class_name(),
                handle.shape
            ),
            &RGB2HSV_ERROR_INVALID_INPUT,
        ));
    }
    Ok(())
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
    use crate::builtins::common::{gpu_helpers, test_support};
    use futures::executor::block_on;
    use runmat_value::{IntegerStorage, Tensor};

    fn call(tensor: Tensor) -> BuiltinResult<Tensor> {
        let Value::Tensor(out) =
            block_on(rgb2hsv_builtin(Value::Tensor(tensor), Vec::new())).expect("rgb2hsv")
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
        assert_eq!(out.shape, vec![1, 1, 3]);
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
        assert_eq!(out.shape, vec![4, 3]);
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
            let err = block_on(rgb2hsv_builtin(Value::Tensor(input), Vec::new()))
                .expect_err("unsupported integer class");
            assert_eq!(err.identifier(), RGB2HSV_ERROR_INVALID_INPUT.identifier);
        }
    }

    #[test]
    fn rejects_grayscale_shape() {
        let gray = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = block_on(rgb2hsv_builtin(Value::Tensor(gray), Vec::new())).unwrap_err();
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
        assert_eq!(RGB2HSV_INTEGER_CAPABILITIES.len(), 2);
        assert_eq!(
            RGB2HSV_INTEGER_CAPABILITIES[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::Documented
        );
        assert_eq!(
            RGB2HSV_INTEGER_CAPABILITIES[1].inputs[0].classes,
            &common::REJECTED_IMAGE_INTEGER_CLASSES
        );
    }

    #[test]
    fn rgb2hsv_gpu_fallback_restores_double_output_and_preserves_source() {
        test_support::with_test_provider(|provider| {
            let rgb =
                Tensor::new_integer(IntegerStorage::U8(vec![255, 0, 0]), vec![1, 1, 3]).unwrap();
            let source = gpu_helpers::upload_tensor(provider, &rgb).expect("upload rgb");
            let source =
                source.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let result = block_on(rgb2hsv_builtin(
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
        let handle = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1, 3],
            device_id: u32::MAX - 12,
            buffer_id: 1,
            descriptor: Default::default(),
        }
        .with_numeric_descriptor(
            runmat_accelerate_api::NumericElementType::U32,
            runmat_accelerate_api::GpuTensorStorage::Real,
        );
        let err = block_on(rgb2hsv_builtin(
            Value::GpuTensor(handle.clone()),
            Vec::new(),
        ))
        .expect_err("unsupported resident integer");
        runmat_accelerate_api::clear_handle_metadata(&handle);
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
        let err = block_on(rgb2hsv_builtin(Value::Num(1.0), vec![Value::Num(2.0)]))
            .expect_err("expected argument error");
        assert_eq!(err.identifier(), RGB2HSV_ERROR_TOO_MANY_INPUTS.identifier);
    }
}
