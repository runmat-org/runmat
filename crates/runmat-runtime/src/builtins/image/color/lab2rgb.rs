//! MATLAB-compatible default-D65 `lab2rgb` conversion.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, NumericStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::image::color::common;
use crate::builtins::image::color::type_resolvers::lab2rgb_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "lab2rgb";
const XN: f64 = 0.95047;
const YN: f64 = 1.0;
const ZN: f64 = 1.08883;
const EPSILON: f64 = 216.0 / 24389.0;
const KAPPA: f64 = 24389.0 / 27.0;

const LAB2RGB_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "RGB",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "sRGB image or colormap converted from CIE L*a*b* input.",
}];

const LAB2RGB_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "LAB",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "CIE L*a*b* image or Nx3 colormap values.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "OutputType name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "double, single, uint8, or uint16 output class.",
    },
];

const LAB2RGB_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "RGB = lab2rgb(LAB, OutputType=type)",
    inputs: &LAB2RGB_INPUTS,
    outputs: &LAB2RGB_OUTPUT,
}];

const LAB2RGB_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LAB2RGB.TOO_MANY_INPUTS",
    identifier: Some("RunMat:lab2rgb:TooManyInputs"),
    when: "Too many name-value arguments are supplied.",
    message: "lab2rgb: too many input arguments",
};

const LAB2RGB_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LAB2RGB.INVALID_INPUT",
    identifier: Some("RunMat:lab2rgb:InvalidInput"),
    when: "Input cannot be interpreted as an MxNx3 L*a*b* image or Nx3 colormap.",
    message: "lab2rgb: invalid input",
};

const LAB2RGB_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LAB2RGB.INTERNAL",
    identifier: Some("RunMat:lab2rgb:Internal"),
    when: "RGB output tensor construction fails internally.",
    message: "lab2rgb: internal conversion failure",
};
const LAB2RGB_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LAB2RGB.INVALID_OPTION",
    identifier: Some("RunMat:lab2rgb:InvalidOption"),
    when: "OutputType is malformed or unsupported.",
    message: "lab2rgb: invalid option",
};

const LAB2RGB_ERRORS: [BuiltinErrorDescriptor; 4] = [
    LAB2RGB_ERROR_TOO_MANY_INPUTS,
    LAB2RGB_ERROR_INVALID_INPUT,
    LAB2RGB_ERROR_INTERNAL,
    LAB2RGB_ERROR_INVALID_OPTION,
];

pub const LAB2RGB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LAB2RGB_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LAB2RGB_ERRORS,
};

const LAB2RGB_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "lab2rgb-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "lab2rgb with an explicitly requested gpuArray input is a RunMat extension because the compatibility target documents no interactive GPU Arrays capability",
    error_identifier: Some("RunMat:compatibility:Lab2rgbExplicitGpuExtension"),
};
pub const LAB2RGB_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [LAB2RGB_EXPLICIT_GPU_EXTENSION];

const LAB2RGB_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "LAB",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The compatibility target accepts only single or double LAB input. Every integer class rejects from host or resident metadata before conversion or gather.",
}];
pub const LAB2RGB_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "RGB = lab2rgb(LAB, OutputType=integer_type)",
    inputs: &LAB2RGB_INTEGER_INPUT,
    computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
    output_class: BuiltinIntegerOutputClassRule::OptionDependent,
    overflow: BuiltinIntegerOverflowRule::Saturate,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
    notes: "Integer LAB input is rejected. For floating LAB input, OutputType uint8 or uint16 clamps normalized RGB to [0,1], rounds after scaling by the class maximum, and constructs authoritative native integer storage; floating output preserves out-of-gamut values. Automatic residency may gather transparently, while the gated explicit-gpuArray extension restores output to the exact owner or errors.",
}];

fn lab2rgb_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    lab2rgb_error_with_message(error.message, error)
}

fn lab2rgb_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn lab2rgb_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        lab2rgb_error_with_message(err.message().to_string(), fallback)
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::color::lab2rgb")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("lab2rgb"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host implementation uses the default D65/sRGB conversion path. Floating output retains out-of-gamut values; integer OutputType clips during class conversion.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::color::lab2rgb")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not fused yet; conversion is channel-coupled and piecewise.",
};

#[runtime_builtin(
    name = "lab2rgb",
    category = "image/color",
    summary = "Convert CIE L*a*b* values to sRGB (D65 white point).",
    keywords = "lab2rgb,lab,cielab,rgb,color,image,colormap",
    accel = "sink",
    type_resolver(lab2rgb_type),
    descriptor(crate::builtins::image::color::lab2rgb::LAB2RGB_DESCRIPTOR),
    extensions(crate::builtins::image::color::lab2rgb::LAB2RGB_EXTENSIONS),
    integer_capabilities(crate::builtins::image::color::lab2rgb::LAB2RGB_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::image::color::lab2rgb"
)]
async fn lab2rgb_builtin(lab: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let output_type = parse_output_type(&rest)?;
    let mut resident_sources = Vec::new();
    if let Value::GpuTensor(handle) = &lab {
        if runmat_accelerate_api::handle_is_explicit(handle) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &LAB2RGB_EXPLICIT_GPU_EXTENSION,
                NAME,
            )?;
        }
        if runmat_accelerate_api::handle_integer_type(handle).is_some()
            || runmat_accelerate_api::handle_is_logical(handle)
        {
            return Err(lab2rgb_error_with_message(
                "lab2rgb: integer and logical LAB input is not supported; expected single or double",
                &LAB2RGB_ERROR_INVALID_INPUT,
            ));
        }
        let dtype = common::resident_numeric_dtype(handle, NAME)
            .map_err(|err| lab2rgb_map_error(err, &LAB2RGB_ERROR_INVALID_INPUT))?;
        if !matches!(dtype, NumericDType::F32 | NumericDType::F64) {
            return Err(lab2rgb_error_with_message(
                format!(
                    "lab2rgb: {} input is not supported; expected single or double",
                    dtype.class_name()
                ),
                &LAB2RGB_ERROR_INVALID_INPUT,
            ));
        }
        resident_sources.push(handle.clone());
    }
    let logical_input = match &lab {
        Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_logical(handle),
        _ => false,
    };
    let resident_guard = common::protect_resident_inputs(&resident_sources);
    let tensor = common::gather_tensor(NAME, lab)
        .await
        .map_err(|err| lab2rgb_map_error(err, &LAB2RGB_ERROR_INVALID_INPUT))?;
    resident_guard.restore();
    let layout = common::color_layout(&tensor, NAME)
        .map_err(|err| lab2rgb_map_error(err, &LAB2RGB_ERROR_INVALID_INPUT))?;
    let dtype = tensor.numeric_dtype();
    if logical_input || !matches!(dtype, NumericDType::F32 | NumericDType::F64) {
        return Err(lab2rgb_error_with_message(
            format!(
                "lab2rgb: {} input is not supported; expected single or double",
                if logical_input {
                    "logical"
                } else {
                    dtype.class_name()
                }
            ),
            &LAB2RGB_ERROR_INVALID_INPUT,
        ));
    }
    let values = common::tensor_values_f64(&tensor);
    let mut data = vec![0.0; values.len()];
    for pixel in 0..layout.pixels() {
        let l = values[layout.index(pixel, 0)];
        let a = values[layout.index(pixel, 1)];
        let b = values[layout.index(pixel, 2)];
        let (r, g, blue) = lab_to_rgb_unit(l, a, b);
        data[layout.index(pixel, 0)] = cast_float(r, dtype);
        data[layout.index(pixel, 1)] = cast_float(g, dtype);
        data[layout.index(pixel, 2)] = cast_float(blue, dtype);
    }
    let output_type = output_type.unwrap_or(dtype);
    let storage = lab2rgb_output_storage(data, output_type);
    let out = Tensor::from_numeric_storage(storage, layout.output_shape())
        .map_err(|err| lab2rgb_error_with_message(err, &LAB2RGB_ERROR_INTERNAL))?;
    common::restore_resident_numeric_result_for_sources(
        &resident_sources,
        common::image_value_from_tensor(out),
        NAME,
    )
}

fn parse_output_type(rest: &[Value]) -> BuiltinResult<Option<NumericDType>> {
    if rest.len() > 2 {
        return Err(lab2rgb_error(&LAB2RGB_ERROR_TOO_MANY_INPUTS));
    }
    if rest.is_empty() {
        return Ok(None);
    }
    if rest.len() != 2
        || !tensor::value_to_string(&rest[0])
            .is_some_and(|name| name.eq_ignore_ascii_case("OutputType"))
    {
        return Err(lab2rgb_error(&LAB2RGB_ERROR_INVALID_OPTION));
    }
    let output = tensor::value_to_string(&rest[1])
        .ok_or_else(|| lab2rgb_error(&LAB2RGB_ERROR_INVALID_OPTION))?;
    match output.to_ascii_lowercase().as_str() {
        "double" => Ok(Some(NumericDType::F64)),
        "single" => Ok(Some(NumericDType::F32)),
        "uint8" => Ok(Some(NumericDType::U8)),
        "uint16" => Ok(Some(NumericDType::U16)),
        _ => Err(lab2rgb_error(&LAB2RGB_ERROR_INVALID_OPTION)),
    }
}

fn lab2rgb_output_storage(values: Vec<f64>, dtype: NumericDType) -> NumericStorage {
    match dtype {
        NumericDType::F64 => NumericStorage::F64(values),
        NumericDType::F32 => {
            NumericStorage::F32(values.into_iter().map(|value| value as f32).collect())
        }
        NumericDType::U8 => NumericStorage::U8(
            values
                .into_iter()
                .map(|value| normalized_integer_output(value, u8::MAX as f64) as u8)
                .collect(),
        ),
        NumericDType::U16 => NumericStorage::U16(
            values
                .into_iter()
                .map(|value| normalized_integer_output(value, u16::MAX as f64) as u16)
                .collect(),
        ),
        _ => unreachable!("lab2rgb OutputType parser accepts only floating, uint8, or uint16"),
    }
}

fn normalized_integer_output(value: f64, maximum: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else {
        (value.clamp(0.0, 1.0) * maximum).round()
    }
}

pub(crate) fn lab_to_rgb_unit(l: f64, a: f64, b: f64) -> (f64, f64, f64) {
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    let x = XN * lab_f_inv(fx);
    let y = YN * lab_f_inv(fy);
    let z = ZN * lab_f_inv(fz);
    let r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let blue = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    (linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(blue))
}

fn lab_f_inv(value: f64) -> f64 {
    let cubed = value * value * value;
    if cubed > EPSILON {
        cubed
    } else {
        (116.0 * value - 16.0) / KAPPA
    }
}

pub(crate) fn linear_to_srgb(value: f64) -> f64 {
    if value <= 0.0031308 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
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
    use crate::builtins::common::{gpu_helpers, test_support};
    use futures::executor::block_on;
    use runmat_builtins::{IntegerStorage, LogicalArray, Tensor};

    fn call(tensor: Tensor) -> BuiltinResult<Tensor> {
        let value = block_on(lab2rgb_builtin(Value::Tensor(tensor), Vec::new()))?;
        let Value::Tensor(out) = value else {
            panic!("expected tensor");
        };
        Ok(out)
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    fn values(tensor: &Tensor) -> Vec<f64> {
        tensor.materialize_f64()
    }

    #[test]
    fn converts_white_lab_to_rgb() {
        let lab = Tensor::new(vec![100.0, 0.0, 0.0], vec![1, 1, 3]).unwrap();
        let out = call(lab).unwrap();
        let values = values(&out);
        assert_close(values[0], 1.0, 1e-4);
        assert_close(values[1], 1.0, 1e-4);
        assert_close(values[2], 1.0, 1e-4);
    }

    #[test]
    fn converts_lab_colormap_references_to_rgb() {
        let lab = Tensor::new(
            vec![
                0.0, 53.2408, 87.7347, 32.2970, 0.0, 80.0925, -86.1827, 79.1875, 0.0, 67.2032,
                83.1793, -107.8602,
            ],
            vec![4, 3],
        )
        .unwrap();
        let out = call(lab).unwrap();
        assert_eq!(out.shape, vec![4, 3]);
        let expected = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        for (actual, expected) in values(&out).iter().zip(expected) {
            assert_close(*actual, expected, 1e-3);
        }
    }

    #[test]
    fn preserves_single_precision_metadata() {
        let lab = Tensor::new_with_dtype(vec![100.0, 0.0, 0.0], vec![1, 1, 3], NumericDType::F32)
            .unwrap();
        let out = call(lab).unwrap();
        assert_eq!(out.numeric_dtype(), NumericDType::F32);
        let values = values(&out);
        assert_close(values[0], 1.0, 1e-4);
        assert_close(values[1], 1.0, 1e-4);
        assert_close(values[2], 1.0, 1e-4);
    }

    #[test]
    fn output_type_constructs_authoritative_uint8_and_uint16_storage() {
        let lab = Tensor::new(vec![70.0, 5.0, 10.0], vec![1, 1, 3]).unwrap();
        for (name, dtype) in [("uint8", NumericDType::U8), ("uint16", NumericDType::U16)] {
            let value = block_on(lab2rgb_builtin(
                Value::Tensor(lab.clone()),
                vec![Value::from("OutputType"), Value::from(name)],
            ))
            .expect("integer OutputType");
            let Value::Tensor(output) = value else {
                panic!("expected tensor output");
            };
            assert_eq!(output.numeric_dtype(), dtype);
            assert!(output.integer_storage().is_some());
        }
    }

    #[test]
    fn explicit_resident_input_restores_output_to_exact_owner() {
        test_support::with_test_provider(|provider| {
            let lab = Tensor::new(vec![70.0, 5.0, 10.0], vec![1, 1, 3]).unwrap();
            let source = gpu_helpers::upload_tensor(provider, &lab).expect("upload LAB");
            runmat_accelerate_api::mark_handle_explicit(&source);
            let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
            let value = block_on(lab2rgb_builtin(
                Value::GpuTensor(source.clone()),
                vec![Value::from("OutputType"), Value::from("uint8")],
            ))
            .expect("explicit resident lab2rgb");
            let Value::GpuTensor(output) = value else {
                panic!("explicit gpuArray output must remain resident");
            };
            assert_eq!(output.device_id, source.device_id);
            assert_ne!(output.buffer_id, source.buffer_id);
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&output),
                Some(runmat_accelerate_api::IntegerElementType::U8)
            );
            assert!(runmat_accelerate_api::handle_is_explicit(&output));
            provider.free(&output).expect("free output");
            provider.free(&source).expect("free source");
            runmat_accelerate_api::clear_handle_metadata(&output);
            runmat_accelerate_api::clear_handle_metadata(&source);
        });
    }

    #[test]
    fn floating_output_preserves_out_of_gamut_components() {
        let lab = Tensor::new(vec![50.0, 200.0, 200.0], vec![1, 1, 3]).unwrap();
        let output = call(lab).unwrap();
        assert!(output
            .materialize_f64()
            .iter()
            .any(|value| *value < 0.0 || *value > 1.0));
    }

    #[test]
    fn lab2rgb_rejects_every_integer_class_and_logical() {
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
            let lab = Tensor::new_integer(storage, vec![1, 1, 3]).unwrap();
            let err = call(lab).unwrap_err();
            assert_eq!(err.identifier(), LAB2RGB_ERROR_INVALID_INPUT.identifier);
        }

        let logical = LogicalArray::new(vec![1, 0, 0], vec![1, 1, 3]).unwrap();
        let err = block_on(lab2rgb_builtin(Value::LogicalArray(logical), Vec::new())).unwrap_err();
        assert_eq!(err.identifier(), LAB2RGB_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn lab2rgb_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = LAB2RGB_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(labels, vec!["RGB = lab2rgb(LAB, OutputType=type)"]);
    }

    #[test]
    fn lab2rgb_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = LAB2RGB_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.LAB2RGB.TOO_MANY_INPUTS"));
        assert!(codes.contains(&"RM.LAB2RGB.INVALID_INPUT"));
        assert!(codes.contains(&"RM.LAB2RGB.INTERNAL"));
        assert!(codes.contains(&"RM.LAB2RGB.INVALID_OPTION"));
    }

    #[test]
    fn lab2rgb_too_many_args_uses_stable_identifier() {
        let err = block_on(lab2rgb_builtin(
            Value::Num(1.0),
            vec![Value::Num(2.0), Value::Num(3.0), Value::Num(4.0)],
        ))
        .expect_err("expected argument error");
        assert_eq!(err.identifier(), LAB2RGB_ERROR_TOO_MANY_INPUTS.identifier);
    }

    #[test]
    fn resident_integer_input_rejects_before_provider_lookup() {
        let handle = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1, 3],
            device_id: u32::MAX - 21,
            buffer_id: u64::MAX - 21,
            descriptor: Default::default(),
        };
        runmat_accelerate_api::set_handle_integer_type(
            &handle,
            runmat_accelerate_api::IntegerElementType::U8,
        );
        let error = block_on(lab2rgb_builtin(Value::GpuTensor(handle), Vec::new()))
            .expect_err("integer resident LAB must reject");
        assert_eq!(error.identifier(), LAB2RGB_ERROR_INVALID_INPUT.identifier);
        assert!(!error.message().to_ascii_lowercase().contains("provider"));
    }

    #[test]
    fn resident_floating_metadata_must_match_exact_owner_before_download() {
        test_support::with_test_provider(|provider| {
            let lab =
                Tensor::new_with_dtype(vec![70.0, 5.0, 10.0], vec![1, 1, 3], NumericDType::F32)
                    .expect("LAB");
            let source = gpu_helpers::upload_tensor(provider, &lab).expect("upload LAB");
            let wrong_precision = match provider.precision() {
                runmat_accelerate_api::ProviderPrecision::F32 => {
                    runmat_accelerate_api::ProviderPrecision::F64
                }
                runmat_accelerate_api::ProviderPrecision::F64 => {
                    runmat_accelerate_api::ProviderPrecision::F32
                }
            };
            runmat_accelerate_api::set_handle_precision(&source, wrong_precision);
            let error = block_on(lab2rgb_builtin(
                Value::GpuTensor(source.clone()),
                Vec::new(),
            ))
            .expect_err("stale floating precision must reject before download");
            assert_eq!(error.identifier(), LAB2RGB_ERROR_INVALID_INPUT.identifier);
            runmat_accelerate_api::set_handle_precision(&source, provider.precision());
            provider.free(&source).expect("free source");
            runmat_accelerate_api::clear_handle_metadata(&source);
        });
    }
}
