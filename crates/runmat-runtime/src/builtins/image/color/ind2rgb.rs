//! MATLAB-compatible `ind2rgb` indexed-image conversion.

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
use crate::builtins::image::color::type_resolvers::ind2rgb_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "ind2rgb";

const IND2RGB_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "RGB",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RGB truecolor image converted from indexed image and colormap.",
}];

const IND2RGB_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexed image values.",
    },
    BuiltinParamDescriptor {
        name: "map",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Nx3 colormap.",
    },
];

const IND2RGB_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "RGB = ind2rgb(X, map)",
    inputs: &IND2RGB_INPUTS,
    outputs: &IND2RGB_OUTPUT,
}];

const IND2RGB_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2RGB.TOO_MANY_INPUTS",
    identifier: Some("RunMat:ind2rgb:TooManyInputs"),
    when: "More than two input arguments are supplied.",
    message: "ind2rgb: too many input arguments",
};

const IND2RGB_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2RGB.INVALID_INPUT",
    identifier: Some("RunMat:ind2rgb:InvalidInput"),
    when: "Inputs cannot be interpreted as numeric indexed image and numeric colormap tensors.",
    message: "ind2rgb: invalid input",
};

const IND2RGB_ERROR_INVALID_COLORMAP: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2RGB.INVALID_COLORMAP",
    identifier: Some("RunMat:ind2rgb:InvalidColormap"),
    when: "The map argument is not an Nx3 colormap.",
    message: "ind2rgb: map must be an Nx3 colormap",
};

const IND2RGB_ERROR_INVALID_INDEX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2RGB.INVALID_INDEX",
    identifier: Some("RunMat:ind2rgb:InvalidIndex"),
    when: "At least one floating index value is not finite or integral.",
    message: "ind2rgb: floating index values must be finite integers",
};

const IND2RGB_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2RGB.INTERNAL",
    identifier: Some("RunMat:ind2rgb:Internal"),
    when: "RGB output tensor construction fails internally.",
    message: "ind2rgb: internal conversion failure",
};

const IND2RGB_ERRORS: [BuiltinErrorDescriptor; 5] = [
    IND2RGB_ERROR_TOO_MANY_INPUTS,
    IND2RGB_ERROR_INVALID_INPUT,
    IND2RGB_ERROR_INVALID_COLORMAP,
    IND2RGB_ERROR_INVALID_INDEX,
    IND2RGB_ERROR_INTERNAL,
];

pub const IND2RGB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IND2RGB_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IND2RGB_ERRORS,
};

const IND2RGB_DOCUMENTED_INDEX_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &common::DOCUMENTED_IMAGE_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "uint8 and uint16 indexed images use documented zero-based indices and are clipped to the colormap extent before lookup.",
    }];
const IND2RGB_REJECTED_INDEX_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &common::REJECTED_IMAGE_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Other native integer classes are outside the documented indexed-image surface and reject before a resident value is downloaded.",
    }];
const IND2RGB_REJECTED_MAP_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "map",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented colormap is double; typed-integer colormaps reject before lookup or resident download.",
    }];
pub const IND2RGB_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "RGB = ind2rgb(integer_X, map)",
        inputs: &IND2RGB_DOCUMENTED_INDEX_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Authoritative uint8/uint16 indices are zero-based and clipped exactly; documented GPU input is downloaded non-destructively and the double RGB result is restored through its owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "ind2rgb(unsupported_integer_X, map)",
        inputs: &IND2RGB_REJECTED_INDEX_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Unsupported indexed-image classes reject consistently on host and from resident dtype metadata.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "ind2rgb(X, integer_map)",
        inputs: &IND2RGB_REJECTED_MAP_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "All typed-integer colormaps reject because the documented map class is double.",
    },
];

fn ind2rgb_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    ind2rgb_error_with_message(error.message, error)
}

fn ind2rgb_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn ind2rgb_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        ind2rgb_error_with_message(err.message().to_string(), fallback)
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::color::ind2rgb")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("ind2rgb"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Owner-aware host fallback preserves indexed-image class rules and restores documented GPU results to the source provider.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::color::ind2rgb")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not fused yet; colormap lookup changes rank and depends on index class.",
};

#[runtime_builtin(
    name = "ind2rgb",
    category = "image/color",
    summary = "Convert indexed images and colormaps to RGB.",
    keywords = "ind2rgb,indexed image,colormap,rgb,image",
    accel = "sink",
    type_resolver(ind2rgb_type),
    descriptor(crate::builtins::image::color::ind2rgb::IND2RGB_DESCRIPTOR),
    integer_capabilities(crate::builtins::image::color::ind2rgb::IND2RGB_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::image::color::ind2rgb"
)]
async fn ind2rgb_builtin(indexed: Value, map: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(ind2rgb_error(&IND2RGB_ERROR_TOO_MANY_INPUTS));
    }
    let resident_sources: Vec<_> = [&indexed, &map]
        .into_iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        })
        .collect();
    let indexed_is_logical = match &indexed {
        Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_logical(handle),
        _ => false,
    };
    let map_is_logical = match &map {
        Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_logical(handle),
        _ => false,
    };
    if let Value::GpuTensor(handle) = &indexed {
        let dtype = common::resident_numeric_dtype(handle, NAME)
            .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INVALID_INPUT))?;
        if indexed_is_logical
            || !matches!(
                dtype,
                NumericDType::F32 | NumericDType::F64 | NumericDType::U8 | NumericDType::U16
            )
        {
            return Err(ind2rgb_error_with_message(
                format!(
                    "ind2rgb: {} indexed image is not supported; expected single, double, uint8, or uint16",
                    if indexed_is_logical { "logical" } else { "resident numeric" }
                ),
                &IND2RGB_ERROR_INVALID_INPUT,
            ));
        }
        if handle.shape.len() != 2 {
            return Err(ind2rgb_error_with_message(
                "ind2rgb: indexed image must be an MxN matrix",
                &IND2RGB_ERROR_INVALID_INPUT,
            ));
        }
    }
    if let Value::GpuTensor(handle) = &map {
        let dtype = common::resident_numeric_dtype(handle, NAME)
            .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INVALID_COLORMAP))?;
        if map_is_logical || dtype != NumericDType::F64 {
            return Err(ind2rgb_error_with_message(
                "ind2rgb: resident colormap is not supported; expected double",
                &IND2RGB_ERROR_INVALID_COLORMAP,
            ));
        }
        if handle.shape.len() != 2 || handle.shape.get(1) != Some(&3) || handle.shape[0] == 0 {
            return Err(ind2rgb_error(&IND2RGB_ERROR_INVALID_COLORMAP));
        }
    }
    let resident_guard = common::protect_resident_inputs(&resident_sources);
    let indexed = common::gather_tensor(NAME, indexed)
        .await
        .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INVALID_INPUT))?;
    let map = common::gather_tensor(NAME, map)
        .await
        .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INVALID_INPUT))?;
    resident_guard.restore();
    let indexed_dtype = indexed.numeric_dtype();
    if indexed.shape.len() != 2 {
        return Err(ind2rgb_error_with_message(
            "ind2rgb: indexed image must be an MxN matrix",
            &IND2RGB_ERROR_INVALID_INPUT,
        ));
    }
    if indexed_is_logical
        || !matches!(
            indexed_dtype,
            NumericDType::F32 | NumericDType::F64 | NumericDType::U8 | NumericDType::U16
        )
    {
        return Err(ind2rgb_error_with_message(
            format!(
                "ind2rgb: {} indexed image is not supported; expected single, double, uint8, or uint16",
                if indexed_is_logical {
                    "logical"
                } else {
                    indexed_dtype.class_name()
                }
            ),
            &IND2RGB_ERROR_INVALID_INPUT,
        ));
    }
    let map_dtype = map.numeric_dtype();
    if map_is_logical || map_dtype != NumericDType::F64 {
        return Err(ind2rgb_error_with_message(
            format!(
                "ind2rgb: {} colormap is not supported; expected double",
                if map_is_logical {
                    "logical"
                } else {
                    map_dtype.class_name()
                }
            ),
            &IND2RGB_ERROR_INVALID_COLORMAP,
        ));
    }
    let layout = common::color_layout(&map, NAME)
        .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INVALID_COLORMAP))?;
    let common::ColorLayout::Colormap { rows: map_rows } = layout else {
        return Err(ind2rgb_error(&IND2RGB_ERROR_INVALID_COLORMAP));
    };

    let pixels = indexed.len();
    let mut shape = indexed.shape.clone();
    shape.push(3);
    let indexed_values = common::tensor_values_f64(&indexed);
    let map_values = common::tensor_values_f64(&map);
    let mut data = vec![0.0; pixels * 3];
    for (pixel, raw_index) in indexed_values.iter().copied().enumerate() {
        let map_index = map_index(raw_index, indexed_dtype, map_rows)
            .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INVALID_INDEX))?;
        for channel in 0..3 {
            data[pixel + pixels * channel] = map_values[layout.index(map_index, channel)];
        }
    }

    let out = common::tensor_with_dtype(data, shape, NumericDType::F64, NAME)
        .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INTERNAL))?;
    common::restore_resident_numeric_result_for_sources(
        &resident_sources,
        common::image_value_from_tensor(out),
        NAME,
    )
}

fn map_index(value: f64, dtype: NumericDType, map_rows: usize) -> BuiltinResult<usize> {
    if !value.is_finite()
        || (matches!(dtype, NumericDType::F32 | NumericDType::F64) && value.fract() != 0.0)
    {
        return Err(ind2rgb_error(&IND2RGB_ERROR_INVALID_INDEX));
    }
    let index = if matches!(dtype, NumericDType::U8 | NumericDType::U16) {
        (value as isize).clamp(0, map_rows as isize - 1)
    } else {
        (value as isize).clamp(1, map_rows as isize) - 1
    };
    Ok(index as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::{gpu_helpers, test_support};
    use futures::executor::block_on;
    use runmat_value::{IntegerStorage, LogicalArray, Tensor};

    fn call(indexed: Tensor, map: Tensor) -> BuiltinResult<Tensor> {
        let value = block_on(ind2rgb_builtin(
            Value::Tensor(indexed),
            Value::Tensor(map),
            Vec::new(),
        ))?;
        let Value::Tensor(out) = value else {
            panic!("expected tensor");
        };
        Ok(out)
    }

    fn values(tensor: &Tensor) -> Vec<f64> {
        tensor.materialize_f64()
    }

    #[test]
    fn converts_one_based_double_indices() {
        let indexed = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let map = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]).unwrap();
        let out = call(indexed, map).unwrap();
        assert_eq!(out.shape, vec![1, 2, 3]);
        assert_eq!(out.numeric_dtype(), NumericDType::F64);
        assert_eq!(values(&out), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn converts_zero_based_uint8_indices() {
        let indexed = Tensor::new_with_dtype(vec![0.0, 1.0], vec![1, 2], NumericDType::U8).unwrap();
        let map = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]).unwrap();
        let out = call(indexed, map).unwrap();
        assert_eq!(values(&out), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn accepts_single_indices_and_returns_double() {
        let indexed =
            Tensor::new_with_dtype(vec![1.0, 2.0], vec![2, 1], NumericDType::F32).unwrap();
        let map = Tensor::new(vec![1.0, 0.0, 0.0, 0.5, 0.0, 1.0], vec![2, 3]).unwrap();
        let out = call(indexed, map).unwrap();
        assert_eq!(out.shape, vec![2, 1, 3]);
        assert_eq!(out.numeric_dtype(), NumericDType::F64);
        assert_eq!(values(&out), vec![1.0, 0.0, 0.0, 0.5, 0.0, 1.0]);
    }

    #[test]
    fn ind2rgb_reads_typed_integer_indices_exactly() {
        let indexed = Tensor::new_integer(IntegerStorage::U16(vec![0, 1]), vec![2, 1]).unwrap();
        let map = Tensor::new(vec![1.0, 0.0, 0.0, 0.5, 0.0, 1.0], vec![2, 3]).unwrap();

        let out = call(indexed, map).unwrap();

        assert_eq!(out.shape, vec![2, 1, 3]);
        assert_eq!(out.numeric_dtype(), NumericDType::F64);
        assert_eq!(values(&out), vec![1.0, 0.0, 0.0, 0.5, 0.0, 1.0]);
    }

    #[test]
    fn clips_indices_to_colormap_range() {
        let indexed = Tensor::new(vec![-10.0, 100.0], vec![1, 2]).unwrap();
        let map = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]).unwrap();
        let out = call(indexed, map).unwrap();
        assert_eq!(values(&out), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn rejects_nonintegral_floating_indices_instead_of_rounding_extension() {
        let indexed = Tensor::new(vec![1.5], vec![1, 1]).unwrap();
        let map = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let err = call(indexed, map).expect_err("fractional floating index");
        assert_eq!(err.identifier(), IND2RGB_ERROR_INVALID_INDEX.identifier);
    }

    #[test]
    fn rejects_nonscalar_index_stack_shape() {
        let indexed = Tensor::new(vec![1.0, 1.0], vec![1, 1, 2]).unwrap();
        let map = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let err = call(indexed, map).expect_err("indexed image must be a matrix");
        assert!(err.message().contains("MxN matrix"));
    }

    #[test]
    fn rejects_unsupported_index_classes() {
        for storage in [
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ] {
            let indexed = Tensor::new_integer(storage, vec![1, 1]).unwrap();
            let map = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
            let err = call(indexed, map).unwrap_err();
            assert!(err.message().contains("indexed image is not supported"));
        }
    }

    #[test]
    fn rejects_non_double_colormap() {
        let indexed = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let map =
            Tensor::new_with_dtype(vec![1.0, 0.0, 0.0], vec![1, 3], NumericDType::F32).unwrap();
        let err = call(indexed, map).unwrap_err();
        assert!(err.message().contains("colormap is not supported"));
    }

    #[test]
    fn rejects_logical_indexed_image_and_colormap() {
        let logical_indexed = LogicalArray::new(vec![1], vec![1, 1]).unwrap();
        let map = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let err = block_on(ind2rgb_builtin(
            Value::LogicalArray(logical_indexed),
            Value::Tensor(map),
            Vec::new(),
        ))
        .unwrap_err();
        assert!(err.message().contains("logical indexed image"));

        let indexed = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let logical_map = LogicalArray::new(vec![1, 0, 0], vec![1, 3]).unwrap();
        let err = block_on(ind2rgb_builtin(
            Value::Tensor(indexed),
            Value::LogicalArray(logical_map),
            Vec::new(),
        ))
        .unwrap_err();
        assert!(err.message().contains("logical colormap"));
    }

    #[test]
    fn rejects_non_colormap_map_shape() {
        let indexed = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let map = Tensor::new(vec![1.0; 12], vec![2, 2, 3]).unwrap();
        let err = block_on(ind2rgb_builtin(
            Value::Tensor(indexed),
            Value::Tensor(map),
            Vec::new(),
        ))
        .unwrap_err();
        assert!(err.message().contains("map must be an Nx3 colormap"));
    }

    #[test]
    fn ind2rgb_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = IND2RGB_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(labels, vec!["RGB = ind2rgb(X, map)"]);
    }

    #[test]
    fn ind2rgb_integer_capabilities_distinguish_supported_indices_and_rejected_maps() {
        assert_eq!(IND2RGB_INTEGER_CAPABILITIES.len(), 3);
        assert_eq!(
            IND2RGB_INTEGER_CAPABILITIES[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::Documented
        );
        assert_eq!(
            IND2RGB_INTEGER_CAPABILITIES[2].inputs[0].availability,
            BuiltinIntegerInputAvailability::Rejected
        );
    }

    #[test]
    fn ind2rgb_gpu_fallback_restores_to_owner_and_preserves_source() {
        test_support::with_test_provider(|provider| {
            let indexed = Tensor::new_integer(IntegerStorage::U8(vec![0, 1]), vec![1, 2]).unwrap();
            let source = gpu_helpers::upload_tensor(provider, &indexed).expect("upload indexed");
            let source =
                source.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let map = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]).unwrap();

            let result = block_on(ind2rgb_builtin(
                Value::GpuTensor(source.clone()),
                Value::Tensor(map),
                Vec::new(),
            ))
            .expect("ind2rgb");
            let Value::GpuTensor(output) = result else {
                panic!("expected restored gpu output");
            };
            assert_eq!(output.shape, vec![1, 2, 3]);
            assert!(runmat_accelerate_api::provider_for_handle(&output)
                .is_some_and(|owner| std::ptr::eq(owner, provider)));
            assert_eq!(
                test_support::gather(Value::GpuTensor(output))
                    .unwrap()
                    .materialize_f64(),
                vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            );
            assert_eq!(
                test_support::gather(Value::GpuTensor(source))
                    .unwrap()
                    .integer_storage(),
                Some(&IntegerStorage::U8(vec![0, 1]))
            );
        });
    }

    #[test]
    fn ind2rgb_rejects_unsupported_resident_integer_before_download() {
        let handle = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX - 10,
            buffer_id: 1,
            descriptor: Default::default(),
        }
        .with_numeric_descriptor(
            runmat_accelerate_api::NumericElementType::I16,
            runmat_accelerate_api::GpuTensorStorage::Real,
        );
        let map = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let err = block_on(ind2rgb_builtin(
            Value::GpuTensor(handle.clone()),
            Value::Tensor(map),
            Vec::new(),
        ))
        .expect_err("unsupported resident integer");
        runmat_accelerate_api::clear_handle_metadata(&handle);
        assert_eq!(err.identifier(), IND2RGB_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn ind2rgb_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = IND2RGB_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.IND2RGB.TOO_MANY_INPUTS"));
        assert!(codes.contains(&"RM.IND2RGB.INVALID_INPUT"));
        assert!(codes.contains(&"RM.IND2RGB.INVALID_COLORMAP"));
        assert!(codes.contains(&"RM.IND2RGB.INVALID_INDEX"));
        assert!(codes.contains(&"RM.IND2RGB.INTERNAL"));
    }

    #[test]
    fn ind2rgb_too_many_args_uses_stable_identifier() {
        let err = block_on(ind2rgb_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            vec![Value::Num(3.0)],
        ))
        .expect_err("expected argument error");
        assert_eq!(err.identifier(), IND2RGB_ERROR_TOO_MANY_INPUTS.identifier);
    }
}
