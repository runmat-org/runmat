//! MATLAB-compatible `ind2rgb` indexed-image conversion.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
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
    when: "At least one index value is not finite.",
    message: "ind2rgb: index values must be finite",
};

const IND2RGB_ERROR_INDEX_OUT_OF_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2RGB.INDEX_OUT_OF_RANGE",
    identifier: Some("RunMat:ind2rgb:IndexOutOfRange"),
    when: "At least one index value falls outside the colormap bounds.",
    message: "ind2rgb: index is outside the colormap",
};

const IND2RGB_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2RGB.INTERNAL",
    identifier: Some("RunMat:ind2rgb:Internal"),
    when: "RGB output tensor construction fails internally.",
    message: "ind2rgb: internal conversion failure",
};

const IND2RGB_ERRORS: [BuiltinErrorDescriptor; 6] = [
    IND2RGB_ERROR_TOO_MANY_INPUTS,
    IND2RGB_ERROR_INVALID_INPUT,
    IND2RGB_ERROR_INVALID_COLORMAP,
    IND2RGB_ERROR_INVALID_INDEX,
    IND2RGB_ERROR_INDEX_OUT_OF_RANGE,
    IND2RGB_ERROR_INTERNAL,
];

pub const IND2RGB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IND2RGB_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IND2RGB_ERRORS,
};

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

fn ind2rgb_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    ind2rgb_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
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
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host implementation preserves indexed-image class rules for uint8/uint16 index arrays.",
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
    summary = "Convert an indexed image and colormap to an RGB truecolor image.",
    keywords = "ind2rgb,indexed image,colormap,rgb,image",
    accel = "sink",
    type_resolver(ind2rgb_type),
    descriptor(crate::builtins::image::color::ind2rgb::IND2RGB_DESCRIPTOR),
    builtin_path = "crate::builtins::image::color::ind2rgb"
)]
async fn ind2rgb_builtin(indexed: Value, map: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(ind2rgb_error(&IND2RGB_ERROR_TOO_MANY_INPUTS));
    }
    let indexed = common::gather_tensor(NAME, indexed)
        .await
        .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INVALID_INPUT))?;
    let map = common::gather_tensor(NAME, map)
        .await
        .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INVALID_INPUT))?;
    let layout = common::color_layout(&map, NAME)
        .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INVALID_COLORMAP))?;
    let common::ColorLayout::Colormap { rows: map_rows } = layout else {
        return Err(ind2rgb_error(&IND2RGB_ERROR_INVALID_COLORMAP));
    };

    let pixels = indexed.data.len();
    let mut shape = indexed.shape.clone();
    shape.push(3);
    let dtype = match map.dtype {
        NumericDType::F32 => NumericDType::F32,
        _ => NumericDType::F64,
    };
    let mut data = vec![0.0; pixels * 3];
    for (pixel, raw_index) in indexed.data.iter().copied().enumerate() {
        let map_index = map_index(raw_index, indexed.dtype, map_rows)
            .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INVALID_INDEX))?;
        for channel in 0..3 {
            let value = common::unit_value(map.data[layout.index(map_index, channel)], map.dtype);
            data[pixel + pixels * channel] = if matches!(dtype, NumericDType::F32) {
                (value as f32) as f64
            } else {
                value
            };
        }
    }

    let out = common::tensor_with_dtype(data, shape, dtype, NAME)
        .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INTERNAL))?;
    Ok(common::image_value_from_tensor(out))
}

fn map_index(value: f64, dtype: NumericDType, map_rows: usize) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(ind2rgb_error(&IND2RGB_ERROR_INVALID_INDEX));
    }
    let index = if matches!(dtype, NumericDType::U8 | NumericDType::U16) {
        value.round() as isize
    } else {
        value.round() as isize - 1
    };
    if index < 0 || index as usize >= map_rows {
        return Err(ind2rgb_error_with_detail(
            &IND2RGB_ERROR_INDEX_OUT_OF_RANGE,
            format!("index {} is outside the colormap", value),
        ));
    }
    Ok(index as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;

    fn call(indexed: Tensor, map: Tensor) -> BuiltinResult<Tensor> {
        let Value::Tensor(out) = block_on(ind2rgb_builtin(
            Value::Tensor(indexed),
            Value::Tensor(map),
            Vec::new(),
        ))
        .expect("ind2rgb") else {
            panic!("expected tensor");
        };
        Ok(out)
    }

    #[test]
    fn converts_one_based_double_indices() {
        let indexed = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let map = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]).unwrap();
        let out = call(indexed, map).unwrap();
        assert_eq!(out.shape, vec![1, 2, 3]);
        assert_eq!(out.data, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn converts_zero_based_uint8_indices() {
        let indexed = Tensor::new_with_dtype(vec![0.0, 1.0], vec![1, 2], NumericDType::U8).unwrap();
        let map = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]).unwrap();
        let out = call(indexed, map).unwrap();
        assert_eq!(out.data, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn scales_uint8_colormap_values_to_unit_rgb() {
        let indexed = Tensor::new_with_dtype(vec![0.0, 1.0], vec![2, 1], NumericDType::U8).unwrap();
        let map = Tensor::new_with_dtype(
            vec![255.0, 0.0, 0.0, 128.0, 0.0, 255.0],
            vec![2, 3],
            NumericDType::U8,
        )
        .unwrap();
        let out = call(indexed, map).unwrap();
        assert_eq!(out.shape, vec![2, 1, 3]);
        assert_eq!(out.dtype, NumericDType::F64);
        assert_eq!(out.data, vec![1.0, 0.0, 0.0, 128.0 / 255.0, 0.0, 1.0]);
    }

    #[test]
    fn rejects_out_of_range_indices() {
        let indexed = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let map = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let err = block_on(ind2rgb_builtin(
            Value::Tensor(indexed),
            Value::Tensor(map),
            Vec::new(),
        ))
        .unwrap_err();
        assert!(err.message().contains("outside the colormap"));
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
        assert!(codes.contains(&"RM.IND2RGB.INDEX_OUT_OF_RANGE"));
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
