//! `gray2rgb` compatibility helper for replicating grayscale images into RGB.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::image::color::common;
use crate::builtins::image::color::type_resolvers::gray2rgb_type;
use crate::BuiltinResult;

const NAME: &str = "gray2rgb";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::color::gray2rgb")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("gray2rgb"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host implementation replicates grayscale planes while preserving logical image dtype metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::color::gray2rgb")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not fused yet; output rank changes from MxN to MxNx3.",
};

#[runtime_builtin(
    name = "gray2rgb",
    category = "image/color",
    summary = "Replicate a grayscale image into an RGB truecolor image.",
    keywords = "gray2rgb,gray,grayscale,rgb,image",
    accel = "sink",
    type_resolver(gray2rgb_type),
    builtin_path = "crate::builtins::image::color::gray2rgb"
)]
async fn gray2rgb_builtin(gray: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(common::builtin_error(
            NAME,
            "gray2rgb: too many input arguments",
        ));
    }
    let tensor = common::gather_tensor(NAME, gray).await?;
    let (rows, cols) = common::grayscale_shape(&tensor, NAME)?;
    let pixels = rows * cols;
    let mut data = vec![0.0; pixels * 3];
    for channel in 0..3 {
        data[channel * pixels..(channel + 1) * pixels].copy_from_slice(&tensor.data);
    }
    let out = common::tensor_with_dtype(data, vec![rows, cols, 3], tensor.dtype, NAME)?;
    Ok(common::image_value_from_tensor(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{NumericDType, Tensor};

    fn call(value: Value) -> BuiltinResult<Value> {
        block_on(gray2rgb_builtin(value, Vec::new()))
    }

    #[test]
    fn replicates_grayscale_planes() {
        let gray =
            Tensor::new_with_dtype(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], NumericDType::U8).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(gray)).expect("gray2rgb") else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape, vec![2, 2, 3]);
        assert_eq!(out.dtype, NumericDType::U8);
        assert_eq!(
            out.data,
            vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn rejects_truecolor_input() {
        let rgb = Tensor::new(vec![1.0; 12], vec![2, 2, 3]).unwrap();
        let err = call(Value::Tensor(rgb)).unwrap_err();
        assert!(err.message().contains("expected an MxN grayscale image"));
    }
}
