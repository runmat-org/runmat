//! MATLAB-compatible `rgb2hsv` conversion.

use runmat_builtins::{NumericDType, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::image::color::common;
use crate::builtins::image::color::type_resolvers::same_shape_type;
use crate::BuiltinResult;

const NAME: &str = "rgb2hsv";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::color::rgb2hsv")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("rgb2hsv"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host implementation; float RGB/HSV GPU providers are tracked separately.",
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
    summary = "Convert RGB values to HSV color space.",
    keywords = "rgb2hsv,rgb,hsv,color,image,colormap",
    accel = "sink",
    type_resolver(same_shape_type),
    builtin_path = "crate::builtins::image::color::rgb2hsv"
)]
async fn rgb2hsv_builtin(rgb: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(common::builtin_error(
            NAME,
            "rgb2hsv: too many input arguments",
        ));
    }
    let tensor = common::gather_tensor(NAME, rgb).await?;
    let layout = common::color_layout(&tensor, NAME)?;
    let dtype = common::image_output_dtype(tensor.dtype);
    let mut data = vec![0.0; tensor.data.len()];
    for pixel in 0..layout.pixels() {
        let r = common::clamp01(common::unit_value(
            tensor.data[layout.index(pixel, 0)],
            tensor.dtype,
        ));
        let g = common::clamp01(common::unit_value(
            tensor.data[layout.index(pixel, 1)],
            tensor.dtype,
        ));
        let b = common::clamp01(common::unit_value(
            tensor.data[layout.index(pixel, 2)],
            tensor.dtype,
        ));
        let (h, s, v) = rgb_to_hsv_unit(r, g, b);
        data[layout.index(pixel, 0)] = cast_float(h, dtype);
        data[layout.index(pixel, 1)] = cast_float(s, dtype);
        data[layout.index(pixel, 2)] = cast_float(v, dtype);
    }
    Ok(common::image_value_from_tensor(common::tensor_with_dtype(
        data,
        layout.output_shape(),
        dtype,
        NAME,
    )?))
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
    use futures::executor::block_on;
    use runmat_builtins::Tensor;

    #[test]
    fn converts_red_to_hsv() {
        let rgb = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 1, 3]).unwrap();
        let Value::Tensor(out) =
            block_on(rgb2hsv_builtin(Value::Tensor(rgb), Vec::new())).expect("rgb2hsv")
        else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape, vec![1, 1, 3]);
        assert!((out.data[0] - 0.0).abs() < 1e-12);
        assert!((out.data[1] - 1.0).abs() < 1e-12);
        assert!((out.data[2] - 1.0).abs() < 1e-12);
    }
}
