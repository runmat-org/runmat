//! MATLAB-compatible `hsv2rgb` conversion.

use runmat_builtins::{NumericDType, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::image::color::common;
use crate::builtins::image::color::type_resolvers::same_shape_type;
use crate::BuiltinResult;

const NAME: &str = "hsv2rgb";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::color::hsv2rgb")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("hsv2rgb"),
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::color::hsv2rgb")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not fused yet; conversion operates across HSV channels.",
};

#[runtime_builtin(
    name = "hsv2rgb",
    category = "image/color",
    summary = "Convert HSV values to RGB color space.",
    keywords = "hsv2rgb,hsv,rgb,color,image,colormap",
    accel = "sink",
    type_resolver(same_shape_type),
    builtin_path = "crate::builtins::image::color::hsv2rgb"
)]
async fn hsv2rgb_builtin(hsv: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(common::builtin_error(
            NAME,
            "hsv2rgb: too many input arguments",
        ));
    }
    let tensor = common::gather_tensor(NAME, hsv).await?;
    let layout = common::color_layout(&tensor, NAME)?;
    let dtype = common::image_output_dtype(tensor.dtype);
    let mut data = vec![0.0; tensor.data.len()];
    for pixel in 0..layout.pixels() {
        let h =
            common::unit_value(tensor.data[layout.index(pixel, 0)], tensor.dtype).rem_euclid(1.0);
        let s = common::clamp01(common::unit_value(
            tensor.data[layout.index(pixel, 1)],
            tensor.dtype,
        ));
        let v = common::clamp01(common::unit_value(
            tensor.data[layout.index(pixel, 2)],
            tensor.dtype,
        ));
        let (r, g, b) = hsv_to_rgb_unit(h, s, v);
        data[layout.index(pixel, 0)] = cast_float(r, dtype);
        data[layout.index(pixel, 1)] = cast_float(g, dtype);
        data[layout.index(pixel, 2)] = cast_float(b, dtype);
    }
    Ok(common::image_value_from_tensor(common::tensor_with_dtype(
        data,
        layout.output_shape(),
        dtype,
        NAME,
    )?))
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
    use futures::executor::block_on;
    use runmat_builtins::Tensor;

    #[test]
    fn converts_red_hsv_to_rgb() {
        let hsv = Tensor::new(vec![0.0, 1.0, 1.0], vec![1, 1, 3]).unwrap();
        let Value::Tensor(out) =
            block_on(hsv2rgb_builtin(Value::Tensor(hsv), Vec::new())).expect("hsv2rgb")
        else {
            panic!("expected tensor");
        };
        assert!((out.data[0] - 1.0).abs() < 1e-12);
        assert!((out.data[1] - 0.0).abs() < 1e-12);
        assert!((out.data[2] - 0.0).abs() < 1e-12);
    }
}
