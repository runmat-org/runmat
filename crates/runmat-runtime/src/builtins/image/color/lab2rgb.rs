//! MATLAB-compatible default-D65 `lab2rgb` conversion.

use runmat_builtins::{NumericDType, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::image::color::common;
use crate::builtins::image::color::type_resolvers::same_shape_type;
use crate::BuiltinResult;

const NAME: &str = "lab2rgb";
const XN: f64 = 0.95047;
const YN: f64 = 1.0;
const ZN: f64 = 1.08883;
const EPSILON: f64 = 216.0 / 24389.0;
const KAPPA: f64 = 24389.0 / 27.0;

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
    notes: "Host implementation uses the default D65/sRGB conversion path and clips out-of-gamut RGB values.",
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
    summary = "Convert CIE L*a*b* values to sRGB using D65 white.",
    keywords = "lab2rgb,lab,cielab,rgb,color,image,colormap",
    accel = "sink",
    type_resolver(same_shape_type),
    builtin_path = "crate::builtins::image::color::lab2rgb"
)]
async fn lab2rgb_builtin(lab: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(common::builtin_error(
            NAME,
            "lab2rgb: too many input arguments",
        ));
    }
    let tensor = common::gather_tensor(NAME, lab).await?;
    let layout = common::color_layout(&tensor, NAME)?;
    let dtype = match tensor.dtype {
        NumericDType::F32 => NumericDType::F32,
        _ => NumericDType::F64,
    };
    let mut data = vec![0.0; tensor.data.len()];
    for pixel in 0..layout.pixels() {
        let l = tensor.data[layout.index(pixel, 0)];
        let a = tensor.data[layout.index(pixel, 1)];
        let b = tensor.data[layout.index(pixel, 2)];
        let (r, g, blue) = lab_to_rgb_unit(l, a, b);
        data[layout.index(pixel, 0)] = cast_float(r, dtype);
        data[layout.index(pixel, 1)] = cast_float(g, dtype);
        data[layout.index(pixel, 2)] = cast_float(blue, dtype);
    }
    Ok(common::image_value_from_tensor(common::tensor_with_dtype(
        data,
        layout.output_shape(),
        dtype,
        NAME,
    )?))
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
    (
        common::clamp01(linear_to_srgb(r)),
        common::clamp01(linear_to_srgb(g)),
        common::clamp01(linear_to_srgb(blue)),
    )
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
    use futures::executor::block_on;
    use runmat_builtins::Tensor;

    #[test]
    fn converts_white_lab_to_rgb() {
        let lab = Tensor::new(vec![100.0, 0.0, 0.0], vec![1, 1, 3]).unwrap();
        let Value::Tensor(out) =
            block_on(lab2rgb_builtin(Value::Tensor(lab), Vec::new())).expect("lab2rgb")
        else {
            panic!("expected tensor");
        };
        assert!((out.data[0] - 1.0).abs() < 1e-4);
        assert!((out.data[1] - 1.0).abs() < 1e-4);
        assert!((out.data[2] - 1.0).abs() < 1e-4);
    }
}
