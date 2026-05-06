//! MATLAB-compatible default-D65 `rgb2lab` conversion.

use runmat_builtins::{NumericDType, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::image::color::common;
use crate::builtins::image::color::type_resolvers::same_shape_type;
use crate::BuiltinResult;

const NAME: &str = "rgb2lab";
const XN: f64 = 0.95047;
const YN: f64 = 1.0;
const ZN: f64 = 1.08883;
const EPSILON: f64 = 216.0 / 24389.0;
const KAPPA: f64 = 24389.0 / 27.0;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::color::rgb2lab")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("rgb2lab"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host implementation uses the default sRGB/D65 conversion path.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::color::rgb2lab")]
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
    name = "rgb2lab",
    category = "image/color",
    summary = "Convert sRGB image or colormap values to CIE L*a*b* using D65 white.",
    keywords = "rgb2lab,rgb,lab,cielab,color,image,colormap",
    accel = "sink",
    type_resolver(same_shape_type),
    builtin_path = "crate::builtins::image::color::rgb2lab"
)]
async fn rgb2lab_builtin(rgb: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(common::builtin_error(
            NAME,
            "rgb2lab: too many input arguments",
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
        let (l, a, bstar) = rgb_to_lab_unit(r, g, b);
        data[layout.index(pixel, 0)] = cast_float(l, dtype);
        data[layout.index(pixel, 1)] = cast_float(a, dtype);
        data[layout.index(pixel, 2)] = cast_float(bstar, dtype);
    }
    Ok(common::image_value_from_tensor(common::tensor_with_dtype(
        data,
        layout.output_shape(),
        dtype,
        NAME,
    )?))
}

pub(crate) fn rgb_to_lab_unit(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let r = srgb_to_linear(r);
    let g = srgb_to_linear(g);
    let b = srgb_to_linear(b);
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    let fx = lab_f(x / XN);
    let fy = lab_f(y / YN);
    let fz = lab_f(z / ZN);
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

pub(crate) fn srgb_to_linear(value: f64) -> f64 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn lab_f(value: f64) -> f64 {
    if value > EPSILON {
        value.cbrt()
    } else {
        (KAPPA * value + 16.0) / 116.0
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

    fn call(tensor: Tensor) -> BuiltinResult<Tensor> {
        let Value::Tensor(out) =
            block_on(rgb2lab_builtin(Value::Tensor(tensor), Vec::new())).expect("rgb2lab")
        else {
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

    #[test]
    fn converts_white_to_lab_reference() {
        let rgb = Tensor::new(vec![1.0, 1.0, 1.0], vec![1, 1, 3]).unwrap();
        let out = call(rgb).unwrap();
        assert_close(out.data[0], 100.0, 1e-4);
        assert_close(out.data[1], 0.0, 1e-3);
        assert_close(out.data[2], 0.0, 1e-3);
    }

    #[test]
    fn converts_rgb_colormap_to_lab_references() {
        let rgb = Tensor::new(
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            vec![4, 3],
        )
        .unwrap();
        let out = call(rgb).unwrap();
        assert_eq!(out.shape, vec![4, 3]);
        let expected = vec![
            0.0, 53.2408, 87.7347, 32.2970, 0.0, 80.0925, -86.1827, 79.1875, 0.0, 67.2032, 83.1793,
            -107.8602,
        ];
        for (actual, expected) in out.data.iter().zip(expected) {
            assert_close(*actual, expected, 1e-3);
        }
    }

    #[test]
    fn scales_uint8_rgb_before_lab_conversion() {
        let rgb =
            Tensor::new_with_dtype(vec![255.0, 0.0, 0.0], vec![1, 1, 3], NumericDType::U8).unwrap();
        let out = call(rgb).unwrap();
        assert_eq!(out.dtype, NumericDType::F64);
        assert_close(out.data[0], 53.2408, 1e-3);
        assert_close(out.data[1], 80.0925, 1e-3);
        assert_close(out.data[2], 67.2032, 1e-3);
    }
}
