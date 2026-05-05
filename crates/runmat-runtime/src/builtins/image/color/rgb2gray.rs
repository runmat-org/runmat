//! MATLAB-compatible `rgb2gray` conversion.

use runmat_builtins::{NumericDType, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::image::color::common;
use crate::builtins::image::color::type_resolvers::rgb2gray_type;
use crate::BuiltinResult;

const NAME: &str = "rgb2gray";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::color::rgb2gray")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("rgb2gray"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host implementation preserves integer image dtype semantics; a float GPU provider can be added independently.",
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
    summary = "Convert an RGB image to grayscale using luminance weights.",
    keywords = "rgb2gray,rgb,gray,grayscale,luminance,image",
    accel = "sink",
    type_resolver(rgb2gray_type),
    builtin_path = "crate::builtins::image::color::rgb2gray"
)]
async fn rgb2gray_builtin(rgb: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(common::builtin_error(
            NAME,
            "rgb2gray: too many input arguments",
        ));
    }
    let tensor = common::gather_tensor(NAME, rgb).await?;
    let out = rgb2gray_tensor(&tensor)?;
    Ok(common::image_value_from_tensor(out))
}

fn rgb2gray_tensor(rgb: &Tensor) -> BuiltinResult<Tensor> {
    let common::ColorLayout::Truecolor { rows, cols } = common::truecolor_layout(rgb, NAME)? else {
        unreachable!();
    };
    let pixels = rows * cols;
    let mut data = vec![0.0; pixels];
    for (pixel, out) in data.iter_mut().enumerate() {
        let r = common::unit_value(rgb.data[pixel], rgb.dtype);
        let g = common::unit_value(rgb.data[pixel + pixels], rgb.dtype);
        let b = common::unit_value(rgb.data[pixel + 2 * pixels], rgb.dtype);
        let gray = 0.2989 * r + 0.5870 * g + 0.1140 * b;
        *out = common::unit_to_dtype(gray, rgb.dtype);
    }
    let dtype = match rgb.dtype {
        NumericDType::U8 | NumericDType::U16 | NumericDType::F32 => rgb.dtype,
        NumericDType::F64 => NumericDType::F64,
    };
    common::tensor_with_dtype(data, vec![rows, cols], dtype, NAME)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn call(value: Value) -> Value {
        block_on(rgb2gray_builtin(value, Vec::new())).expect("rgb2gray")
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
    fn preserves_2d_shape() {
        let rgb = Tensor::new(vec![1.0; 12], vec![2, 2, 3]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(rgb)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape, vec![2, 2]);
        assert!(out.data.iter().all(|v| (*v - 0.9999).abs() < 1e-12));
    }

    #[test]
    fn rgb2gray_is_registered_with_dispatcher() {
        let rgb = Tensor::new(vec![1.0, 1.0, 1.0], vec![1, 1, 3]).unwrap();
        let result = block_on(crate::call_builtin_async(NAME, &[Value::Tensor(rgb)]))
            .expect("rgb2gray registered");
        assert!(matches!(result, Value::Num(value) if (value - 0.9999).abs() < 1e-12));
    }
}
