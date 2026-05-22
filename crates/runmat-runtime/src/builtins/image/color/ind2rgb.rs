//! MATLAB-compatible `ind2rgb` indexed-image conversion.

use runmat_builtins::{NumericDType, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::image::color::common;
use crate::builtins::image::color::type_resolvers::ind2rgb_type;
use crate::BuiltinResult;

const NAME: &str = "ind2rgb";

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
    builtin_path = "crate::builtins::image::color::ind2rgb"
)]
async fn ind2rgb_builtin(indexed: Value, map: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(common::builtin_error(
            NAME,
            "ind2rgb: too many input arguments",
        ));
    }
    let indexed = common::gather_tensor(NAME, indexed).await?;
    let map = common::gather_tensor(NAME, map).await?;
    let layout = common::color_layout(&map, NAME)?;
    let common::ColorLayout::Colormap { rows: map_rows } = layout else {
        return Err(common::builtin_error(
            NAME,
            "ind2rgb: map must be an Nx3 colormap",
        ));
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
        let map_index = map_index(raw_index, indexed.dtype, map_rows)?;
        for channel in 0..3 {
            let value = common::unit_value(map.data[layout.index(map_index, channel)], map.dtype);
            data[pixel + pixels * channel] = if matches!(dtype, NumericDType::F32) {
                (value as f32) as f64
            } else {
                value
            };
        }
    }

    Ok(common::image_value_from_tensor(common::tensor_with_dtype(
        data, shape, dtype, NAME,
    )?))
}

fn map_index(value: f64, dtype: NumericDType, map_rows: usize) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(common::builtin_error(
            NAME,
            "ind2rgb: index values must be finite",
        ));
    }
    let index = if matches!(dtype, NumericDType::U8 | NumericDType::U16) {
        value.round() as isize
    } else {
        value.round() as isize - 1
    };
    if index < 0 || index as usize >= map_rows {
        return Err(common::builtin_error(
            NAME,
            format!("ind2rgb: index {} is outside the colormap", value),
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
}
