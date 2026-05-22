//! MATLAB-compatible `im2double` image class conversion.

use runmat_builtins::{IntValue, NumericDType, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::image::color::common;
use crate::builtins::image::color::type_resolvers::same_shape_type;
use crate::builtins::introspection::class::class_name_for_value;
use crate::BuiltinResult;

const NAME: &str = "im2double";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::color::im2double")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Image class scaling runs on the host today so uint8/uint16 image semantics are preserved.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::color::im2double")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not fused yet; integer image dtype metadata is host-side.",
};

#[runtime_builtin(
    name = "im2double",
    category = "image/color",
    summary = "Convert image data to double precision, scaling integer images into [0,1].",
    keywords = "im2double,image,convert,double,uint8,uint16",
    accel = "sink",
    type_resolver(same_shape_type),
    builtin_path = "crate::builtins::image::color::im2double"
)]
async fn im2double_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(common::builtin_error(
            NAME,
            "im2double: too many input arguments",
        ));
    }
    let value = common::gather_value(NAME, &value).await?;
    match value {
        Value::Tensor(tensor) => Ok(common::image_value_from_tensor(im2double_tensor(tensor)?)),
        Value::LogicalArray(array) => {
            let tensor = tensor::logical_to_tensor(&array)
                .map_err(|err| common::builtin_error(NAME, format!("im2double: {err}")))?;
            Ok(common::image_value_from_tensor(tensor))
        }
        Value::Int(IntValue::U8(v)) => Ok(Value::Num(v as f64 / 255.0)),
        Value::Int(IntValue::U16(v)) => Ok(Value::Num(v as f64 / 65535.0)),
        Value::Int(v) => Ok(Value::Num(v.to_f64())),
        Value::Num(v) => Ok(Value::Num(v)),
        Value::Bool(v) => Ok(Value::Num(if v { 1.0 } else { 0.0 })),
        other => Err(common::builtin_error(
            NAME,
            format!(
                "im2double: unsupported input type {}",
                class_name_for_value(&other)
            ),
        )),
    }
}

fn im2double_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let scale = common::dtype_max(tensor.dtype);
    let data = if matches!(tensor.dtype, NumericDType::U8 | NumericDType::U16) {
        tensor.data.iter().map(|&value| value / scale).collect()
    } else {
        tensor.data
    };
    common::tensor_with_dtype(data, tensor.shape, NumericDType::F64, NAME)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::LogicalArray;

    fn call(value: Value) -> Value {
        block_on(im2double_builtin(value, Vec::new())).expect("im2double")
    }

    #[test]
    fn scales_uint8_tensor_to_unit_double() {
        let input =
            Tensor::new_with_dtype(vec![0.0, 128.0, 255.0], vec![1, 3], NumericDType::U8).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(input)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.dtype, NumericDType::F64);
        assert_eq!(out.data[0], 0.0);
        assert!((out.data[1] - 128.0 / 255.0).abs() < 1e-12);
        assert_eq!(out.data[2], 1.0);
    }

    #[test]
    fn scales_uint16_scalar() {
        assert_eq!(call(Value::Int(IntValue::U16(65535))), Value::Num(1.0));
    }

    #[test]
    fn preserves_float_values_and_shape() {
        let input =
            Tensor::new_with_dtype(vec![-0.25, 0.0, 0.5, 1.25], vec![2, 2], NumericDType::F32)
                .unwrap();
        let Value::Tensor(out) = call(Value::Tensor(input)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.dtype, NumericDType::F64);
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.data, vec![-0.25, 0.0, 0.5, 1.25]);
    }

    #[test]
    fn converts_logical_array_to_double_zeros_and_ones() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let Value::Tensor(out) = call(Value::LogicalArray(logical)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.dtype, NumericDType::F64);
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.data, vec![1.0, 0.0, 1.0, 0.0]);
    }
}
