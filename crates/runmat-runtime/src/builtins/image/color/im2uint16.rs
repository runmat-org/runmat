//! MATLAB-compatible `im2uint16` image class conversion.

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

const NAME: &str = "im2uint16";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::color::im2uint16")]
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
    notes: "Host conversion preserves MATLAB image scaling semantics for uint16 outputs.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::color::im2uint16")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not fused yet; uint16 image dtype metadata is host-side.",
};

#[runtime_builtin(
    name = "im2uint16",
    category = "image/color",
    summary = "Convert image data to uint16 using MATLAB image scaling rules.",
    keywords = "im2uint16,image,convert,uint16,double,uint8",
    accel = "sink",
    type_resolver(same_shape_type),
    builtin_path = "crate::builtins::image::color::im2uint16"
)]
async fn im2uint16_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(common::builtin_error(
            NAME,
            "im2uint16: too many input arguments",
        ));
    }
    let value = common::gather_value(NAME, &value).await?;
    match value {
        Value::Tensor(tensor) => Ok(common::image_value_from_tensor(im2uint16_tensor(tensor)?)),
        Value::LogicalArray(array) => {
            let tensor = tensor::logical_to_tensor(&array)
                .map_err(|err| common::builtin_error(NAME, format!("im2uint16: {err}")))?;
            Ok(common::image_value_from_tensor(im2uint16_tensor(tensor)?))
        }
        Value::Int(IntValue::U16(v)) => Ok(Value::Int(IntValue::U16(v))),
        Value::Int(IntValue::U8(v)) => Ok(Value::Int(IntValue::U16((v as u16) * 257))),
        Value::Int(v) => Ok(Value::Int(IntValue::U16(
            common::clamp_round(v.to_f64(), 65535.0) as u16,
        ))),
        Value::Num(v) => Ok(Value::Int(IntValue::U16(common::unit_to_dtype(
            common::clamp01(v),
            NumericDType::U16,
        ) as u16))),
        Value::Bool(v) => Ok(Value::Int(IntValue::U16(if v { 65535 } else { 0 }))),
        other => Err(common::builtin_error(
            NAME,
            format!(
                "im2uint16: unsupported input type {}",
                class_name_for_value(&other)
            ),
        )),
    }
}

fn im2uint16_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data = match tensor.dtype {
        NumericDType::U16 => tensor.data,
        NumericDType::U8 => tensor.data.iter().map(|&value| value * 257.0).collect(),
        NumericDType::F32 | NumericDType::F64 => tensor
            .data
            .iter()
            .map(|&value| common::unit_to_dtype(common::clamp01(value), NumericDType::U16))
            .collect(),
    };
    common::tensor_with_dtype(data, tensor.shape, NumericDType::U16, NAME)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::LogicalArray;

    fn call(value: Value) -> Value {
        block_on(im2uint16_builtin(value, Vec::new())).expect("im2uint16")
    }

    #[test]
    fn scales_double_to_uint16_image_range() {
        assert_eq!(call(Value::Num(1.0)), Value::Int(IntValue::U16(65535)));
    }

    #[test]
    fn expands_uint8_tensor_to_uint16_range() {
        let input = Tensor::new_with_dtype(vec![0.0, 255.0], vec![1, 2], NumericDType::U8).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(input)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.dtype, NumericDType::U16);
        assert_eq!(out.data, vec![0.0, 65535.0]);
    }

    #[test]
    fn clamps_and_rounds_float_tensor_to_uint16() {
        let input = Tensor::new(vec![-0.1, 0.0, 0.5, 1.0, 1.2, f64::NAN], vec![2, 3]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(input)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.dtype, NumericDType::U16);
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(out.data, vec![0.0, 0.0, 32768.0, 65535.0, 65535.0, 0.0]);
    }

    #[test]
    fn converts_logical_array_to_uint16_extrema() {
        let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).unwrap();
        let Value::Tensor(out) = call(Value::LogicalArray(logical)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.dtype, NumericDType::U16);
        assert_eq!(out.data, vec![0.0, 65535.0, 65535.0, 0.0]);
    }
}
