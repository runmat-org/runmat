//! MATLAB-compatible `isrow` builtin.

use crate::builtins::common::shape::{effective_rank, value_dimensions};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, ResolveContext, Type,
    Value,
};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::introspection::isrow")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isrow",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Reads shape metadata without provider access and returns a host logical scalar.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::introspection::isrow"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isrow",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query; not fused into GPU kernels.",
};

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when input has one row.",
}];

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to inspect.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isrow(A)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

const ERRORS: [BuiltinErrorDescriptor; 0] = [];

pub const ISROW_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const ISROW_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "isrow is a universal shape predicate; integer class and values are irrelevant and resident shape metadata is read without gathering payload data.",
};

fn bool_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Bool
}

#[runtime_builtin(
    name = "isrow",
    category = "array/introspection",
    summary = "Return true when an array has exactly one row.",
    keywords = "isrow,row vector,shape,metadata",
    accel = "metadata",
    type_resolver(bool_type),
    descriptor(crate::builtins::array::introspection::isrow::ISROW_DESCRIPTOR),
    integer_audit(crate::builtins::array::introspection::isrow::ISROW_INTEGER_AUDIT),
    builtin_path = "crate::builtins::array::introspection::isrow"
)]
async fn isrow_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let dims = value_dimensions(&value).await?;
    Ok(Value::Bool(
        effective_rank(&dims) <= 2 && dims.first().copied().unwrap_or(1) == 1,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;

    #[test]
    fn detects_single_row_shape() {
        let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let col = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        assert_eq!(
            block_on(isrow_builtin(Value::Tensor(row))).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            block_on(isrow_builtin(Value::Tensor(col))).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            block_on(isrow_builtin(Value::Num(1.0))).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn ignores_trailing_singletons_but_rejects_effective_higher_rank() {
        let row = Tensor::new(vec![1.0, 2.0], vec![1, 2, 1]).unwrap();
        assert_eq!(
            block_on(isrow_builtin(Value::Tensor(row))).unwrap(),
            Value::Bool(true)
        );
        let higher = Tensor::new(vec![1.0, 2.0], vec![1, 1, 2]).unwrap();
        assert_eq!(
            block_on(isrow_builtin(Value::Tensor(higher))).unwrap(),
            Value::Bool(false)
        );
    }
}
