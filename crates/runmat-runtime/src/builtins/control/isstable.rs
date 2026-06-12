//! Stability test for SISO transfer-function models.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::tf_model::TfModel;
use crate::builtins::control::type_resolvers::isstable_type;
use crate::BuiltinResult;

const ISSTABLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when all continuous poles are in the open left-half plane or all discrete poles are inside the unit circle.",
}];
const ISSTABLE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO tf model.",
}];
const ISSTABLE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isstable(sys)",
    inputs: &ISSTABLE_INPUTS,
    outputs: &ISSTABLE_OUTPUT,
}];
const ISSTABLE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.ISSTABLE.INVALID_MODEL",
        identifier: Some("RunMat:isstable:InvalidModel"),
        when: "Input system is not a valid SISO tf object.",
        message: "isstable: invalid model",
    },
    BuiltinErrorDescriptor {
        code: "RM.ISSTABLE.UNSUPPORTED_MODEL",
        identifier: Some("RunMat:isstable:UnsupportedModel"),
        when: "Model form is unsupported.",
        message: "isstable: unsupported model",
    },
    BuiltinErrorDescriptor {
        code: "RM.ISSTABLE.INTERNAL",
        identifier: Some("RunMat:isstable:Internal"),
        when: "Pole calculation failed.",
        message: "isstable: internal error",
    },
];
pub const ISSTABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISSTABLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISSTABLE_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::isstable")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isstable",
    op_kind: GpuOpKind::Custom("control-stability"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "isstable analyzes host-side transfer-function metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::isstable")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isstable",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "isstable returns scalar model metadata and is not fused.",
};

#[runtime_builtin(
    name = "isstable",
    category = "control",
    summary = "Test stability of SISO transfer-function models.",
    keywords = "isstable,control system,stability,poles,tf",
    type_resolver(isstable_type),
    descriptor(crate::builtins::control::isstable::ISSTABLE_DESCRIPTOR),
    builtin_path = "crate::builtins::control::isstable"
)]
async fn isstable_builtin(sys: Value) -> BuiltinResult<Value> {
    let model = TfModel::from_value_async(sys, "isstable").await?;
    Ok(Value::Bool(model.is_stable()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;

    #[test]
    fn stable_continuous_model_returns_true() {
        let sys = block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Num(1.0),
                Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0], vec![1, 3]).unwrap()),
            ],
        ))
        .expect("tf");
        assert_eq!(
            block_on(isstable_builtin(sys)).expect("isstable"),
            Value::Bool(true)
        );
    }
}
