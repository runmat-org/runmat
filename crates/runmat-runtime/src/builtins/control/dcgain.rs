//! DC gain for SISO transfer-function models.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::tf_model::{output_complex_scalar, TfModel};
use crate::builtins::control::type_resolvers::dcgain_type;
use crate::BuiltinResult;

const DCGAIN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "gain",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Steady-state gain evaluated at s=0 for continuous-time or z=1 for discrete-time.",
}];
const DCGAIN_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO tf model.",
}];
const DCGAIN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "gain = dcgain(sys)",
    inputs: &DCGAIN_INPUTS,
    outputs: &DCGAIN_OUTPUT,
}];
const DCGAIN_ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.DCGAIN.INVALID_ARGUMENT",
        identifier: Some("RunMat:dcgain:InvalidArgument"),
        when: "Input does not match supported invocation forms.",
        message: "dcgain: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.DCGAIN.INVALID_MODEL",
        identifier: Some("RunMat:dcgain:InvalidModel"),
        when: "Input system is not a valid SISO tf object.",
        message: "dcgain: invalid model",
    },
    BuiltinErrorDescriptor {
        code: "RM.DCGAIN.UNSUPPORTED_MODEL",
        identifier: Some("RunMat:dcgain:UnsupportedModel"),
        when: "Model form is not supported by the current implementation.",
        message: "dcgain: unsupported model",
    },
    BuiltinErrorDescriptor {
        code: "RM.DCGAIN.INTERNAL",
        identifier: Some("RunMat:dcgain:Internal"),
        when: "Gain evaluation failed internally.",
        message: "dcgain: internal error",
    },
];
pub const DCGAIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DCGAIN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DCGAIN_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::dcgain")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "dcgain",
    op_kind: GpuOpKind::Custom("control-dc-gain"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "dcgain evaluates host-side transfer-function metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::dcgain")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "dcgain",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "dcgain is scalar model analysis and is not fused.",
};

#[runtime_builtin(
    name = "dcgain",
    category = "control",
    summary = "Evaluate steady-state gain of SISO transfer-function models.",
    keywords = "dcgain,control system,steady state,transfer function,tf",
    type_resolver(dcgain_type),
    descriptor(crate::builtins::control::dcgain::DCGAIN_DESCRIPTOR),
    builtin_path = "crate::builtins::control::dcgain"
)]
async fn dcgain_builtin(sys: Value) -> BuiltinResult<Value> {
    let model = TfModel::from_value_async(sys, "dcgain").await?;
    Ok(output_complex_scalar(model.dc_gain()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;

    #[test]
    fn continuous_dcgain_evaluates_at_zero() {
        let sys = block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Num(2.0),
                Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap()),
            ],
        ))
        .expect("tf");
        let Value::Num(gain) = block_on(dcgain_builtin(sys)).expect("dcgain") else {
            panic!("expected scalar gain");
        };
        assert!((gain - 2.0 / 3.0).abs() < 1.0e-12);
    }
}
