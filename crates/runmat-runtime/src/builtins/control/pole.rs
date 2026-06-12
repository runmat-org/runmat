//! Pole extraction for SISO transfer-function models.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::tf_model::{output_complex_column, TfModel};
use crate::builtins::control::type_resolvers::pole_type;
use crate::BuiltinResult;

const POLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Poles of the SISO tf model as a column vector.",
}];
const POLE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO tf model.",
}];
const POLE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "p = pole(sys)",
    inputs: &POLE_INPUTS,
    outputs: &POLE_OUTPUT,
}];
const POLE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.POLE.INVALID_MODEL",
        identifier: Some("RunMat:pole:InvalidModel"),
        when: "Input system is not a valid SISO tf object.",
        message: "pole: invalid model",
    },
    BuiltinErrorDescriptor {
        code: "RM.POLE.UNSUPPORTED_MODEL",
        identifier: Some("RunMat:pole:UnsupportedModel"),
        when: "Model form is unsupported.",
        message: "pole: unsupported model",
    },
    BuiltinErrorDescriptor {
        code: "RM.POLE.INTERNAL",
        identifier: Some("RunMat:pole:Internal"),
        when: "Root calculation or output construction failed.",
        message: "pole: internal error",
    },
];
pub const POLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &POLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &POLE_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::pole")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "pole",
    op_kind: GpuOpKind::Custom("control-poles"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "pole computes roots from host-side transfer-function metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::pole")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "pole",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "pole is model analysis and is not fused.",
};

#[runtime_builtin(
    name = "pole",
    category = "control",
    summary = "Return poles of SISO transfer-function models.",
    keywords = "pole,poles,control system,stability,transfer function,tf",
    type_resolver(pole_type),
    descriptor(crate::builtins::control::pole::POLE_DESCRIPTOR),
    builtin_path = "crate::builtins::control::pole"
)]
async fn pole_builtin(sys: Value) -> BuiltinResult<Value> {
    let model = TfModel::from_value_async(sys, "pole").await?;
    output_complex_column(model.poles()?, "pole")
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;

    #[test]
    fn pole_returns_roots_of_denominator() {
        let sys = block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Num(1.0),
                Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0], vec![1, 3]).unwrap()),
            ],
        ))
        .expect("tf");
        let Value::Tensor(poles) = block_on(pole_builtin(sys)).expect("pole") else {
            panic!("expected real poles");
        };
        assert_eq!(poles.shape, vec![2, 1]);
        assert!(poles.data.iter().any(|p| (*p + 1.0).abs() < 1.0e-8));
        assert!(poles.data.iter().any(|p| (*p + 2.0).abs() < 1.0e-8));
    }
}
