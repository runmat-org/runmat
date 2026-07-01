//! Zero extraction for SISO transfer-function control models.

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
use crate::builtins::control::type_resolvers::zero_type;
use crate::BuiltinResult;

const ZERO_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "z",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zeros of the SISO tf model as a column vector.",
}];
const ZERO_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO tf model.",
}];
const ZERO_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "z = zero(sys)",
    inputs: &ZERO_INPUTS,
    outputs: &ZERO_OUTPUT,
}];
const ZERO_ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.ZERO.INVALID_MODEL",
        identifier: Some("RunMat:zero:InvalidModel"),
        when: "Input system is not a valid SISO tf object.",
        message: "zero: invalid model",
    },
    BuiltinErrorDescriptor {
        code: "RM.ZERO.UNSUPPORTED_MODEL",
        identifier: Some("RunMat:zero:UnsupportedModel"),
        when: "Model form is unsupported.",
        message: "zero: unsupported model",
    },
    BuiltinErrorDescriptor {
        code: "RM.ZERO.INVALID_ARGUMENT",
        identifier: Some("RunMat:zero:InvalidArgument"),
        when: "Model metadata or arguments are malformed.",
        message: "zero: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.ZERO.INTERNAL",
        identifier: Some("RunMat:zero:Internal"),
        when: "Root calculation or output construction failed.",
        message: "zero: internal error",
    },
];
pub const ZERO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ZERO_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ZERO_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::zero")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "zero",
    op_kind: GpuOpKind::Custom("control-zeros"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "zero computes roots from host-side transfer-function metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::zero")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "zero",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "zero is model analysis and is not fused.",
};

#[runtime_builtin(
    name = "zero",
    category = "control",
    summary = "Return zeros of SISO transfer-function models.",
    keywords = "zero,zeros,control system,transfer function,tf",
    type_resolver(zero_type),
    descriptor(crate::builtins::control::zero::ZERO_DESCRIPTOR),
    builtin_path = "crate::builtins::control::zero"
)]
async fn zero_builtin(sys: Value) -> BuiltinResult<Value> {
    let model = TfModel::from_value_async(sys, "zero").await?;
    output_complex_column(model.zeros()?, "zero")
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;

    #[test]
    fn zero_returns_roots_of_numerator() {
        let sys = block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0], vec![1, 3]).unwrap()),
                Value::Tensor(Tensor::new(vec![1.0, 4.0], vec![1, 2]).unwrap()),
            ],
        ))
        .expect("tf");
        let Value::Tensor(zeros) = block_on(zero_builtin(sys)).expect("zero") else {
            panic!("expected real zeros");
        };
        assert_eq!(zeros.shape, vec![2, 1]);
        assert!(zeros.data.iter().any(|z| (*z + 1.0).abs() < 1.0e-8));
        assert!(zeros.data.iter().any(|z| (*z + 2.0).abs() < 1.0e-8));
    }

    #[test]
    fn zero_returns_complex_conjugate_roots() {
        let sys = block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Tensor(Tensor::new(vec![1.0, 0.0, 1.0], vec![1, 3]).unwrap()),
                Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            ],
        ))
        .expect("tf");
        let Value::ComplexTensor(zeros) = block_on(zero_builtin(sys)).expect("zero") else {
            panic!("expected complex zeros");
        };
        assert_eq!(zeros.shape, vec![2, 1]);
        assert!(zeros.data.iter().all(|(re, _)| re.abs() < 1.0e-8));
        assert!(zeros.data.iter().any(|(_, im)| (*im - 1.0).abs() < 1.0e-8));
        assert!(zeros.data.iter().any(|(_, im)| (*im + 1.0).abs() < 1.0e-8));
    }

    #[test]
    fn zero_static_gain_returns_empty_column() {
        let sys = block_on(crate::call_builtin_async(
            "tf",
            &[Value::Num(5.0), Value::Num(2.0)],
        ))
        .expect("tf");
        let Value::Tensor(zeros) = block_on(zero_builtin(sys)).expect("zero") else {
            panic!("expected real empty column");
        };
        assert_eq!(zeros.shape, vec![0, 1]);
        assert!(zeros.data.is_empty());
    }
}
