//! DC gain for SISO transfer-function models.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, Value,
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

const DCGAIN_INTEGER_SYS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "sys",
    classes: &[],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "dcgain accepts an LTI model object. Integer coefficient support belongs to model constructors and does not make an integer array a valid sys input.",
}];

pub const DCGAIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "k = dcgain(integer_sys)",
        inputs: &DCGAIN_INTEGER_SYS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer input is inapplicable at the dynamic-system object boundary and rejects before provider access. Coefficients have already crossed the constructor's model-numeric boundary.",
    }];

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
    integer_capabilities(crate::builtins::control::dcgain::DCGAIN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::control::dcgain"
)]
async fn dcgain_builtin(sys: Value) -> BuiltinResult<Value> {
    let model = TfModel::from_value(sys, "dcgain")?;
    Ok(output_complex_scalar(model.dc_gain()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerStorage, Tensor};

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

    #[test]
    fn dcgain_returns_infinity_for_continuous_and_discrete_integrators() {
        for (numerator, denominator, sample_time, expected_negative) in [
            (
                Value::Num(2.0),
                Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
                None,
                false,
            ),
            (
                Value::Num(-2.0),
                Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
                None,
                true,
            ),
            (
                Value::Num(3.0),
                Value::Tensor(Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap()),
                Some(0.1),
                false,
            ),
        ] {
            let mut arguments = vec![numerator, denominator];
            if let Some(sample_time) = sample_time {
                arguments.push(Value::Num(sample_time));
            }
            let sys = block_on(crate::call_builtin_async("tf", &arguments)).expect("tf");
            let Value::Num(gain) = block_on(dcgain_builtin(sys)).expect("dcgain") else {
                panic!("expected real infinite gain");
            };
            assert!(gain.is_infinite());
            assert_eq!(gain.is_sign_negative(), expected_negative);
        }
    }

    #[test]
    fn dcgain_rejects_integer_sys_without_conflating_integer_coefficients() {
        for sys in [
            Value::Int(IntValue::I64(i64::MAX)),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap(),
            ),
        ] {
            let error = block_on(dcgain_builtin(sys)).expect_err("integer sys rejection");
            assert_eq!(error.identifier(), Some("RunMat:dcgain:InvalidModel"));
        }

        let sys = block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Int(IntValue::U8(2)),
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::I16(vec![1, 3]), vec![1, 2]).unwrap(),
                ),
            ],
        ))
        .expect("tf with integer coefficients");
        let Value::Num(gain) = block_on(dcgain_builtin(sys)).expect("dcgain model object") else {
            panic!("real scalar gain");
        };
        assert!((gain - 2.0 / 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn dcgain_resident_nonobject_rejects_before_provider_access() {
        let handle = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX - 610,
            descriptor: Default::default(),
        }
        .with_numeric_descriptor(
            runmat_accelerate_api::NumericElementType::U64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        );
        let error = block_on(dcgain_builtin(Value::GpuTensor(handle)))
            .expect_err("resident nonobject rejection");
        assert_eq!(error.identifier(), Some("RunMat:dcgain:InvalidModel"));
    }

    #[test]
    fn dcgain_integer_capability_is_an_object_boundary_rejection() {
        assert_eq!(DCGAIN_INTEGER_CAPABILITIES.len(), 1);
        assert!(DCGAIN_INTEGER_CAPABILITIES[0].inputs[0].classes.is_empty());
        assert_eq!(
            DCGAIN_INTEGER_CAPABILITIES[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::Rejected
        );
    }
}
