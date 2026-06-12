//! MATLAB-compatible SISO `feedback` interconnection for transfer functions.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::tf_model::{scalar_f64, two_models_ordered};
use crate::builtins::control::type_resolvers::feedback_type;
use crate::{BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "feedback";

const FEEDBACK_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Closed-loop SISO transfer-function object.",
}];
const FEEDBACK_PARAM_SYS1: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sys1",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Forward-path SISO transfer-function model.",
};
const FEEDBACK_PARAM_SYS2: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sys2",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("1"),
    description: "Feedback-path SISO transfer function or scalar gain.",
};
const FEEDBACK_PARAM_SIGN: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sign",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("-1"),
    description: "-1 for negative feedback or +1 for positive feedback.",
};
const FEEDBACK_INPUT_SYS: [BuiltinParamDescriptor; 1] = [FEEDBACK_PARAM_SYS1];
const FEEDBACK_INPUT_SYS_OTHER: [BuiltinParamDescriptor; 2] =
    [FEEDBACK_PARAM_SYS1, FEEDBACK_PARAM_SYS2];
const FEEDBACK_INPUT_SYS_OTHER_SIGN: [BuiltinParamDescriptor; 3] = [
    FEEDBACK_PARAM_SYS1,
    FEEDBACK_PARAM_SYS2,
    FEEDBACK_PARAM_SIGN,
];
const FEEDBACK_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "sys = feedback(sys1)",
        inputs: &FEEDBACK_INPUT_SYS,
        outputs: &FEEDBACK_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "sys = feedback(sys1, sys2)",
        inputs: &FEEDBACK_INPUT_SYS_OTHER,
        outputs: &FEEDBACK_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "sys = feedback(sys1, sys2, sign)",
        inputs: &FEEDBACK_INPUT_SYS_OTHER_SIGN,
        outputs: &FEEDBACK_OUTPUT,
    },
];
const FEEDBACK_ERRORS: [BuiltinErrorDescriptor; 5] = [
    BuiltinErrorDescriptor {
        code: "RM.FEEDBACK.INVALID_ARGUMENT",
        identifier: Some("RunMat:feedback:InvalidArgument"),
        when: "Inputs do not match supported feedback invocation forms.",
        message: "feedback: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.FEEDBACK.INVALID_MODEL",
        identifier: Some("RunMat:feedback:InvalidModel"),
        when: "Input systems are not supported SISO transfer-function models.",
        message: "feedback: invalid model",
    },
    BuiltinErrorDescriptor {
        code: "RM.FEEDBACK.INVALID_SIGN",
        identifier: Some("RunMat:feedback:InvalidSign"),
        when: "Feedback sign is not -1 or +1.",
        message: "feedback: sign must be -1 or +1",
    },
    BuiltinErrorDescriptor {
        code: "RM.FEEDBACK.UNSUPPORTED_MODEL",
        identifier: Some("RunMat:feedback:UnsupportedModel"),
        when: "A supported-looking model uses unsupported delays or incompatible sample times.",
        message: "feedback: unsupported model",
    },
    BuiltinErrorDescriptor {
        code: "RM.FEEDBACK.INTERNAL",
        identifier: Some("RunMat:feedback:Internal"),
        when: "Closed-loop transfer-function assembly failed.",
        message: "feedback: internal error",
    },
];
pub const FEEDBACK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FEEDBACK_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FEEDBACK_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::feedback")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "feedback",
    op_kind: GpuOpKind::Custom("control-feedback-interconnection"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "SISO transfer-function interconnection runs on host-side metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::feedback")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "feedback",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "feedback creates a transfer-function object and terminates numeric fusion chains.",
};

fn feedback_error(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = crate::build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "feedback",
    category = "control",
    summary = "Form SISO feedback interconnections for transfer-function models.",
    keywords = "feedback,control system,closed loop,transfer function,tf",
    type_resolver(feedback_type),
    descriptor(crate::builtins::control::feedback::FEEDBACK_DESCRIPTOR),
    builtin_path = "crate::builtins::control::feedback"
)]
async fn feedback_builtin(sys1: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 2 {
        return Err(feedback_error(
            "feedback: expected feedback(sys1), feedback(sys1, sys2), or feedback(sys1, sys2, sign)",
            &FEEDBACK_ERRORS[0],
        ));
    }
    let sys2 = rest.first().cloned().unwrap_or(Value::Num(1.0));
    let sign = match rest.get(1) {
        Some(value) => {
            let gathered = crate::dispatcher::gather_if_needed_async(value).await?;
            scalar_f64(&gathered, "sign", BUILTIN_NAME)?
        }
        None => -1.0,
    };
    if sign != -1.0 && sign != 1.0 {
        return Err(feedback_error(
            "feedback: sign must be -1 or +1",
            &FEEDBACK_ERRORS[2],
        ));
    }

    let (forward, feedback_path) = two_models_ordered(sys1, sys2, BUILTIN_NAME).await?;
    forward
        .feedback(&feedback_path, sign)?
        .to_value(BUILTIN_NAME)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;

    fn tf(num: Vec<f64>, den: Vec<f64>) -> Value {
        block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Tensor(Tensor::new(num.clone(), vec![1, num.len()]).unwrap()),
                Value::Tensor(Tensor::new(den.clone(), vec![1, den.len()]).unwrap()),
            ],
        ))
        .expect("tf")
    }

    fn coeff(value: &Value, field: &str) -> Vec<f64> {
        let Value::Object(object) = value else {
            panic!("expected tf object");
        };
        let Value::Tensor(tensor) = object.properties.get(field).expect(field) else {
            panic!("expected tensor");
        };
        tensor.data.clone()
    }

    #[test]
    fn unity_negative_feedback_forms_closed_loop() {
        let g = tf(vec![2.0], vec![1.0, 3.0]);
        let out = block_on(feedback_builtin(g, vec![Value::Num(1.0)])).expect("feedback");
        assert_eq!(coeff(&out, "Numerator"), vec![2.0]);
        assert_eq!(coeff(&out, "Denominator"), vec![1.0, 5.0]);
    }

    #[test]
    fn positive_feedback_subtracts_loop_gain() {
        let g = tf(vec![2.0], vec![1.0, 3.0]);
        let out = block_on(feedback_builtin(g, vec![Value::Num(1.0), Value::Num(1.0)]))
            .expect("feedback");
        assert_eq!(coeff(&out, "Numerator"), vec![2.0]);
        assert_eq!(coeff(&out, "Denominator"), vec![1.0, 1.0]);
    }
}
