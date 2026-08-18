//! MATLAB-compatible SISO `feedback` interconnection for transfer functions.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
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

const SINGLE_SYSTEM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feedback-single-system",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "feedback(sys1) with an implicit unity feedback path is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeedbackSingleSystemExtension"),
};
const SCALAR_FORWARD_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feedback-scalar-forward-system",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "feedback with a scalar forward-path system is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeedbackScalarForwardSystemExtension"),
};
const SCALAR_SYSTEMS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feedback-scalar-scalar-systems",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "feedback with scalar forward and feedback-path systems is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeedbackScalarScalarSystemsExtension"),
};
const INTEGER_GAIN_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feedback-integer-scalar-gain",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "feedback with a typed-integer scalar system gain is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeedbackIntegerScalarGainExtension"),
};
const INTEGER_SIGN_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feedback-integer-sign",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "feedback with a typed-integer sign is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeedbackIntegerSignExtension"),
};
const LOGICAL_NUMERIC_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feedback-logical-numeric-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "feedback with a logical numeric system or sign is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeedbackLogicalNumericInputExtension"),
};
const RESIDENT_NUMERIC_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feedback-resident-numeric-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "feedback with a resident numeric system or sign is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeedbackResidentNumericInputExtension"),
};
const SINGLE_NUMERIC_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feedback-single-numeric-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "feedback with a native-single numeric system or sign is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeedbackSingleNumericInputExtension"),
};

pub const EXTENSIONS: [BuiltinExtensionDescriptor; 8] = [
    SINGLE_SYSTEM_EXTENSION,
    SCALAR_FORWARD_EXTENSION,
    SCALAR_SYSTEMS_EXTENSION,
    INTEGER_GAIN_EXTENSION,
    INTEGER_SIGN_EXTENSION,
    LOGICAL_NUMERIC_EXTENSION,
    RESIDENT_NUMERIC_EXTENSION,
    SINGLE_NUMERIC_EXTENSION,
];

const INTEGER_GAIN_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "sys1",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A typed-integer forward gain is independently gated, then converted once to the floating/complex tf coefficient domain.",
    },
    BuiltinIntegerInputCapability {
        name: "sys2",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A typed-integer feedback-path gain is independently gated, then converted once to the floating/complex tf coefficient domain.",
    },
];
const INTEGER_SIGN_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sign",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The typed-integer sign is independently gated and must be exactly -1 or +1; unsigned classes can express only +1.",
    }];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "sys = feedback(integer_gain, sys2) or feedback(sys1, integer_gain)",
        inputs: &INTEGER_GAIN_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "All eight typed integer classes are RunMat-only scalar gains. The result is a host tf object; int64 and uint64 values outside binary64's exact integer range can round at the explicit floating boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "sys = feedback(sys1, sys2, integer_sign)",
        inputs: &INTEGER_SIGN_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "All eight typed integer classes are admitted as a RunMat-only sign control, but the exact valid set remains -1 or +1. The result is a host tf object.",
    },
];

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
    extensions(crate::builtins::control::feedback::EXTENSIONS),
    integer_capabilities(crate::builtins::control::feedback::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::control::feedback"
)]
async fn feedback_builtin(sys1: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 2 {
        return Err(feedback_error(
            "feedback: expected feedback(sys1), feedback(sys1, sys2), or feedback(sys1, sys2, sign)",
            &FEEDBACK_ERRORS[0],
        ));
    }
    ensure_extensions_enabled(&sys1, &rest)?;
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

fn ensure_extensions_enabled(sys1: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if rest.is_empty() {
        ensure_extension(&SINGLE_SYSTEM_EXTENSION)?;
    }

    let sys2 = rest.first();
    let sys1_is_scalar = is_scalar_system_value(sys1);
    if sys1_is_scalar && sys2.is_some_and(is_scalar_system_value) {
        ensure_extension(&SCALAR_SYSTEMS_EXTENSION)?;
    }
    if sys1_is_scalar {
        ensure_extension(&SCALAR_FORWARD_EXTENSION)?;
    }

    if is_typed_integer_value(sys1) || sys2.is_some_and(is_typed_integer_value) {
        ensure_extension(&INTEGER_GAIN_EXTENSION)?;
    }
    if rest.get(1).is_some_and(is_typed_integer_value) {
        ensure_extension(&INTEGER_SIGN_EXTENSION)?;
    }
    if is_logical_value(sys1) || rest.iter().any(is_logical_value) {
        ensure_extension(&LOGICAL_NUMERIC_EXTENSION)?;
    }
    if is_resident_value(sys1) || rest.iter().any(is_resident_value) {
        ensure_extension(&RESIDENT_NUMERIC_EXTENSION)?;
    }
    if is_single_numeric_value(sys1) || rest.iter().any(is_single_numeric_value) {
        ensure_extension(&SINGLE_NUMERIC_EXTENSION)?;
    }
    Ok(())
}

fn ensure_extension(extension: &'static BuiltinExtensionDescriptor) -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(extension, BUILTIN_NAME)
}

fn is_scalar_system_value(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => true,
        Value::Tensor(tensor) => tensor.len() == 1,
        Value::ComplexTensor(tensor) => tensor.len() == 1,
        Value::LogicalArray(logical) => logical.data.len() == 1,
        Value::GpuTensor(handle) => handle.shape.iter().product::<usize>() == 1,
        _ => false,
    }
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_single_numeric_value(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == runmat_builtins::NumericDType::F32)
        || matches!(value, Value::ComplexTensor(tensor) if tensor.numeric_dtype() == runmat_builtins::NumericDType::F32)
}

fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

fn is_resident_value(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{
        ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray, NumericDType,
        Tensor,
    };

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
        tensor.materialize_f64().clone()
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

    #[test]
    fn integer_metadata_covers_gain_and_sign_for_all_classes() {
        assert_eq!(INTEGER_CAPABILITIES.len(), 2);
        assert_eq!(INTEGER_CAPABILITIES[0].inputs.len(), 2);
        assert_eq!(INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
        assert_eq!(INTEGER_CAPABILITIES[0].inputs[1].classes.len(), 8);
        assert_eq!(INTEGER_CAPABILITIES[1].inputs[0].classes.len(), 8);
        assert_eq!(
            INTEGER_CAPABILITIES[0].computation_domain,
            BuiltinIntegerComputationDomain::FloatingPoint
        );
        assert_eq!(
            INTEGER_CAPABILITIES[1].computation_domain,
            BuiltinIntegerComputationDomain::Structural
        );
    }

    #[test]
    fn integer_gain_and_sign_support_all_eight_classes_in_runmat_mode() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let gains = [
            IntValue::I8(1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(1),
        ];
        for gain in gains {
            let out = block_on(feedback_builtin(
                tf(vec![2.0], vec![1.0, 3.0]),
                vec![Value::Int(gain.clone()), Value::Int(gain)],
            ))
            .expect("typed-integer gain and sign");
            assert_eq!(coeff(&out, "Denominator"), vec![1.0, 1.0]);
        }
    }

    #[test]
    fn integer_tensor_gain_uses_explicit_binary64_boundary() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let wide = (1_u64 << 53) + 1;
        let gain = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1])
            .expect("integer scalar tensor");
        let out = block_on(feedback_builtin(
            tf(vec![1.0], vec![1.0]),
            vec![Value::Tensor(gain)],
        ))
        .expect("wide integer gain");
        assert_eq!(coeff(&out, "Denominator"), vec![(wide as f64) + 1.0]);
    }

    #[test]
    fn integer_sign_keeps_the_exact_plus_or_minus_one_contract() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let negative = block_on(feedback_builtin(
            tf(vec![2.0], vec![1.0, 3.0]),
            vec![Value::Num(1.0), Value::Int(IntValue::I64(-1))],
        ))
        .expect("negative integer sign");
        assert_eq!(coeff(&negative, "Denominator"), vec![1.0, 5.0]);

        let error = block_on(feedback_builtin(
            tf(vec![2.0], vec![1.0, 3.0]),
            vec![Value::Num(1.0), Value::Int(IntValue::U64(0))],
        ))
        .expect_err("zero is not a feedback sign");
        assert_eq!(error.identifier(), FEEDBACK_ERRORS[2].identifier);
    }

    #[test]
    fn logical_numeric_inputs_are_a_separate_runmat_extension() {
        let g = tf(vec![2.0], vec![1.0, 3.0]);
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(feedback_builtin(g.clone(), vec![Value::Bool(true)]))
                .expect_err("strict mode rejects logical gain");
            assert_eq!(
                error.identifier(),
                LOGICAL_NUMERIC_EXTENSION.error_identifier
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let logical = LogicalArray::new(vec![1], vec![1, 1]).expect("logical scalar");
            let out = block_on(feedback_builtin(g, vec![Value::LogicalArray(logical)]))
                .expect("logical gain");
            assert_eq!(coeff(&out, "Denominator"), vec![1.0, 5.0]);
        }
    }

    #[test]
    fn runmat_only_forms_have_stable_independent_guards() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);

        let documented = block_on(feedback_builtin(
            tf(vec![2.0], vec![1.0, 3.0]),
            vec![Value::Num(1.0), Value::Num(-1.0)],
        ))
        .expect("tf plus ordinary double gain and sign remains admitted");
        assert_eq!(coeff(&documented, "Denominator"), vec![1.0, 5.0]);

        let error = block_on(feedback_builtin(tf(vec![1.0], vec![1.0]), vec![]))
            .expect_err("one-argument form");
        assert_eq!(error.identifier(), SINGLE_SYSTEM_EXTENSION.error_identifier);

        let error = block_on(feedback_builtin(
            Value::Num(1.0),
            vec![tf(vec![1.0], vec![1.0])],
        ))
        .expect_err("scalar forward form");
        assert_eq!(
            error.identifier(),
            SCALAR_FORWARD_EXTENSION.error_identifier
        );

        let error = block_on(feedback_builtin(Value::Num(1.0), vec![Value::Num(1.0)]))
            .expect_err("scalar-scalar form");
        assert_eq!(
            error.identifier(),
            SCALAR_SYSTEMS_EXTENSION.error_identifier
        );

        let error = block_on(feedback_builtin(
            tf(vec![1.0], vec![1.0]),
            vec![Value::Int(IntValue::I16(1))],
        ))
        .expect_err("integer gain");
        assert_eq!(error.identifier(), INTEGER_GAIN_EXTENSION.error_identifier);

        let error = block_on(feedback_builtin(
            tf(vec![1.0], vec![1.0]),
            vec![Value::Num(1.0), Value::Int(IntValue::I8(-1))],
        ))
        .expect_err("integer sign");
        assert_eq!(error.identifier(), INTEGER_SIGN_EXTENSION.error_identifier);

        let complex_integer = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I16(vec![1]), IntegerStorage::I16(vec![0]))
                .expect("paired integer storage"),
            vec![1, 1],
        )
        .expect("complex integer scalar");
        let error = block_on(feedback_builtin(
            tf(vec![1.0], vec![1.0]),
            vec![Value::ComplexTensor(complex_integer)],
        ))
        .expect_err("complex integer gain");
        assert_eq!(error.identifier(), INTEGER_GAIN_EXTENSION.error_identifier);

        let single = Tensor::new_with_dtype(vec![1.0], vec![1, 1], NumericDType::F32)
            .expect("single scalar");
        let error = block_on(feedback_builtin(
            tf(vec![1.0], vec![1.0]),
            vec![Value::Tensor(single)],
        ))
        .expect_err("single gain");
        assert_eq!(
            error.identifier(),
            SINGLE_NUMERIC_EXTENSION.error_identifier
        );
    }

    #[test]
    fn resident_gate_runs_before_provider_access() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 9_404_001,
            descriptor: Default::default(),
        });
        let error = block_on(feedback_builtin(tf(vec![1.0], vec![1.0]), vec![resident]))
            .expect_err("resident gate");
        assert_eq!(
            error.identifier(),
            RESIDENT_NUMERIC_EXTENSION.error_identifier
        );
    }
}
