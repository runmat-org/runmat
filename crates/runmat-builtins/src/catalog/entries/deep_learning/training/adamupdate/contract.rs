use crate::*;

const INPUTS: [BuiltinParamDescriptor; 9] = [
    BuiltinParamDescriptor { name: "params", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None, description: "Nonempty real double/single parameter array or host dlarray; plain resident gpuArray uses the provider path." },
    BuiltinParamDescriptor { name: "grad", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None, description: "Finite gradient with the parameter shape." },
    BuiltinParamDescriptor { name: "averageGrad", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None, description: "First-moment state with matching shape, or empty numeric state to initialize to zero." },
    BuiltinParamDescriptor { name: "averageSqGrad", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None, description: "Second-moment state with matching shape, or empty numeric state to initialize to zero." },
    BuiltinParamDescriptor { name: "iteration", ty: BuiltinParamType::NumericScalar, arity: BuiltinParamArity::Required, default: None, description: "Positive exact integer iteration count, decoded without a floating-point round trip." },
    BuiltinParamDescriptor { name: "learnRate", ty: BuiltinParamType::NumericScalar, arity: BuiltinParamArity::Optional, default: Some("0.001"), description: "Positive finite learning rate." },
    BuiltinParamDescriptor { name: "gradDecay", ty: BuiltinParamType::NumericScalar, arity: BuiltinParamArity::Optional, default: Some("0.9"), description: "Finite first-moment decay in [0, 1)." },
    BuiltinParamDescriptor { name: "sqGradDecay", ty: BuiltinParamType::NumericScalar, arity: BuiltinParamArity::Optional, default: Some("0.999"), description: "Finite second-moment decay in [0, 1)." },
    BuiltinParamDescriptor { name: "epsilon", ty: BuiltinParamType::NumericScalar, arity: BuiltinParamArity::Optional, default: Some("1e-8"), description: "Positive finite denominator offset." },
];
const OUTPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "params",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Updated parameters, preserving supported input representation.",
    },
    BuiltinParamDescriptor {
        name: "averageGrad",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Updated first-moment state.",
    },
    BuiltinParamDescriptor {
        name: "averageSqGrad",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Updated second-moment state.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[params, averageGrad, averageSqGrad] = adamupdate(params, grad, averageGrad, averageSqGrad, iteration, learnRate, gradDecay, sqGradDecay, epsilon)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const ADAMUPDATE_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.INVALID_INPUT",
    identifier: Some("RunMat:deepLearning:InvalidInput"),
    when: "Argument or requested-output count is invalid, a numeric class or shape is unsupported, a control is out of range, or an update produces a nonfinite value.",
    message: "adamupdate: invalid parameters, optimizer state, controls, or output count",
};
pub const ADAMUPDATE_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.UNSUPPORTED",
    identifier: Some("RunMat:deepLearning:Unsupported"),
    when: "An explicit gpuArray input has no supported resident provider path, or a provider-owned input cannot be updated and its permitted automatic fallback cannot materialize it.",
    message: "adamupdate: provider-backed optimizer update is unavailable",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [ADAMUPDATE_INVALID_INPUT, ADAMUPDATE_UNSUPPORTED];
pub const ADAMUPDATE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const ADAMUPDATE_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "adamupdate-typed-integer-scalar",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Typed integer scalar operands and optimizer controls extend the documented single/double adamupdate input classes.",
    error_identifier: Some("RunMat:compatibility:AdamupdateIntegerScalarExtension"),
};
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [ADAMUPDATE_INTEGER_EXTENSION];

const ITERATION: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "iteration",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "A positive typed integer iteration is a RunMat extension, range-checked and decoded exactly before bias correction.",
}];
const SCALAR_CONTROLS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "scalar parameters, gradients, moments, learnRate, gradDecay, sqGradDecay, epsilon",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed integer scalar operands and controls are converted at the floating-point Adam arithmetic boundary; integer tensors are not supported.",
}];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [BuiltinIntegerCapabilityDescriptor {
    form: "adamupdate(params, grad, averageGrad, averageSqGrad, integer_iteration, ...)",
    inputs: &ITERATION,
    computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::FunctionSpecific,
    overload: BuiltinIntegerOverloadKind::Multiple,
    notes: "The integer controls do not make integer parameter tensors admissible; parameter and gradient tensors remain floating point.",
}, BuiltinIntegerCapabilityDescriptor {
    form: "adamupdate(integer_scalar_or_control, ...)",
    inputs: &SCALAR_CONTROLS,
    computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::FunctionSpecific,
    overload: BuiltinIntegerOverloadKind::Multiple,
    notes: "Only scalar typed integers are admitted outside the structural iteration argument; integer tensors reject.",
}];
