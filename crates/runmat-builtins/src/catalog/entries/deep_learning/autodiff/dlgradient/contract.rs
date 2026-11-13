use crate::*;

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "loss",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar dlarray produced on the active dlfeval tape.",
    },
    BuiltinParamDescriptor {
        name: "targets",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "One or more traced dlarray values, learnables trees, or dlnetwork objects.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "gradients",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "One gradient value or tree per requested target.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "g = dlgradient(loss, target)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[g1, g2, ...] = dlgradient(loss, target1, target2, ...)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
];

pub const DLGRADIENT_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.INVALID_INPUT",
    identifier: Some("RunMat:deepLearning:InvalidInput"),
    when: "No target is supplied; more outputs than targets are requested; loss is not a scalar traced dlarray; or a target is not traced or has an unsupported structure.",
    message: "dlgradient requires a scalar traced loss and valid traced targets",
};
pub const DLGRADIENT_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.UNSUPPORTED",
    identifier: Some("RunMat:deepLearning:Unsupported"),
    when: "A provider-resident dlarray would require unavailable resident tape kernels, or a requested gradient reduction is outside the scalar-broadcast and supported-layer subset.",
    message: "dlgradient cannot differentiate this provider or operation form",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [DLGRADIENT_INVALID_INPUT, DLGRADIENT_UNSUPPORTED];
pub const DLGRADIENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_TARGET: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "loss, targets",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The host autodiff tape accepts traced double or single payloads; native integer payloads are never silently widened through floating-point differentiation.",
}];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "dlgradient(integer_loss_or_target)",
        inputs: &INTEGER_TARGET,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Explicit conversion to double or single is required before constructing differentiable dlarray Data.",
    }];
