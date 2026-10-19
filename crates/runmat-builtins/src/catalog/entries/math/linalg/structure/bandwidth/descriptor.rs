use crate::*;

pub const BANDWIDTH_OUTPUT_BW: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "lower",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Lower bandwidth scalar.",
    },
    BuiltinParamDescriptor {
        name: "upper",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Upper bandwidth scalar.",
    },
];

pub const BANDWIDTH_OUTPUT_SCALAR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Selected lower or upper bandwidth scalar.",
}];

pub const BANDWIDTH_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix.",
}];

pub const BANDWIDTH_INPUTS_SELECTOR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix.",
    },
    BuiltinParamDescriptor {
        name: "selector",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Selector string: \"lower\" or \"upper\".",
    },
];

pub const BANDWIDTH_INPUTS_TF: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "sys",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero-delay scalar SISO tf model.",
    },
    BuiltinParamDescriptor {
        name: "dbdrop",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("-3"),
        description: "Finite negative gain drop in decibels.",
    },
];

pub const BANDWIDTH_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "[lower, upper] = bandwidth(A)",
        inputs: &BANDWIDTH_INPUTS,
        outputs: &BANDWIDTH_OUTPUT_BW,
    },
    BuiltinSignatureDescriptor {
        label: "b = bandwidth(A, selector)",
        inputs: &BANDWIDTH_INPUTS_SELECTOR,
        outputs: &BANDWIDTH_OUTPUT_SCALAR,
    },
    BuiltinSignatureDescriptor {
        label: "w = bandwidth(sys, dbdrop)",
        inputs: &BANDWIDTH_INPUTS_TF,
        outputs: &BANDWIDTH_OUTPUT_SCALAR,
    },
];

pub const BANDWIDTH_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BANDWIDTH.INVALID_ARGUMENT",
    identifier: Some("RunMat:bandwidth:InvalidArgument"),
    when: "Selector argument is invalid or argument count exceeds supported forms.",
    message: "bandwidth: invalid argument",
};

pub const BANDWIDTH_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BANDWIDTH.INVALID_INPUT",
    identifier: Some("RunMat:bandwidth:InvalidInput"),
    when: "Input type/shape cannot be processed as a numeric or logical 2-D matrix.",
    message: "bandwidth: invalid input",
};

pub const BANDWIDTH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BANDWIDTH.INTERNAL",
    identifier: Some("RunMat:bandwidth:Internal"),
    when: "Runtime fails while constructing intermediate tensors or values.",
    message: "bandwidth: internal runtime failure",
};

pub const BANDWIDTH_ERROR_UNSUPPORTED_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BANDWIDTH.UNSUPPORTED_MODEL",
    identifier: Some("RunMat:bandwidth:UnsupportedModel"),
    when: "A transfer function is well formed but outside the current zero-delay scalar SISO physical-frequency subset.",
    message: "bandwidth: unsupported model",
};

pub const BANDWIDTH_ERRORS: [BuiltinErrorDescriptor; 4] = [
    BANDWIDTH_ERROR_INVALID_ARGUMENT,
    BANDWIDTH_ERROR_INVALID_INPUT,
    BANDWIDTH_ERROR_INTERNAL,
    BANDWIDTH_ERROR_UNSUPPORTED_MODEL,
];

pub const BANDWIDTH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BANDWIDTH_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BANDWIDTH_ERRORS,
};

pub const BANDWIDTH_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "bandwidth-integer-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "bandwidth with typed-integer input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BandwidthIntegerInputExtension"),
    };
pub const BANDWIDTH_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "bandwidth-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "bandwidth with logical input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BandwidthLogicalInputExtension"),
    };
pub const BANDWIDTH_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    BANDWIDTH_INTEGER_INPUT_EXTENSION,
    BANDWIDTH_LOGICAL_INPUT_EXTENSION,
];

pub const BANDWIDTH_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented matrix domain is single/double; RunMat mode additionally accepts every real integer class and paired complex-integer storage.",
    }];
pub const BANDWIDTH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[lower, upper] = bandwidth(integer_A, ...)",
        inputs: &BANDWIDTH_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Host integer storage is scanned exactly for nonzero structure. Resident integer input is gated before provider dispatch, gathered exactly, and returns host double scalar metadata.",
    }];
