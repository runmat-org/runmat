use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Options struct for optimization solvers.",
}];
const PAIRS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option field name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option value.",
    },
    BuiltinParamDescriptor {
        name: "additional_pairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional option name/value pairs, admitted as complete pairs.",
    },
];
const OLD_PAIRS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "oldopts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Existing options struct to update.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Option field name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option value(s) and additional name/value pairs.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "options = optimset('fminsearch')",
        inputs: &[BuiltinParamDescriptor {
            name: "solver",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Supported solver name or handle (fminsearch).",
        }],
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "options = optimset(oldopts, newopts)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "oldopts",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Existing structure or [].",
            },
            BuiltinParamDescriptor {
                name: "newopts",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Structure of nonempty replacement values.",
            },
        ],
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "options = optimset()",
        inputs: &[],
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "options = optimset(name, value, ...)",
        inputs: &PAIRS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "options = optimset(oldopts, name, value, ...)",
        inputs: &OLD_PAIRS,
        outputs: &OUTPUT,
    },
];
pub const OPTIMSET_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMSET.INVALID_ARGUMENT",
    identifier: Some("RunMat:optimset:InvalidArgument"),
    when: "Name/value argument grammar is invalid.",
    message: "optimset: invalid argument",
};
pub const OPTIMSET_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMSET.INVALID_INPUT",
    identifier: Some("RunMat:optimset:InvalidInput"),
    when: "Option field names are not valid string scalars or payload materialization fails.",
    message: "optimset: invalid input",
};
pub const OPTIMSET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        OPTIMSET_ERROR_INVALID_ARGUMENT,
        OPTIMSET_ERROR_INVALID_INPUT,
    ],
};

pub const OPTIMSET_INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "optimset-integer-option",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "optimset with native-class integer option payloads is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:OptimsetIntegerOptionExtension"),
    };
pub const OPTIMSET_RESIDENT_OPTION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "optimset-resident-option",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "optimset preserving explicit gpuArray option payloads is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:OptimsetResidentOptionExtension"),
    };
pub const OPTIMSET_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    OPTIMSET_INTEGER_OPTION_EXTENSION,
    OPTIMSET_RESIDENT_OPTION_EXTENSION,
];

const INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "option value", classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Documented numeric options use single or double; RunMat preserves exact integer payloads in the returned struct.",
}];
pub const OPTIMSET_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "options = optimset(___, name, integer_value, ___)",
    inputs: &INTEGER_INPUT,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::PreserveInput,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::StructuralParameter,
    notes: "Typed integer payloads are gated and retained exactly; automatic resident payloads gather while explicit resident preservation is separately gated.",
}];
