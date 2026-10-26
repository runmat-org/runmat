use crate::*;

pub const OPTIMOPTIONS_INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "optimoptions-integer-option",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "optimoptions with native-class integer option values is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:OptimoptionsIntegerOptionExtension"),
    };
pub const OPTIMOPTIONS_RESIDENT_OPTION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "optimoptions-resident-option",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "optimoptions with explicit gpuArray option values is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:OptimoptionsResidentOptionExtension"),
    };
pub const OPTIMOPTIONS_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    OPTIMOPTIONS_INTEGER_OPTION_EXTENSION,
    OPTIMOPTIONS_RESIDENT_OPTION_EXTENSION,
];

const INTEGER_FLOATING_OPTION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "floating option value",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Optimization tolerances are documented in floating classes; typed integers are gated and must convert exactly.",
    }];
const INTEGER_STRUCTURAL_OPTION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "count, bandwidth, flag or sparsity-pattern option value",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed iteration/evaluation/PCG/node/feasible-point counts and finite bandwidths are RunMat-only controls checked exactly against the platform size range and published in their native signed/unsigned class; scalar integer arrays normalize to native scalars. Typed flags are checked as 0/1 before logical publication. Native integer HessPattern carriers retain their class/storage; the model owner validates their pattern meaning.",
    }];
pub const OPTIMOPTIONS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "options = optimoptions(___, floating_name, integer_value, ___)", inputs: &INTEGER_FLOATING_OPTION_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "RunMat's supported compatibility subset normalizes accepted integer tolerances to double only after an exact representability check." },
    BuiltinIntegerCapabilityDescriptor { form: "options = optimoptions(___, structural_name, integer_value, ___)", inputs: &INTEGER_STRUCTURAL_OPTION_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Counts use exact integer-to-usize parsing and retain native class; 0/1 controls become logical fields. Ordinary floating/default count values remain double. This describes RunMat count/publication behavior, not unverified MATLAB integer-overload parity; RunMat returns a struct rather than MATLAB's options object." },
];

const OPTIMOPTIONS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Options struct for optimization solvers.",
}];

const OPTIMOPTIONS_INPUTS_SOLVER: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "solver",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A supported solver name or its named function handle.",
}];

const OPTIMOPTIONS_INPUTS_SOLVER_PAIRS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "solver",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "A supported solver name or its named function handle.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
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

const OPTIMOPTIONS_INPUTS_EXISTING_PAIRS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "oldopts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Existing options struct to update.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Option field name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option value(s), additional name/value pairs, or another options struct.",
    },
];

const OPTIMOPTIONS_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "options = optimoptions(solver)",
        inputs: &OPTIMOPTIONS_INPUTS_SOLVER,
        outputs: &OPTIMOPTIONS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "options = optimoptions(solver, name, value, ...)",
        inputs: &OPTIMOPTIONS_INPUTS_SOLVER_PAIRS,
        outputs: &OPTIMOPTIONS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "options = optimoptions(oldopts, name, value, ...)",
        inputs: &OPTIMOPTIONS_INPUTS_EXISTING_PAIRS,
        outputs: &OPTIMOPTIONS_OUTPUT,
    },
];

pub const OPTIMOPTIONS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMOPTIONS.INVALID_ARGUMENT",
    identifier: Some("RunMat:optimoptions:InvalidArgument"),
    when: "Argument grammar does not match supported optimoptions forms.",
    message: "optimoptions: invalid argument",
};
pub const OPTIMOPTIONS_ERROR_INVALID_SOLVER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMOPTIONS.INVALID_SOLVER",
    identifier: Some("RunMat:optimoptions:InvalidSolver"),
    when: "The solver argument is not one of the supported optimization builtins.",
    message: "optimoptions: invalid solver",
};
pub const OPTIMOPTIONS_ERROR_INVALID_OPTION_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMOPTIONS.INVALID_OPTION_NAME",
    identifier: Some("RunMat:optimoptions:InvalidOptionName"),
    when: "An option name is not a text scalar.",
    message: "optimoptions: invalid option name",
};
pub const OPTIMOPTIONS_ERROR_MISSING_OPTION_VALUE: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.OPTIMOPTIONS.MISSING_OPTION_VALUE",
        identifier: Some("RunMat:optimoptions:MissingOptionValue"),
        when: "A name-value option key is not followed by a value.",
        message: "optimoptions: missing option value",
    };
pub const OPTIMOPTIONS_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMOPTIONS.UNKNOWN_OPTION",
    identifier: Some("RunMat:optimoptions:UnknownOption"),
    when: "An option name is not supported by the selected solver.",
    message: "optimoptions: unknown option",
};
pub const OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.OPTIMOPTIONS.INVALID_OPTION_VALUE",
        identifier: Some("RunMat:optimoptions:InvalidOptionValue"),
        when: "An option value fails type or domain validation.",
        message: "optimoptions: invalid option value",
    };
pub const OPTIMOPTIONS_ERROR_FLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMOPTIONS.FLOW",
    identifier: Some("RunMat:optimoptions:Flow"),
    when: "Nested flow fails while gathering input values.",
    message: "optimoptions: flow failure",
};

const OPTIMOPTIONS_ERRORS: [BuiltinErrorDescriptor; 7] = [
    OPTIMOPTIONS_ERROR_INVALID_ARGUMENT,
    OPTIMOPTIONS_ERROR_INVALID_SOLVER,
    OPTIMOPTIONS_ERROR_INVALID_OPTION_NAME,
    OPTIMOPTIONS_ERROR_MISSING_OPTION_VALUE,
    OPTIMOPTIONS_ERROR_UNKNOWN_OPTION,
    OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
    OPTIMOPTIONS_ERROR_FLOW,
];

pub const OPTIMOPTIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OPTIMOPTIONS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &OPTIMOPTIONS_ERRORS,
};
