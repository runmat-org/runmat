use crate::*;

pub const COND_INTEGER_MATRIX_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cond-integer-matrix",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cond with a typed-integer matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CondIntegerMatrixExtension"),
};

pub const COND_LOGICAL_MATRIX_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cond-logical-matrix",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cond with a logical matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CondLogicalMatrixExtension"),
};

pub const COND_INTEGER_NORM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cond-integer-norm-selector",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cond with a typed-integer norm selector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CondIntegerNormSelectorExtension"),
};

pub const COND_LOGICAL_NORM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cond-logical-norm-selector",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cond with a logical norm selector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CondLogicalNormSelectorExtension"),
};

pub const COND_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    COND_INTEGER_MATRIX_EXTENSION,
    COND_LOGICAL_MATRIX_EXTENSION,
    COND_INTEGER_NORM_EXTENSION,
    COND_LOGICAL_NORM_EXTENSION,
];

pub const COND_INTEGER_MATRIX_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "RunMat mode accepts integer matrices only when every element is exactly representable at the floating linear-algebra boundary.",
}];

pub const COND_INTEGER_NORM_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "p",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "Integer norm selectors are decoded structurally and accept only exact values 1 or 2.",
    }];

pub const COND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "c = cond(integer_A, p)", inputs: &COND_INTEGER_MATRIX_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "This RunMat extension requires exact binary64 representability before host execution; the result is double on host, while a resident scalar uses the owning provider's selected precision." },
    BuiltinIntegerCapabilityDescriptor { form: "c = cond(A, integer_p)", inputs: &COND_INTEGER_NORM_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Host typed-integer selectors do not enter matrix arithmetic and wide values are rejected without floating conversion; resident norm selectors are not supported." },
];

pub const COND_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "c",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Condition number estimate of A.",
}];

pub const COND_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix.",
}];

pub const COND_INPUTS_NORM: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix.",
    },
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Norm selector (1, 2, inf, or \"fro\").",
    },
];

pub const COND_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "c = cond(A)",
        inputs: &COND_INPUTS,
        outputs: &COND_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "c = cond(A, p)",
        inputs: &COND_INPUTS_NORM,
        outputs: &COND_OUTPUT,
    },
];

pub const COND_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COND.INVALID_ARGUMENT",
    identifier: Some("RunMat:cond:InvalidArgument"),
    when: "Norm selector argument is malformed or unsupported.",
    message: "cond: invalid argument",
};

pub const COND_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COND.INVALID_INPUT",
    identifier: Some("RunMat:cond:InvalidInput"),
    when: "Input shape/type cannot be processed for condition-number evaluation.",
    message: "cond: invalid input",
};

pub const COND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COND.INTERNAL",
    identifier: Some("RunMat:cond:Internal"),
    when: "Runtime fails while computing cond or executing fallback/upload paths.",
    message: "cond: internal runtime failure",
};

pub const COND_ERRORS: [BuiltinErrorDescriptor; 3] = [
    COND_ERROR_INVALID_ARGUMENT,
    COND_ERROR_INVALID_INPUT,
    COND_ERROR_INTERNAL,
];

pub const COND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COND_ERRORS,
};
