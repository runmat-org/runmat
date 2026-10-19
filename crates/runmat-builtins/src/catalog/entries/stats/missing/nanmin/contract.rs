use crate::*;

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric data.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Supported dimension, normalization, or output-class controls.",
    },
];
const PAIRWISE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First numeric operand.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second numeric operand, equal-shaped or scalar.",
    },
];
const REDUCTION_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric data to reduce.",
    },
    BuiltinParamDescriptor {
        name: "placeholder",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: Some("[]"),
        description: "Empty placeholder distinguishing reduction from pairwise comparison.",
    },
    BuiltinParamDescriptor {
        name: "dim_or_options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional dimension or all selector.",
    },
];
const VALUE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "NaN-omitting result.",
}];
const REDUCTION_OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "NaN-omitting reduction result.",
    },
    BuiltinParamDescriptor {
        name: "I",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based indices of selected minima.",
    },
];
pub const NANMIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "y = nanmin(A, ...)",
            inputs: &INPUTS,
            outputs: &REDUCTION_OUTPUTS,
        },
        BuiltinSignatureDescriptor {
            label: "y = nanmin(A, B)",
            inputs: &PAIRWISE_INPUTS,
            outputs: &VALUE_OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "y = nanmin(A, [], dim_or_options)",
            inputs: &REDUCTION_INPUTS,
            outputs: &REDUCTION_OUTPUTS,
        },
    ],
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        BuiltinErrorDescriptor {
            code: "RM.MISSING.INVALID_ARGUMENT",
            identifier: Some("RunMat:missing:InvalidArgument"),
            when: "The input count or alias-specific argument form is invalid.",
            message: "missing-value builtin: invalid argument",
        },
        BuiltinErrorDescriptor {
            code: "RM.MISSING.UNSUPPORTED_TYPE",
            identifier: Some("RunMat:missing:UnsupportedType"),
            when: "The data class has no supported alias form.",
            message: "missing-value builtin: unsupported input type",
        },
        BuiltinErrorDescriptor {
            code: "RM.MISSING.INTERNAL",
            identifier: Some("RunMat:missing:InternalError"),
            when: "Missing-value wrapper materialization fails.",
            message: "missing-value builtin: internal error",
        },
        BuiltinErrorDescriptor {
            code: "RM.MIN.INVALID_ARGUMENT",
            identifier: Some("RunMat:min:InvalidArgument"),
            when:
                "The owning min reduction rejects its admitted data, options, or provider result.",
            message: "min: delegated reduction error",
        },
        BuiltinErrorDescriptor {
            code: "RM.MIN.INVALID_INPUT",
            identifier: Some("RunMat:min:InvalidInput"),
            when:
                "The owning min reduction rejects its admitted data, options, or provider result.",
            message: "min: delegated reduction error",
        },
        BuiltinErrorDescriptor {
            code: "RM.MIN.SIZE_MISMATCH",
            identifier: Some("RunMat:min:SizeMismatch"),
            when:
                "The owning min reduction rejects its admitted data, options, or provider result.",
            message: "min: delegated reduction error",
        },
        BuiltinErrorDescriptor {
            code: "RM.MIN.INTERNAL",
            identifier: Some("RunMat:min:Internal"),
            when:
                "The owning min reduction rejects its admitted data, options, or provider result.",
            message: "min: delegated reduction error",
        },
    ],
};
pub const NANMIN_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "nanmin-typed-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "nanmin with a typed-integer data or control input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NanminTypedIntegerInputExtension"),
};
pub const NANMIN_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [NANMIN_INTEGER_EXTENSION];
pub const NANMIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "y = nanmin(A, args...) or nanmin(A, B)",
        inputs: &[BuiltinIntegerInputCapability {
            name: "A_or_control", classes: &ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
            notes: "Typed-integer data and controls are admitted only by this RunMat extension.",
        }],
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat's typed-integer reduction and compatible pairwise forms retain exact storage; omit-NaN resident reduction uses the parent's host fallback and pairwise input follows same-class-or-scalar-double rules.",
    }];
