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

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "NaN-omitting result.",
}];
pub const NANMEDIAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "y = nanmedian(A, ...)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
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
            code: "RM.MEDIAN.INVALID_ARGUMENT",
            identifier: Some("RunMat:median:InvalidArgument"),
            when: "The owning median reduction rejects its admitted data, options, or provider result.",
            message: "median: delegated reduction error",
        },
        BuiltinErrorDescriptor {
            code: "RM.MEDIAN.INVALID_INPUT",
            identifier: Some("RunMat:median:InvalidInput"),
            when: "The owning median reduction rejects its admitted data, options, or provider result.",
            message: "median: delegated reduction error",
        },
        BuiltinErrorDescriptor {
            code: "RM.MEDIAN.INTERNAL",
            identifier: Some("RunMat:median:Internal"),
            when: "The owning median reduction rejects its admitted data, options, or provider result.",
            message: "median: delegated reduction error",
        },
    ],
};
pub const NANMEDIAN_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "nanmedian-typed-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "nanmedian with a typed-integer data or control input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NanmedianTypedIntegerInputExtension"),
};
pub const NANMEDIAN_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [NANMEDIAN_INTEGER_EXTENSION];
pub const NANMEDIAN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "y = nanmedian(A, args...)",
        inputs: &[BuiltinIntegerInputCapability {
            name: "A_or_control",
            classes: &ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
            notes: "Typed-integer data and controls are admitted only by this RunMat extension.",
        }],
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat routes typed-integer input through median with omitnan; exact host reduction preserves all eight classes and supported resident fallback restores output.",
    }];
