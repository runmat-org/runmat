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
pub const NANMEAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "y = nanmean(A, ...)",
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
            code: "RM.MEAN.INVALID_ARGUMENT",
            identifier: Some("RunMat:mean:InvalidArgument"),
            when:
                "The owning mean reduction rejects its admitted data, options, or provider result.",
            message: "mean: delegated reduction error",
        },
        BuiltinErrorDescriptor {
            code: "RM.MEAN.INVALID_INPUT",
            identifier: Some("RunMat:mean:InvalidInput"),
            when:
                "The owning mean reduction rejects its admitted data, options, or provider result.",
            message: "mean: delegated reduction error",
        },
        BuiltinErrorDescriptor {
            code: "RM.MEAN.INTERNAL",
            identifier: Some("RunMat:mean:Internal"),
            when:
                "The owning mean reduction rejects its admitted data, options, or provider result.",
            message: "mean: delegated reduction error",
        },
    ],
};
pub const NANMEAN_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "nanmean-typed-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "nanmean with a typed-integer data or control input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NanmeanTypedIntegerInputExtension"),
};
pub const NANMEAN_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [NANMEAN_INTEGER_EXTENSION];
pub const NANMEAN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "y = nanmean(A, args...)",
        inputs: &[BuiltinIntegerInputCapability {
            name: "A_or_control",
            classes: &ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
            notes: "Typed-integer data and controls are admitted only by this RunMat extension.",
        }],
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes:
            "RunMat routes typed-integer forms through mean with omitnan: default or double output is double, native output preserves the input class, and modern mean-only options remain extension syntax.",
    },
];
