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
pub const NANSUM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "y = nansum(A, ...)",
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
            code: "RM.SUM.INVALID_ARGUMENT",
            identifier: Some("RunMat:sum:InvalidArgument"),
            when:
                "The owning sum reduction rejects its admitted data, options, or provider result.",
            message: "sum: delegated reduction error",
        },
        BuiltinErrorDescriptor {
            code: "RM.SUM.INVALID_INPUT",
            identifier: Some("RunMat:sum:InvalidInput"),
            when:
                "The owning sum reduction rejects its admitted data, options, or provider result.",
            message: "sum: delegated reduction error",
        },
        BuiltinErrorDescriptor {
            code: "RM.SUM.INTERNAL",
            identifier: Some("RunMat:sum:Internal"),
            when:
                "The owning sum reduction rejects its admitted data, options, or provider result.",
            message: "sum: delegated reduction error",
        },
    ],
};
pub const NANSUM_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "nansum-typed-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "nansum with a typed-integer data or control input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NansumTypedIntegerInputExtension"),
};
pub const NANSUM_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [NANSUM_INTEGER_EXTENSION];
pub const NANSUM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "y = nansum(A, args...)",
        inputs: &[BuiltinIntegerInputCapability {
            name: "A_or_control", classes: &ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
            notes: "Typed-integer data and controls are admitted only by this RunMat extension.",
        }],
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat routes typed-integer forms through sum with omitnan: default or double output is double, native output preserves the input class with saturating accumulation, and modern sum-only options remain extension syntax.",
    }];
