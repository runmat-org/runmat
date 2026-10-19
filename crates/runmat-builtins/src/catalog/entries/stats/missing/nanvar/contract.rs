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
pub const NANVAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "y = nanvar(A, ...)",
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
            code: "RM.VAR.INVALID_ARGUMENT",
            identifier: Some("RunMat:var:InvalidArgument"),
            when:
                "The owning var reduction rejects its admitted data, options, or provider result.",
            message: "var: delegated reduction error",
        },
        BuiltinErrorDescriptor {
            code: "RM.VAR.INVALID_INPUT",
            identifier: Some("RunMat:var:InvalidInput"),
            when:
                "The owning var reduction rejects its admitted data, options, or provider result.",
            message: "var: delegated reduction error",
        },
        BuiltinErrorDescriptor {
            code: "RM.VAR.COMPLEX_UNSUPPORTED",
            identifier: Some("RunMat:var:ComplexUnsupported"),
            when:
                "The owning var reduction rejects its admitted data, options, or provider result.",
            message: "var: delegated reduction error",
        },
        BuiltinErrorDescriptor {
            code: "RM.VAR.INTERNAL",
            identifier: Some("RunMat:var:Internal"),
            when:
                "The owning var reduction rejects its admitted data, options, or provider result.",
            message: "var: delegated reduction error",
        },
    ],
};
pub const NANVAR_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "nanvar-typed-integer-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "nanvar with a typed-integer normalization or dimension control is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NanvarTypedIntegerControlExtension"),
};
pub const NANVAR_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [NANVAR_INTEGER_EXTENSION];
pub const NANVAR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "y = nanvar(integer_A, ...)",
        inputs: &[BuiltinIntegerInputCapability {
            name: "A",
            classes: &[],
            availability: BuiltinIntegerInputAvailability::Rejected,
            scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes: "Typed-integer data is rejected before host or provider reduction.",
        }],
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed-integer data is not admitted in either compatibility mode.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = nanvar(A, typed_integer_control)",
        inputs: &[BuiltinIntegerInputCapability {
            name: "normalization_or_dimension",
            classes: &ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
            notes: "Typed integer controls are decoded exactly in RunMat mode.",
        }],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes:
            "With floating input, exact typed-integer normalization and dimension controls are RunMat-only; non-scalar weighted variance remains separately unsupported.",
    },
];
