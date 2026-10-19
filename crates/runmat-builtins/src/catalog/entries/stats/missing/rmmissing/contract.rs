use crate::*;

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, text, or table data.",
    },
    BuiltinParamDescriptor {
        name: "dim_or_options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional row/column selection or dimension.",
    },
];
const VALUE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input after removing missing entries, rows, or columns.",
}];
const OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "R",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input after removing missing entries, rows, or columns.",
    },
    BuiltinParamDescriptor {
        name: "TF",
        ty: BuiltinParamType::LogicalArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Removed vector entries, rows, or columns.",
    },
];
pub const RMMISSING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "R = rmmissing(A, ...)",
            inputs: &INPUTS,
            outputs: &VALUE_OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "[R, TF] = rmmissing(A, ...)",
            inputs: &INPUTS,
            outputs: &OUTPUTS,
        },
    ],
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        BuiltinErrorDescriptor {
            code: "RM.MISSING.INVALID_ARGUMENT",
            identifier: Some("RunMat:missing:InvalidArgument"),
            when: "The dimension, option, output count, or resident input is invalid.",
            message: "missing-value builtin: invalid argument",
        },
        BuiltinErrorDescriptor {
            code: "RM.MISSING.UNSUPPORTED_TYPE",
            identifier: Some("RunMat:missing:UnsupportedType"),
            when: "The input has no supported removal representation.",
            message: "missing-value builtin: unsupported input type",
        },
        BuiltinErrorDescriptor {
            code: "RM.MISSING.INTERNAL",
            identifier: Some("RunMat:missing:InternalError"),
            when: "Table, array, or provider restoration fails.",
            message: "missing-value builtin: internal error",
        },
    ],
};
pub const RMMISSING_INTEGER_DIM_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "rmmissing-integer-dimension",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "rmmissing accepts a typed-integer dimension control as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RmmissingIntegerDimensionExtension"),
    };
pub const RMMISSING_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [RMMISSING_INTEGER_DIM_EXTENSION];
const INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes:
        "All eight integer classes have no standard missing element and retain their exact storage.",
}];
const INTEGER_DIMENSION: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "dim",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "A typed integer dimension is decoded exactly as a structural control.",
}];
pub const RMMISSING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[R, TF] = rmmissing(integer_A, ...)",
        inputs: &INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer A is unchanged, TF is all false, and admitted resident outputs restore through the input owner.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "R = rmmissing(A, integer_dim)",
        inputs: &INTEGER_DIMENSION,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Compatibility admission precedes provider access; the dimension never crosses a floating boundary.",
    },
];
