use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scanned numeric values or a character array for character-only formats.",
    },
    BuiltinParamDescriptor {
        name: "count",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Number of scanned values, returned as double.",
    },
    BuiltinParamDescriptor {
        name: "errmsg",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Scan diagnostic character vector.",
    },
    BuiltinParamDescriptor {
        name: "nextindex",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based next character index, returned as double.",
    },
];

const INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "text",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text.",
    },
    BuiltinParamDescriptor {
        name: "format",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("%f"),
        description: "Scan format.",
    },
    BuiltinParamDescriptor {
        name: "sizeA",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional positive scalar or two-element output size and read limit.",
    },
];

pub const SSCANF_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SSCANF.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:sscanf:TooManyOutputs"),
    when: "More than four outputs are requested.",
    message: "sscanf: too many output arguments",
};

pub const SSCANF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "[A, count, errmsg, nextindex] = sscanf(text, format, size)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    }],
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[SSCANF_ERROR_TOO_MANY_OUTPUTS],
};
const SSCANF_INTEGER_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sizeA",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The optional output-size scalar or two-element vector accepts every built-in integer class and is decoded directly from native storage.",
    }];
pub const SSCANF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "A = sscanf(text, format, integer_sizeA)",
        inputs: &SSCANF_INTEGER_SIZE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "sizeA limits admitted reads and controls output shape. Pure %ld/%li scans return int64, pure %lu/%lo/%lx scans return uint64, character-only formats return char, and mixed numeric/character formats return double.",
    }];
