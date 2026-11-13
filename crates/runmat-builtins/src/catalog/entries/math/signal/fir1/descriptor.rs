use crate::*;
const FIR1_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "FIR numerator coefficient row vector.",
}];

const FIR1_INPUTS_CORE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter order.",
    },
    BuiltinParamDescriptor {
        name: "Wn",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Normalized cutoff frequency or two-element band.",
    },
];

const FIR1_INPUTS_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter order.",
    },
    BuiltinParamDescriptor {
        name: "Wn",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Normalized cutoff frequency or two-element band.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Filter type, window vector, or scale/noscale option.",
    },
];

const FIR1_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "b = fir1(n, Wn)",
        inputs: &FIR1_INPUTS_CORE,
        outputs: &FIR1_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "b = fir1(n, Wn, ftype)",
        inputs: &FIR1_INPUTS_OPTIONS,
        outputs: &FIR1_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "b = fir1(n, Wn, ftype, window, scaleopt)",
        inputs: &FIR1_INPUTS_OPTIONS,
        outputs: &FIR1_OUTPUT,
    },
];

pub const FIR1_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIR1.ARG_COUNT",
    identifier: Some("RunMat:fir1:ArgCount"),
    when: "Fewer than two arguments are supplied.",
    message: "fir1: expected fir1(n, Wn, ...)",
};

pub const FIR1_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIR1.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:fir1:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "fir1: too many output arguments",
};

pub const FIR1_ERROR_INVALID_ORDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIR1.INVALID_ORDER",
    identifier: Some("RunMat:fir1:InvalidOrder"),
    when: "The order is not a finite nonnegative integer scalar.",
    message: "fir1: filter order must be a nonnegative integer",
};

pub const FIR1_ERROR_INVALID_FREQUENCY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIR1.INVALID_FREQUENCY",
    identifier: Some("RunMat:fir1:InvalidFrequency"),
    when: "Wn is not a valid normalized scalar or increasing two-element band.",
    message: "fir1: invalid cutoff frequency",
};

pub const FIR1_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIR1.INVALID_OPTION",
    identifier: Some("RunMat:fir1:InvalidOption"),
    when: "A filter type, window vector, or scaling option is unsupported.",
    message: "fir1: invalid option",
};

pub const FIR1_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIR1.INTERNAL",
    identifier: Some("RunMat:fir1:Internal"),
    when: "Coefficient materialization fails internally.",
    message: "fir1: internal error",
};

const FIR1_ERRORS: [BuiltinErrorDescriptor; 6] = [
    FIR1_ERROR_ARG_COUNT,
    FIR1_ERROR_TOO_MANY_OUTPUTS,
    FIR1_ERROR_INVALID_ORDER,
    FIR1_ERROR_INVALID_FREQUENCY,
    FIR1_ERROR_INVALID_OPTION,
    FIR1_ERROR_INTERNAL,
];

macro_rules! fir1_extension {
    ($name:ident, $id:literal, $description:literal, $error:literal) => {
        pub const $name: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: $description,
            error_identifier: Some($error),
        };
    };
}

fir1_extension!(
    FIR1_INTEGER_ORDER_EXTENSION,
    "fir1-integer-order",
    "fir1 with a typed-integer filter order is a RunMat extension",
    "RunMat:compatibility:Fir1IntegerOrderExtension"
);
fir1_extension!(
    FIR1_LOGICAL_ORDER_EXTENSION,
    "fir1-logical-order",
    "fir1 with a logical filter order is a RunMat extension",
    "RunMat:compatibility:Fir1LogicalOrderExtension"
);
fir1_extension!(
    FIR1_SINGLE_ORDER_EXTENSION,
    "fir1-single-order",
    "fir1 with a single-precision filter order is a RunMat extension",
    "RunMat:compatibility:Fir1SingleOrderExtension"
);
fir1_extension!(
    FIR1_INTEGER_CUTOFF_EXTENSION,
    "fir1-integer-cutoff",
    "fir1 with typed-integer cutoff frequencies is a RunMat extension",
    "RunMat:compatibility:Fir1IntegerCutoffExtension"
);
fir1_extension!(
    FIR1_LOGICAL_CUTOFF_EXTENSION,
    "fir1-logical-cutoff",
    "fir1 with logical cutoff frequencies is a RunMat extension",
    "RunMat:compatibility:Fir1LogicalCutoffExtension"
);
fir1_extension!(
    FIR1_SINGLE_CUTOFF_EXTENSION,
    "fir1-single-cutoff",
    "fir1 with single-precision cutoff frequencies is a RunMat extension",
    "RunMat:compatibility:Fir1SingleCutoffExtension"
);
fir1_extension!(
    FIR1_INTEGER_WINDOW_EXTENSION,
    "fir1-integer-window",
    "fir1 with a typed-integer window is a RunMat extension",
    "RunMat:compatibility:Fir1IntegerWindowExtension"
);
fir1_extension!(
    FIR1_LOGICAL_WINDOW_EXTENSION,
    "fir1-logical-window",
    "fir1 with a logical window is a RunMat extension",
    "RunMat:compatibility:Fir1LogicalWindowExtension"
);
fir1_extension!(
    FIR1_SINGLE_WINDOW_EXTENSION,
    "fir1-single-window",
    "fir1 with a single-precision window is a RunMat extension",
    "RunMat:compatibility:Fir1SingleWindowExtension"
);

pub const FIR1_EXTENSIONS: [BuiltinExtensionDescriptor; 9] = [
    FIR1_INTEGER_ORDER_EXTENSION,
    FIR1_LOGICAL_ORDER_EXTENSION,
    FIR1_SINGLE_ORDER_EXTENSION,
    FIR1_INTEGER_CUTOFF_EXTENSION,
    FIR1_LOGICAL_CUTOFF_EXTENSION,
    FIR1_SINGLE_CUTOFF_EXTENSION,
    FIR1_INTEGER_WINDOW_EXTENSION,
    FIR1_LOGICAL_WINDOW_EXTENSION,
    FIR1_SINGLE_WINDOW_EXTENSION,
];

const FIR1_INTEGER_ORDER_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "n",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented order class is double; RunMat mode parses every typed-integer scalar exactly as a structural filter order.",
    },
];
const FIR1_INTEGER_CUTOFF_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "Wn",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed cutoff storage is admitted only in RunMat mode and must be exactly representable at the binary64 FIR-design boundary.",
    },
];
const FIR1_INTEGER_WINDOW_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "window",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed window values remain authoritative through admission and must be exactly representable before binary64 window multiplication.",
    },
];

pub const FIR1_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "b = fir1(integer_n,Wn,...)",
        inputs: &FIR1_INTEGER_ORDER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The order is decoded exactly in every supported integer class; resident order input gathers through its owner after compatibility admission, and coefficient output remains a host double row vector.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "b = fir1(n,integer_Wn,...)",
        inputs: &FIR1_INTEGER_CUTOFF_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer cutoff values cross one checked binary64 boundary before normalized-frequency validation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "b = fir1(n,Wn,integer_window,...)",
        inputs: &FIR1_INTEGER_WINDOW_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer window values cross one checked binary64 boundary before coefficient multiplication.",
    },
];

pub const FIR1_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FIR1_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FIR1_ERRORS,
};
