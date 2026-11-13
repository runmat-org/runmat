//! Boxplot public forms, diagnostics and integer/compatibility capability metadata.
use crate::*;

const OUTPUT_HANDLES: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Row vector of graphics handles for created box plot primitives.",
};

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric vector or matrix data.",
};

const PARAM_G: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "g",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Grouping vector or cell array of grouping vectors.",
};

const PARAM_AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nameValuePairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Box plot name-value options.",
};

const INPUT_X: [BuiltinParamDescriptor; 1] = [PARAM_X];
const INPUT_X_OPTIONS: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_OPTIONS];
const INPUT_X_G: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_G];
const INPUT_X_G_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_G, PARAM_OPTIONS];
const INPUT_AX_X: [BuiltinParamDescriptor; 2] = [PARAM_AX, PARAM_X];
const INPUT_AX_X_G_OPTIONS: [BuiltinParamDescriptor; 4] =
    [PARAM_AX, PARAM_X, PARAM_G, PARAM_OPTIONS];
const OUTPUT_H: [BuiltinParamDescriptor; 1] = [OUTPUT_HANDLES];

const SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "boxplot(x)",
        inputs: &INPUT_X,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "boxplot(x, g)",
        inputs: &INPUT_X_G,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "boxplot(___, Name, Value)",
        inputs: &INPUT_X_OPTIONS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "boxplot(ax, ___)",
        inputs: &INPUT_AX_X,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "boxplot(ax, x, g, Name, Value)",
        inputs: &INPUT_AX_X_G_OPTIONS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "h = boxplot(___)",
        inputs: &INPUT_X,
        outputs: &OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "h = boxplot(x, g, Name, Value)",
        inputs: &INPUT_X_G_OPTIONS,
        outputs: &OUTPUT_H,
    },
];

pub const BOXPLOT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BOXPLOT.INVALID_ARGUMENT",
    identifier: Some("RunMat:boxplot:InvalidArgument"),
    when: "Inputs, grouping variables, or name-value options are malformed.",
    message: "boxplot: invalid argument",
};

pub const BOXPLOT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BOXPLOT.INTERNAL",
    identifier: Some("RunMat:boxplot:Internal"),
    when: "RunMat cannot construct box statistics or plotting primitives.",
    message: "boxplot: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] =
    [BOXPLOT_ERROR_INVALID_ARGUMENT, BOXPLOT_ERROR_INTERNAL];

pub const BOXPLOT_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "boxplot-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "boxplot with integer sample data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BoxplotIntegerDataExtension"),
};

pub const BOXPLOT_LOGICAL_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "boxplot-logical-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "boxplot with logical sample data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BoxplotLogicalDataExtension"),
};

pub const BOXPLOT_INTEGER_GROUP_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "boxplot-integer-group",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "boxplot with integer grouping data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BoxplotIntegerGroupExtension"),
    };

pub const BOXPLOT_LOGICAL_GROUP_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "boxplot-logical-group",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "boxplot with logical grouping data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BoxplotLogicalGroupExtension"),
    };

pub const BOXPLOT_INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "boxplot-integer-option",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "boxplot with an integer numeric option is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BoxplotIntegerOptionExtension"),
    };

pub const BOXPLOT_LOGICAL_OPTION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "boxplot-logical-option",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "boxplot with a logical numeric option is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BoxplotLogicalOptionExtension"),
    };

pub const BOXPLOT_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "boxplot-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "boxplot with gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BoxplotGpuInputExtension"),
};

pub const BOXPLOT_EXTENSIONS: [BuiltinExtensionDescriptor; 7] = [
    BOXPLOT_INTEGER_DATA_EXTENSION,
    BOXPLOT_LOGICAL_DATA_EXTENSION,
    BOXPLOT_INTEGER_GROUP_EXTENSION,
    BOXPLOT_LOGICAL_GROUP_EXTENSION,
    BOXPLOT_INTEGER_OPTION_EXTENSION,
    BOXPLOT_LOGICAL_OPTION_EXTENSION,
    BOXPLOT_GPU_INPUT_EXTENSION,
];

const BOXPLOT_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat accepts every integer class as sample data; the public sample-data domain is single or double.",
    }];

const BOXPLOT_INTEGER_GROUP_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "g or ColorGroup",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat accepts every integer class as direct or cell-contained grouping data and retains exact integer labels; the public numeric grouping domain is single or double.",
    }];

const BOXPLOT_INTEGER_OPTION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "numeric Name-Value option",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat accepts integer numeric option values; visualization controls cross an explicit floating boundary while numeric labels and grouping options retain exact text or identity. Public numeric option domains are single or double.",
    }];

pub const BOXPLOT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "boxplot(x) with integer sample data",
        inputs: &BOXPLOT_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only visualization form; integer observations cross one deliberate binary64 statistics/rendering boundary and the optional result is a host graphics-handle vector.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "boxplot(x,g) or boxplot(___,ColorGroup,g) with integer grouping data",
        inputs: &BOXPLOT_INTEGER_GROUP_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only grouping form; integer group identity, order, and labels remain exact for every supported class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "boxplot(___,Name,Value) with integer numeric option values",
        inputs: &BOXPLOT_INTEGER_OPTION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only option form; geometry, color, limit, and scale controls cross one deliberate binary64 boundary after option-specific validation, while labels and grouping identity remain exact.",
    },
];

pub const BOXPLOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
