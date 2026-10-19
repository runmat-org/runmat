use crate::*;
pub const GPU_CDATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "heatmap-gpu-cdata",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "heatmap with GPU-resident CData is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HeatmapGpuCDataExtension"),
};

pub const HEATMAP_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [GPU_CDATA_EXTENSION];

const fn documented_integer_input(
    name: &'static str,
    notes: &'static str,
) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes,
    }
}

const INTEGER_CDATA: [BuiltinIntegerInputCapability; 1] = [documented_integer_input(
    "CData",
    "Matrix ColorData accepts every built-in integer class and remains authoritative on the chart object.",
)];
const INTEGER_XVALUES: [BuiltinIntegerInputCapability; 1] = [documented_integer_input(
    "XValues",
    "Numeric constructor labels accept every built-in integer class and are formatted from authoritative storage.",
)];
const INTEGER_YVALUES: [BuiltinIntegerInputCapability; 1] = [documented_integer_input(
    "YValues",
    "Numeric constructor labels accept every built-in integer class and are formatted from authoritative storage.",
)];
const INTEGER_FONT_SIZE: [BuiltinIntegerInputCapability; 1] = [documented_integer_input(
    "FontSize",
    "The documented positive numeric font size is bounded before the client graphics conversion.",
)];
const INTEGER_COLORBAR_VISIBLE: [BuiltinIntegerInputCapability; 1] = [documented_integer_input(
    "ColorbarVisible",
    "The documented numeric on/off value is validated exactly as zero or one.",
)];
const INTEGER_GRID_VISIBLE: [BuiltinIntegerInputCapability; 1] = [documented_integer_input(
    "GridVisible",
    "The documented numeric on/off value is validated exactly as zero or one.",
)];
const INTEGER_COLOR_LIMITS: [BuiltinIntegerInputCapability; 1] = [documented_integer_input(
    "ColorLimits",
    "The documented two-element increasing limit vector is compared in authoritative storage before the renderer conversion.",
)];
const INTEGER_TABLE_VALUES: [BuiltinIntegerInputCapability; 3] = [
    documented_integer_input(
        "XVariable",
        "Integer table axis values group by exact integer identity before labels are formatted.",
    ),
    documented_integer_input(
        "YVariable",
        "Integer table axis values group by exact integer identity before labels are formatted.",
    ),
    documented_integer_input(
        "ColorVariable",
        "Summary methods use floating aggregation; ColorMethod none retains exact native integer ColorData for a complete axis-pair grid and rejects nonrepresentable wide values when an incomplete grid requires NaN cells.",
    ),
];

pub const HEATMAP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 8] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = heatmap(integer_CData)",
        inputs: &INTEGER_CDATA,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Exact ColorData is retained on the HeatmapChart; integer-aware normalization precedes the explicit floating client-renderer boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = heatmap(integer_XValues, YValues, CData)",
        inputs: &INTEGER_XVALUES,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer labels are converted directly to exact decimal strings; the output is an opaque chart handle.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = heatmap(XValues, integer_YValues, CData)",
        inputs: &INTEGER_YVALUES,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer labels are converted directly to exact decimal strings; the output is an opaque chart handle.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = heatmap(..., 'FontSize', integer_size)",
        inputs: &INTEGER_FONT_SIZE,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The exact positive scalar is range checked before conversion to the client font-size representation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = heatmap(..., 'ColorbarVisible', integer_on_off)",
        inputs: &INTEGER_COLORBAR_VISIBLE,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Only exact integer zero and one are accepted.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = heatmap(..., 'GridVisible', integer_on_off)",
        inputs: &INTEGER_GRID_VISIBLE,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Only exact integer zero and one are accepted.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = heatmap(..., 'ColorLimits', integer_limits)",
        inputs: &INTEGER_COLOR_LIMITS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The two authoritative endpoints are ordered before their explicit client-renderer conversion.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = heatmap(T, XVariable, YVariable, 'ColorVariable', CVariable)",
        inputs: &INTEGER_TABLE_VALUES,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Table grouping keys preserve native integer identity. Summary values enter double ColorData; none retains exact native integer ColorData on complete grids, while incomplete grids need representable double values for NaN holes.",
    },
];

const HEATMAP_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the heatmap chart.",
}];

const HEATMAP_INPUTS_CDATA: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "CData",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "M-by-N numeric matrix of color values.",
}];

const HEATMAP_INPUTS_XYCDATA: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "XValues",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Column labels (length N).",
    },
    BuiltinParamDescriptor {
        name: "YValues",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Row labels (length M).",
    },
    BuiltinParamDescriptor {
        name: "CData",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "M-by-N numeric matrix of color values.",
    },
];
const HEATMAP_INPUTS_CDATA_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "CData",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "M-by-N numeric matrix of color values.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Heatmap property name/value pairs.",
    },
];

const HEATMAP_INPUTS_XYCDATA_PROPS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "XValues",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Column labels (length N).",
    },
    BuiltinParamDescriptor {
        name: "YValues",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Row labels (length M).",
    },
    BuiltinParamDescriptor {
        name: "CData",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "M-by-N numeric matrix of color values.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Heatmap property name/value pairs.",
    },
];

const HEATMAP_INPUTS_TABLE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "SourceTable",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Table or timetable whose rows are grouped into cells.",
    },
    BuiltinParamDescriptor {
        name: "XVariable",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Exactly one table variable for heatmap columns.",
    },
    BuiltinParamDescriptor {
        name: "YVariable",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Exactly one table variable for heatmap rows.",
    },
];

const HEATMAP_INPUTS_TABLE_PROPS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "SourceTable",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Table or timetable whose rows are grouped into cells.",
    },
    BuiltinParamDescriptor {
        name: "XVariable",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Exactly one table variable for heatmap columns.",
    },
    BuiltinParamDescriptor {
        name: "YVariable",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Exactly one table variable for heatmap rows.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "ColorVariable, ColorMethod, and heatmap property pairs.",
    },
];

const HEATMAP_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "h = heatmap(CData)",
        inputs: &HEATMAP_INPUTS_CDATA,
        outputs: &HEATMAP_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = heatmap(CData, Name, Value, ...)",
        inputs: &HEATMAP_INPUTS_CDATA_PROPS,
        outputs: &HEATMAP_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = heatmap(XValues, YValues, CData)",
        inputs: &HEATMAP_INPUTS_XYCDATA,
        outputs: &HEATMAP_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = heatmap(XValues, YValues, CData, Name, Value, ...)",
        inputs: &HEATMAP_INPUTS_XYCDATA_PROPS,
        outputs: &HEATMAP_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = heatmap(SourceTable, XVariable, YVariable)",
        inputs: &HEATMAP_INPUTS_TABLE,
        outputs: &HEATMAP_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = heatmap(SourceTable, XVariable, YVariable, Name, Value, ...)",
        inputs: &HEATMAP_INPUTS_TABLE_PROPS,
        outputs: &HEATMAP_OUTPUT_HANDLE,
    },
];

pub const HEATMAP_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HEATMAP.INVALID_ARGUMENT",
    identifier: Some("RunMat:heatmap:InvalidArgument"),
    when: "CData/label/property inputs are malformed or incompatible.",
    message: "heatmap: invalid argument",
};

pub const HEATMAP_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HEATMAP.INTERNAL",
    identifier: Some("RunMat:heatmap:Internal"),
    when: "Internal render/surface construction fails.",
    message: "heatmap: internal operation failed",
};

pub const HEATMAP_ERROR_UNSUPPORTED_COLOR_DATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HEATMAP.UNSUPPORTED_COLOR_DATA",
    identifier: Some("RunMat:heatmap:UnsupportedColorData"),
    when: "An incomplete table grid needs missing ColorData cells but its wide integer observations cannot be represented exactly with the available numeric storage.",
    message: "heatmap: exact ColorData with missing grid cells is unsupported",
};

const HEATMAP_ERRORS: [BuiltinErrorDescriptor; 3] = [
    HEATMAP_ERROR_INVALID_ARGUMENT,
    HEATMAP_ERROR_INTERNAL,
    HEATMAP_ERROR_UNSUPPORTED_COLOR_DATA,
];

pub const HEATMAP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HEATMAP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HEATMAP_ERRORS,
};
