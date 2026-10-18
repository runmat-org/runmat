use crate::*;

pub const LEGEND_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Legend labels, commands, and style properties have no integer numeric-data role; graphics-object targets are opaque handles rather than integer values.",
};

const OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Legend object handle.",
}];
const INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
};
const INPUTS_AX: [BuiltinParamDescriptor; 1] = [AX];
const MODE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Legend mode token: 'on'|'off'|'show'|'hide'|'boxon'|'boxoff'.",
};
const INPUTS_MODE: [BuiltinParamDescriptor; 1] = [MODE];
const INPUTS_AX_MODE: [BuiltinParamDescriptor; 2] = [AX, MODE];
const LABELS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "labels_or_pairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Labels (string/cell/string-array) with optional trailing Name/Value legend style pairs.",
};
const INPUTS_LABELS: [BuiltinParamDescriptor; 1] = [LABELS];
const INPUTS_AX_LABELS: [BuiltinParamDescriptor; 2] = [AX, LABELS];
const SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "legend()",
        inputs: &INPUTS_NONE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "legend(mode)",
        inputs: &INPUTS_MODE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "legend(labels..., Name, Value, ...)",
        inputs: &INPUTS_LABELS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "h = legend()",
        inputs: &INPUTS_NONE,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = legend(ax)",
        inputs: &INPUTS_AX,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = legend(mode)",
        inputs: &INPUTS_MODE,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = legend(ax, mode)",
        inputs: &INPUTS_AX_MODE,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = legend(labels..., Name, Value, ...)",
        inputs: &INPUTS_LABELS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = legend(ax, labels..., Name, Value, ...)",
        inputs: &INPUTS_AX_LABELS,
        outputs: &OUTPUT_HANDLE,
    },
];
pub const LEGEND_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LEGEND.INVALID_ARGUMENT",
    identifier: Some("RunMat:legend:InvalidArgument"),
    when: "Legend arguments are malformed, unsupported, or have invalid labels/properties.",
    message: "legend: invalid argument",
};
pub const LEGEND_ERROR_INVALID_AXES: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LEGEND.INVALID_AXES",
    identifier: Some("RunMat:legend:InvalidAxes"),
    when: "Resolved axes target is invalid or out of range.",
    message: "legend: invalid axes target",
};
pub const LEGEND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LEGEND.INTERNAL",
    identifier: Some("RunMat:legend:Internal"),
    when: "Internal legend state operation fails.",
    message: "legend: internal operation failed",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    LEGEND_ERROR_INVALID_ARGUMENT,
    LEGEND_ERROR_INVALID_AXES,
    LEGEND_ERROR_INTERNAL,
];
pub const LEGEND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
