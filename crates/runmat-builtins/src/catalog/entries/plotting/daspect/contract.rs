use crate::*;
const DASPECT_OUTPUT_RATIO: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ratio",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current data aspect ratio [dx dy dz].",
}];

const DASPECT_OUTPUT_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current data aspect ratio mode, 'auto' or 'manual'.",
}];

const DASPECT_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const DASPECT_INPUTS_RATIO: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ratio",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Three-element positive numeric data aspect ratio.",
}];

const DASPECT_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Mode string: 'auto', 'manual', or 'mode'.",
}];

const DASPECT_INPUTS_AX_RATIO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "ratio_or_mode",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Ratio vector or mode string.",
    },
];

const DASPECT_INPUTS_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
}];

const DASPECT_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "ratio = daspect()",
        inputs: &DASPECT_INPUTS_NONE,
        outputs: &DASPECT_OUTPUT_RATIO,
    },
    BuiltinSignatureDescriptor {
        label: "daspect(ratio)",
        inputs: &DASPECT_INPUTS_RATIO,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "mode = daspect('mode')",
        inputs: &DASPECT_INPUTS_MODE,
        outputs: &DASPECT_OUTPUT_MODE,
    },
    BuiltinSignatureDescriptor {
        label: "daspect(mode)",
        inputs: &DASPECT_INPUTS_MODE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "ratio = daspect(ax)",
        inputs: &DASPECT_INPUTS_AX,
        outputs: &DASPECT_OUTPUT_RATIO,
    },
    BuiltinSignatureDescriptor {
        label: "mode = daspect(ax, 'mode')",
        inputs: &DASPECT_INPUTS_AX_RATIO,
        outputs: &DASPECT_OUTPUT_MODE,
    },
    BuiltinSignatureDescriptor {
        label: "daspect(ax, ratio_or_mode)",
        inputs: &DASPECT_INPUTS_AX_RATIO,
        outputs: &[],
    },
];

const DASPECT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DASPECT.INVALID_ARGUMENT",
    identifier: Some("RunMat:daspect:InvalidArgument"),
    when: "Argument count, axes handle, ratio vector, or mode string is invalid.",
    message: "daspect: invalid argument",
};

const DASPECT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DASPECT.INTERNAL",
    identifier: Some("RunMat:daspect:Internal"),
    when: "Internal plotting state update fails.",
    message: "daspect: internal operation failed",
};

const DASPECT_ERRORS: [BuiltinErrorDescriptor; 3] = [
    DASPECT_ERROR_INVALID_ARGUMENT,
    DASPECT_ERROR_INTERNAL,
    DASPECT_ERROR_TOO_MANY_OUTPUTS,
];

pub const DASPECT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DASPECT_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DASPECT_ERRORS,
};

pub const DASPECT_NUMERIC_AXES_ALIAS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "daspect-numeric-axes-handle-alias",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "daspect with RunMat's numeric encoded axes handle is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DaspectNumericAxesAliasExtension"),
    };
pub const DASPECT_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [DASPECT_NUMERIC_AXES_ALIAS_EXTENSION];

const DASPECT_INTEGER_RATIO: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ratio",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight positive integer classes are documented. Shape and positivity are checked from authoritative integer storage before conversion to the host double graphics-property boundary.",
    }];
const DASPECT_INTEGER_AXES_ALIAS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ax",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed integers are accepted only as exact scalar aliases for RunMat's encoded axes handles and are separately compatibility gated.",
    }];
const DASPECT_RESIDENT_INTEGER: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "resident ratio or ax",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Resident numeric arrays are rejected without provider lookup, download, or upload.",
    }];

pub const DASPECT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "daspect(integer_ratio)", inputs: &DASPECT_INTEGER_RATIO, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The documented integer vector is validated exactly, then converted once into the double-valued DataAspectRatio property; the setter returns no value." },
    BuiltinIntegerCapabilityDescriptor { form: "daspect(integer_ax, ratio_or_mode)", inputs: &DASPECT_INTEGER_AXES_ALIAS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "RunMat-only exact numeric handle alias; the public contract accepts an Axes object." },
    BuiltinIntegerCapabilityDescriptor { form: "daspect(resident_integer, ...)", inputs: &DASPECT_RESIDENT_INTEGER, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Graphics metadata control never gathers a resident numeric array to reinterpret it as a ratio or handle." },
];

pub const DASPECT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DASPECT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:daspect:TooManyOutputs"),
    when: "More than one query output is requested.",
    message: "daspect: too many output arguments",
};
