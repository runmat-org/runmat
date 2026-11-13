use crate::*;

pub(super) const AUDIOINFO_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "info",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Audio metadata structure.",
}];
pub(super) const AUDIOINFO_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Audio file path.",
}];
pub(super) const AUDIOINFO_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "info = audioinfo(filename)",
        inputs: &AUDIOINFO_INPUTS,
        outputs: &AUDIOINFO_OUTPUTS,
    }];

pub const AUDIOINFO_ERROR_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AUDIOINFO.ARGUMENT",
    identifier: Some("RunMat:audioinfo:InvalidArgument"),
    when: "Filename is missing or cannot be interpreted as a scalar path.",
    message: "audioinfo: invalid filename",
};
pub const AUDIOINFO_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AUDIOINFO.IO",
    identifier: Some("RunMat:audioinfo:Io"),
    when: "The audio file cannot be read.",
    message: "audioinfo: unable to read file",
};
pub const AUDIOINFO_ERROR_FORMAT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AUDIOINFO.FORMAT",
    identifier: Some("RunMat:audioinfo:UnsupportedFormat"),
    when: "The file is not a supported audio container or has malformed metadata.",
    message: "audioinfo: unsupported or invalid audio file",
};
pub(super) const AUDIOINFO_ERRORS: [BuiltinErrorDescriptor; 3] = [
    AUDIOINFO_ERROR_ARGUMENT,
    AUDIOINFO_ERROR_IO,
    AUDIOINFO_ERROR_FORMAT,
];

pub const AUDIOINFO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &AUDIOINFO_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &AUDIOINFO_ERRORS,
};

pub(super) const AUDIOREAD_OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Decoded samples as an N-by-C matrix.",
    },
    BuiltinParamDescriptor {
        name: "Fs",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Sample rate in Hz.",
    },
];
pub(super) const AUDIOREAD_OUTPUTS_SAMPLES: [BuiltinParamDescriptor; 1] =
    [BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Decoded samples as an N-by-C matrix.",
    }];
pub(super) const AUDIOREAD_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] =
    [BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Audio file path.",
    }];
pub(super) const AUDIOREAD_INPUTS_RANGE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Audio file path.",
    },
    BuiltinParamDescriptor {
        name: "samples",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based inclusive frame range [first last].",
    },
];
pub(super) const AUDIOREAD_INPUTS_DATATYPE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Audio file path.",
    },
    BuiltinParamDescriptor {
        name: "datatype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output class: double or native.",
    },
];
pub(super) const AUDIOREAD_INPUTS_RANGE_NATIVE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Audio file path.",
    },
    BuiltinParamDescriptor {
        name: "samples",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based inclusive frame range [first last].",
    },
    BuiltinParamDescriptor {
        name: "datatype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Output class; \"native\" preserves representable source classes.",
    },
];
pub(super) const AUDIOREAD_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "y = audioread(filename)",
        inputs: &AUDIOREAD_INPUTS_FILENAME,
        outputs: &AUDIOREAD_OUTPUTS_SAMPLES,
    },
    BuiltinSignatureDescriptor {
        label: "[y, Fs] = audioread(filename)",
        inputs: &AUDIOREAD_INPUTS_FILENAME,
        outputs: &AUDIOREAD_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "y = audioread(filename, samples)",
        inputs: &AUDIOREAD_INPUTS_RANGE,
        outputs: &AUDIOREAD_OUTPUTS_SAMPLES,
    },
    BuiltinSignatureDescriptor {
        label: "[y, Fs] = audioread(filename, samples)",
        inputs: &AUDIOREAD_INPUTS_RANGE,
        outputs: &AUDIOREAD_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "y = audioread(filename, datatype)",
        inputs: &AUDIOREAD_INPUTS_DATATYPE,
        outputs: &AUDIOREAD_OUTPUTS_SAMPLES,
    },
    BuiltinSignatureDescriptor {
        label: "[y, Fs] = audioread(filename, samples, datatype)",
        inputs: &AUDIOREAD_INPUTS_RANGE_NATIVE,
        outputs: &AUDIOREAD_OUTPUTS,
    },
];
pub const AUDIOREAD_ERROR_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AUDIOREAD.ARGUMENT",
    identifier: Some("RunMat:audioread:InvalidArgument"),
    when: "Filename, sample range, datatype, or output count is invalid.",
    message: "audioread: invalid argument",
};
pub const AUDIOREAD_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AUDIOREAD.IO",
    identifier: Some("RunMat:audioread:Io"),
    when: "The audio file cannot be read.",
    message: "audioread: unable to read file",
};
pub const AUDIOREAD_ERROR_FORMAT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AUDIOREAD.FORMAT",
    identifier: Some("RunMat:audioread:UnsupportedFormat"),
    when: "The file is not a supported audio container, uses unsupported audio coding, or is malformed.",
    message: "audioread: unsupported or invalid audio file",
};
pub(super) const AUDIOREAD_ERRORS: [BuiltinErrorDescriptor; 3] = [
    AUDIOREAD_ERROR_ARGUMENT,
    AUDIOREAD_ERROR_IO,
    AUDIOREAD_ERROR_FORMAT,
];

pub const AUDIOREAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &AUDIOREAD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &AUDIOREAD_ERRORS,
};

pub(super) const AUDIOREAD_NATIVE_INTEGER_CLASSES: [BuiltinIntegerClass; 3] = [
    BuiltinIntegerClass::Uint8,
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Int32,
];
pub(super) const AUDIOREAD_NATIVE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "native_audio_payload",
        classes: &AUDIOREAD_NATIVE_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "For documented integer-coded files, datatype=\"native\" follows the PCM container width: uint8 for 8-bit unsigned, int16 for 16-bit signed, and int32 for 24-bit left-aligned or 32-bit signed containers. Extensible valid-bit precision may be smaller than its container.",
    }];
pub(super) const AUDIOREAD_REJECTED_INTEGER_SAMPLE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "samples",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented two-element sample range has data type double; typed-integer scalar or array controls reject rather than widening the public surface.",
    }];
pub const AUDIOREAD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "y = audioread(filename, ..., \"native\")",
        inputs: &AUDIOREAD_NATIVE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Native integer samples are decoded directly into authoritative storage with file-format-dependent class and column-major N-by-C channel layout; Fs remains double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = audioread(filename, integer_samples, datatype)",
        inputs: &AUDIOREAD_REJECTED_INTEGER_SAMPLE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed-integer sample-range controls are not part of the documented double-only range contract and reject before filesystem I/O; audioread has no interactive GPU-array surface.",
    },
];
