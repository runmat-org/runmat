use crate::*;

pub const DATACURSORMODE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATACURSORMODE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATACURSORMODE_ERRORS,
};

pub const DATACURSORMODE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "dcm = datacursormode()",
        inputs: &DATACURSORMODE_INPUTS_NONE,
        outputs: &DATACURSORMODE_OUTPUT_OBJECT,
    },
    BuiltinSignatureDescriptor {
        label: "datacursormode()",
        inputs: &DATACURSORMODE_INPUTS_NONE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "datacursormode(option)",
        inputs: &DATACURSORMODE_INPUTS_OPTION,
        outputs: &[],
    },
];

pub const DATACURSORMODE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

pub const DATACURSORMODE_OUTPUT_OBJECT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dcm",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Data cursor mode object for the current or specified figure.",
}];

pub const DATACURSORMODE_INPUTS_OPTION: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Mode option: on, off, or toggle.",
}];

pub const DATACURSORMODE_ERRORS: [BuiltinErrorDescriptor; 2] = [
    DATACURSORMODE_ERROR_INVALID_ARGUMENT,
    DATACURSORMODE_ERROR_INTERNAL,
];

pub const DATACURSORMODE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATACURSORMODE.INVALID_ARGUMENT",
    identifier: Some("RunMat:datacursormode:InvalidArgument"),
    when: "Argument count, figure handle, option, or property is invalid.",
    message: "datacursormode: invalid argument",
};

pub const DATACURSORMODE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATACURSORMODE.INTERNAL",
    identifier: Some("RunMat:datacursormode:Internal"),
    when: "Internal data cursor state or handle-object update fails.",
    message: "datacursormode: internal operation failed",
};

pub const DATACURSORMODE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    DATACURSORMODE_DATACURSOR_NUMERIC_HANDLE_EXTENSION,
    DATACURSORMODE_DATACURSOR_STATUS_OUTPUT_EXTENSION,
    DATACURSORMODE_DATACURSOR_ONOFF_ALIASES_EXTENSION,
];

pub const DATACURSORMODE_DATACURSOR_NUMERIC_HANDLE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datacursormode-numeric-handle-alias",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "datacursormode with RunMat numeric graphics-handle aliases is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatacursormodeNumericHandleExtension"),
    };

pub const DATACURSORMODE_DATACURSOR_STATUS_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datacursormode-status-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "status output from a datacursormode command form is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatacursormodeStatusOutputExtension"),
    };

pub const DATACURSORMODE_DATACURSOR_ONOFF_ALIASES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datacursormode-onoff-aliases",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "nonstandard datacursormode on/off numeric and text aliases are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatacursormodeOnOffAliasesExtension"),
    };

pub const DATACURSORMODE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "dcm = datacursormode(integer_fig)", inputs: &DATACURSORMODE_DATACURSOR_INTEGER_HANDLE, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "RunMat-only exact numeric handle alias for the current numeric graphics registry; native Figure-object admission is not implemented." },
    BuiltinIntegerCapabilityDescriptor { form: "set(dcm, 'Enable', integer_state)", inputs: &DATACURSORMODE_DATACURSOR_INTEGER_ENABLE, computation_domain: BuiltinIntegerComputationDomain::Predicate, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "All eight classes are admitted only for authoritative scalar zero or one." },
    BuiltinIntegerCapabilityDescriptor { form: "datacursormode(resident_integer, ...)", inputs: &DATACURSORMODE_DATACURSOR_RESIDENT_INTEGER, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Interaction metadata never gathers a resident array to reinterpret it as a handle or switch." },
];

pub const DATACURSORMODE_DATACURSOR_INTEGER_HANDLE: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fig",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed integers are exact scalar aliases for RunMat's numeric figure registry and are separately compatibility gated; this extension-only form is not advertised as a public signature.",
    }];

pub const DATACURSORMODE_DATACURSOR_INTEGER_ENABLE: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Enable",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented numeric on/off property accepts only exact scalar 0 or 1; other numeric truthiness is a separately gated alias.",
    }];

pub const DATACURSORMODE_DATACURSOR_RESIDENT_INTEGER: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "resident fig or property",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Resident numeric arrays reject without provider access.",
    }];
