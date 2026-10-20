use crate::*;

pub const DATA_TIP_TEXT_ROW_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATA_TIP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATA_TIP_ERRORS,
};

pub const DATA_TIP_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "row = dataTipTextRow(label, value)",
        inputs: &DATA_TIP_INPUTS_LABEL_VALUE,
        outputs: &DATA_TIP_OUTPUT_ROW,
    },
    BuiltinSignatureDescriptor {
        label: "row = dataTipTextRow(label, value, format)",
        inputs: &DATA_TIP_INPUTS_LABEL_VALUE_FORMAT,
        outputs: &DATA_TIP_OUTPUT_ROW,
    },
];

pub const DATA_TIP_INPUTS_LABEL_VALUE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "label",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Display label for the data-tip row.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value source, expression, or data vector shown by the row.",
    },
];

pub const DATA_TIP_OUTPUT_ROW: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "row",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Data-tip text row object with Label, Value, and Format properties.",
}];

pub const DATA_TIP_INPUTS_LABEL_VALUE_FORMAT: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "label",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Display label for the data-tip row.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value source, expression, or data vector shown by the row.",
    },
    BuiltinParamDescriptor {
        name: "format",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"auto\""),
        description: "Optional numeric, datetime, duration, or text display format.",
    },
];

pub const DATA_TIP_ERRORS: [BuiltinErrorDescriptor; 2] =
    [DATA_TIP_ERROR_INVALID_ARGUMENT, DATA_TIP_ERROR_INTERNAL];

pub const DATA_TIP_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATATIPTEXTROW.INVALID_ARGUMENT",
    identifier: Some("RunMat:dataTipTextRow:InvalidArgument"),
    when: "The label/format text is not scalar, or the argument count is not supported.",
    message: "dataTipTextRow: invalid argument",
};

pub const DATA_TIP_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATATIPTEXTROW.INTERNAL",
    identifier: Some("RunMat:dataTipTextRow:Internal"),
    when: "Internal data-tip row object construction fails.",
    message: "dataTipTextRow: internal error",
};

pub const DATA_TIP_TEXT_ROW_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    DATA_TIP_RESIDENT_VALUE_EXTENSION,
    DATA_TIP_NONVECTOR_VALUE_EXTENSION,
];

pub const DATA_TIP_RESIDENT_VALUE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "data-tip-text-row-resident-value",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "dataTipTextRow preserving a resident numeric Value source is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DataTipTextRowResidentValueExtension"),
    };

pub const DATA_TIP_NONVECTOR_VALUE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "data-tip-text-row-nonvector-value",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "dataTipTextRow with a nonvector numeric Value source is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DataTipTextRowNonvectorValueExtension"),
    };

pub const DATA_TIP_TEXT_ROW_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "row = dataTipTextRow(label, integer_vector, format?)", inputs: &DATA_TIP_INTEGER_VECTOR, computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Inferred vector support stores the exact integer Value object as metadata; formatting occurs later in the data-tip renderer." },
    BuiltinIntegerCapabilityDescriptor { form: "row = dataTipTextRow(label, integer_nonvector, format?)", inputs: &DATA_TIP_INTEGER_NONVECTOR, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "RunMat-only nonvector metadata preservation is independently gated." },
    BuiltinIntegerCapabilityDescriptor { form: "row = dataTipTextRow(label, resident_integer, format?)", inputs: &DATA_TIP_INTEGER_RESIDENT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::Multiple, notes: "RunMat-only resident metadata preservation performs no gather, upload, or provider dispatch." },
];

pub const DATA_TIP_INTEGER_VECTOR: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "value",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public Value contract says vector without a per-class table. RunMat classifies integer vectors as an evidence-based inference and preserves authoritative storage structurally without conversion.",
    }];

pub const DATA_TIP_INTEGER_NONVECTOR: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "value",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Nonvector integer Value sources are preserved only behind the separate RunMat extension gate.",
    }];

pub const DATA_TIP_INTEGER_RESIDENT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "resident value",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Resident integer Value metadata is preserved without provider access only behind the separate resident-value extension gate.",
    }];
