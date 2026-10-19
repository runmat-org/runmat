use crate::*;

const ANY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host table with file-derived row count and variable classes.",
}];
const PARQUETREAD_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Parquet file path.",
}];
const PARQUETREAD_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Parquet file path.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Options such as SelectedVariableNames, RowGroups, or OutputType.",
    },
];
const PARQUETREAD_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "T = parquetread(filename)",
        inputs: &PARQUETREAD_INPUTS_FILENAME,
        outputs: &ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "T = parquetread(filename, nameValuePairs...)",
        inputs: &PARQUETREAD_INPUTS_NAME_VALUE,
        outputs: &ANY_OUTPUT,
    },
];
pub const PARQUETREAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PARQUETREAD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "RowGroups",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "One-based row-group indices are decoded from native integer storage and checked against platform and opened-file bounds.",
}];
pub const PARQUETREAD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "T = parquetread(filename, 'RowGroups', integer_indices)",
    inputs: &INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::StructuralParameter,
    notes: "Indices choose native file row groups, not a numerical GPU kernel. Imported integer columns retain their file-derived native classes.",
}];
