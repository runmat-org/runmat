use crate::*;

const PARQUET_DATASTORE_INTEGER_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ReadSize, BlockSize, location, or other constructor argument",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented ReadSize and BlockSize numeric datatype is double, while locations and other constructor options are text, logical, or objects. Typed-integer host or resident arguments reject before gather and cannot be silently ignored.",
    }];
pub const PARQUET_DATASTORE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "pds = parquetDatastore(location, typed_integer_argument...)",
        inputs: &PARQUET_DATASTORE_INTEGER_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The constructor records supported host metadata only. Integer Parquet columns belong to parquetread/read datastore operations, not to this constructor input surface.",
    }];

pub const PARQUET_DATASTORE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "ds = parquetDatastore(files, Name, Value...)",
            inputs: &[
                BuiltinParamDescriptor { name: "files", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None, description: "File specification retained without reading, expanding, or validating its location." },
                BuiltinParamDescriptor { name: "Name,Value", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Variadic, default: None, description: "ReadSize is 'rowgroup' (default), 'file', or a positive double integer; BlockSize is a positive double integer (default 128000000)." },
            ],
            outputs: &[BuiltinParamDescriptor { name: "descriptor", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None, description: "Scalar host value descriptor; not a live iterator or UI handle." }],
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[super::super::super::shared::TABLE_ERROR_INVALID_ARGUMENT],
};
