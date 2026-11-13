use crate::*;

const ANY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "info",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar metadata struct with file-derived counts and schema fields.",
}];
const PARQUETINFO_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Parquet file path to inspect.",
}];
const PARQUETINFO_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "info = parquetinfo(filename)",
    inputs: &PARQUETINFO_INPUTS,
    outputs: &ANY_OUTPUT,
}];
pub const PARQUETINFO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PARQUETINFO_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
