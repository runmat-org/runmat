use crate::*;

const SOURCE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "source",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Source file, directory, or wildcard pattern.",
};
const DESTINATION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "destination",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Destination file or directory path.",
};
const FORCE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "flag",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("'f'"),
    description: "Case-insensitive 'f' flag that permits replacement of an existing target.",
};
const STATUS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Double scalar 1 after success or 0 after an operational failure.",
};
const MESSAGE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "msg",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Character-row diagnostic, empty after success.",
};
const MESSAGE_ID: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "msgID",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Character-row diagnostic identifier, empty after success.",
};
const OUTPUTS: &[BuiltinParamDescriptor] = &[STATUS, MESSAGE, MESSAGE_ID];

pub(super) const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "[status, msg, msgID] = copyfile(source, destination)",
        inputs: &[SOURCE, DESTINATION],
        outputs: OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[status, msg, msgID] = copyfile(source, destination, flag)",
        inputs: &[SOURCE, DESTINATION, FORCE],
        outputs: OUTPUTS,
    },
];
