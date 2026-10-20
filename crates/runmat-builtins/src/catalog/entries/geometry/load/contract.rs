use crate::*;

pub const GEOMETRY_ASSET_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("geometry.Asset");

pub const INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "geometry.load accepts a scalar text path and a text Units option; imported numeric geometry is not a direct integer argument.",
};

const INPUT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "path",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar path to an importable geometry file.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "Optional 'Units' and 'm', 'mm', or 'in': the source unit when the file declares none.",
    },
];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "asset",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host geometry.Asset object.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "asset = geometry.load(path, Name, Value)",
    inputs: &INPUT,
    outputs: &OUTPUT,
}];
pub const ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.LOAD.IO",
    identifier: Some("RunMat:geometry:load:IoFailure"),
    when: "The geometry file cannot be read.",
    message: "geometry.load: failed to read geometry file",
};
pub const ERROR_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.LOAD.INVALID_PATH",
    identifier: Some("RunMat:geometry:load:InvalidPath"),
    when: "The path is not scalar string text or a character row, or an option is not a supported Units Name, Value pair.",
    message: "geometry.load: invalid path",
};
pub const ERROR_OPERATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.LOAD.OPERATION_FAILED",
    identifier: Some("RunMat:geometry:load:OperationFailed"),
    when: "The geometry importer rejects or cannot import the file.",
    message: "geometry.load: operation failed",
};
pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.LOAD.INTERNAL",
    identifier: Some("RunMat:geometry:load:Internal"),
    when: "The geometry asset cannot be converted to a RunMat object.",
    message: "geometry.load: internal error",
};
pub const ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.LOAD.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:geometry:load:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "geometry.load: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    ERROR_INPUT,
    ERROR_IO,
    ERROR_OPERATION,
    ERROR_INTERNAL,
    ERROR_OUTPUTS,
];
pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
