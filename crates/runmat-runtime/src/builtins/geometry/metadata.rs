use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

pub const GEOMETRY_ASSET_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("geometry.Asset");
pub(super) const GEOMETRY_INSPECT_RESULT_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("geometry.InspectResult");
pub const GEOMETRY_ASSET_JSON_PROPERTY: &str = "__runmat_geometry_asset_json";
pub(super) const GEOMETRY_LOAD_NAME: &str = "geometry.load";
pub(super) const GEOMETRY_INSPECT_NAME: &str = "geometry.inspect";
pub(super) const GEOMETRY_LIST_REGIONS_NAME: &str = "geometry.listRegions";
pub(super) const GEOMETRY_MESHES_NAME: &str = "geometry.meshes";

pub const GEOMETRY_LIST_REGIONS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "geometry.listRegions accepts only a geometry.Asset object and returns imported-region metadata; it has no direct integer data or control argument.",
    };
pub const GEOMETRY_MESHES_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "geometry.meshes accepts only a geometry.Asset object and returns host mesh structs; vertices, 1-based faces/triangles, and region ranges are deliberate double-valued API outputs rather than a typed-integer input surface.",
    };

pub(super) const STRUCT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Operation result as a struct.",
}];
pub(super) const PATH_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "path",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Path to the geometry file.",
}];

pub(super) const GEOMETRY_LOAD_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "asset = geometry.load(path)",
        inputs: &PATH_INPUT,
        outputs: &STRUCT_OUTPUT,
    }];
pub(super) const GEOMETRY_INSPECT_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "info = geometry.inspect(path)",
        inputs: &PATH_INPUT,
        outputs: &STRUCT_OUTPUT,
    }];
pub(super) const GEOMETRY_LIST_REGIONS_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "regions = geometry.listRegions(asset)",
        inputs: &STRUCT_OUTPUT,
        outputs: &STRUCT_OUTPUT,
    }];
pub(super) const GEOMETRY_MESHES_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "meshes = geometry.meshes(asset)",
        inputs: &STRUCT_OUTPUT,
        outputs: &STRUCT_OUTPUT,
    }];

pub(super) const GEOMETRY_LOAD_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.LOAD.IO",
    identifier: Some("RunMat:geometry:load:IoFailure"),
    when: "The geometry file cannot be read.",
    message: "geometry.load: failed to read geometry file",
};
pub(super) const GEOMETRY_LOAD_ERROR_OPERATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.LOAD.OPERATION_FAILED",
    identifier: Some("RunMat:geometry:load:OperationFailed"),
    when: "The geometry load operation rejects or cannot import the file.",
    message: "geometry.load: operation failed",
};
pub(super) const GEOMETRY_LOAD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.LOAD.INTERNAL",
    identifier: Some("RunMat:geometry:load:Internal"),
    when: "The loaded geometry asset cannot be converted to a RunMat value.",
    message: "geometry.load: internal error",
};
pub(super) const GEOMETRY_LOAD_ERRORS: [BuiltinErrorDescriptor; 3] = [
    GEOMETRY_LOAD_ERROR_IO,
    GEOMETRY_LOAD_ERROR_OPERATION,
    GEOMETRY_LOAD_ERROR_INTERNAL,
];
pub const GEOMETRY_LOAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GEOMETRY_LOAD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GEOMETRY_LOAD_ERRORS,
};

pub(super) const GEOMETRY_INSPECT_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.INSPECT.IO",
    identifier: Some("RunMat:geometry:inspect:IoFailure"),
    when: "The geometry file cannot be read.",
    message: "geometry.inspect: failed to read geometry file",
};
pub(super) const GEOMETRY_INSPECT_ERROR_OPERATION: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.GEOMETRY.INSPECT.OPERATION_FAILED",
        identifier: Some("RunMat:geometry:inspect:OperationFailed"),
        when: "The geometry inspection operation fails.",
        message: "geometry.inspect: operation failed",
    };
pub(super) const GEOMETRY_INSPECT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.INSPECT.INTERNAL",
    identifier: Some("RunMat:geometry:inspect:Internal"),
    when: "The inspection result cannot be converted to a RunMat value.",
    message: "geometry.inspect: internal error",
};
pub(super) const GEOMETRY_INSPECT_ERRORS: [BuiltinErrorDescriptor; 3] = [
    GEOMETRY_INSPECT_ERROR_IO,
    GEOMETRY_INSPECT_ERROR_OPERATION,
    GEOMETRY_INSPECT_ERROR_INTERNAL,
];
pub const GEOMETRY_INSPECT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GEOMETRY_INSPECT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GEOMETRY_INSPECT_ERRORS,
};
pub const GEOMETRY_LIST_REGIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GEOMETRY_LIST_REGIONS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GEOMETRY_INSPECT_ERRORS,
};
pub const GEOMETRY_MESHES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GEOMETRY_MESHES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GEOMETRY_INSPECT_ERRORS,
};
