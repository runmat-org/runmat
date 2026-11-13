use crate::*;

pub const MESH_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MESH.INVALID_ARGUMENT",
    identifier: Some("RunMat:mesh:InvalidArgument"),
    when: "Surface input arrays or style name/value arguments are invalid.",
    message: "mesh: invalid argument",
};

pub const MESH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MESH.INTERNAL",
    identifier: Some("RunMat:mesh:Internal"),
    when: "Internal surface generation/render preparation fails unexpectedly.",
    message: "mesh: internal operation failed",
};

pub(super) const MESH_ERRORS: [BuiltinErrorDescriptor; 3] = [
    MESH_ERROR_INVALID_ARGUMENT,
    MESH_ERROR_INTERNAL,
    COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
