use crate::*;

pub const AREA_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AREA.INVALID_ARGUMENT",
    identifier: Some("RunMat:area:InvalidArgument"),
    when: "Input data, style tokens, or name/value options are invalid.",
    message: "area: invalid argument",
};

pub const AREA_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AREA.INTERNAL",
    identifier: Some("RunMat:area:Internal"),
    when: "Renderer/GPU conversion fails during chart construction.",
    message: "area: internal operation failed",
};

pub(super) const AREA_ERRORS: [BuiltinErrorDescriptor; 3] = [
    AREA_ERROR_INVALID_ARGUMENT,
    AREA_ERROR_INTERNAL,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
