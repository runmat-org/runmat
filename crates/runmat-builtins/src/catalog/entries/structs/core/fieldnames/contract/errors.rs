use crate::BuiltinErrorDescriptor;

pub const FIELDNAMES_ERROR_INVALID_TARGET: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIELDNAMES.INVALID_TARGET",
    identifier: Some("RunMat:fieldnames:InvalidTarget"),
    when: "Input is not a structure or supported RunMat object.",
    message: "fieldnames: expected struct, struct array, or supported object",
};
pub const FIELDNAMES_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIELDNAMES.INTERNAL",
    identifier: Some("RunMat:fieldnames:InternalError"),
    when: "Output construction or internal metadata processing fails.",
    message: "fieldnames: internal error",
};

pub(super) const ERRORS: &[BuiltinErrorDescriptor] =
    &[FIELDNAMES_ERROR_INVALID_TARGET, FIELDNAMES_ERROR_INTERNAL];
