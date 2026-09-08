use crate::BuiltinErrorDescriptor;

pub const FIELDNAMES_ERROR_INVALID_TARGET: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIELDNAMES.INVALID_TARGET",
    identifier: Some("RunMat:fieldnames:InvalidTarget"),
    when: "Input is not a structure, represented structure array, or supported RunMat object.",
    message: "fieldnames: expected struct, struct array, or supported object",
};
pub const FIELDNAMES_ERROR_STRUCT_ARRAY_CONTENTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIELDNAMES.STRUCT_ARRAY_CONTENTS",
    identifier: Some("RunMat:fieldnames:StructArrayContents"),
    when: "A represented structure array contains a non-structure element.",
    message: "fieldnames: expected struct array contents to be structs",
};
pub const FIELDNAMES_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIELDNAMES.INTERNAL",
    identifier: Some("RunMat:fieldnames:InternalError"),
    when: "Output construction or internal metadata processing fails.",
    message: "fieldnames: internal error",
};

pub(super) const ERRORS: &[BuiltinErrorDescriptor] = &[
    FIELDNAMES_ERROR_INVALID_TARGET,
    FIELDNAMES_ERROR_STRUCT_ARRAY_CONTENTS,
    FIELDNAMES_ERROR_INTERNAL,
];
