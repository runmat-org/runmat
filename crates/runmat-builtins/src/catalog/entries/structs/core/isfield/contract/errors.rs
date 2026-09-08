use crate::BuiltinErrorDescriptor;

pub const ISFIELD_ERROR_FIELD_NAME_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISFIELD.FIELD_NAME_TYPE",
    identifier: Some("RunMat:isfield:FieldNameType"),
    when: "Field-name input is not a string scalar, string array, character row, or cell collection of scalar text.",
    message: "isfield: field names must be strings, string arrays, or cell arrays of character vectors",
};
pub const ISFIELD_ERROR_CELL_ELEMENT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISFIELD.CELL_ELEMENT_TYPE",
    identifier: Some("RunMat:isfield:CellElementType"),
    when: "A field-name cell element is not scalar string or character-row text.",
    message: "isfield: cell array elements must be character vectors or strings",
};
pub const ISFIELD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISFIELD.INTERNAL",
    identifier: Some("RunMat:isfield:InternalError"),
    when: "Logical result construction fails.",
    message: "isfield: internal error",
};

pub(super) const ERRORS: &[BuiltinErrorDescriptor] = &[
    ISFIELD_ERROR_FIELD_NAME_TYPE,
    ISFIELD_ERROR_CELL_ELEMENT_TYPE,
    ISFIELD_ERROR_INTERNAL,
];
