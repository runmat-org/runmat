use crate::BuiltinErrorDescriptor;

pub const RMFIELD_ERROR_NOT_ENOUGH_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RMFIELD.NOT_ENOUGH_INPUTS",
    identifier: Some("RunMat:rmfield:NotEnoughInputs"),
    when: "No field-name argument is supplied.",
    message: "rmfield: not enough input arguments",
};
pub const RMFIELD_ERROR_INVALID_TARGET: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RMFIELD.INVALID_TARGET",
    identifier: Some("RunMat:rmfield:InvalidTarget"),
    when: "The first input is not a structure.",
    message: "rmfield: expected struct or struct array",
};
pub const RMFIELD_ERROR_FIELD_NAME_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RMFIELD.FIELD_NAME_TYPE",
    identifier: Some("RunMat:rmfield:FieldNameType"),
    when: "A field-name input is not a character row, string value, string array, or cell collection of scalar text.",
    message: "rmfield: field names must be string scalars, character vectors, or single-element string arrays",
};
pub const RMFIELD_ERROR_FIELD_NAME_EMPTY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RMFIELD.FIELD_NAME_EMPTY",
    identifier: Some("RunMat:rmfield:FieldNameEmpty"),
    when: "A field name is empty.",
    message: "rmfield: field names must be nonempty character vectors or strings",
};
pub const RMFIELD_ERROR_MISSING_FIELD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RMFIELD.MISSING_FIELD",
    identifier: Some("RunMat:rmfield:MissingField"),
    when: "A requested field does not exist on every input structure element.",
    message: "Reference to non-existent field",
};
pub const RMFIELD_ERROR_REBUILD_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RMFIELD.REBUILD_FAILED",
    identifier: Some("RunMat:rmfield:RebuildFailed"),
    when: "The updated structure array cannot retain its original shape.",
    message: "rmfield: failed to rebuild struct array",
};

pub(super) const ERRORS: &[BuiltinErrorDescriptor] = &[
    RMFIELD_ERROR_NOT_ENOUGH_INPUTS,
    RMFIELD_ERROR_INVALID_TARGET,
    RMFIELD_ERROR_FIELD_NAME_TYPE,
    RMFIELD_ERROR_FIELD_NAME_EMPTY,
    RMFIELD_ERROR_MISSING_FIELD,
    RMFIELD_ERROR_REBUILD_FAILED,
];
