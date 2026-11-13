use crate::BuiltinErrorDescriptor;

pub const SETFIELD_ERROR_NOT_ENOUGH_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETFIELD.NOT_ENOUGH_INPUTS",
    identifier: Some("RunMat:setfield:NotEnoughInputs"),
    when: "A field path and value are not supplied.",
    message: "setfield: expected at least one field name and a value",
};
pub const SETFIELD_ERROR_FIELD_EXPECTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETFIELD.FIELD_EXPECTED",
    identifier: Some("RunMat:setfield:FieldExpected"),
    when: "A selector is not followed by a field name.",
    message: "setfield: expected field name arguments",
};
pub const SETFIELD_ERROR_INDEX_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETFIELD.INDEX_INVALID",
    identifier: Some("RunMat:setfield:InvalidIndex"),
    when: "An index component is invalid.",
    message: "setfield: invalid index element",
};
pub const SETFIELD_ERROR_INDEX_SELECTOR_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETFIELD.INDEX_SELECTOR_TYPE",
    identifier: Some("RunMat:setfield:IndexSelectorType"),
    when: "An index selector is not a cell array.",
    message: "setfield: indices must be provided in a cell array",
};
pub const SETFIELD_ERROR_INDEX_SHAPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETFIELD.INDEX_SHAPE",
    identifier: Some("RunMat:setfield:IndexShape"),
    when: "The index rank or shape is unsupported for the target.",
    message: "setfield: unsupported index shape for target value",
};
pub const SETFIELD_ERROR_FIELD_NAME_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETFIELD.FIELD_NAME_TYPE",
    identifier: Some("RunMat:setfield:FieldNameType"),
    when: "A field name is not scalar text.",
    message: "setfield: expected field name",
};
pub const SETFIELD_ERROR_NON_STRUCT_ASSIGNMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETFIELD.NON_STRUCT_ASSIGNMENT",
    identifier: Some("RunMat:setfield:NonStructAssignment"),
    when: "Field assignment targets an unsupported value.",
    message: "Struct contents assignment to a non-struct object is not supported.",
};
pub const SETFIELD_ERROR_INDEX_OUT_OF_BOUNDS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETFIELD.INDEX_OUT_OF_BOUNDS",
    identifier: Some("RunMat:setfield:IndexOutOfBounds"),
    when: "An index exceeds the target bounds.",
    message: "Index exceeds the number of array elements.",
};
pub const SETFIELD_ERROR_MISSING_FIELD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETFIELD.MISSING_FIELD",
    identifier: Some("RunMat:setfield:MissingField"),
    when: "Indexed assignment targets an absent field that must already exist.",
    message: "Reference to non-existent field",
};
pub const SETFIELD_ERROR_PROPERTY_PRIVATE_ACCESS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETFIELD.PROPERTY_PRIVATE_ACCESS",
    identifier: Some("RunMat:PropertyPrivateAccess"),
    when: "A property is inaccessible from the current context.",
    message: "setfield: private property access denied",
};
pub const SETFIELD_ERROR_PROPERTY_STATIC_ACCESS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETFIELD.PROPERTY_STATIC_ACCESS",
    identifier: Some("RunMat:PropertyStaticAccess"),
    when: "A static property is assigned through an instance.",
    message: "setfield: static property access denied",
};
pub const SETFIELD_ERROR_OBJECT_PROPERTY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETFIELD.OBJECT_PROPERTY",
    identifier: Some("RunMat:setfield:ObjectProperty"),
    when: "Object property assignment is invalid.",
    message: "setfield: invalid object property operation",
};
pub const SETFIELD_ERROR_INVALID_HANDLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETFIELD.INVALID_HANDLE",
    identifier: Some("RunMat:setfield:InvalidHandle"),
    when: "A handle is invalid or deleted.",
    message: "setfield: invalid or deleted handle object",
};
pub const SETFIELD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETFIELD.INTERNAL",
    identifier: Some("RunMat:setfield:InternalError"),
    when: "Execution cannot complete the assignment.",
    message: "setfield: internal error",
};
pub const SETFIELD_ERRORS: &[BuiltinErrorDescriptor] = &[
    SETFIELD_ERROR_TOO_MANY_OUTPUTS,
    SETFIELD_ERROR_NOT_ENOUGH_INPUTS,
    SETFIELD_ERROR_FIELD_EXPECTED,
    SETFIELD_ERROR_INDEX_SELECTOR_TYPE,
    SETFIELD_ERROR_INDEX_INVALID,
    SETFIELD_ERROR_INDEX_SHAPE,
    SETFIELD_ERROR_FIELD_NAME_TYPE,
    SETFIELD_ERROR_NON_STRUCT_ASSIGNMENT,
    SETFIELD_ERROR_INDEX_OUT_OF_BOUNDS,
    SETFIELD_ERROR_MISSING_FIELD,
    SETFIELD_ERROR_PROPERTY_PRIVATE_ACCESS,
    SETFIELD_ERROR_PROPERTY_STATIC_ACCESS,
    SETFIELD_ERROR_OBJECT_PROPERTY,
    SETFIELD_ERROR_INVALID_HANDLE,
    SETFIELD_ERROR_INTERNAL,
];
pub const SETFIELD_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETFIELD.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:setfield:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "setfield: too many output arguments",
};
