use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

const OUTPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Selected field or property value.",
}];
const DIRECT: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Structure or supported RunMat object-family value.",
    },
    BuiltinParamDescriptor {
        name: "field",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Field name.",
    },
];
const PATH: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Structure or supported RunMat object-family value.",
    },
    BuiltinParamDescriptor {
        name: "path",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Field names with optional cell-array index selectors.",
    },
];
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "value = getfield(S, field)",
        inputs: DIRECT,
        outputs: OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "value = getfield(S, field1, ..., fieldN)",
        inputs: PATH,
        outputs: OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "value = getfield(S, idx, field1, ..., fieldN)",
        inputs: PATH,
        outputs: OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "value = getfield(S, idx, field1, idx1, ..., fieldN, idxN)",
        inputs: PATH,
        outputs: OUTPUT,
    },
];

pub const GETFIELD_ERROR_NOT_ENOUGH_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.NOT_ENOUGH_INPUTS",
    identifier: Some("RunMat:getfield:NotEnoughInputs"),
    when: "No field path is supplied.",
    message: "getfield: expected at least one field name",
};
pub const GETFIELD_ERROR_FIELD_EXPECTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.FIELD_EXPECTED",
    identifier: Some("RunMat:getfield:FieldExpected"),
    when: "A selector is not followed by a field name.",
    message: "getfield: expected field name arguments",
};
pub const GETFIELD_ERROR_INDEX_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.INDEX_INVALID",
    identifier: Some("RunMat:getfield:InvalidIndex"),
    when: "An index component is invalid.",
    message: "getfield: invalid index element",
};
pub const GETFIELD_ERROR_INDEX_SELECTOR_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.INDEX_SELECTOR_TYPE",
    identifier: Some("RunMat:getfield:IndexSelectorType"),
    when: "An index selector is not a cell array.",
    message: "getfield: indices must be provided in a cell array",
};
pub const GETFIELD_ERROR_INDEX_SHAPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.INDEX_SHAPE",
    identifier: Some("RunMat:getfield:IndexShape"),
    when: "The index rank or shape is unsupported for the target.",
    message: "getfield: unsupported index shape for target value",
};
pub const GETFIELD_ERROR_FIELD_NAME_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.FIELD_NAME_TYPE",
    identifier: Some("RunMat:getfield:FieldNameType"),
    when: "A field name is not scalar text.",
    message: "getfield: expected field name",
};
pub const GETFIELD_ERROR_NON_STRUCT_REFERENCE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.NON_STRUCT_REFERENCE",
    identifier: Some("RunMat:getfield:NonStructReference"),
    when: "Field lookup targets an unsupported value.",
    message: "Struct contents reference from a non-struct array object.",
};
pub const GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.INDEX_OUT_OF_BOUNDS",
    identifier: Some("RunMat:getfield:IndexOutOfBounds"),
    when: "An index exceeds the target bounds.",
    message: "Index exceeds the number of array elements.",
};
pub const GETFIELD_ERROR_MISSING_FIELD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.MISSING_FIELD",
    identifier: Some("RunMat:getfield:MissingField"),
    when: "A requested field is absent.",
    message: "Reference to non-existent field",
};
pub const GETFIELD_ERROR_PROPERTY_PRIVATE_ACCESS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.PROPERTY_PRIVATE_ACCESS",
    identifier: Some("RunMat:PropertyPrivateAccess"),
    when: "A property is inaccessible from the current context.",
    message: "You cannot get this property from the current context.",
};
pub const GETFIELD_ERROR_OBJECT_PROPERTY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.OBJECT_PROPERTY",
    identifier: Some("RunMat:getfield:ObjectProperty"),
    when: "Object property access is invalid.",
    message: "getfield: invalid object property access",
};
pub const GETFIELD_ERROR_INVALID_HANDLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.INVALID_HANDLE",
    identifier: Some("RunMat:getfield:InvalidHandle"),
    when: "A handle is invalid or deleted.",
    message: "Invalid or deleted handle object",
};
pub const GETFIELD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.INTERNAL",
    identifier: Some("RunMat:getfield:InternalError"),
    when: "Execution cannot complete the access.",
    message: "getfield: internal error",
};
pub const GETFIELD_ERRORS: &[BuiltinErrorDescriptor] = &[
    GETFIELD_ERROR_NOT_ENOUGH_INPUTS,
    GETFIELD_ERROR_FIELD_EXPECTED,
    GETFIELD_ERROR_INDEX_SELECTOR_TYPE,
    GETFIELD_ERROR_INDEX_INVALID,
    GETFIELD_ERROR_INDEX_SHAPE,
    GETFIELD_ERROR_FIELD_NAME_TYPE,
    GETFIELD_ERROR_NON_STRUCT_REFERENCE,
    GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS,
    GETFIELD_ERROR_MISSING_FIELD,
    GETFIELD_ERROR_PROPERTY_PRIVATE_ACCESS,
    GETFIELD_ERROR_OBJECT_PROPERTY,
    GETFIELD_ERROR_INVALID_HANDLE,
    GETFIELD_ERROR_INTERNAL,
];

pub const GETFIELD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: GETFIELD_ERRORS,
};
