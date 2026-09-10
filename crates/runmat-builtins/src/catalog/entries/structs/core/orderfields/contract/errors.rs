use crate::BuiltinErrorDescriptor;

macro_rules! error {
    ($name:ident, $code:literal, $identifier:literal, $when:literal, $message:literal) => {
        pub const $name: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: $code,
            identifier: Some($identifier),
            when: $when,
            message: $message,
        };
    };
}

error!(
    ORDERFIELDS_ERROR_TOO_MANY_INPUTS,
    "RM.ORDERFIELDS.TOO_MANY_INPUTS",
    "orderfields:TooManyInputs",
    "More than two inputs are supplied.",
    "orderfields: expected at most two input arguments"
);
error!(
    ORDERFIELDS_ERROR_INVALID_TARGET,
    "RM.ORDERFIELDS.INVALID_INPUT",
    "orderfields:InvalidInput",
    "The first input is not a structure.",
    "orderfields: first argument must be a struct or struct array"
);
error!(
    ORDERFIELDS_ERROR_EMPTY_STRUCT_ARRAY,
    "RM.ORDERFIELDS.EMPTY_STRUCT_ARRAY",
    "orderfields:EmptyStructArray",
    "An empty structure array is given a nonempty reference structure.",
    "orderfields: empty struct arrays cannot adopt a non-empty reference order"
);
error!(
    ORDERFIELDS_ERROR_NO_FIELDS,
    "RM.ORDERFIELDS.NO_FIELDS",
    "orderfields:NoFields",
    "A nonempty name or numeric order is supplied for a structure array without field metadata.",
    "orderfields: struct array has no fields to reorder"
);
error!(
    ORDERFIELDS_ERROR_INVALID_REFERENCE,
    "RM.ORDERFIELDS.INVALID_REFERENCE",
    "orderfields:InvalidReference",
    "A reference structure array contains a non-structure element or inconsistent field schema.",
    "orderfields: reference struct array must contain structures with one field schema"
);
error!(
    ORDERFIELDS_ERROR_INVALID_NAME_LIST,
    "RM.ORDERFIELDS.INVALID_NAME_LIST",
    "orderfields:InvalidFieldNameList",
    "A field-name collection contains a value other than scalar text.",
    "orderfields: field names must be string scalars or character vectors"
);
error!(
    ORDERFIELDS_ERROR_EMPTY_FIELD_NAME,
    "RM.ORDERFIELDS.EMPTY_FIELD_NAME",
    "orderfields:EmptyFieldName",
    "A requested field name is empty.",
    "orderfields: field names must be nonempty"
);
error!(
    ORDERFIELDS_ERROR_INVALID_PERMUTATION,
    "RM.ORDERFIELDS.INVALID_PERMUTATION",
    "orderfields:InvalidPermutation",
    "The numeric order does not contain exactly one position for every field.",
    "orderfields: index vector must permute every field exactly once"
);
error!(
    ORDERFIELDS_ERROR_INDEX_NOT_INTEGER,
    "RM.ORDERFIELDS.INDEX_NOT_INTEGER",
    "orderfields:IndexNotInteger",
    "A numeric order contains a noninteger value.",
    "orderfields: index vector must contain integers"
);
error!(
    ORDERFIELDS_ERROR_INDEX_OUT_OF_RANGE,
    "RM.ORDERFIELDS.INDEX_OUT_OF_RANGE",
    "orderfields:IndexOutOfRange",
    "A numeric order contains a position outside the field range.",
    "orderfields: index vector element out of range"
);
error!(
    ORDERFIELDS_ERROR_DUPLICATE_INDEX,
    "RM.ORDERFIELDS.INDEX_DUPLICATE",
    "orderfields:DuplicateIndex",
    "A numeric order repeats a field position.",
    "orderfields: index vector contains duplicate positions"
);
error!(
    ORDERFIELDS_ERROR_FIELD_MISMATCH,
    "RM.ORDERFIELDS.FIELD_MISMATCH",
    "orderfields:FieldMismatch",
    "A reference structure has a different field set.",
    "orderfields: field names must match the struct exactly"
);
error!(
    ORDERFIELDS_ERROR_UNKNOWN_FIELD,
    "RM.ORDERFIELDS.UNKNOWN_FIELD",
    "orderfields:UnknownField",
    "A requested name is absent from the input structure.",
    "orderfields: unknown field in requested order"
);
error!(
    ORDERFIELDS_ERROR_DUPLICATE_FIELD,
    "RM.ORDERFIELDS.DUPLICATE_FIELD",
    "orderfields:DuplicateField",
    "A requested field name occurs more than once.",
    "orderfields: duplicate field in requested order"
);
error!(
    ORDERFIELDS_ERROR_INVALID_ORDER,
    "RM.ORDERFIELDS.INVALID_ORDER_ARGUMENT",
    "orderfields:InvalidOrderArgument",
    "The second input is not a supported order description.",
    "orderfields: unrecognised ordering argument"
);
error!(
    ORDERFIELDS_ERROR_REBUILD_FAILED,
    "RM.ORDERFIELDS.REBUILD_FAILED",
    "orderfields:RebuildFailed",
    "The structure array cannot retain its original shape.",
    "orderfields: failed to rebuild struct array"
);

pub(super) const ERRORS: &[BuiltinErrorDescriptor] = &[
    ORDERFIELDS_ERROR_TOO_MANY_INPUTS,
    ORDERFIELDS_ERROR_INVALID_TARGET,
    ORDERFIELDS_ERROR_EMPTY_STRUCT_ARRAY,
    ORDERFIELDS_ERROR_NO_FIELDS,
    ORDERFIELDS_ERROR_INVALID_REFERENCE,
    ORDERFIELDS_ERROR_INVALID_NAME_LIST,
    ORDERFIELDS_ERROR_EMPTY_FIELD_NAME,
    ORDERFIELDS_ERROR_INVALID_PERMUTATION,
    ORDERFIELDS_ERROR_INDEX_NOT_INTEGER,
    ORDERFIELDS_ERROR_INDEX_OUT_OF_RANGE,
    ORDERFIELDS_ERROR_DUPLICATE_INDEX,
    ORDERFIELDS_ERROR_FIELD_MISMATCH,
    ORDERFIELDS_ERROR_UNKNOWN_FIELD,
    ORDERFIELDS_ERROR_DUPLICATE_FIELD,
    ORDERFIELDS_ERROR_INVALID_ORDER,
    ORDERFIELDS_ERROR_REBUILD_FAILED,
];
