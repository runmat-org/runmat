use crate::*;
pub const DYNAMIC_ERROR_INVALID_TARGET: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DYNAMICPROPS.INVALID_TARGET",
    identifier: Some("RunMat:dynamicprops:InvalidTarget"),
    when: "The target is not a valid handle object for the selected operation.",
    message: "dynamicprops: target must be a valid handle object",
};
pub const DYNAMIC_ERROR_INVALID_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DYNAMICPROPS.INVALID_NAME",
    identifier: Some("RunMat:dynamicprops:InvalidName"),
    when: "The property name is empty or not a valid identifier.",
    message: "dynamicprops: invalid property name",
};
pub const DYNAMIC_ERROR_DUPLICATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DYNAMICPROPS.DUPLICATE_PROPERTY",
    identifier: Some("RunMat:dynamicprops:DuplicateProperty"),
    when: "A class-defined or dynamic property has the requested name.",
    message: "dynamicprops: property already exists",
};
pub const DYNAMIC_ERROR_INVALID_METADATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DYNAMICPROPS.INVALID_METADATA",
    identifier: Some("RunMat:dynamicprops:InvalidMetadata"),
    when: "The dynamic-property metadata handle is invalid, stale, or malformed.",
    message: "dynamicprops: invalid dynamic property metadata",
};
pub const DYNAMIC_ERROR_UNSUPPORTED_METADATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DYNAMICPROPS.UNSUPPORTED_METADATA_PROPERTY",
    identifier: Some("RunMat:dynamicprops:UnsupportedMetadataProperty"),
    when: "A metadata assignment uses an unsupported field or value.",
    message: "dynamicprops: unsupported dynamic property metadata assignment",
};
