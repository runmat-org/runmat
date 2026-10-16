use crate::*;

pub(crate) const OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "opts",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Typed delimited-text or spreadsheet import-options object.",
};
pub(crate) const SELECTION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "selection",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Variable name, index, or logical selector.",
};
pub(crate) const PAIRS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nameValuePairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Supported per-variable name/value controls.",
};
pub(crate) const TYPE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "type",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One type or a type per selected variable.",
};
pub(crate) const UPDATED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "updatedOpts",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "An updated typed import-options object; the input is not mutated.",
}];
pub(crate) const VARIABLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "variableOpts",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Selected typed variable-options object or object array.",
}];
pub(crate) const INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMPORT_OPTIONS.INVALID_ARGUMENT",
    identifier: Some("RunMat:ImportOptions:InvalidArgument"),
    when: "The options object, selector, type, or variable control is unsupported or malformed.",
    message: "invalid import-options argument",
};
