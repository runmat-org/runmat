//! Shared descriptor vocabulary for this runtime mechanism.
use crate::*;
pub const MEMOIZED_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "memoizedFcn",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "MemoizedFunction object.",
}];

pub const MEMOIZE_ERROR_INVALID_OBJECT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMOIZE.INVALID_OBJECT",
    identifier: Some("RunMat:memoize:InvalidObject"),
    when: "A memoized-function method receives a non-MemoizedFunction object.",
    message: "memoize: expected MemoizedFunction object",
};

pub const MEMOIZE_ERROR_INVALID_INDEX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMOIZE.INVALID_INDEX",
    identifier: Some("RunMat:memoize:InvalidIndex"),
    when: "The MemoizedFunction object is indexed with an unsupported indexing form.",
    message: "memoize: unsupported MemoizedFunction indexing",
};

pub const MEMOIZE_ERROR_GC: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMOIZE.GC",
    identifier: Some("RunMat:memoize:GcFailure"),
    when: "The memoized object target cannot be allocated or accessed.",
    message: "memoize: internal object storage failed",
};
