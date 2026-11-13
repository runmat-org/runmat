use crate::*;

pub(super) const RECEIVER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "m",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One sealed memmapfile receiver issued by the selected filesystem provider.",
};

pub(super) const SUBSCRIPT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Standard structure subscript describing a public property or mapped Data selection.",
};

pub(super) const INVALID_SUBSCRIPT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMMAPFILE.METHOD.INVALID_SUBSCRIPT",
    identifier: Some("RunMat:InvalidObjectSubscriptPath"),
    when: "S is not a valid standard object subscript path.",
    message: "memmapfile: invalid standard subscript path",
};

pub(super) const CANCELLED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMMAPFILE.METHOD.CANCELLED",
    identifier: Some("RunMat:parallel:Cancelled"),
    when: "The execution service cancels a pending selector, gather, or mapped-range operation.",
    message: "memmapfile: mapped operation cancelled",
};

pub(super) const PROVIDER_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMMAPFILE.METHOD.PROVIDER_ERROR",
    identifier: None,
    when: "A resident index or rhs fails to materialize; its original provider diagnostic and cause are retained.",
    message: "memmapfile: provider materialization failed",
};
