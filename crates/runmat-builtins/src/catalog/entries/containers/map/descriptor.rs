use crate::*;

const CONTAINERS_MAP_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "containers.Map handle object.",
}];

const CONTAINERS_MAP_INPUTS_KEYS_VALUES: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "keys",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Key container (cell, string/char, or numeric vector).",
    },
    BuiltinParamDescriptor {
        name: "values",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value container aligned with keys.",
    },
];

const CONTAINERS_MAP_INPUTS_KEYS_VALUES_OPTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "keys",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Key container (cell, string/char, or numeric vector).",
    },
    BuiltinParamDescriptor {
        name: "values",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value container aligned with keys.",
    },
    BuiltinParamDescriptor {
        name: "UniformValues",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Literal UniformValues option name.",
    },
    BuiltinParamDescriptor {
        name: "isUniform",
        ty: BuiltinParamType::LogicalArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Logical scalar selecting uniform-value validation.",
    },
];

const CONTAINERS_MAP_INPUTS_OPTIONS_ONLY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "The required KeyType and ValueType name/value pairs, in either order.",
}];

const CONTAINERS_MAP_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "M = containers.Map()",
        inputs: &[],
        outputs: &CONTAINERS_MAP_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = containers.Map(keys, values)",
        inputs: &CONTAINERS_MAP_INPUTS_KEYS_VALUES,
        outputs: &CONTAINERS_MAP_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = containers.Map(keys, values, 'UniformValues', isUniform)",
        inputs: &CONTAINERS_MAP_INPUTS_KEYS_VALUES_OPTS,
        outputs: &CONTAINERS_MAP_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = containers.Map('KeyType', kType, 'ValueType', vType)",
        inputs: &CONTAINERS_MAP_INPUTS_OPTIONS_ONLY,
        outputs: &CONTAINERS_MAP_OUTPUT,
    },
];

const CONTAINERS_MAP_METHOD_INPUT_MAP: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "containers.Map handle object.",
}];

const CONTAINERS_MAP_KEYS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "K",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Row cell array containing map keys.",
}];

const CONTAINERS_MAP_VALUES_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "V",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Row cell array containing map values.",
}];
const CONTAINERS_MAP_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const CONTAINERS_MAP_INPUTS_KEY_SPEC: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "containers.Map handle object.",
    },
    BuiltinParamDescriptor {
        name: "keySet",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Key scalar or key collection to query/mutate.",
    },
];

const CONTAINERS_MAP_ISKEY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical membership result for each key.",
}];

const CONTAINERS_MAP_INPUTS_SUBSREF: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "containers.Map handle object.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard substruct-compatible indexing path.",
    },
];

const CONTAINERS_MAP_INPUTS_SUBSASGN: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "containers.Map handle object.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard substruct-compatible assignment path.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Assigned value (scalar or collection).",
    },
];

const CONTAINERS_MAP_SUBSREF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Lookup/property value result.",
}];

const CONTAINERS_MAP_KEYS_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "K = containers.Map.keys(M)",
        inputs: &CONTAINERS_MAP_METHOD_INPUT_MAP,
        outputs: &CONTAINERS_MAP_KEYS_OUTPUT,
    }];

const CONTAINERS_MAP_VALUES_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "V = containers.Map.values(M)",
        inputs: &CONTAINERS_MAP_METHOD_INPUT_MAP,
        outputs: &CONTAINERS_MAP_VALUES_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "V = containers.Map.values(M, keySet)",
        inputs: &CONTAINERS_MAP_INPUTS_KEY_SPEC,
        outputs: &CONTAINERS_MAP_VALUES_OUTPUT,
    },
];

const CONTAINERS_MAP_ISKEY_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "tf = containers.Map.isKey(M, keySet)",
        inputs: &CONTAINERS_MAP_INPUTS_KEY_SPEC,
        outputs: &CONTAINERS_MAP_ISKEY_OUTPUT,
    }];

const CONTAINERS_MAP_REMOVE_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "containers.Map.remove(M, keySet)",
        inputs: &CONTAINERS_MAP_INPUTS_KEY_SPEC,
        outputs: &CONTAINERS_MAP_OUTPUTS_NONE,
    }];

const CONTAINERS_MAP_SUBSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "value = containers.Map.subsref(M, S)",
        inputs: &CONTAINERS_MAP_INPUTS_SUBSREF,
        outputs: &CONTAINERS_MAP_SUBSREF_OUTPUT,
    }];

const CONTAINERS_MAP_SUBSASGN_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "M = containers.Map.subsasgn(M, S, rhs)",
        inputs: &CONTAINERS_MAP_INPUTS_SUBSASGN,
        outputs: &CONTAINERS_MAP_OUTPUT,
    }];

pub const CONTAINERS_MAP_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTAINERS_MAP.INVALID_ARGUMENT",
    identifier: Some("RunMat:containers.Map:InvalidArgument"),
    when: "Map constructor/method inputs, option grammar, or key/value payloads are invalid.",
    message: "containers.Map: invalid argument",
};

pub const CONTAINERS_MAP_ERROR_MISSING_KEY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTAINERS_MAP.MISSING_KEY",
    identifier: Some("RunMat:containers.Map:MissingKey"),
    when: "Lookup/removal targets a key that is not present in the map.",
    message: "containers.Map: The specified key is not present in this container.",
};

pub const CONTAINERS_MAP_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTAINERS_MAP.INTERNAL",
    identifier: Some("RunMat:containers.Map:Internal"),
    when: "Map registry/storage operations fail unexpectedly.",
    message: "containers.Map: internal operation failed",
};

const CONTAINERS_MAP_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CONTAINERS_MAP_ERROR_INVALID_ARGUMENT,
    CONTAINERS_MAP_ERROR_MISSING_KEY,
    CONTAINERS_MAP_ERROR_INTERNAL,
];

pub const CONTAINERS_MAP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONTAINERS_MAP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONTAINERS_MAP_ERRORS,
};

pub const CONTAINERS_MAP_KEYS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONTAINERS_MAP_KEYS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONTAINERS_MAP_ERRORS,
};

pub const CONTAINERS_MAP_VALUES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONTAINERS_MAP_VALUES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONTAINERS_MAP_ERRORS,
};

pub const CONTAINERS_MAP_ISKEY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONTAINERS_MAP_ISKEY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONTAINERS_MAP_ERRORS,
};

pub const CONTAINERS_MAP_REMOVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONTAINERS_MAP_REMOVE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONTAINERS_MAP_ERRORS,
};

pub const CONTAINERS_MAP_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONTAINERS_MAP_SUBSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONTAINERS_MAP_ERRORS,
};

pub const CONTAINERS_MAP_SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONTAINERS_MAP_SUBSASGN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONTAINERS_MAP_ERRORS,
};
