//! MATLAB-compatible `containers.Map` constructor and methods for RunMat.

use runmat_types::MemberAccess;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{
    atomic::{AtomicBool, Ordering as AtomicOrdering},
    Arc,
};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerClass, BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_gc::{GcHandle, GcRoot, RootId, Trace, Tracer};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, HandleRef, IntValue, IntegerStorage, LogicalArray, NumericDType, ObjectInstance,
    StructValue, Tensor, Value,
};

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::containers::type_resolvers::{
    map_cell_type, map_handle_type, map_is_key_type, map_unknown_type,
};
use crate::{
    build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError, OBJECT_INDEX_BRACE,
    OBJECT_INDEX_MEMBER, OBJECT_INDEX_PAREN, OBJECT_SUBSASGN_METHOD, OBJECT_SUBSREF_METHOD,
};

const CLASS_NAME: &str = "containers.Map";
const BUILTIN_CONSTRUCTOR: &str = "containers.Map";
const BUILTIN_KEYS: &str = "containers.Map.keys";
const BUILTIN_VALUES: &str = "containers.Map.values";
const BUILTIN_IS_KEY: &str = "containers.Map.isKey";
const BUILTIN_REMOVE: &str = "containers.Map.remove";
const BUILTIN_SUBSREF: &str = "containers.Map.subsref";
const BUILTIN_SUBSASGN: &str = "containers.Map.subsasgn";

fn contains_resident_value(value: &Value) -> bool {
    match value {
        Value::GpuTensor(_) => true,
        Value::Cell(cell) => cell.data.iter().any(contains_resident_value),
        Value::OutputList(values) => values.iter().any(contains_resident_value),
        _ => false,
    }
}

fn ensure_resident_extension(
    value: &Value,
    extension: &BuiltinExtensionDescriptor,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if contains_resident_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)?;
    }
    Ok(())
}

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

const CONTAINERS_MAP_INPUTS_SUBSREF: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "containers.Map handle object.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing kind: (), ., or {}.",
    },
    BuiltinParamDescriptor {
        name: "payload",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing payload cell/property argument.",
    },
];

const CONTAINERS_MAP_INPUTS_SUBSASGN: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "containers.Map handle object.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Assignment kind: (), ., or {}.",
    },
    BuiltinParamDescriptor {
        name: "payload",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Assignment payload cell/property argument.",
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
        label: "value = containers.Map.subsref(M, kind, payload)",
        inputs: &CONTAINERS_MAP_INPUTS_SUBSREF,
        outputs: &CONTAINERS_MAP_SUBSREF_OUTPUT,
    }];

const CONTAINERS_MAP_SUBSASGN_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "M = containers.Map.subsasgn(M, kind, payload, rhs)",
        inputs: &CONTAINERS_MAP_INPUTS_SUBSASGN,
        outputs: &CONTAINERS_MAP_OUTPUT,
    }];

const CONTAINERS_MAP_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTAINERS_MAP.INVALID_ARGUMENT",
    identifier: Some("RunMat:containers.Map:InvalidArgument"),
    when: "Map constructor/method inputs, option grammar, or key/value payloads are invalid.",
    message: "containers.Map: invalid argument",
};

const CONTAINERS_MAP_ERROR_MISSING_KEY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTAINERS_MAP.MISSING_KEY",
    identifier: Some("RunMat:containers.Map:MissingKey"),
    when: "Lookup/removal targets a key that is not present in the map.",
    message: "containers.Map: The specified key is not present in this container.",
};

const CONTAINERS_MAP_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
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

const MAP_KEY_INTEGER_CLASSES: [BuiltinIntegerClass; 4] = [
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Uint32,
    BuiltinIntegerClass::Int64,
    BuiltinIntegerClass::Uint64,
];
const MAP_KEY_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "keySet",
    classes: &MAP_KEY_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Public numeric Map keys are scalar single/double/int32/uint32/int64/uint64; these four integer key classes retain exact native identity.",
}];
const MAP_VALUE_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "valueSet",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Every integer class is a documented ValueType and retains its exact class and payload.",
}];
const MAP_STORED_KEY_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "stored keys",
    classes: &MAP_KEY_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer keys are returned in their declared key class inside the output cell array.",
}];
const MAP_STORED_VALUE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "stored values",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer values pass through the host Map store and output cells with exact class and value.",
    }];
const MAP_ASSIGN_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "keySet",
        classes: &MAP_KEY_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public integer Map keys are int32/uint32/int64/uint64 and retain exact native identity.",
    },
    BuiltinIntegerInputCapability {
        name: "rhs",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All integer rhs classes are exact for ValueType='any' and their matching declared integer ValueType.",
    },
];
const MAP_RESIDENT_KEY_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "resident keySet",
        classes: &MAP_KEY_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Resident int32/uint32/int64/uint64 keys are a gated RunMat extension and gather before host Map lookup.",
    }];
const MAP_RESIDENT_VALUE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "resident valueSet or rhs",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Resident integer values are a gated RunMat extension and gather before exact host storage or declared ValueType conversion.",
    }];
const MAP_RESIDENT_ASSIGN_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "resident keySet",
        classes: &MAP_KEY_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Resident integer keys are gated and gathered before host key normalization.",
    },
    BuiltinIntegerInputCapability {
        name: "resident rhs",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "Resident integer rhs values are gated and gathered before host ValueType conversion.",
    },
];

pub const CONTAINERS_MAP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "M = containers.Map(integer_keySet, valueSet)",
        inputs: &MAP_KEY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "KeyType is inferred exactly for int32/uint32/int64/uint64; unsupported int8/uint8/int16/uint16 key arrays reject rather than alias through double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "M = containers.Map(keySet, integer_valueSet)",
        inputs: &MAP_VALUE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Uniform constructor values infer their exact integer ValueType; ValueType='any' and explicit integer ValueType preserve or deliberately cast native storage.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "M = containers.Map(resident_integer_keySet, valueSet)",
        inputs: &MAP_RESIDENT_KEY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only resident keys gather after compatibility admission and before host key-type inference and exact identity storage.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "M = containers.Map(keySet, resident_integer_valueSet)",
        inputs: &MAP_RESIDENT_VALUE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only resident values gather after compatibility admission; the resulting Map and all outputs remain host-resident.",
    },
];
pub const CONTAINERS_MAP_KEYS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "K = containers.Map.keys(M_with_integer_keys)",
        inputs: &MAP_STORED_KEY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The row cell contains exact typed scalar keys; the Map itself is host-resident.",
    }];
pub const CONTAINERS_MAP_VALUES_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "V = containers.Map.values(M_with_integer_values, keySet?)",
        inputs: &MAP_STORED_VALUE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "All-values output is a row cell; selected-values output matches keySet cell shape and preserves each stored integer class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "V = containers.Map.values(M, resident_integer_keySet)",
        inputs: &MAP_RESIDENT_KEY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "RunMat-only resident selected keys gather after compatibility admission; the host cell output preserves keySet shape.",
    },
];
pub const CONTAINERS_MAP_ISKEY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "tf = containers.Map.isKey(M, integer_keySet)",
        inputs: &MAP_KEY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Scalar input returns scalar logical; a cell keySet returns a same-shape logical array with exact wide-key identity.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "tf = containers.Map.isKey(M, resident_integer_keySet)",
        inputs: &MAP_RESIDENT_KEY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "RunMat-only resident keys gather after compatibility admission; the logical result is host-resident and shape preserving.",
    },
];
pub const CONTAINERS_MAP_REMOVE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "containers.Map.remove(M, integer_keySet)",
        inputs: &MAP_KEY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Removal mutates the host handle object and performs exact native key lookup.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "containers.Map.remove(M, resident_integer_keySet)",
        inputs: &MAP_RESIDENT_KEY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only resident keys gather after compatibility admission before the host handle is mutated.",
    },
];
pub const CONTAINERS_MAP_SUBSREF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "count = containers.Map.subsref(M, '.', 'Count')",
        inputs: &[],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The read-only Count property is a scalar uint64 and is derived from the host Map entry count without floating conversion.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "value = containers.Map.subsref(M, '()', integer_key)",
        inputs: &MAP_KEY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The integer controls lookup only; output class is the stored value class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "integer_value = containers.Map.subsref(M_with_integer_value, '()', key)",
        inputs: &MAP_STORED_VALUE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes:
            "Lookup returns the exact stored integer scalar or array without floating conversion.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "value = containers.Map.subsref(M, '()', resident_integer_key)",
        inputs: &MAP_RESIDENT_KEY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only resident keys gather after compatibility admission; lookup returns the host-stored value.",
    },
];
pub const CONTAINERS_MAP_SUBSASGN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "containers.Map.subsasgn(M, '()', integer_key, integer_rhs)",
        inputs: &MAP_ASSIGN_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Exact key identity is independent of exact rhs storage; a declared integer ValueType applies MATLAB integer conversion before storage.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "containers.Map.subsasgn(M, '()', resident_integer_key, resident_integer_rhs)",
        inputs: &MAP_RESIDENT_ASSIGN_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only resident assignment data gathers after compatibility admission and before host key normalization and rhs conversion.",
    },
];

const MAP_RESIDENT_CONSTRUCTOR_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "containers-map-resident-constructor-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "containers.Map gathering resident constructor keys or values is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ContainersMapResidentConstructorExtension"),
};
const MAP_RESIDENT_ISKEY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "containers-map-resident-iskey-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "containers.Map.isKey gathering resident keys is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ContainersMapResidentIsKeyExtension"),
};
const MAP_RESIDENT_VALUES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "containers-map-resident-values-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "containers.Map.values gathering resident selected keys is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ContainersMapResidentValuesExtension"),
};
const MAP_RESIDENT_REMOVE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "containers-map-resident-remove-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "containers.Map.remove gathering resident keys is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ContainersMapResidentRemoveExtension"),
};
const MAP_RESIDENT_SUBSREF_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "containers-map-resident-subsref-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "containers.Map.subsref gathering resident keys is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ContainersMapResidentSubsrefExtension"),
};
const MAP_RESIDENT_SUBSASGN_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "containers-map-resident-subsasgn-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "containers.Map.subsasgn gathering resident keys or rhs values is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ContainersMapResidentSubsasgnExtension"),
};
pub const CONTAINERS_MAP_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [MAP_RESIDENT_CONSTRUCTOR_EXTENSION];
pub const CONTAINERS_MAP_ISKEY_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [MAP_RESIDENT_ISKEY_EXTENSION];
pub const CONTAINERS_MAP_VALUES_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [MAP_RESIDENT_VALUES_EXTENSION];
pub const CONTAINERS_MAP_REMOVE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [MAP_RESIDENT_REMOVE_EXTENSION];
pub const CONTAINERS_MAP_SUBSREF_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [MAP_RESIDENT_SUBSREF_EXTENSION];
pub const CONTAINERS_MAP_SUBSASGN_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [MAP_RESIDENT_SUBSASGN_EXTENSION];

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "containers.Map",
    op_kind: GpuOpKind::Custom("map"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Map storage and outputs are host-resident; resident inputs gather only through explicitly gated RunMat extensions.",
};

fn map_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    builtin: &'static str,
) -> RuntimeError {
    let raw = detail.as_ref().trim();
    let normalized = raw
        .strip_prefix("containers.Map:")
        .map(str::trim)
        .unwrap_or(raw);
    let message = if normalized.is_empty() {
        error.message.to_string()
    } else {
        format!("{}: {}", error.message, normalized)
    };
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_descriptor_error(
    error: &'static BuiltinErrorDescriptor,
    builtin: &'static str,
) -> RuntimeError {
    map_error_with_detail(error, "", builtin)
}

fn map_invalid(detail: impl AsRef<str>, builtin: &'static str) -> RuntimeError {
    map_error_with_detail(&CONTAINERS_MAP_ERROR_INVALID_ARGUMENT, detail, builtin)
}

fn map_internal(detail: impl AsRef<str>, builtin: &'static str) -> RuntimeError {
    map_error_with_detail(&CONTAINERS_MAP_ERROR_INTERNAL, detail, builtin)
}

fn map_error(message: impl Into<String>, builtin: &'static str) -> RuntimeError {
    map_invalid(message.into(), builtin)
}

fn attach_builtin_context(mut error: RuntimeError, builtin: &'static str) -> RuntimeError {
    if error.context.builtin.is_none() {
        error.context = error.context.with_builtin(builtin);
    }
    error
}

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "containers.Map",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Handles act as fusion sinks; map construction terminates GPU fusion plans.",
};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static MAP_REGISTRY: RefCell<HashMap<u64, MapStore>> = RefCell::new(HashMap::new());
    static MAP_ROOT_STATE: RefCell<Option<MapRootState>> = const { RefCell::new(None) };
}

static CONTAINERS_MAP_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(CLASS_NAME);

struct MapRootState {
    root_id: RootId,
    active: Arc<AtomicBool>,
}

struct MapRegistryRoot {
    active: Arc<AtomicBool>,
}

impl GcRoot for MapRegistryRoot {
    fn scan(&self) -> Vec<GcHandle> {
        struct RootCollector {
            roots: Vec<GcHandle>,
        }

        impl Tracer for RootCollector {
            fn mark(&mut self, handle: GcHandle) {
                self.roots.push(handle);
            }
        }

        MAP_REGISTRY.with(|registry| {
            let registry = registry.borrow();
            let mut collector = RootCollector { roots: Vec::new() };
            for store in registry.values() {
                if let Some(storage) = store.storage {
                    collector.mark(storage);
                }
                for entry in &store.entries {
                    entry.key_value.trace(&mut collector);
                    entry.value.trace(&mut collector);
                }
            }
            collector.roots
        })
    }

    fn description(&self) -> String {
        "containers.Map registry values".to_string()
    }

    fn is_active(&self) -> bool {
        self.active.load(AtomicOrdering::Acquire)
            && MAP_REGISTRY.with(|registry| !registry.borrow().is_empty())
    }
}

fn ensure_map_registry_root_registered(builtin: &'static str) -> BuiltinResult<()> {
    MAP_ROOT_STATE.with(|state| {
        if state
            .borrow()
            .as_ref()
            .is_some_and(|state| state.active.load(AtomicOrdering::Acquire))
        {
            return Ok(());
        }

        let active = Arc::new(AtomicBool::new(true));
        let root_id = runmat_gc::gc_register_root(Box::new(MapRegistryRoot {
            active: Arc::clone(&active),
        }))
        .map_err(|e| {
            map_internal(
                format!("containers.Map: failed to register GC root: {e}"),
                builtin,
            )
        })?;
        *state.borrow_mut() = Some(MapRootState { root_id, active });
        Ok(())
    })
}

fn deactivate_map_registry_root_if_empty() {
    let empty = MAP_REGISTRY.with(|registry| {
        registry
            .try_borrow()
            .map(|registry| registry.is_empty())
            .unwrap_or(false)
    });
    if empty {
        MAP_ROOT_STATE.with(|state| {
            if let Some(state) = state.borrow_mut().take() {
                state.active.store(false, AtomicOrdering::Release);
                if let Err(err) = runmat_gc::gc_unregister_root(state.root_id) {
                    log::warn!("containers.Map: failed to unregister empty registry root: {err}");
                }
            }
        });
    }
}

fn ensure_containers_map_class_registered() {
    CONTAINERS_MAP_CLASS_REGISTERED.ensure(|| {
        let mut properties = HashMap::new();
        for name in ["Count", "KeyType", "ValueType"] {
            properties.insert(
                name.to_string(),
                crate::class_registry::RuntimeProperty {
                    name: name.to_string(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: true,
                    get_access: MemberAccess::Public,
                    set_access: MemberAccess::Private,
                    default_value: None,
                },
            );
        }

        let mut methods = HashMap::new();
        for (name, function_name) in [
            ("keys", BUILTIN_KEYS),
            ("values", BUILTIN_VALUES),
            ("isKey", BUILTIN_IS_KEY),
            ("remove", BUILTIN_REMOVE),
            (OBJECT_SUBSREF_METHOD, BUILTIN_SUBSREF),
            (OBJECT_SUBSASGN_METHOD, BUILTIN_SUBSASGN),
        ] {
            methods.insert(
                name.to_string(),
                crate::class_registry::RuntimeMethod {
                    name: name.to_string(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: function_name.to_string(),
                    implicit_class_argument: None,
                },
            );
        }

        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: CLASS_NAME.to_string(),
            parent: None,
            properties,
            methods,
        });
    });
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeyType {
    Char,
    Double,
    Single,
    Int32,
    UInt32,
    Int64,
    UInt64,
}

impl KeyType {
    fn matlab_name(self) -> &'static str {
        match self {
            KeyType::Char => "char",
            KeyType::Double => "double",
            KeyType::Single => "single",
            KeyType::Int32 => "int32",
            KeyType::UInt32 => "uint32",
            KeyType::Int64 => "int64",
            KeyType::UInt64 => "uint64",
        }
    }

    fn parse(value: &Value, builtin: &'static str) -> BuiltinResult<Self> {
        let text = string_from_value(value, "containers.Map: expected a KeyType string", builtin)?;
        match text.to_ascii_lowercase().as_str() {
            "char" | "character" => Ok(KeyType::Char),
            "double" => Ok(KeyType::Double),
            "single" => Ok(KeyType::Single),
            "int32" => Ok(KeyType::Int32),
            "uint32" => Ok(KeyType::UInt32),
            "int64" => Ok(KeyType::Int64),
            "uint64" => Ok(KeyType::UInt64),
            other => Err(map_error(
                format!(
                    "containers.Map: unsupported KeyType '{other}'. Valid types: char, double, single, int32, uint32, int64, uint64."
                ),
                builtin,
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueType {
    Any,
    Char,
    Double,
    Single,
    Logical,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
}

impl ValueType {
    fn matlab_name(self) -> &'static str {
        match self {
            ValueType::Any => "any",
            ValueType::Char => "char",
            ValueType::Double => "double",
            ValueType::Single => "single",
            ValueType::Logical => "logical",
            ValueType::Int8 => "int8",
            ValueType::UInt8 => "uint8",
            ValueType::Int16 => "int16",
            ValueType::UInt16 => "uint16",
            ValueType::Int32 => "int32",
            ValueType::UInt32 => "uint32",
            ValueType::Int64 => "int64",
            ValueType::UInt64 => "uint64",
        }
    }

    fn parse(value: &Value, builtin: &'static str) -> BuiltinResult<Self> {
        let text = string_from_value(
            value,
            "containers.Map: expected a ValueType string",
            builtin,
        )?;
        match text.to_ascii_lowercase().as_str() {
            "any" => Ok(ValueType::Any),
            "char" | "character" => Ok(ValueType::Char),
            "double" => Ok(ValueType::Double),
            "single" => Ok(ValueType::Single),
            "logical" => Ok(ValueType::Logical),
            "int8" => Ok(ValueType::Int8),
            "uint8" => Ok(ValueType::UInt8),
            "int16" => Ok(ValueType::Int16),
            "uint16" => Ok(ValueType::UInt16),
            "int32" => Ok(ValueType::Int32),
            "uint32" => Ok(ValueType::UInt32),
            "int64" => Ok(ValueType::Int64),
            "uint64" => Ok(ValueType::UInt64),
            other => Err(map_error(
                format!(
                    "containers.Map: unsupported ValueType '{other}'. Valid types: any, char, logical, double, single, int8, uint8, int16, uint16, int32, uint32, int64, uint64."
                ),
                builtin,
            )),
        }
    }

    fn normalize(&self, value: Value, builtin: &'static str) -> BuiltinResult<Value> {
        match self {
            ValueType::Any => Ok(value),
            ValueType::Char => {
                let chars = char_array_from_value(&value, builtin)?;
                Ok(Value::CharArray(chars))
            }
            ValueType::Double => normalize_numeric_value(value, NumericDType::F64, builtin),
            ValueType::Single => normalize_numeric_value(value, NumericDType::F32, builtin),
            ValueType::Logical => normalize_logical_value(value, builtin),
            integer_type => normalize_integer_value(value, *integer_type, builtin),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum NormalizedKey {
    String(String),
    Float(u64),
    Int(i64),
    UInt(u64),
}

#[derive(Clone)]
struct MapEntry {
    normalized: NormalizedKey,
    key_value: Value,
    value: Value,
}

struct MapStore {
    storage: Option<GcHandle>,
    key_type: KeyType,
    value_type: ValueType,
    uniform_values: bool,
    uniform_class: Option<ValueClass>,
    entries: Vec<MapEntry>,
    index: HashMap<NormalizedKey, usize>,
}

impl MapStore {
    fn new(key_type: KeyType, value_type: ValueType, uniform_values: bool) -> Self {
        Self {
            storage: None,
            key_type,
            value_type,
            uniform_values,
            uniform_class: None,
            entries: Vec::new(),
            index: HashMap::new(),
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn contains(&self, key: &NormalizedKey) -> bool {
        self.index.contains_key(key)
    }

    fn get(&self, key: &NormalizedKey) -> Option<Value> {
        self.index
            .get(key)
            .map(|&idx| self.entries[idx].value.clone())
    }

    fn insert_new(&mut self, mut entry: MapEntry, builtin: &'static str) -> BuiltinResult<()> {
        if self.index.contains_key(&entry.normalized) {
            return Err(map_error(
                "containers.Map: Duplicate key name was provided.",
                builtin,
            ));
        }
        entry.value = self.normalize_value(entry.value, builtin)?;
        self.track_uniform_class(&entry.value, builtin)?;
        let idx = self.entries.len();
        self.entries.push(entry.clone());
        self.index.insert(entry.normalized, idx);
        Ok(())
    }

    fn set(&mut self, mut entry: MapEntry, builtin: &'static str) -> BuiltinResult<()> {
        entry.value = self.normalize_value(entry.value, builtin)?;
        self.track_uniform_class(&entry.value, builtin)?;
        if let Some(&idx) = self.index.get(&entry.normalized) {
            self.entries[idx].value = entry.value.clone();
            self.entries[idx].key_value = entry.key_value;
        } else {
            let idx = self.entries.len();
            self.entries.push(entry.clone());
            self.index.insert(entry.normalized, idx);
        }
        Ok(())
    }

    fn remove(&mut self, key: &NormalizedKey, builtin: &'static str) -> BuiltinResult<()> {
        let idx = match self.index.get(key) {
            Some(&idx) => idx,
            None => {
                return Err(map_descriptor_error(
                    &CONTAINERS_MAP_ERROR_MISSING_KEY,
                    builtin,
                ));
            }
        };
        self.entries.remove(idx);
        self.index.clear();
        for (pos, entry) in self.entries.iter().enumerate() {
            self.index.insert(entry.normalized.clone(), pos);
        }
        if self.entries.is_empty() {
            self.uniform_class = None;
        }
        Ok(())
    }

    fn keys(&self) -> Vec<Value> {
        self.sorted_entries()
            .into_iter()
            .map(|entry| entry.key_value.clone())
            .collect()
    }

    fn values(&self) -> Vec<Value> {
        self.sorted_entries()
            .into_iter()
            .map(|entry| entry.value.clone())
            .collect()
    }

    fn sorted_entries(&self) -> Vec<&MapEntry> {
        let mut entries = self.entries.iter().collect::<Vec<_>>();
        entries.sort_by(|left, right| match (&left.normalized, &right.normalized) {
            (NormalizedKey::String(left), NormalizedKey::String(right)) => left.cmp(right),
            (NormalizedKey::Int(left), NormalizedKey::Int(right)) => left.cmp(right),
            (NormalizedKey::UInt(left), NormalizedKey::UInt(right)) => left.cmp(right),
            (NormalizedKey::Float(left), NormalizedKey::Float(right)) => f64::from_bits(*left)
                .partial_cmp(&f64::from_bits(*right))
                .unwrap_or(std::cmp::Ordering::Equal),
            _ => std::cmp::Ordering::Equal,
        });
        entries
    }

    fn normalize_value(&self, value: Value, builtin: &'static str) -> BuiltinResult<Value> {
        self.value_type.normalize(value, builtin)
    }

    fn track_uniform_class(&mut self, value: &Value, builtin: &'static str) -> BuiltinResult<()> {
        if !self.uniform_values {
            return Ok(());
        }
        let class = ValueClass::from_value(value);
        if let Some(existing) = &self.uniform_class {
            if existing != &class {
                return Err(map_error(
                    "containers.Map: UniformValues=true requires all values to share the same MATLAB class.",
                    builtin,
                ));
            }
        } else {
            self.uniform_class = Some(class);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ValueClass {
    Char,
    String,
    Numeric(NumericDType),
    Logical,
    Cell,
    Struct,
    Object,
    Other(&'static str),
}

impl ValueClass {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::CharArray(_) => ValueClass::Char,
            Value::String(_) | Value::StringArray(_) => ValueClass::String,
            Value::Num(_) => ValueClass::Numeric(NumericDType::F64),
            Value::Tensor(tensor) => ValueClass::Numeric(tensor.numeric_dtype()),
            Value::ComplexTensor(tensor) => ValueClass::Numeric(tensor.numeric_dtype()),
            Value::Bool(_) | Value::LogicalArray(_) => ValueClass::Logical,
            Value::Int(value) => ValueClass::Numeric(int_value_dtype(value)),
            Value::Cell(_) => ValueClass::Cell,
            Value::Struct(_) => ValueClass::Struct,
            Value::ObjectArray(_)
            | Value::Object(_)
            | Value::HandleObject(_)
            | Value::Listener(_) => ValueClass::Object,
            _ => ValueClass::Other("other"),
        }
    }
}

struct ConstructorArgs {
    key_type: KeyType,
    value_type: ValueType,
    uniform_values: bool,
    keys: Vec<KeyCandidate>,
    values: Vec<Value>,
}

struct KeyCandidate {
    normalized: NormalizedKey,
    canonical: Value,
}

#[runtime_builtin(
    name = "containers.Map",
    category = "containers/map",
    summary = "Create key-value dictionary objects.",
    keywords = "map,containers.Map,dictionary,hash map,lookup",
    accel = "metadata",
    sink = true,
    type_resolver(map_handle_type),
    descriptor(crate::builtins::containers::map::containers_map::CONTAINERS_MAP_DESCRIPTOR),
    extensions(crate::builtins::containers::map::containers_map::CONTAINERS_MAP_EXTENSIONS),
    integer_capabilities(
        crate::builtins::containers::map::containers_map::CONTAINERS_MAP_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.iter().any(contains_resident_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &MAP_RESIDENT_CONSTRUCTOR_EXTENSION,
            BUILTIN_CONSTRUCTOR,
        )?;
    }
    let mut host_args = Vec::with_capacity(args.len());
    for value in args {
        host_args.push(
            gather_if_needed_async(&value)
                .await
                .map_err(|err| attach_builtin_context(err, BUILTIN_CONSTRUCTOR))?,
        );
    }
    let parsed = parse_constructor_args(host_args, BUILTIN_CONSTRUCTOR).await?;
    let store = build_store(parsed, BUILTIN_CONSTRUCTOR)?;
    allocate_handle(store, BUILTIN_CONSTRUCTOR)
}

#[runtime_builtin(
    name = "containers.Map.keys",
    type_resolver(map_cell_type),
    descriptor(crate::builtins::containers::map::containers_map::CONTAINERS_MAP_KEYS_DESCRIPTOR),
    integer_capabilities(
        crate::builtins::containers::map::containers_map::CONTAINERS_MAP_KEYS_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_keys(map: Value) -> crate::BuiltinResult<Value> {
    with_store(&map, BUILTIN_KEYS, |store| {
        let values = store.keys();
        make_row_cell(values, BUILTIN_KEYS)
    })
}

#[runtime_builtin(
    name = "containers.Map.values",
    type_resolver(map_cell_type),
    descriptor(crate::builtins::containers::map::containers_map::CONTAINERS_MAP_VALUES_DESCRIPTOR),
    extensions(crate::builtins::containers::map::containers_map::CONTAINERS_MAP_VALUES_EXTENSIONS),
    integer_capabilities(crate::builtins::containers::map::containers_map::CONTAINERS_MAP_VALUES_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_values(map: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.iter().any(contains_resident_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &MAP_RESIDENT_VALUES_EXTENSION,
            BUILTIN_VALUES,
        )?;
    }
    if rest.is_empty() {
        return with_store(&map, BUILTIN_VALUES, |store| {
            make_row_cell(store.values(), BUILTIN_VALUES)
        });
    }
    if rest.len() != 1 {
        return Err(map_error(
            "containers.Map: values expects M or M,keySet",
            BUILTIN_VALUES,
        ));
    }
    let Value::Cell(key_set) = &rest[0] else {
        return Err(map_error(
            "containers.Map: values keySet must be a cell array",
            BUILTIN_VALUES,
        ));
    };
    let mut keys = Vec::with_capacity(key_set.data.len());
    for key in &key_set.data {
        keys.push(
            gather_if_needed_async(key)
                .await
                .map_err(|err| attach_builtin_context(err, BUILTIN_VALUES))?,
        );
    }
    with_store(&map, BUILTIN_VALUES, |store| {
        let mut values = Vec::with_capacity(keys.len());
        for key in &keys {
            let normalized = normalize_key(key, store.key_type, BUILTIN_VALUES)?;
            values.push(store.get(&normalized).ok_or_else(|| {
                map_descriptor_error(&CONTAINERS_MAP_ERROR_MISSING_KEY, BUILTIN_VALUES)
            })?);
        }
        crate::make_cell_with_shape(values, vec![key_set.rows, key_set.cols])
            .map_err(|err| map_error(format!("containers.Map: {err}"), BUILTIN_VALUES))
    })
}

#[runtime_builtin(
    name = "containers.Map.isKey",
    type_resolver(map_is_key_type),
    descriptor(crate::builtins::containers::map::containers_map::CONTAINERS_MAP_ISKEY_DESCRIPTOR),
    extensions(crate::builtins::containers::map::containers_map::CONTAINERS_MAP_ISKEY_EXTENSIONS),
    integer_capabilities(
        crate::builtins::containers::map::containers_map::CONTAINERS_MAP_ISKEY_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_is_key(map: Value, key_spec: Value) -> crate::BuiltinResult<Value> {
    ensure_resident_extension(&key_spec, &MAP_RESIDENT_ISKEY_EXTENSION, BUILTIN_IS_KEY)?;
    let key_type = with_store(&map, BUILTIN_IS_KEY, |store| Ok(store.key_type))?;
    let collection = collect_key_spec(&key_spec, key_type, BUILTIN_IS_KEY).await?;
    with_store(&map, BUILTIN_IS_KEY, |store| {
        let mut flags = Vec::with_capacity(collection.values.len());
        for value in &collection.values {
            let normalized = normalize_key(value, store.key_type, BUILTIN_IS_KEY)?;
            flags.push(store.contains(&normalized));
        }
        if collection.values.len() == 1 {
            Ok(Value::Bool(flags[0]))
        } else {
            let data: Vec<u8> = flags.into_iter().map(|b| if b { 1 } else { 0 }).collect();
            let logical = LogicalArray::new(data, collection.shape)
                .map_err(|e| map_error(format!("containers.Map: {e}"), BUILTIN_IS_KEY))?;
            Ok(Value::LogicalArray(logical))
        }
    })
}

#[runtime_builtin(
    name = "containers.Map.remove",
    type_resolver(map_handle_type),
    descriptor(crate::builtins::containers::map::containers_map::CONTAINERS_MAP_REMOVE_DESCRIPTOR),
    extensions(crate::builtins::containers::map::containers_map::CONTAINERS_MAP_REMOVE_EXTENSIONS),
    integer_capabilities(crate::builtins::containers::map::containers_map::CONTAINERS_MAP_REMOVE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_remove(map: Value, key_spec: Value) -> crate::BuiltinResult<Value> {
    ensure_resident_extension(&key_spec, &MAP_RESIDENT_REMOVE_EXTENSION, BUILTIN_REMOVE)?;
    let key_type = with_store(&map, BUILTIN_REMOVE, |store| Ok(store.key_type))?;
    let collection = collect_key_spec(&key_spec, key_type, BUILTIN_REMOVE).await?;
    with_store_mut(&map, BUILTIN_REMOVE, |store| {
        for value in &collection.values {
            let normalized = normalize_key(value, store.key_type, BUILTIN_REMOVE)?;
            store.remove(&normalized, BUILTIN_REMOVE)?;
        }
        Ok(())
    })?;
    Ok(map)
}

#[runtime_builtin(
    name = "containers.Map.subsref",
    type_resolver(map_unknown_type),
    descriptor(
        crate::builtins::containers::map::containers_map::CONTAINERS_MAP_SUBSREF_DESCRIPTOR
    ),
    extensions(crate::builtins::containers::map::containers_map::CONTAINERS_MAP_SUBSREF_EXTENSIONS),
    integer_capabilities(crate::builtins::containers::map::containers_map::CONTAINERS_MAP_SUBSREF_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_subsref(
    map: Value,
    kind: String,
    payload: Value,
) -> crate::BuiltinResult<Value> {
    ensure_resident_extension(&payload, &MAP_RESIDENT_SUBSREF_EXTENSION, BUILTIN_SUBSREF)?;
    if !matches!(map, Value::HandleObject(_)) {
        return Err(map_error(
            format!("containers.Map: subsref expects a containers.Map handle, got {map:?}"),
            BUILTIN_SUBSREF,
        ));
    }
    match kind.as_str() {
        OBJECT_INDEX_PAREN => {
            let mut args = extract_key_arguments(&payload, BUILTIN_SUBSREF)?;
            if args.is_empty() {
                return Err(map_error(
                    "containers.Map: indexing requires at least one key",
                    BUILTIN_SUBSREF,
                ));
            }
            if args.len() != 1 {
                return Err(map_error(
                    "containers.Map: indexing expects a single key argument",
                    BUILTIN_SUBSREF,
                ));
            }
            let key_arg = args.remove(0);
            let key_type = with_store(&map, BUILTIN_SUBSREF, |store| Ok(store.key_type))?;
            let collection = collect_key_spec(&key_arg, key_type, BUILTIN_SUBSREF).await?;
            if collection.values.len() != 1 {
                return Err(map_error(
                    "containers.Map: indexing requires exactly one scalar key",
                    BUILTIN_SUBSREF,
                ));
            }
            with_store(&map, BUILTIN_SUBSREF, |store| {
                let normalized =
                    normalize_key(&collection.values[0], store.key_type, BUILTIN_SUBSREF)?;
                store.get(&normalized).ok_or_else(|| {
                    map_descriptor_error(&CONTAINERS_MAP_ERROR_MISSING_KEY, BUILTIN_SUBSREF)
                })
            })
        }
        OBJECT_INDEX_MEMBER => {
            let field = string_from_value(
                &payload,
                "containers.Map: property name must be text",
                BUILTIN_SUBSREF,
            )?;
            with_store(&map, BUILTIN_SUBSREF, |store| {
                match field.to_ascii_lowercase().as_str() {
                    "count" => Ok(Value::Int(IntValue::U64(store.len() as u64))),
                    "keytype" => char_array_value(store.key_type.matlab_name(), BUILTIN_SUBSREF),
                    "valuetype" => {
                        char_array_value(store.value_type.matlab_name(), BUILTIN_SUBSREF)
                    }
                    other => Err(map_error(
                        format!("containers.Map: no such property '{other}'"),
                        BUILTIN_SUBSREF,
                    )),
                }
            })
        }
        OBJECT_INDEX_BRACE => Err(map_error(
            "containers.Map: curly-brace indexing is not supported.",
            BUILTIN_SUBSREF,
        )),
        other => Err(map_error(
            format!("containers.Map: unsupported indexing kind '{other}'"),
            BUILTIN_SUBSREF,
        )),
    }
}

#[runtime_builtin(
    name = "containers.Map.subsasgn",
    type_resolver(map_handle_type),
    descriptor(
        crate::builtins::containers::map::containers_map::CONTAINERS_MAP_SUBSASGN_DESCRIPTOR
    ),
    extensions(crate::builtins::containers::map::containers_map::CONTAINERS_MAP_SUBSASGN_EXTENSIONS),
    integer_capabilities(crate::builtins::containers::map::containers_map::CONTAINERS_MAP_SUBSASGN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_subsasgn(
    map: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
    if contains_resident_value(&payload) || contains_resident_value(&rhs) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &MAP_RESIDENT_SUBSASGN_EXTENSION,
            BUILTIN_SUBSASGN,
        )?;
    }
    if !matches!(map, Value::HandleObject(_)) {
        return Err(map_error(
            format!("containers.Map: subsasgn expects a containers.Map handle, got {map:?}"),
            BUILTIN_SUBSASGN,
        ));
    }
    match kind.as_str() {
        OBJECT_INDEX_PAREN => {
            let mut args = extract_key_arguments(&payload, BUILTIN_SUBSASGN)?;
            if args.is_empty() {
                return Err(map_error(
                    "containers.Map: assignment requires at least one key",
                    BUILTIN_SUBSASGN,
                ));
            }
            if args.len() != 1 {
                return Err(map_error(
                    "containers.Map: assignment expects a single key argument",
                    BUILTIN_SUBSASGN,
                ));
            }
            let key_arg = args.remove(0);
            let key_type = with_store(&map, BUILTIN_SUBSASGN, |store| Ok(store.key_type))?;
            let KeyCollection {
                values: key_values, ..
            } = collect_key_spec(&key_arg, key_type, BUILTIN_SUBSASGN).await?;
            if key_values.len() != 1 {
                return Err(map_error(
                    "containers.Map: assignment requires exactly one scalar key",
                    BUILTIN_SUBSASGN,
                ));
            }
            let values =
                expand_assignment_values(rhs.clone(), key_values.len(), BUILTIN_SUBSASGN).await?;
            with_store_mut(&map, BUILTIN_SUBSASGN, move |store| {
                for (key_raw, value) in key_values.into_iter().zip(values.into_iter()) {
                    let (normalized, canonical) =
                        canonicalize_key(key_raw, store.key_type, BUILTIN_SUBSASGN)?;
                    let entry = MapEntry {
                        normalized,
                        key_value: canonical,
                        value,
                    };
                    store.set(entry, BUILTIN_SUBSASGN)?;
                }
                Ok(())
            })?;
            Ok(map)
        }
        OBJECT_INDEX_MEMBER => Err(map_error(
            "containers.Map: property assignments are not supported.",
            BUILTIN_SUBSASGN,
        )),
        OBJECT_INDEX_BRACE => Err(map_error(
            "containers.Map: curly-brace assignment is not supported.",
            BUILTIN_SUBSASGN,
        )),
        other => Err(map_error(
            format!("containers.Map: unsupported assignment kind '{other}'"),
            BUILTIN_SUBSASGN,
        )),
    }
}

async fn parse_constructor_args(
    args: Vec<Value>,
    builtin: &'static str,
) -> BuiltinResult<ConstructorArgs> {
    let mut index = 0usize;
    let mut keys_input: Option<Value> = None;
    let mut values_input: Option<Value> = None;

    if index < args.len() && keyword_of(&args[index]).is_none() {
        if args.len() < 2 {
            return Err(map_error(
                "containers.Map: constructor requires both keys and values when either is provided.",
                builtin,
            ));
        }
        keys_input = Some(args[index].clone());
        values_input = Some(args[index + 1].clone());
        index += 2;
    }

    let has_data = keys_input.is_some();
    let mut key_type = KeyType::Char;
    let mut value_type = ValueType::Any;
    let mut uniform_values = has_data;
    let mut key_type_explicit = false;
    let mut value_type_explicit = false;
    while index < args.len() {
        let keyword = keyword_of(&args[index]).ok_or_else(|| {
            map_error(
                "containers.Map: expected option name (e.g. 'KeyType')",
                builtin,
            )
        })?;
        index += 1;
        let Some(value) = args.get(index) else {
            return Err(map_error(
                format!("containers.Map: missing value for option '{keyword}'"),
                builtin,
            ));
        };
        index += 1;
        match keyword.as_str() {
            "keytype" => {
                if has_data {
                    return Err(map_error(
                        "containers.Map: KeyType is only valid for an empty Map constructor",
                        builtin,
                    ));
                }
                key_type = KeyType::parse(value, builtin)?;
                key_type_explicit = true;
            }
            "valuetype" => {
                if has_data {
                    return Err(map_error(
                        "containers.Map: ValueType is only valid for an empty Map constructor",
                        builtin,
                    ));
                }
                value_type = ValueType::parse(value, builtin)?;
                value_type_explicit = true;
            }
            "uniformvalues" => {
                if !has_data {
                    return Err(map_error(
                        "containers.Map: UniformValues requires keySet and valueSet",
                        builtin,
                    ));
                }
                uniform_values = bool_from_value(
                    value,
                    "containers.Map: UniformValues must be logical",
                    builtin,
                )?
            }
            other => {
                return Err(map_error(
                    format!("containers.Map: unrecognised option '{other}'"),
                    builtin,
                ));
            }
        }
    }

    if !has_data && (key_type_explicit != value_type_explicit) {
        return Err(map_error(
            "containers.Map: KeyType and ValueType must both be specified for an empty typed Map",
            builtin,
        ));
    }
    if let Some(keys) = &keys_input {
        key_type = infer_key_type(keys, builtin)?;
    }
    if let Some(values) = &values_input {
        value_type = if uniform_values {
            infer_value_type(values)
        } else {
            ValueType::Any
        };
    }

    let keys = match keys_input {
        Some(value) => prepare_keys(value, key_type, builtin).await?,
        None => Vec::new(),
    };

    let values = match values_input {
        Some(value) => prepare_values(value, builtin).await?,
        None => Vec::new(),
    };

    if keys.len() != values.len() {
        return Err(map_error(
            format!(
                "containers.Map: number of keys ({}) must match number of values ({})",
                keys.len(),
                values.len()
            ),
            builtin,
        ));
    }

    Ok(ConstructorArgs {
        key_type,
        value_type,
        uniform_values,
        keys,
        values,
    })
}

fn infer_key_type(value: &Value, builtin: &'static str) -> BuiltinResult<KeyType> {
    match value {
        Value::CharArray(_) | Value::StringArray(_) | Value::String(_) => Ok(KeyType::Char),
        Value::Cell(cell)
            if cell.data.iter().all(|value| {
                matches!(value, Value::CharArray(_) | Value::String(_) | Value::StringArray(_))
            }) =>
        {
            Ok(KeyType::Char)
        }
        Value::Num(_) => Ok(KeyType::Double),
        Value::Tensor(tensor) => key_type_from_dtype(tensor.numeric_dtype()).ok_or_else(|| {
            map_error(
                "containers.Map: numeric key arrays must be double, single, int32, uint32, int64, or uint64",
                builtin,
            )
        }),
        Value::Int(value) => key_type_from_dtype(int_value_dtype(value)).ok_or_else(|| {
            map_error(
                "containers.Map: integer keys must be int32, uint32, int64, or uint64",
                builtin,
            )
        }),
        Value::GpuTensor(handle) => {
            if let Some(integer_type) = runmat_accelerate_api::handle_integer_type(handle) {
                match integer_type {
                    runmat_accelerate_api::IntegerElementType::I32 => Ok(KeyType::Int32),
                    runmat_accelerate_api::IntegerElementType::U32 => Ok(KeyType::UInt32),
                    runmat_accelerate_api::IntegerElementType::I64 => Ok(KeyType::Int64),
                    runmat_accelerate_api::IntegerElementType::U64 => Ok(KeyType::UInt64),
                    _ => Err(map_error(
                        "containers.Map: resident integer keys must be int32, uint32, int64, or uint64",
                        builtin,
                    )),
                }
            } else if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
            {
                Ok(KeyType::Single)
            } else {
                Ok(KeyType::Double)
            }
        }
        _ => Err(map_error(
            "containers.Map: keys must be a numeric array, cell array of character vectors, or string array",
            builtin,
        )),
    }
}

fn key_type_from_dtype(dtype: NumericDType) -> Option<KeyType> {
    match dtype {
        NumericDType::F64 => Some(KeyType::Double),
        NumericDType::F32 => Some(KeyType::Single),
        NumericDType::I32 => Some(KeyType::Int32),
        NumericDType::U32 => Some(KeyType::UInt32),
        NumericDType::I64 => Some(KeyType::Int64),
        NumericDType::U64 => Some(KeyType::UInt64),
        _ => None,
    }
}

fn int_value_dtype(value: &IntValue) -> NumericDType {
    IntegerStorage::from_scalar(value.clone()).numeric_dtype()
}

fn infer_value_type(value: &Value) -> ValueType {
    match value {
        Value::Num(_) => ValueType::Double,
        Value::Tensor(tensor) => value_type_from_dtype(tensor.numeric_dtype()),
        Value::Int(value) => value_type_from_dtype(int_value_dtype(value)),
        Value::Bool(_) | Value::LogicalArray(_) => ValueType::Logical,
        Value::CharArray(_) => ValueType::Char,
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_is_logical(handle) {
                ValueType::Logical
            } else if let Some(integer_type) = runmat_accelerate_api::handle_integer_type(handle) {
                match integer_type {
                    runmat_accelerate_api::IntegerElementType::I8 => ValueType::Int8,
                    runmat_accelerate_api::IntegerElementType::U8 => ValueType::UInt8,
                    runmat_accelerate_api::IntegerElementType::I16 => ValueType::Int16,
                    runmat_accelerate_api::IntegerElementType::U16 => ValueType::UInt16,
                    runmat_accelerate_api::IntegerElementType::I32 => ValueType::Int32,
                    runmat_accelerate_api::IntegerElementType::U32 => ValueType::UInt32,
                    runmat_accelerate_api::IntegerElementType::I64 => ValueType::Int64,
                    runmat_accelerate_api::IntegerElementType::U64 => ValueType::UInt64,
                }
            } else if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
            {
                ValueType::Single
            } else {
                ValueType::Double
            }
        }
        Value::Cell(cell) => {
            let mut iter = cell.data.iter();
            let Some(first) = iter.next() else {
                return ValueType::Any;
            };
            let inferred = infer_value_type(first);
            if inferred != ValueType::Any && iter.all(|value| infer_value_type(value) == inferred) {
                inferred
            } else {
                ValueType::Any
            }
        }
        _ => ValueType::Any,
    }
}

fn value_type_from_dtype(dtype: NumericDType) -> ValueType {
    match dtype {
        NumericDType::F64 => ValueType::Double,
        NumericDType::F32 => ValueType::Single,
        NumericDType::I8 => ValueType::Int8,
        NumericDType::U8 => ValueType::UInt8,
        NumericDType::I16 => ValueType::Int16,
        NumericDType::U16 => ValueType::UInt16,
        NumericDType::I32 => ValueType::Int32,
        NumericDType::U32 => ValueType::UInt32,
        NumericDType::I64 => ValueType::Int64,
        NumericDType::U64 => ValueType::UInt64,
    }
}

fn build_store(args: ConstructorArgs, builtin: &'static str) -> BuiltinResult<MapStore> {
    let mut store = MapStore::new(args.key_type, args.value_type, args.uniform_values);
    for (candidate, value) in args.keys.into_iter().zip(args.values.into_iter()) {
        store.insert_new(
            MapEntry {
                normalized: candidate.normalized,
                key_value: candidate.canonical,
                value,
            },
            builtin,
        )?;
    }
    Ok(store)
}

fn allocate_handle(store: MapStore, builtin: &'static str) -> BuiltinResult<Value> {
    ensure_containers_map_class_registered();

    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    ensure_map_registry_root_registered(builtin)?;
    MAP_REGISTRY.with(|registry| {
        registry
            .try_borrow_mut()
            .map_err(|_| map_internal("containers.Map: registry is already borrowed", builtin))?
            .insert(id, store);
        Ok::<(), RuntimeError>(())
    })?;
    let mut storage = ObjectInstance::new(CLASS_NAME.to_string());
    storage
        .properties
        .insert("id".to_string(), Value::Int(IntValue::U64(id)));
    let gc = match runmat_gc::gc_allocate(Value::Object(storage)) {
        Ok(gc) => gc,
        Err(e) => {
            MAP_REGISTRY.with(|registry| {
                if let Ok(mut registry) = registry.try_borrow_mut() {
                    registry.remove(&id);
                }
            });
            deactivate_map_registry_root_if_empty();
            return Err(map_error(format!("containers.Map: {e}"), builtin));
        }
    };
    MAP_REGISTRY.with(|registry| {
        let mut registry = registry
            .try_borrow_mut()
            .map_err(|_| map_internal("containers.Map: registry is already borrowed", builtin))?;
        let store = registry
            .get_mut(&id)
            .ok_or_else(|| map_internal("containers.Map: internal storage not found", builtin))?;
        store.storage = Some(gc);
        Ok::<(), RuntimeError>(())
    })?;
    Ok(Value::HandleObject(HandleRef {
        class_name: CLASS_NAME.to_string(),
        target: gc,
        valid: true,
    }))
}

fn with_store<F, R>(map: &Value, builtin: &'static str, f: F) -> BuiltinResult<R>
where
    F: FnOnce(&MapStore) -> BuiltinResult<R>,
{
    let handle = extract_handle(map, builtin)?;
    ensure_handle(handle, builtin)?;
    let id = map_id(handle, builtin)?;
    MAP_REGISTRY.with(|registry| {
        let registry = registry
            .try_borrow()
            .map_err(|_| map_internal("containers.Map: registry already borrowed", builtin))?;
        let store = registry
            .get(&id)
            .ok_or_else(|| map_internal("containers.Map: internal storage not found", builtin))?;
        f(store)
    })
}

fn with_store_mut<F, R>(map: &Value, builtin: &'static str, f: F) -> BuiltinResult<R>
where
    F: FnOnce(&mut MapStore) -> BuiltinResult<R>,
{
    let handle = extract_handle(map, builtin)?;
    ensure_handle(handle, builtin)?;
    let id = map_id(handle, builtin)?;
    MAP_REGISTRY.with(|registry| {
        let mut registry = registry
            .try_borrow_mut()
            .map_err(|_| map_internal("containers.Map: registry already borrowed", builtin))?;
        let store = registry
            .get_mut(&id)
            .ok_or_else(|| map_internal("containers.Map: internal storage not found", builtin))?;
        f(store)
    })
}

fn extract_handle<'a>(value: &'a Value, builtin: &'static str) -> BuiltinResult<&'a HandleRef> {
    match value {
        Value::HandleObject(handle) => Ok(handle),
        _ => Err(map_error(
            "containers.Map: expected a containers.Map handle",
            builtin,
        )),
    }
}

fn ensure_handle(handle: &HandleRef, builtin: &'static str) -> BuiltinResult<()> {
    if !crate::is_handle_valid(handle) {
        return Err(map_error("containers.Map: handle is invalid", builtin));
    }
    if handle.class_name != CLASS_NAME {
        return Err(map_error(
            format!(
                "containers.Map: expected handle of class '{}', got '{}'",
                CLASS_NAME, handle.class_name
            ),
            builtin,
        ));
    }
    Ok(())
}

fn map_id(handle: &HandleRef, builtin: &'static str) -> BuiltinResult<u64> {
    let storage = runmat_gc::gc_clone_value(&handle.target).map_err(|e| {
        map_internal(
            format!("containers.Map: invalid handle storage: {e}"),
            builtin,
        )
    })?;
    let id_value = match &storage {
        Value::Object(object) if object.class_name == CLASS_NAME => object.properties.get("id"),
        Value::Struct(StructValue { fields }) => fields.get("id"),
        other => {
            return Err(map_internal(
                format!("containers.Map: internal storage has unexpected shape {other:?}"),
                builtin,
            ));
        }
    };
    match id_value {
        Some(Value::Int(IntValue::U64(id))) => Ok(*id),
        Some(Value::Int(other)) => {
            let id = other.to_i64();
            if id < 0 {
                Err(map_internal(
                    "containers.Map: negative map identifier",
                    builtin,
                ))
            } else {
                Ok(id as u64)
            }
        }
        Some(Value::Num(n)) if n.is_finite() && *n >= 0.0 && n.fract() == 0.0 => {
            if *n >= u64::MAX as f64 {
                Err(map_internal(
                    "containers.Map: map identifier out of range",
                    builtin,
                ))
            } else {
                Ok(*n as u64)
            }
        }
        _ => Err(map_internal(
            "containers.Map: corrupted storage identifier",
            builtin,
        )),
    }
}

async fn prepare_keys(
    value: Value,
    key_type: KeyType,
    builtin: &'static str,
) -> BuiltinResult<Vec<KeyCandidate>> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(|err| attach_builtin_context(err, builtin))?;
    let flattened = flatten_keys(&host, key_type, builtin).await?;
    let mut out = Vec::with_capacity(flattened.len());
    for raw_key in flattened {
        let (normalized, canonical) = canonicalize_key(raw_key, key_type, builtin)?;
        out.push(KeyCandidate {
            normalized,
            canonical,
        });
    }
    Ok(out)
}

async fn prepare_values(value: Value, builtin: &'static str) -> BuiltinResult<Vec<Value>> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(|err| attach_builtin_context(err, builtin))?;
    flatten_values(&host, builtin).await
}

async fn flatten_keys(
    value: &Value,
    _key_type: KeyType,
    builtin: &'static str,
) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data.len());
            for ptr in &cell.data {
                let element = ptr;
                if matches!(element, Value::Cell(_)) {
                    return Err(map_error(
                        "containers.Map: nested cell arrays are not supported for keys",
                        builtin,
                    ));
                }
                out.push(
                    gather_if_needed_async(element)
                        .await
                        .map_err(|err| attach_builtin_context(err, builtin))?,
                );
            }
            Ok(out)
        }
        Value::StringArray(sa) => Ok(sa
            .data
            .iter()
            .map(|text| Value::String(text.clone()))
            .collect()),
        Value::CharArray(ca) => Ok(char_array_rows(ca, builtin)?),
        Value::LogicalArray(_) => Err(map_error(
            "containers.Map: logical arrays are not supported as Map keys",
            builtin,
        )),
        Value::Tensor(t) => {
            if !t.shape.is_empty()
                && tensor::tensor_element_len(t) != 1
                && !is_vector_shape(&t.shape)
            {
                return Err(map_error(
                    "containers.Map: numeric keys must be scalar or vector shaped",
                    builtin,
                ));
            }
            Ok(tensor_elements_to_values(t))
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => {
            Ok(vec![value.clone()])
        }
        Value::GpuTensor(_) => Err(map_error(
            "containers.Map: GPU keys must be gathered to the host before construction",
            builtin,
        )),
        other => Err(map_error(
            format!("containers.Map: unsupported key container {other:?}"),
            builtin,
        )),
    }
}

async fn flatten_values(value: &Value, builtin: &'static str) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data.len());
            for ptr in &cell.data {
                out.push(
                    gather_if_needed_async(ptr)
                        .await
                        .map_err(|err| attach_builtin_context(err, builtin))?,
                );
            }
            Ok(out)
        }
        Value::StringArray(sa) => Ok(sa
            .data
            .iter()
            .map(|text| Value::String(text.clone()))
            .collect()),
        Value::CharArray(ca) => Ok(char_array_rows(ca, builtin)?),
        Value::LogicalArray(arr) => Ok(arr.data.iter().map(|&b| Value::Bool(b != 0)).collect()),
        Value::Tensor(t) => {
            if !t.shape.is_empty()
                && !is_vector_shape(&t.shape)
                && tensor::tensor_element_len(t) != 1
            {
                return Err(map_error(
                    "containers.Map: numeric values must be scalar or vector shaped",
                    builtin,
                ));
            }
            Ok(tensor_elements_to_values(t))
        }
        _ => Ok(vec![value.clone()]),
    }
}

fn char_array_rows(ca: &CharArray, builtin: &'static str) -> BuiltinResult<Vec<Value>> {
    if ca.rows == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(ca.rows);
    for row in 0..ca.rows {
        let mut text = String::with_capacity(ca.cols);
        for col in 0..ca.cols {
            text.push(ca.data[row * ca.cols + col]);
        }
        let chars: Vec<char> = text.chars().collect();
        let array = CharArray::new(chars.clone(), 1, chars.len())
            .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))?;
        out.push(Value::CharArray(array));
    }
    Ok(out)
}

fn is_vector_shape(shape: &[usize]) -> bool {
    match shape.len() {
        0 => true,
        1 => true,
        2 => shape[0] == 1 || shape[1] == 1,
        _ => false,
    }
}

fn tensor_elements_to_values(tensor: &Tensor) -> Vec<Value> {
    if let Some(storage) = tensor.integer_storage() {
        return storage.exact_values().into_iter().map(Value::Int).collect();
    }
    match tensor.numeric_dtype() {
        NumericDType::F64 => tensor
            .as_f64_slice()
            .expect("double tensor has native double storage")
            .iter()
            .copied()
            .map(Value::Num)
            .collect(),
        NumericDType::F32 => (0..tensor.len())
            .map(|index| {
                let value = match tensor.numeric_value_at(index) {
                    Some(runmat_value::NumericScalar::F32(value)) => value,
                    _ => unreachable!("single tensor has native single storage"),
                };
                Value::Tensor(
                    Tensor::from_f32(vec![value], vec![1, 1])
                        .expect("single scalar tensor construction"),
                )
            })
            .collect(),
        _ => unreachable!("integer tensor returned through exact key/value path"),
    }
}

fn canonicalize_key(
    value: Value,
    key_type: KeyType,
    builtin: &'static str,
) -> BuiltinResult<(NormalizedKey, Value)> {
    let normalized = normalize_key(&value, key_type, builtin)?;
    let canonical = match key_type {
        KeyType::Char => Value::CharArray(char_array_from_value(&value, builtin)?),
        KeyType::Double => Value::Num(numeric_from_value(
            &value,
            "containers.Map: keys must be numeric scalars",
            builtin,
        )?),
        KeyType::Single => Value::Tensor(
            Tensor::from_f32(
                vec![numeric_from_value(
                    &value,
                    "containers.Map: keys must be numeric scalars",
                    builtin,
                )? as f32],
                vec![1, 1],
            )
            .map_err(|err| map_error(format!("containers.Map: {err}"), builtin))?,
        ),
        KeyType::Int32 => Value::Int(IntValue::I32(integer_from_value(
            &value,
            i32::MIN as i64,
            i32::MAX as i64,
            "containers.Map: int32 keys must be integers",
            builtin,
        )? as i32)),
        KeyType::UInt32 => Value::Int(IntValue::U32(unsigned_from_value(
            &value,
            u32::MAX as u64,
            "containers.Map: uint32 keys must be unsigned integers",
            builtin,
        )? as u32)),
        KeyType::Int64 => Value::Int(IntValue::I64(integer_from_value(
            &value,
            i64::MIN,
            i64::MAX,
            "containers.Map: int64 keys must be integers",
            builtin,
        )?)),
        KeyType::UInt64 => Value::Int(IntValue::U64(unsigned_from_value(
            &value,
            u64::MAX,
            "containers.Map: uint64 keys must be unsigned integers",
            builtin,
        )?)),
    };
    Ok((normalized, canonical))
}

fn normalize_key(
    value: &Value,
    key_type: KeyType,
    builtin: &'static str,
) -> BuiltinResult<NormalizedKey> {
    if !value_matches_key_type(value, key_type) {
        return Err(map_error(
            format!(
                "containers.Map: key class must match the map KeyType '{}'",
                key_type.matlab_name()
            ),
            builtin,
        ));
    }
    match key_type {
        KeyType::Char => {
            let text =
                string_from_value(value, "containers.Map: keys must be text scalars", builtin)?;
            Ok(NormalizedKey::String(text))
        }
        KeyType::Double | KeyType::Single => {
            let numeric = numeric_from_value(
                value,
                "containers.Map: keys must be numeric scalars",
                builtin,
            )?;
            if !numeric.is_finite() {
                return Err(map_error(
                    "containers.Map: keys must be finite numeric scalars",
                    builtin,
                ));
            }
            let numeric = if key_type == KeyType::Single {
                f64::from(numeric as f32)
            } else {
                numeric
            };
            let canonical = if numeric == 0.0 { 0.0 } else { numeric };
            Ok(NormalizedKey::Float(canonical.to_bits()))
        }
        KeyType::Int32 | KeyType::Int64 => {
            let bounds = if key_type == KeyType::Int32 {
                (i32::MIN as i64, i32::MAX as i64)
            } else {
                (i64::MIN, i64::MAX)
            };
            let value = integer_from_value(
                value,
                bounds.0,
                bounds.1,
                "containers.Map: integer keys must be whole numbers",
                builtin,
            )?;
            Ok(NormalizedKey::Int(value))
        }
        KeyType::UInt32 | KeyType::UInt64 => {
            let limit = if key_type == KeyType::UInt32 {
                u32::MAX as u64
            } else {
                u64::MAX
            };
            let value = unsigned_from_value(
                value,
                limit,
                "containers.Map: unsigned keys must be non-negative integers",
                builtin,
            )?;
            Ok(NormalizedKey::UInt(value))
        }
    }
}

fn value_matches_key_type(value: &Value, key_type: KeyType) -> bool {
    match key_type {
        KeyType::Char => {
            matches!(
                value,
                Value::CharArray(chars) if chars.rows == 1
            ) || matches!(value, Value::String(_))
                || matches!(value, Value::StringArray(strings) if strings.data.len() == 1)
        }
        KeyType::Double => {
            matches!(value, Value::Num(_))
                || matches!(value, Value::Tensor(tensor) if tensor.len() == 1 && tensor.numeric_dtype() == NumericDType::F64)
        }
        KeyType::Single => {
            matches!(value, Value::Tensor(tensor) if tensor.len() == 1 && tensor.numeric_dtype() == NumericDType::F32)
        }
        KeyType::Int32 => {
            matches!(value, Value::Int(IntValue::I32(_)))
                || matches!(value, Value::Tensor(tensor) if tensor.len() == 1 && tensor.numeric_dtype() == NumericDType::I32)
        }
        KeyType::UInt32 => {
            matches!(value, Value::Int(IntValue::U32(_)))
                || matches!(value, Value::Tensor(tensor) if tensor.len() == 1 && tensor.numeric_dtype() == NumericDType::U32)
        }
        KeyType::Int64 => {
            matches!(value, Value::Int(IntValue::I64(_)))
                || matches!(value, Value::Tensor(tensor) if tensor.len() == 1 && tensor.numeric_dtype() == NumericDType::I64)
        }
        KeyType::UInt64 => {
            matches!(value, Value::Int(IntValue::U64(_)))
                || matches!(value, Value::Tensor(tensor) if tensor.len() == 1 && tensor.numeric_dtype() == NumericDType::U64)
        }
    }
}

fn string_from_value(value: &Value, context: &str, builtin: &'static str) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        _ => Err(map_error(context, builtin)),
    }
}

fn char_array_from_value(value: &Value, builtin: &'static str) -> BuiltinResult<CharArray> {
    match value {
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.clone()),
        Value::String(s) => {
            let chars: Vec<char> = s.chars().collect();
            CharArray::new(chars.clone(), 1, chars.len())
                .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))
        }
        Value::StringArray(sa) if sa.data.len() == 1 => {
            let chars: Vec<char> = sa.data[0].chars().collect();
            CharArray::new(chars.clone(), 1, chars.len())
                .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))
        }
        _ => Err(map_error(
            "containers.Map: keys must be character vectors",
            builtin,
        )),
    }
}

fn char_array_value(text: &str, builtin: &'static str) -> BuiltinResult<Value> {
    let chars: Vec<char> = text.chars().collect();
    CharArray::new(chars.clone(), 1, chars.len())
        .map(Value::CharArray)
        .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))
}

fn normalize_numeric_value(
    value: Value,
    dtype: NumericDType,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match value {
        Value::Num(value) if dtype == NumericDType::F64 => Ok(Value::Num(value)),
        Value::Num(value) => Tensor::from_f32(vec![value as f32], vec![1, 1])
            .map(Value::Tensor)
            .map_err(|err| map_error(format!("containers.Map: {err}"), builtin)),
        Value::Tensor(tensor) if tensor.len() == 1 => {
            Ok(Value::Tensor(tensor::coerce_tensor_dtype(tensor, dtype)))
        }
        Value::Int(value) => normalize_numeric_value(Value::Num(value.to_f64()), dtype, builtin),
        Value::Bool(value) => {
            normalize_numeric_value(Value::Num(if value { 1.0 } else { 0.0 }), dtype, builtin)
        }
        Value::LogicalArray(arr) if arr.data.len() == 1 => {
            let data: Vec<f64> = arr
                .data
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            let tensor = Tensor::new(data, arr.shape.clone())
                .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))?;
            Ok(Value::Tensor(tensor::coerce_tensor_dtype(tensor, dtype)))
        }
        Value::Cell(_)
        | Value::SparseTensor(_)
        | Value::Struct(_)
        | Value::ObjectArray(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_)
        | Value::Symbolic(_)
        | Value::SymbolicArray(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Foreign(_)
        | Value::GpuTensor(_)
        | Value::OutputList(_)
        | Value::Tensor(_)
        | Value::LogicalArray(_) => Err(map_error(
            "containers.Map: values must be numeric scalars when ValueType is 'double' or 'single'",
            builtin,
        )),
    }
}

fn normalize_logical_value(value: Value, builtin: &'static str) -> BuiltinResult<Value> {
    match value {
        Value::Bool(_) => Ok(value),
        Value::LogicalArray(ref array) if array.data.len() == 1 => Ok(value),
        Value::Int(i) => Ok(Value::Bool(!i.is_zero())),
        Value::Num(n) => Ok(Value::Bool(n != 0.0)),
        Value::Tensor(t) if t.len() == 1 => {
            let flags: Vec<u8> = if let Some(storage) = t.integer_storage() {
                storage
                    .exact_values()
                    .into_iter()
                    .map(|value| if value.is_zero() { 0 } else { 1 })
                    .collect()
            } else {
                (0..t.len())
                    .map(|index| {
                        if tensor::tensor_value_f64(&t, index) != 0.0 {
                            1
                        } else {
                            0
                        }
                    })
                    .collect()
            };
            let logical = LogicalArray::new(flags, t.shape.clone())
                .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))?;
            Ok(Value::LogicalArray(logical))
        }
        Value::CharArray(_)
        | Value::SparseTensor(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::Struct(_)
        | Value::Cell(_)
        | Value::ObjectArray(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_)
        | Value::Symbolic(_)
        | Value::SymbolicArray(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Foreign(_)
        | Value::GpuTensor(_)
        | Value::OutputList(_)
        | Value::Tensor(_)
        | Value::LogicalArray(_) => Err(map_error(
            "containers.Map: values must be logical scalars when ValueType is 'logical'",
            builtin,
        )),
    }
}

fn integer_dtype_for_value_type(value_type: ValueType) -> Option<NumericDType> {
    match value_type {
        ValueType::Int8 => Some(NumericDType::I8),
        ValueType::UInt8 => Some(NumericDType::U8),
        ValueType::Int16 => Some(NumericDType::I16),
        ValueType::UInt16 => Some(NumericDType::U16),
        ValueType::Int32 => Some(NumericDType::I32),
        ValueType::UInt32 => Some(NumericDType::U32),
        ValueType::Int64 => Some(NumericDType::I64),
        ValueType::UInt64 => Some(NumericDType::U64),
        _ => None,
    }
}

fn normalize_integer_value(
    value: Value,
    value_type: ValueType,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let dtype = integer_dtype_for_value_type(value_type)
        .ok_or_else(|| map_internal("containers.Map: invalid integer ValueType", builtin))?;
    let (tensor, scalar) = match value {
        Value::Tensor(tensor) if tensor.len() == 1 => (tensor, true),
        Value::Int(value) => (
            Tensor::new_integer(integer_storage_from_scalar(value), vec![1, 1])
                .map_err(|err| map_error(format!("containers.Map: {err}"), builtin))?,
            true,
        ),
        Value::Num(value) => (
            Tensor::new(vec![value], vec![1, 1])
                .map_err(|err| map_error(format!("containers.Map: {err}"), builtin))?,
            true,
        ),
        Value::Bool(value) => (
            Tensor::new(vec![if value { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|err| map_error(format!("containers.Map: {err}"), builtin))?,
            true,
        ),
        Value::LogicalArray(array) if array.data.len() == 1 => (
            Tensor::new(
                array
                    .data
                    .iter()
                    .map(|value| f64::from(*value != 0))
                    .collect(),
                array.shape,
            )
            .map_err(|err| map_error(format!("containers.Map: {err}"), builtin))?,
            true,
        ),
        _ => {
            return Err(map_error(
                "containers.Map: value cannot be converted to the declared integer ValueType",
                builtin,
            ))
        }
    };
    let converted = tensor::coerce_tensor_dtype(tensor, dtype);
    if scalar {
        let value = converted
            .integer_storage()
            .and_then(|storage| storage.value_at(0))
            .ok_or_else(|| {
                map_internal("containers.Map: integer scalar conversion failed", builtin)
            })?;
        Ok(Value::Int(value))
    } else {
        Ok(Value::Tensor(converted))
    }
}

fn integer_storage_from_scalar(value: IntValue) -> IntegerStorage {
    match value {
        IntValue::I8(value) => IntegerStorage::I8(vec![value]),
        IntValue::I16(value) => IntegerStorage::I16(vec![value]),
        IntValue::I32(value) => IntegerStorage::I32(vec![value]),
        IntValue::I64(value) => IntegerStorage::I64(vec![value]),
        IntValue::U8(value) => IntegerStorage::U8(vec![value]),
        IntValue::U16(value) => IntegerStorage::U16(vec![value]),
        IntValue::U32(value) => IntegerStorage::U32(vec![value]),
        IntValue::U64(value) => IntegerStorage::U64(vec![value]),
    }
}

fn numeric_from_value(value: &Value, context: &str, builtin: &'static str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Ok(tensor::tensor_value_f64(t, 0)),
        Value::LogicalArray(arr) if arr.data.len() == 1 => {
            Ok(if arr.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => Err(map_error(context, builtin)),
    }
}

fn integer_from_value(
    value: &Value,
    min: i64,
    max: i64,
    context: &str,
    builtin: &'static str,
) -> BuiltinResult<i64> {
    match value {
        Value::Int(i) => {
            let Some(v) = i.try_to_i64() else {
                return Err(map_error(context, builtin));
            };
            if v < min || v > max {
                return Err(map_error(context, builtin));
            }
            Ok(v)
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                let Some(v) = storage.value_at(0).and_then(|value| value.try_to_i64()) else {
                    return Err(map_error(context, builtin));
                };
                if v < min || v > max {
                    return Err(map_error(context, builtin));
                }
                Ok(v)
            } else {
                integer_from_value(
                    &Value::Num(tensor::tensor_value_f64(t, 0)),
                    min,
                    max,
                    context,
                    builtin,
                )
            }
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(map_error(context, builtin));
            }
            if (*n < min as f64) || (*n > max as f64) {
                return Err(map_error(context, builtin));
            }
            if (n.round() - n).abs() > f64::EPSILON {
                return Err(map_error(context, builtin));
            }
            Ok(n.round() as i64)
        }
        Value::Bool(b) => {
            let v = if *b { 1 } else { 0 };
            if v < min || v > max {
                return Err(map_error(context, builtin));
            }
            Ok(v)
        }
        _ => Err(map_error(context, builtin)),
    }
}

fn unsigned_from_value(
    value: &Value,
    max: u64,
    context: &str,
    builtin: &'static str,
) -> BuiltinResult<u64> {
    match value {
        Value::Int(i) => {
            let Some(v) = i.try_to_u64() else {
                return Err(map_error(context, builtin));
            };
            if v > max {
                return Err(map_error(context, builtin));
            }
            Ok(v)
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                let Some(v) = storage.value_at(0).and_then(|value| value.try_to_u64()) else {
                    return Err(map_error(context, builtin));
                };
                if v > max {
                    return Err(map_error(context, builtin));
                }
                Ok(v)
            } else {
                unsigned_from_value(
                    &Value::Num(tensor::tensor_value_f64(t, 0)),
                    max,
                    context,
                    builtin,
                )
            }
        }
        Value::Num(n) => {
            if !n.is_finite() || *n < 0.0 || *n > max as f64 {
                return Err(map_error(context, builtin));
            }
            if (n.round() - n).abs() > f64::EPSILON {
                return Err(map_error(context, builtin));
            }
            Ok(n.round() as u64)
        }
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        _ => Err(map_error(context, builtin)),
    }
}

fn bool_from_value(value: &Value, context: &str, builtin: &'static str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::LogicalArray(arr) if arr.data.len() == 1 => Ok(arr.data[0] != 0),
        Value::Int(i) => Ok(!i.is_zero()),
        Value::Num(n) => Ok(*n != 0.0),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                Ok(!storage
                    .value_at(0)
                    .ok_or_else(|| map_error(context, builtin))?
                    .is_zero())
            } else {
                Ok(tensor::tensor_value_f64(t, 0) != 0.0)
            }
        }
        _ => Err(map_error(context, builtin)),
    }
}

fn make_row_cell(values: Vec<Value>, builtin: &'static str) -> BuiltinResult<Value> {
    let cols = values.len();
    crate::make_cell_with_shape(values, vec![1, cols])
        .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))
}

fn extract_key_arguments(payload: &Value, builtin: &'static str) -> BuiltinResult<Vec<Value>> {
    match payload {
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data.len());
            for ptr in &cell.data {
                out.push(ptr.clone());
            }
            Ok(out)
        }
        other => Err(map_error(
            format!("containers.Map: expected key arguments in a cell array, got {other:?}"),
            builtin,
        )),
    }
}

async fn expand_assignment_values(
    value: Value,
    expected: usize,
    builtin: &'static str,
) -> BuiltinResult<Vec<Value>> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(|err| attach_builtin_context(err, builtin))?;
    if expected == 1 {
        Ok(vec![host])
    } else {
        let values = flatten_values(&host, builtin).await?;
        if values.len() != expected {
            return Err(map_error(
                format!(
                    "containers.Map: assignment with {} keys requires {} values (got {})",
                    expected,
                    expected,
                    values.len()
                ),
                builtin,
            ));
        }
        Ok(values)
    }
}

struct KeyCollection {
    values: Vec<Value>,
    shape: Vec<usize>,
}

async fn collect_key_spec(
    value: &Value,
    key_type: KeyType,
    builtin: &'static str,
) -> BuiltinResult<KeyCollection> {
    let host = gather_if_needed_async(value)
        .await
        .map_err(|err| attach_builtin_context(err, builtin))?;
    match &host {
        Value::Cell(cell) => {
            let mut values = Vec::with_capacity(cell.data.len());
            for ptr in &cell.data {
                values.push(
                    gather_if_needed_async(ptr)
                        .await
                        .map_err(|err| attach_builtin_context(err, builtin))?,
                );
            }
            Ok(KeyCollection {
                values,
                shape: vec![cell.rows, cell.cols],
            })
        }
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(KeyCollection {
            values: vec![Value::String(sa.data[0].clone())],
            shape: vec![1, 1],
        }),
        Value::CharArray(ca) if ca.rows == 1 => Ok(KeyCollection {
            values: vec![Value::CharArray(ca.clone())],
            shape: vec![1, 1],
        }),
        Value::Tensor(t) if key_type != KeyType::Char && t.len() == 1 => Ok(KeyCollection {
            values: tensor_elements_to_values(t),
            shape: vec![1, 1],
        }),
        Value::StringArray(_) | Value::CharArray(_) | Value::Tensor(_) => Err(map_error(
            "containers.Map: multiple keys must be supplied in a cell array",
            builtin,
        )),
        _ => Ok(KeyCollection {
            values: vec![host.clone()],
            shape: vec![1, 1],
        }),
    }
}

pub fn map_length(value: &Value) -> Option<usize> {
    if let Value::HandleObject(handle) = value {
        if crate::is_handle_valid(handle) && handle.class_name == CLASS_NAME {
            if let Ok(id) = map_id(handle, BUILTIN_CONSTRUCTOR) {
                return MAP_REGISTRY.with(|registry| {
                    registry
                        .try_borrow()
                        .ok()
                        .and_then(|registry| registry.get(&id).map(|store| store.len()))
                });
            }
        }
    }
    None
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::{gpu_helpers, test_support};
    use futures::executor::block_on;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::IntegerStorage;

    fn error_message(err: crate::RuntimeError) -> String {
        err.message.clone()
    }

    fn containers_map_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::containers_map_builtin(args))
    }

    fn containers_map_keys(map: Value) -> BuiltinResult<Value> {
        block_on(super::containers_map_keys(map))
    }

    fn containers_map_values(map: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::containers_map_values(map, rest))
    }

    fn containers_map_is_key(map: Value, key_spec: Value) -> BuiltinResult<Value> {
        block_on(super::containers_map_is_key(map, key_spec))
    }

    fn containers_map_remove(map: Value, key_spec: Value) -> BuiltinResult<Value> {
        block_on(super::containers_map_remove(map, key_spec))
    }

    fn containers_map_subsref(map: Value, kind: String, payload: Value) -> BuiltinResult<Value> {
        block_on(super::containers_map_subsref(map, kind, payload))
    }

    fn containers_map_subsasgn(
        map: Value,
        kind: String,
        payload: Value,
        rhs: Value,
    ) -> BuiltinResult<Value> {
        block_on(super::containers_map_subsasgn(map, kind, payload, rhs))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn construct_empty_map_defaults() {
        let map = containers_map_builtin(Vec::new()).expect("map");
        let count = containers_map_subsref(
            map.clone(),
            ".".to_string(),
            Value::from("Count".to_string()),
        )
        .expect("Count");
        assert_eq!(count, Value::Int(IntValue::U64(0)));

        let key_type = containers_map_subsref(
            map.clone(),
            ".".to_string(),
            Value::from("KeyType".to_string()),
        )
        .expect("KeyType");
        assert_eq!(
            key_type,
            Value::CharArray(CharArray::new("char".chars().collect(), 1, 4).unwrap())
        );

        let value_type = containers_map_subsref(
            map.clone(),
            ".".to_string(),
            Value::from("ValueType".to_string()),
        )
        .expect("ValueType");
        assert_eq!(
            value_type,
            Value::CharArray(CharArray::new("any".chars().collect(), 1, 3).unwrap())
        );
    }

    #[test]
    fn map_type_resolvers_basics() {
        let ctx = ResolveContext::new(Vec::new());
        assert_eq!(map_handle_type(&[Type::Unknown], &ctx), Type::Unknown);
        assert_eq!(map_cell_type(&[], &ctx), Type::cell());
        assert_eq!(map_is_key_type(&[Type::String], &ctx), Type::logical());
        assert_eq!(map_unknown_type(&[], &ctx), Type::Unknown);
    }

    #[test]
    fn containers_map_descriptor_includes_constructor_and_method_signatures() {
        let constructor_labels: Vec<&str> = CONTAINERS_MAP_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(constructor_labels.contains(&"M = containers.Map()"));
        assert!(constructor_labels.contains(&"M = containers.Map(keys, values)"));

        let method_labels: Vec<&str> = CONTAINERS_MAP_SUBSREF_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(method_labels.contains(&"value = containers.Map.subsref(M, kind, payload)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn constructor_with_cells_lookup() {
        let keys = crate::make_cell(vec![Value::from("apple"), Value::from("pear")], 1, 2).unwrap();
        let values = crate::make_cell(vec![Value::Num(5.0), Value::Num(7.0)], 1, 2).unwrap();
        let map = containers_map_builtin(vec![keys, values]).expect("map");
        let apple = containers_map_subsref(
            map.clone(),
            "()".to_string(),
            crate::make_cell(vec![Value::from("apple")], 1, 1).unwrap(),
        )
        .expect("lookup");
        assert_eq!(apple, Value::Num(5.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn constructor_rejects_duplicate_keys() {
        let keys = crate::make_cell(vec![Value::from("dup"), Value::from("dup")], 1, 2).unwrap();
        let values = crate::make_cell(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
        let err = containers_map_builtin(vec![keys, values]).expect_err("duplicate check");
        let message = error_message(err);
        assert!(message.contains("Duplicate key name"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn constructor_errors_when_value_count_mismatch() {
        let keys = crate::make_cell(vec![Value::from("a"), Value::from("b")], 1, 2).unwrap();
        let values = crate::make_cell(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err = containers_map_builtin(vec![keys, values]).expect_err("count mismatch");
        let message = error_message(err);
        assert!(message.contains("number of keys"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn comparison_method_rejects_unknown_values() {
        let keys = crate::make_cell(vec![Value::from("a")], 1, 1).unwrap();
        let values = crate::make_cell(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err = containers_map_builtin(vec![
            keys,
            values,
            Value::from("ComparisonMethod"),
            Value::from("caseinsensitive"),
        ])
        .expect_err("comparison method");
        let message = error_message(err);
        assert!(message.contains("unrecognised option"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn key_type_single_roundtrip() {
        let map = containers_map_builtin(vec![
            Value::from("KeyType"),
            Value::from("single"),
            Value::from("ValueType"),
            Value::from("any"),
        ])
        .expect("map");
        let key_type = containers_map_subsref(map.clone(), ".".to_string(), Value::from("KeyType"))
            .expect("keytype");
        assert_eq!(
            key_type,
            Value::CharArray(CharArray::new("single".chars().collect(), 1, 6).unwrap())
        );

        let single_key = Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap());
        let payload = crate::make_cell(vec![single_key], 1, 1).unwrap();
        let map = containers_map_subsasgn(map, "()".to_string(), payload.clone(), Value::Num(7.0))
            .expect("assign");
        let value = containers_map_subsref(map, "()".to_string(), payload).expect("lookup");
        assert!(matches!(value, Value::Num(n) if (n - 7.0).abs() < 1e-12));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn value_type_double_converts_integers() {
        let map = containers_map_builtin(vec![
            Value::from("KeyType"),
            Value::from("double"),
            Value::from("ValueType"),
            Value::from("double"),
        ])
        .expect("map");
        let payload = crate::make_cell(vec![Value::Num(1.0)], 1, 1).unwrap();
        let map = containers_map_subsasgn(
            map,
            "()".to_string(),
            payload.clone(),
            Value::Int(IntValue::I32(7)),
        )
        .expect("assign");
        let value = containers_map_subsref(map, "()".to_string(), payload).expect("lookup");
        assert!(matches!(value, Value::Num(n) if (n - 7.0).abs() < 1e-12));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn value_type_logical_rejects_nonscalar_numeric_arrays() {
        let tensor = Tensor::new(vec![0.0, 2.0, -3.0], vec![3, 1]).unwrap();
        let map = containers_map_builtin(vec![
            Value::from("KeyType"),
            Value::from("char"),
            Value::from("ValueType"),
            Value::from("logical"),
        ])
        .expect("map");
        let payload = crate::make_cell(vec![Value::from("mask")], 1, 1).unwrap();
        let error = containers_map_subsasgn(map, "()".to_string(), payload, Value::Tensor(tensor))
            .expect_err("declared logical values must be scalar");
        assert!(error.to_string().contains("logical scalars"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uniform_values_enforced_on_assignment() {
        let map = containers_map_builtin(vec![
            crate::make_cell(vec![Value::from("x")], 1, 1).unwrap(),
            crate::make_cell(vec![Value::Num(1.0)], 1, 1).unwrap(),
        ])
        .expect("map");
        let payload = crate::make_cell(vec![Value::from("x")], 1, 1).unwrap();
        let err = containers_map_subsasgn(map, "()".to_string(), payload, Value::from("text"))
            .expect_err("uniform enforcement");
        let message = error_message(err);
        assert!(message.contains("numeric scalar"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assignment_updates_and_inserts() {
        let map = containers_map_builtin(Vec::new()).expect("map");
        let payload = crate::make_cell(vec![Value::from("alpha")], 1, 1).unwrap();
        let updated = containers_map_subsasgn(
            map.clone(),
            "()".to_string(),
            payload.clone(),
            Value::Num(1.0),
        )
        .expect("assign");
        let updated = containers_map_subsasgn(
            updated.clone(),
            "()".to_string(),
            payload.clone(),
            Value::Num(5.0),
        )
        .expect("update");
        let beta_payload = crate::make_cell(vec![Value::from("beta")], 1, 1).unwrap();
        let updated = containers_map_subsasgn(
            updated.clone(),
            "()".to_string(),
            beta_payload,
            Value::Num(9.0),
        )
        .expect("insert");
        let value = containers_map_subsref(updated, "()".to_string(), payload).expect("lookup");
        assert_eq!(value, Value::Num(5.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn subsref_rejects_multiple_keys() {
        let keys = crate::make_cell(
            vec![Value::from("a"), Value::from("b"), Value::from("c")],
            1,
            3,
        )
        .unwrap();
        let values = crate::make_cell(
            vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
            1,
            3,
        )
        .unwrap();
        let map = containers_map_builtin(vec![keys, values]).expect("map");
        let request = crate::make_cell(vec![Value::from("a"), Value::from("c")], 1, 2).unwrap();
        let payload = crate::make_cell(vec![request], 1, 1).unwrap();
        let error = containers_map_subsref(map, "()".to_string(), payload).unwrap_err();
        assert!(error.to_string().contains("exactly one scalar key"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn subsref_rejects_empty_key_collection() {
        let keys = crate::make_cell(vec![Value::from("z")], 1, 1).unwrap();
        let values = crate::make_cell(vec![Value::Num(42.0)], 1, 1).unwrap();
        let map = containers_map_builtin(vec![keys, values]).expect("map");
        let empty_keys = crate::make_cell(Vec::new(), 1, 0).unwrap();
        let payload = crate::make_cell(vec![empty_keys], 1, 1).unwrap();
        let error = containers_map_subsref(map, "()".to_string(), payload).unwrap_err();
        assert!(error.to_string().contains("exactly one scalar key"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn subsasgn_rejects_multiple_keys() {
        let keys = crate::make_cell(vec![Value::from("a"), Value::from("b")], 1, 2).unwrap();
        let values = crate::make_cell(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
        let map = containers_map_builtin(vec![keys, values]).expect("map");
        let key_spec = crate::make_cell(vec![Value::from("a"), Value::from("b")], 1, 2).unwrap();
        let payload = crate::make_cell(vec![key_spec], 1, 1).unwrap();
        let new_values = crate::make_cell(vec![Value::Num(10.0), Value::Num(20.0)], 1, 2).unwrap();
        let error =
            containers_map_subsasgn(map, "()".to_string(), payload, new_values).unwrap_err();
        assert!(error.to_string().contains("exactly one scalar key"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assignment_value_count_mismatch_errors() {
        let keys = crate::make_cell(vec![Value::from("x"), Value::from("y")], 1, 2).unwrap();
        let values = crate::make_cell(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
        let map = containers_map_builtin(vec![keys, values]).expect("map");
        let key_spec = crate::make_cell(vec![Value::from("x"), Value::from("y")], 1, 2).unwrap();
        let payload = crate::make_cell(vec![key_spec], 1, 1).unwrap();
        let rhs = crate::make_cell(vec![Value::Num(99.0)], 1, 1).unwrap();
        let err =
            containers_map_subsasgn(map, "()".to_string(), payload, rhs).expect_err("value count");
        let message = error_message(err);
        assert!(message.contains("exactly one scalar key"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn subsasgn_rejects_empty_key_collection() {
        let keys = crate::make_cell(vec![Value::from("root")], 1, 1).unwrap();
        let values = crate::make_cell(vec![Value::Num(7.0)], 1, 1).unwrap();
        let map = containers_map_builtin(vec![keys, values]).expect("map");
        let empty_keys = crate::make_cell(Vec::new(), 1, 0).unwrap();
        let payload = crate::make_cell(vec![empty_keys], 1, 1).unwrap();
        let rhs = crate::make_cell(Vec::new(), 1, 0).unwrap();
        let error = containers_map_subsasgn(map, "()".to_string(), payload, rhs).unwrap_err();
        assert!(error.to_string().contains("exactly one scalar key"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn keys_values_iskey_remove() {
        let keys = crate::make_cell(
            vec![Value::from("a"), Value::from("b"), Value::from("c")],
            1,
            3,
        )
        .unwrap();
        let values = crate::make_cell(
            vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
            1,
            3,
        )
        .unwrap();
        let map = containers_map_builtin(vec![keys, values]).expect("map");
        let key_list = containers_map_keys(map.clone()).expect("keys");
        match key_list {
            Value::Cell(cell) => assert_eq!(cell.data.len(), 3),
            other => panic!("expected cell array, got {other:?}"),
        }
        let mask = containers_map_is_key(
            map.clone(),
            crate::make_cell(vec![Value::from("a"), Value::from("z")], 1, 2).unwrap(),
        )
        .expect("mask");
        match mask {
            Value::LogicalArray(arr) => {
                assert_eq!(arr.data, vec![1, 0]);
            }
            other => panic!("expected logical array, got {:?}", other),
        }
        let removed = containers_map_remove(
            map.clone(),
            crate::make_cell(vec![Value::from("b")], 1, 1).unwrap(),
        )
        .expect("remove");
        let mask = containers_map_is_key(
            removed,
            crate::make_cell(vec![Value::from("b")], 1, 1).unwrap(),
        )
        .expect("mask");
        assert_eq!(mask, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn remove_missing_key_returns_error() {
        let keys = crate::make_cell(vec![Value::from("key")], 1, 1).unwrap();
        let values = crate::make_cell(vec![Value::Num(1.0)], 1, 1).unwrap();
        let map = containers_map_builtin(vec![keys, values]).expect("map");
        let err = containers_map_remove(
            map,
            crate::make_cell(vec![Value::from("missing")], 1, 1).unwrap(),
        )
        .expect_err("remove missing");
        assert_eq!(
            err.identifier(),
            CONTAINERS_MAP_ERROR_MISSING_KEY.identifier
        );
        let message = error_message(err);
        assert_eq!(message, CONTAINERS_MAP_ERROR_MISSING_KEY.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn length_delegates_to_map_count() {
        let keys = crate::make_cell(
            vec![Value::from("a"), Value::from("b"), Value::from("c")],
            1,
            3,
        )
        .unwrap();
        let values = crate::make_cell(
            vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
            1,
            3,
        )
        .unwrap();
        let map = containers_map_builtin(vec![keys, values]).expect("map");
        assert_eq!(map_length(&map), Some(3));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn map_id_rejects_corrupted_numeric_identifiers() {
        for id_value in [
            Value::Num(1.9),
            Value::Num(f64::INFINITY),
            Value::Num(u64::MAX as f64),
        ] {
            let mut storage = ObjectInstance::new(CLASS_NAME.to_string());
            storage.properties.insert("id".to_string(), id_value);
            let target = runmat_gc::gc_allocate(Value::Object(storage)).expect("storage");
            let handle = HandleRef {
                class_name: CLASS_NAME.to_string(),
                target,
                valid: true,
            };

            let err =
                map_id(&handle, BUILTIN_CONSTRUCTOR).expect_err("corrupted map id should reject");
            assert_eq!(err.identifier(), CONTAINERS_MAP_ERROR_INTERNAL.identifier);
        }
    }

    #[test]
    fn typed_map_keys_preserve_full_uint64_range() {
        assert_eq!(
            unsigned_from_value(
                &Value::Int(IntValue::U64(u64::MAX)),
                u64::MAX,
                "key",
                BUILTIN_CONSTRUCTOR,
            )
            .expect("uint64 key"),
            u64::MAX
        );
        assert!(integer_from_value(
            &Value::Int(IntValue::U64(u64::MAX)),
            i64::MIN,
            i64::MAX,
            "key",
            BUILTIN_CONSTRUCTOR,
        )
        .is_err());
    }

    #[test]
    fn scalar_map_helpers_read_typed_integer_tensor_storage_exactly() {
        let u64_tensor = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
            .expect("uint64 key");
        assert_eq!(
            unsigned_from_value(
                &Value::Tensor(u64_tensor),
                u64::MAX,
                "key",
                BUILTIN_CONSTRUCTOR,
            )
            .expect("uint64 key"),
            u64::MAX
        );

        let i32_tensor =
            Tensor::new_integer(IntegerStorage::I32(vec![-7]), vec![1, 1]).expect("int32 key");
        assert_eq!(
            integer_from_value(
                &Value::Tensor(i32_tensor),
                i32::MIN as i64,
                i32::MAX as i64,
                "key",
                BUILTIN_CONSTRUCTOR,
            )
            .expect("int32 key"),
            -7
        );

        let logical_tensor =
            Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).expect("logical key");
        assert!(
            bool_from_value(&Value::Tensor(logical_tensor), "key", BUILTIN_CONSTRUCTOR,)
                .expect("logical key")
        );
    }

    #[test]
    fn vector_map_helpers_preserve_native_single_and_logicalize_typed_values() {
        let single = Tensor::from_f32(vec![1.25, 2.5], vec![1, 2]).expect("single values");
        let values = tensor_elements_to_values(&single);
        assert_eq!(values.len(), 2);
        for (value, expected) in values.into_iter().zip([1.25_f32, 2.5]) {
            let Value::Tensor(tensor) = value else {
                panic!("expected native-single scalar tensor");
            };
            assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
            assert_eq!(
                tensor.numeric_value_at(0),
                Some(runmat_value::NumericScalar::F32(expected))
            );
        }

        let logical = normalize_logical_value(
            Value::Tensor(
                Tensor::from_f32(vec![0.0, -0.0, 2.0, f32::NAN], vec![2, 2])
                    .expect("single logical source"),
            ),
            BUILTIN_CONSTRUCTOR,
        );
        assert!(logical.is_err());
    }

    #[test]
    fn vector_map_keys_and_values_read_typed_integer_storage_exactly() {
        let keys = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX]),
            vec![1, 2],
        )
        .expect("uint64 keys");
        let values =
            Tensor::new_integer(IntegerStorage::I32(vec![11, 22]), vec![1, 2]).expect("values");

        let map =
            containers_map_builtin(vec![Value::Tensor(keys), Value::Tensor(values)]).expect("map");

        let key_payload = crate::make_cell(
            vec![
                Value::Int(IntValue::U64(u64::MAX - 1)),
                Value::Int(IntValue::U64(u64::MAX)),
            ],
            1,
            2,
        )
        .unwrap();
        let result = containers_map_values(map, vec![key_payload]).expect("selected values");
        match result {
            Value::Cell(cell) => {
                assert_eq!((cell.rows, cell.cols), (1, 2));
                assert_eq!(cell.data[0], Value::Int(IntValue::I32(11)));
                assert_eq!(cell.data[1], Value::Int(IntValue::I32(22)));
            }
            other => panic!("expected cell lookup result, got {:?}", other),
        }
    }

    #[test]
    fn value_type_logical_reads_typed_integer_tensor_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![0, u64::MAX]), vec![1, 2])
            .expect("integer logical source");
        let map = containers_map_builtin(vec![
            Value::from("KeyType"),
            Value::from("char"),
            Value::from("ValueType"),
            Value::from("logical"),
        ])
        .expect("map");
        let payload = crate::make_cell(vec![Value::from("mask")], 1, 1).unwrap();
        let error = containers_map_subsasgn(map, "()".to_string(), payload, Value::Tensor(tensor))
            .expect_err("declared logical ValueType accepts scalar values only");
        assert!(error.to_string().contains("logical scalars"));
    }

    fn all_integer_value_storages() -> Vec<(IntegerStorage, &'static str)> {
        vec![
            (IntegerStorage::I8(vec![i8::MIN, i8::MAX]), "int8"),
            (IntegerStorage::U8(vec![0, u8::MAX]), "uint8"),
            (IntegerStorage::I16(vec![i16::MIN, i16::MAX]), "int16"),
            (IntegerStorage::U16(vec![0, u16::MAX]), "uint16"),
            (IntegerStorage::I32(vec![i32::MIN, i32::MAX]), "int32"),
            (IntegerStorage::U32(vec![0, u32::MAX]), "uint32"),
            (IntegerStorage::I64(vec![i64::MIN, i64::MAX]), "int64"),
            (IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX]), "uint64"),
        ]
    }

    fn public_integer_key_storages() -> Vec<(IntegerStorage, &'static str)> {
        vec![
            (IntegerStorage::I32(vec![i32::MIN, i32::MAX]), "int32"),
            (IntegerStorage::U32(vec![0, u32::MAX]), "uint32"),
            (IntegerStorage::I64(vec![i64::MIN, i64::MAX]), "int64"),
            (IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX]), "uint64"),
        ]
    }

    fn property_text(map: Value, name: &str) -> String {
        let value = containers_map_subsref(map, ".".to_string(), Value::from(name)).unwrap();
        let Value::CharArray(chars) = value else {
            panic!("expected char property");
        };
        chars.data.iter().collect()
    }

    #[test]
    fn constructor_subsref_and_values_preserve_all_integer_value_classes_exactly() {
        for (storage, class_name) in all_integer_value_storages() {
            let expected = storage.exact_values();
            let keys = crate::make_cell(vec![Value::from("a"), Value::from("b")], 1, 2).unwrap();
            let values = Value::Tensor(Tensor::new_integer(storage, vec![1, 2]).unwrap());
            let map = containers_map_builtin(vec![keys, values]).unwrap();
            assert_eq!(property_text(map.clone(), "ValueType"), class_name);

            let payload = crate::make_cell(vec![Value::from("b")], 1, 1).unwrap();
            assert_eq!(
                containers_map_subsref(map.clone(), "()".to_string(), payload).unwrap(),
                Value::Int(expected[1].clone())
            );

            let Value::Cell(all_values) = containers_map_values(map.clone(), Vec::new()).unwrap()
            else {
                panic!("expected values cell");
            };
            assert_eq!(
                all_values.data,
                vec![
                    Value::Int(expected[0].clone()),
                    Value::Int(expected[1].clone())
                ]
            );

            let selected_keys =
                crate::make_cell(vec![Value::from("b"), Value::from("a")], 2, 1).unwrap();
            let Value::Cell(selected) =
                containers_map_values(map, vec![selected_keys]).expect("selected values")
            else {
                panic!("expected selected values cell");
            };
            assert_eq!((selected.rows, selected.cols), (2, 1));
            assert_eq!(
                selected.data,
                vec![
                    Value::Int(expected[1].clone()),
                    Value::Int(expected[0].clone())
                ]
            );
        }
    }

    #[test]
    fn keys_iskey_and_remove_preserve_all_public_integer_key_classes() {
        for (storage, class_name) in public_integer_key_storages() {
            let expected = storage.exact_values();
            let key_tensor =
                Value::Tensor(Tensor::new_integer(storage.clone(), vec![1, 2]).unwrap());
            let value_tensor = Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap());
            let map = containers_map_builtin(vec![key_tensor.clone(), value_tensor]).unwrap();
            assert_eq!(property_text(map.clone(), "KeyType"), class_name);

            let Value::Cell(keys) = containers_map_keys(map.clone()).unwrap() else {
                panic!("expected keys cell");
            };
            assert_eq!(
                keys.data,
                vec![
                    Value::Int(expected[0].clone()),
                    Value::Int(expected[1].clone())
                ]
            );

            let key_cell =
                crate::make_cell(expected.iter().cloned().map(Value::Int).collect(), 1, 2).unwrap();
            let Value::LogicalArray(found) =
                containers_map_is_key(map.clone(), key_cell).expect("isKey")
            else {
                panic!("expected logical array");
            };
            assert_eq!(found.shape, vec![1, 2]);
            assert_eq!(found.data, vec![1, 1]);

            let removed = containers_map_remove(map, Value::Int(expected[1].clone())).unwrap();
            assert_eq!(
                containers_map_is_key(removed, Value::Int(expected[1].clone())).unwrap(),
                Value::Bool(false)
            );
        }
    }

    #[test]
    fn integer_key_methods_reject_class_mismatch_and_binary64_aliases() {
        let wide = 9_007_199_254_740_993_u64;
        let keys = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap(),
        );
        let map = containers_map_builtin(vec![keys, Value::Num(7.0)]).unwrap();
        let rounded_double = Value::Num(wide as f64);
        let payload = crate::make_cell(vec![rounded_double.clone()], 1, 1).unwrap();
        for error in [
            containers_map_subsref(map.clone(), "()".to_string(), payload.clone()).unwrap_err(),
            containers_map_is_key(map.clone(), rounded_double.clone()).unwrap_err(),
            containers_map_remove(map.clone(), rounded_double.clone()).unwrap_err(),
            containers_map_subsasgn(map.clone(), "()".to_string(), payload, Value::Num(9.0))
                .unwrap_err(),
        ] {
            assert!(error.to_string().contains("KeyType 'uint64'"));
        }
        let exact_payload = crate::make_cell(vec![Value::Int(IntValue::U64(wide))], 1, 1).unwrap();
        assert_eq!(
            containers_map_subsref(map, "()".to_string(), exact_payload).unwrap(),
            Value::Num(7.0)
        );
    }

    #[test]
    fn direct_noncell_multi_key_method_inputs_reject() {
        let keys = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I32(vec![1, 2]), vec![1, 2]).unwrap(),
        );
        let map = containers_map_builtin(vec![
            keys.clone(),
            Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap()),
        ])
        .unwrap();
        let is_key_error = containers_map_is_key(map.clone(), keys.clone()).unwrap_err();
        assert!(is_key_error.to_string().contains("cell array"));
        let remove_error = containers_map_remove(map, keys).unwrap_err();
        assert!(remove_error.to_string().contains("cell array"));
    }

    #[test]
    fn subsasgn_explicit_integer_value_types_cover_all_eight_classes() {
        for (storage, class_name) in all_integer_value_storages() {
            let expected = storage.exact_values()[1].clone();
            let map = containers_map_builtin(vec![
                Value::from("KeyType"),
                Value::from("uint64"),
                Value::from("ValueType"),
                Value::from(class_name),
            ])
            .unwrap();
            let payload =
                crate::make_cell(vec![Value::Int(IntValue::U64(u64::MAX))], 1, 1).unwrap();
            let updated = containers_map_subsasgn(
                map,
                "()".to_string(),
                payload.clone(),
                Value::Int(expected.clone()),
            )
            .unwrap();
            assert_eq!(
                containers_map_subsref(updated, "()".to_string(), payload).unwrap(),
                Value::Int(expected)
            );
        }
    }

    #[test]
    fn any_preserves_nonscalar_integer_arrays_and_declared_types_reject_them() {
        for (storage, class_name) in all_integer_value_storages() {
            let expected = storage.clone();
            let array = Value::Tensor(Tensor::new_integer(storage, vec![1, 2]).unwrap());
            let any_map = containers_map_builtin(vec![
                Value::from("KeyType"),
                Value::from("char"),
                Value::from("ValueType"),
                Value::from("any"),
            ])
            .unwrap();
            let payload = crate::make_cell(vec![Value::from("array")], 1, 1).unwrap();
            let any_map =
                containers_map_subsasgn(any_map, "()".to_string(), payload.clone(), array.clone())
                    .unwrap();
            let Value::Tensor(stored) =
                containers_map_subsref(any_map, "()".to_string(), payload.clone()).unwrap()
            else {
                panic!("expected exact integer tensor");
            };
            assert_eq!(stored.integer_storage(), Some(&expected));

            let typed_map = containers_map_builtin(vec![
                Value::from("KeyType"),
                Value::from("char"),
                Value::from("ValueType"),
                Value::from(class_name),
            ])
            .unwrap();
            assert!(containers_map_subsasgn(typed_map, "()".to_string(), payload, array,).is_err());
        }
    }

    #[test]
    fn unsupported_narrow_integer_key_classes_reject_without_double_aliasing() {
        for storage in [
            IntegerStorage::I8(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::U16(vec![1]),
        ] {
            let error = containers_map_builtin(vec![
                Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            ])
            .unwrap_err();
            assert!(error.to_string().contains("int32"));
        }
    }

    #[test]
    fn resident_extensions_gate_before_provider_access_for_integer_surfaces() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 77,
            buffer_id: 99,
            descriptor: Default::default(),
        });
        let char_map = containers_map_builtin(Vec::new()).unwrap();
        let checks = [
            block_on(super::containers_map_builtin(vec![
                resident.clone(),
                Value::Num(1.0),
            ]))
            .unwrap_err(),
            block_on(super::containers_map_is_key(
                char_map.clone(),
                resident.clone(),
            ))
            .unwrap_err(),
            block_on(super::containers_map_remove(
                char_map.clone(),
                resident.clone(),
            ))
            .unwrap_err(),
            block_on(super::containers_map_values(
                char_map.clone(),
                vec![crate::make_cell(vec![resident.clone()], 1, 1).unwrap()],
            ))
            .unwrap_err(),
            block_on(super::containers_map_subsref(
                char_map.clone(),
                "()".to_string(),
                crate::make_cell(vec![resident.clone()], 1, 1).unwrap(),
            ))
            .unwrap_err(),
            block_on(super::containers_map_subsasgn(
                char_map,
                "()".to_string(),
                crate::make_cell(vec![Value::from("x")], 1, 1).unwrap(),
                resident,
            ))
            .unwrap_err(),
        ];
        for error in checks {
            assert!(error
                .identifier()
                .is_some_and(|identifier| identifier
                    .starts_with("RunMat:compatibility:ContainersMapResident")));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn map_constructor_gathers_gpu_values() {
        test_support::with_test_provider(|provider| {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let keys = crate::make_cell(vec![Value::from("alpha")], 1, 1).unwrap();
            let data = vec![1.0, 2.0, 3.0];
            let shape = vec![3, 1];
            let view = runmat_accelerate_api::HostTensorView {
                data: &data,
                shape: &shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let values = crate::make_cell(vec![Value::GpuTensor(handle)], 1, 1).unwrap();
            let map = containers_map_builtin(vec![
                keys,
                values,
                Value::from("UniformValues"),
                Value::Bool(false),
            ])
            .expect("map");
            let payload = crate::make_cell(vec![Value::from("alpha")], 1, 1).unwrap();
            let value = containers_map_subsref(map, "()".to_string(), payload).expect("lookup");
            match value {
                Value::Tensor(t) => {
                    assert_eq!(t.shape, shape);
                    assert_eq!(t.materialize_f64(), data);
                }
                other => panic!("expected tensor, got {:?}", other),
            }
        });
    }

    #[test]
    fn map_resident_integer_keys_and_values_gather_exactly() {
        test_support::with_test_provider(|provider| {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let wide = 9_007_199_254_740_993_u64;
            let key_tensor =
                Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
            let key_handle = gpu_helpers::upload_tensor(provider, &key_tensor).expect("key upload");
            let map = containers_map_builtin(vec![Value::GpuTensor(key_handle), Value::Num(7.0)])
                .expect("resident key constructor");
            let payload = crate::make_cell(vec![Value::Int(IntValue::U64(wide))], 1, 1).unwrap();
            assert_eq!(
                containers_map_subsref(map, "()".to_string(), payload).unwrap(),
                Value::Num(7.0)
            );

            let value_tensor =
                Tensor::new_integer(IntegerStorage::U64(vec![wide, u64::MAX]), vec![2, 1]).unwrap();
            let value_handle =
                gpu_helpers::upload_tensor(provider, &value_tensor).expect("value upload");
            let keys = crate::make_cell(vec![Value::from("wide")], 1, 1).unwrap();
            let values = crate::make_cell(vec![Value::GpuTensor(value_handle)], 1, 1).unwrap();
            let map = containers_map_builtin(vec![
                keys,
                values,
                Value::from("UniformValues"),
                Value::Bool(false),
            ])
            .expect("resident value constructor");
            let payload = crate::make_cell(vec![Value::from("wide")], 1, 1).unwrap();
            let Value::Tensor(gathered) =
                containers_map_subsref(map, "()".to_string(), payload).unwrap()
            else {
                panic!("expected exact integer tensor value");
            };
            assert_eq!(gathered.integer_storage(), value_tensor.integer_storage());
        });
    }
}
