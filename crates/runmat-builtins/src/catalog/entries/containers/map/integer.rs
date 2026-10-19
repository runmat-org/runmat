use crate::*;

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
    classes: &crate::ALL_INTEGER_CLASSES,
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
        classes: &crate::ALL_INTEGER_CLASSES,
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
        classes: &crate::ALL_INTEGER_CLASSES,
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
        classes: &crate::ALL_INTEGER_CLASSES,
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
        classes: &crate::ALL_INTEGER_CLASSES,
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
        form: "S = substruct('.', 'Count'); count = containers.Map.subsref(M, S)",
        inputs: &[],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The read-only Count property is a scalar uint64 and is derived from the host Map entry count without floating conversion.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "S = substruct('()', {integer_key}); value = containers.Map.subsref(M, S)",
        inputs: &MAP_KEY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The integer controls lookup only; output class is the stored value class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "S = substruct('()', {key}); integer_value = containers.Map.subsref(M_with_integer_value, S)",
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
        form: "S = substruct('()', {resident_integer_key}); value = containers.Map.subsref(M, S)",
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
        form: "S = substruct('()', {integer_key}); containers.Map.subsasgn(M, S, integer_rhs)",
        inputs: &MAP_ASSIGN_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Exact key identity is independent of exact rhs storage; a declared integer ValueType applies MATLAB integer conversion before storage.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "S = substruct('()', {resident_integer_key}); containers.Map.subsasgn(M, S, resident_integer_rhs)",
        inputs: &MAP_RESIDENT_ASSIGN_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only resident assignment data gathers after compatibility admission and before host key normalization and rhs conversion.",
    },
];
