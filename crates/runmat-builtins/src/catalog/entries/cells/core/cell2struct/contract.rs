use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};

const OUTPUTS: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Structure or structure array result.",
}];
const INPUTS: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cell array containing field values.",
    },
    BuiltinParamDescriptor {
        name: "fields",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Field names as character data, strings, or a cell array of text scalars.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Dimension whose entries correspond to field names.",
    },
];
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "S = cell2struct(C, fields, dim)",
    inputs: INPUTS,
    outputs: OUTPUTS,
}];

pub const CELL2STRUCT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELL2STRUCT.INVALID_INPUT",
    identifier: Some("RunMat:cell2struct:InvalidInput"),
    when: "Arguments are not a cell array, field-name list, and valid dimension.",
    message: "cell2struct: invalid input",
};
pub const CELL2STRUCT_ERROR_SHAPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELL2STRUCT.SHAPE",
    identifier: Some("RunMat:cell2struct:ShapeMismatch"),
    when: "The field count differs from the selected cell dimension.",
    message: "cell2struct: field count does not match selected dimension",
};
const ERRORS: &[BuiltinErrorDescriptor] =
    &[CELL2STRUCT_ERROR_INVALID_INPUT, CELL2STRUCT_ERROR_SHAPE];

pub const CELL2STRUCT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};

const PAYLOAD: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "C payload",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer payloads move into fields without conversion or provider access.",
}];
const DIMENSION: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "dim",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Typed integer dimensions are decoded exactly and range checked.",
}];
pub const CELL2STRUCT_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[
    BuiltinIntegerCapabilityDescriptor { form: "S = cell2struct(C_with_integer_payload, fields, dim)", inputs: PAYLOAD, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Only container metadata changes; nested host values and resident handles retain identity." },
    BuiltinIntegerCapabilityDescriptor { form: "S = cell2struct(C, fields, integer_dim)", inputs: DIMENSION, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The dimension must be positive and representable by the target platform." },
];
