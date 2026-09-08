use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};

const OUTPUTS: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell array result.",
}];
const INPUT_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
};
const INPUT_DIMS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive integer dimension scalar or vector retained in each cell.",
};
const INPUTS_ONE: &[BuiltinParamDescriptor] = &[INPUT_A];
const INPUTS_DIMS: &[BuiltinParamDescriptor] = &[INPUT_A, INPUT_DIMS];
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "C = num2cell(A)",
        inputs: INPUTS_ONE,
        outputs: OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "C = num2cell(A, dim)",
        inputs: INPUTS_DIMS,
        outputs: OUTPUTS,
    },
];

pub const NUM2CELL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUM2CELL.INVALID_INPUT",
    identifier: Some("RunMat:num2cell:InvalidInput"),
    when: "The input or dimension selector is unsupported.",
    message: "num2cell: invalid input",
};
pub const NUM2CELL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUM2CELL.INTERNAL",
    identifier: Some("RunMat:num2cell:Internal"),
    when: "An internal shape or storage invariant is inconsistent.",
    message: "num2cell: internal shape error",
};
const ERRORS: &[BuiltinErrorDescriptor] = &[NUM2CELL_ERROR_INVALID_INPUT, NUM2CELL_ERROR_INTERNAL];

pub const NUM2CELL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};

const INTEGER_INPUTS: &[BuiltinIntegerInputCapability] = &[
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Scalar cells and grouped blocks retain A's native integer class.",
    },
    BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Dimension values are decoded exactly and range checked.",
    },
];

pub const NUM2CELL_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] =
    &[BuiltinIntegerCapabilityDescriptor {
        form: "C = num2cell(A, dim)",
        inputs: INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Cell construction preserves exact native storage; resident input is gathered before partitioning.",
    }];
