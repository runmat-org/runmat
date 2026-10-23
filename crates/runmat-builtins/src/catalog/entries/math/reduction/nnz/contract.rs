use crate::*;

const NNZ_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Count of nonzero elements.",
}];

const NNZ_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const NNZ_INPUTS_DIM: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dimension selector.",
    },
];

const NNZ_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "N = nnz(A)",
        inputs: &NNZ_INPUTS_CORE,
        outputs: &NNZ_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "N = nnz(A, dim)",
        inputs: &NNZ_INPUTS_DIM,
        outputs: &NNZ_OUTPUT,
    },
];

pub const NNZ_DIMENSION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "nnz-dimension-reduction",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "nnz(A,dim) is a RunMat reduction convenience; the documented MATLAB nnz surface accepts only nnz(A)",
    error_identifier: Some("RunMat:compatibility:NnzDimensionExtension"),
};

pub const NNZ_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [NNZ_DIMENSION_EXTENSION];

const NNZ_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are explicitly documented and zero tests read authoritative native storage.",
    }];

const NNZ_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The entire dimension-reduction overload is RunMat-only; admitted integer dimensions are parsed exactly before reduction.",
    }];

pub const NNZ_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "N = nnz(integer_A)", inputs: &NNZ_INTEGER_DATA_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Every exact nonzero native integer contributes one to the host double result. There is no native resident integer count hook: only automatic placement may take admitted exact-owner host fallback; explicit resident input is rejected before gathering." },
    BuiltinIntegerCapabilityDescriptor { form: "N = nnz(integer_A, integer_dim)", inputs: &NNZ_INTEGER_DIM_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "MATLAB-compatible modes reject the undocumented overload before provider access. RunMat mode returns a host double reduction with MATLAB dimension geometry." },
];

pub const NNZ_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NNZ.INVALID_ARGUMENT",
    identifier: Some("RunMat:nnz:InvalidArgument"),
    when: "Dimension argument grammar is invalid.",
    message: "nnz: invalid argument",
};

pub const NNZ_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NNZ.INVALID_INPUT",
    identifier: Some("RunMat:nnz:InvalidInput"),
    when: "Input type is unsupported for nnz.",
    message: "nnz: invalid input",
};

pub const NNZ_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NNZ.INTERNAL",
    identifier: Some("RunMat:nnz:Internal"),
    when: "Reduction execution fails due to provider, conversion, or shape operations.",
    message: "nnz: internal reduction failure",
};

pub const NNZ_ERRORS: [BuiltinErrorDescriptor; 4] = [
    NNZ_ERROR_INVALID_ARGUMENT,
    NNZ_ERROR_INVALID_INPUT,
    NNZ_ERROR_INTERNAL,
    NNZ_ERROR_TOO_MANY_OUTPUTS,
];

pub const NNZ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NNZ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NNZ_ERRORS,
};

pub const NNZ_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NNZ.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:nnz:TooManyOutputs"),
    when: "More outputs are requested than this public form returns.",
    message: "nnz: too many output arguments",
};
