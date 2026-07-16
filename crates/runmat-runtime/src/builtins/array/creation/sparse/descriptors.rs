use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

use super::{SPARSE_ERRORS, SPARSE_INPUT_A, SPARSE_INPUT_DIMS};

pub(super) const SPARSE_MATRIX_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sparse double matrix.",
}];

const DENSE_VECTOR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "v",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Full column vector.",
}];

const MATRIX_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Full or sparse matrix.",
}];

const DIMS_DENSITY_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "m",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of rows.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of columns.",
    },
    BuiltinParamDescriptor {
        name: "density",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Approximate density in [0,1].",
    },
];

const DIMS_DENSITY_RC_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "m",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of rows.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of columns.",
    },
    BuiltinParamDescriptor {
        name: "density",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Approximate density in [0,1].",
    },
    BuiltinParamDescriptor {
        name: "rc",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reciprocal condition number target or singular-value vector.",
    },
];

const MATRIX_TYPENAME_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Full or sparse matrix.",
    },
    BuiltinParamDescriptor {
        name: "typename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("double"),
        description: "Sparse storage type. RunMat currently supports double sparse storage.",
    },
];

const DIMS_DENSITY_TYPENAME_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "m",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of rows.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of columns.",
    },
    BuiltinParamDescriptor {
        name: "density",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Approximate density in [0,1].",
    },
    BuiltinParamDescriptor {
        name: "typename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("double"),
        description: "Sparse storage type. RunMat currently supports double sparse storage.",
    },
];

const DIMS_DENSITY_RC_TYPENAME_INPUTS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "m",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of rows.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of columns.",
    },
    BuiltinParamDescriptor {
        name: "density",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Approximate density in [0,1].",
    },
    BuiltinParamDescriptor {
        name: "rc",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reciprocal condition number target or singular-value vector.",
    },
    BuiltinParamDescriptor {
        name: "typename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("double"),
        description: "Sparse storage type. RunMat currently supports double sparse storage.",
    },
];

const SPDIAGS_OUTPUT_BOUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Bout",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dense matrix whose columns contain requested diagonals.",
}];

const SPDIAGS_OUTPUT_BOUT_ID: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Bout",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dense matrix whose columns contain requested diagonals.",
    },
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Diagonal numbers corresponding to columns of Bout.",
    },
];

const SPDIAGS_INPUT_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Full or sparse matrix.",
}];

const SPDIAGS_INPUT_A_D: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Full or sparse matrix.",
    },
    BuiltinParamDescriptor {
        name: "d",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Diagonal numbers to extract.",
    },
];

const SPDIAGS_INPUT_CONSTRUCT: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "Bin",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Diagonal values.",
    },
    BuiltinParamDescriptor {
        name: "d",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Diagonal numbers.",
    },
    BuiltinParamDescriptor {
        name: "m",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of rows.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of columns.",
    },
];

const SPDIAGS_INPUT_REPLACE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "Bin",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Replacement diagonal values.",
    },
    BuiltinParamDescriptor {
        name: "d",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Diagonal numbers to replace.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Matrix whose selected diagonals are replaced.",
    },
];

const SPEYE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "S = speye()",
        inputs: &[],
        outputs: &SPARSE_MATRIX_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = speye(n)",
        inputs: &SPARSE_INPUT_A,
        outputs: &SPARSE_MATRIX_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = speye(m, n)",
        inputs: &SPARSE_INPUT_DIMS,
        outputs: &SPARSE_MATRIX_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = speye(sz)",
        inputs: &SPARSE_INPUT_A,
        outputs: &SPARSE_MATRIX_OUTPUT,
    },
];

const NONZEROS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "v = nonzeros(A)",
    inputs: &MATRIX_INPUT,
    outputs: &DENSE_VECTOR_OUTPUT,
}];

const SPONES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "R = spones(S)",
    inputs: &MATRIX_INPUT,
    outputs: &SPARSE_MATRIX_OUTPUT,
}];

const SPRAND_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "R = sprand(S)",
        inputs: &MATRIX_INPUT,
        outputs: &SPARSE_MATRIX_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = sprand(S, typename)",
        inputs: &MATRIX_TYPENAME_INPUTS,
        outputs: &SPARSE_MATRIX_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = sprand(m, n, density)",
        inputs: &DIMS_DENSITY_INPUTS,
        outputs: &SPARSE_MATRIX_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = sprand(m, n, density, typename)",
        inputs: &DIMS_DENSITY_TYPENAME_INPUTS,
        outputs: &SPARSE_MATRIX_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = sprand(m, n, density, rc)",
        inputs: &DIMS_DENSITY_RC_INPUTS,
        outputs: &SPARSE_MATRIX_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = sprand(m, n, density, rc, typename)",
        inputs: &DIMS_DENSITY_RC_TYPENAME_INPUTS,
        outputs: &SPARSE_MATRIX_OUTPUT,
    },
];

const SPDIAGS_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "Bout = spdiags(A)",
        inputs: &SPDIAGS_INPUT_A,
        outputs: &SPDIAGS_OUTPUT_BOUT,
    },
    BuiltinSignatureDescriptor {
        label: "[Bout, id] = spdiags(A)",
        inputs: &SPDIAGS_INPUT_A,
        outputs: &SPDIAGS_OUTPUT_BOUT_ID,
    },
    BuiltinSignatureDescriptor {
        label: "Bout = spdiags(A, d)",
        inputs: &SPDIAGS_INPUT_A_D,
        outputs: &SPDIAGS_OUTPUT_BOUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = spdiags(Bin, d, m, n)",
        inputs: &SPDIAGS_INPUT_CONSTRUCT,
        outputs: &SPARSE_MATRIX_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = spdiags(Bin, d, A)",
        inputs: &SPDIAGS_INPUT_REPLACE,
        outputs: &SPARSE_MATRIX_OUTPUT,
    },
];

pub const SPEYE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPEYE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SPARSE_ERRORS,
};

pub const NONZEROS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NONZEROS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SPARSE_ERRORS,
};

pub const SPONES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPONES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SPARSE_ERRORS,
};

pub const SPRAND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPRAND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SPARSE_ERRORS,
};

pub const SPDIAGS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPDIAGS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SPARSE_ERRORS,
};
