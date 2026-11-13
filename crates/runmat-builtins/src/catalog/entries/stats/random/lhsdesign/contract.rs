//! Typed declarations conserved from the Runtime owner.
use crate::*;

pub const INTEGER_DIMENSION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "lhsdesign-integer-dimension",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "lhsdesign with typed-integer n or p dimensions is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LhsdesignIntegerDimensionExtension"),
};

pub const INTEGER_ITERATIONS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "lhsdesign-integer-iterations",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "lhsdesign with a typed-integer Iterations control is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LhsdesignIntegerIterationsExtension"),
};

pub const INTEGER_SMOOTH_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "lhsdesign-integer-smooth",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "lhsdesign with a typed-integer Smooth boolean alias is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LhsdesignIntegerSmoothExtension"),
};

pub const RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "lhsdesign-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "lhsdesign host fallback for explicit gpuArray inputs is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LhsdesignExplicitGpuInputExtension"),
};

pub const LHSDESIGN_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    INTEGER_DIMENSION_EXTENSION,
    INTEGER_ITERATIONS_EXTENSION,
    INTEGER_SMOOTH_EXTENSION,
    RESIDENT_INPUT_EXTENSION,
];

pub const INTEGER_DIMENSION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n or p",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer design dimensions are gated before gather and decoded exactly into bounded positive host counts.",
    }];

pub const INTEGER_ITERATIONS_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Iterations",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer iteration counts are gated before gather and decoded exactly without a floating round trip.",
    }];

pub const INTEGER_SMOOTH_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Smooth",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The RunMat-only integer alias accepts only exact scalar zero or one.",
    }];

pub const LHSDESIGN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "X = lhsdesign(integer_n, integer_p, ___)",
        inputs: &INTEGER_DIMENSION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "n and p are exact structural counts. Current public documentation does not declare lhsdesign output class, so this metadata does not infer single propagation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "X = lhsdesign(n, p, Iterations=integer_iterations)",
        inputs: &INTEGER_ITERATIONS_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Iterations is a positive structural work bound and does not select output class or residency.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "X = lhsdesign(n, p, Smooth=integer_boolean)",
        inputs: &INTEGER_SMOOTH_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer Smooth aliases are a separately gated RunMat convenience; documented text values remain the compatibility surface.",
    },
];

pub const OUTPUT_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "N-by-P Latin hypercube design.",
}];

pub const PARAM_N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of sample points.",
};

pub const PARAM_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of variables.",
};

pub const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options such as Smooth, Criterion, and Iterations.",
};

pub const INPUTS_N_P: [BuiltinParamDescriptor; 2] = [PARAM_N, PARAM_P];

pub const INPUTS_N_P_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_N, PARAM_P, PARAM_OPTIONS];

pub const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "X = lhsdesign(n, p)",
        inputs: &INPUTS_N_P,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "X = lhsdesign(n, p, Name, Value)",
        inputs: &INPUTS_N_P_OPTIONS,
        outputs: &OUTPUT_X,
    },
];

pub const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LHSDESIGN.INVALID_ARGUMENT",
    identifier: Some("RunMat:lhsdesign:InvalidArgument"),
    when: "Sample counts, dimensions, criteria, or name-value options are malformed.",
    message: "lhsdesign: invalid argument",
};

pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LHSDESIGN.INTERNAL",
    identifier: Some("RunMat:lhsdesign:Internal"),
    when: "Internal random generation or tensor allocation fails.",
    message: "lhsdesign: internal operation failed",
};

pub const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const LHSDESIGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
