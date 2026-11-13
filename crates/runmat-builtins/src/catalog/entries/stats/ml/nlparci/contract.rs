use crate::*;

pub const RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "nlparci-resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "nlparci host fallback for resident real floating inputs is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NlparciResidentInputExtension"),
};

const BETA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "beta",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Estimated real single or double parameter vector.",
};
const RESIDUAL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Residual vector with one value per observation.",
};
const METHOD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "method",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Jacobian or Covar.",
};
const SOURCE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "J or CovB",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Jacobian matrix or estimated covariance matrix.",
};
const ALPHA_NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Alpha",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Alpha option name.",
};
const ALPHA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "alpha",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: Some("0.05"),
    description: "Significance level strictly between zero and one.",
};
const CI: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ci",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "N-by-2 lower and upper confidence interval bounds.",
}];
const INPUTS: [BuiltinParamDescriptor; 4] = [BETA, RESIDUAL, METHOD, SOURCE];
const INPUTS_ALPHA: [BuiltinParamDescriptor; 6] =
    [BETA, RESIDUAL, METHOD, SOURCE, ALPHA_NAME, ALPHA];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "ci = nlparci(beta, r, 'Jacobian', J)",
        inputs: &INPUTS,
        outputs: &CI,
    },
    BuiltinSignatureDescriptor {
        label: "ci = nlparci(beta, r, 'Covar', CovB)",
        inputs: &INPUTS,
        outputs: &CI,
    },
    BuiltinSignatureDescriptor {
        label: "ci = nlparci(beta, r, method, data, 'Alpha', alpha)",
        inputs: &INPUTS_ALPHA,
        outputs: &CI,
    },
];

pub const NLPARCI_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NLPARCI.INVALID_ARGUMENT",
    identifier: Some("RunMat:nlparci:InvalidArgument"),
    when: "The option, class, shape, or significance level is invalid.",
    message: "nlparci: invalid argument",
};
pub const NLPARCI_ERROR_NUMERICAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor { code: "RM.NLPARCI.NUMERICAL", identifier: Some("RunMat:nlparci:NumericalFailure"), when: "There are insufficient complete observations, a rank-deficient Jacobian, or the interval calculation cannot be completed.", message: "nlparci: numerical failure" };
const ERRORS: [BuiltinErrorDescriptor; 2] =
    [NLPARCI_ERROR_INVALID_ARGUMENT, NLPARCI_ERROR_NUMERICAL];
pub const NLPARCI_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const REJECTED_DATA: [BuiltinIntegerInputCapability; 4] = [
    BuiltinIntegerInputCapability {
        name: "beta",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "Public parameter estimates are single or double; native integer storage is rejected.",
    },
    BuiltinIntegerInputCapability {
        name: "r",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Residuals must be floating point.",
    },
    BuiltinIntegerInputCapability {
        name: "J or CovB",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Jacobian and covariance values must be floating point.",
    },
    BuiltinIntegerInputCapability {
        name: "alpha",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The significance level must be a floating scalar.",
    },
];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "ci = nlparci(beta, r, method, data, 'Alpha', alpha)", inputs: &REJECTED_DATA, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "All native integer input roles are rejected before host materialization; scalar numeric literals are ordinary double values." }];
