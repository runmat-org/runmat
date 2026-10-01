use crate::*;
const COV_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A_or_B",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented observation domain is single or double; RunMat mode additionally accepts all eight real integer classes.",
    }];

const COV_INTEGER_NORMALIZATION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "w",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented scalar normalization weight is single or double; RunMat mode additionally accepts typed integer 0 or 1.",
    }];

const COV_INTEGER_VECTOR_WEIGHTS_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "vector_weights",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Observation-weight vectors are a RunMat extension for every numeric class; integer vectors are validated exactly before floating weighted covariance.",
    }];

pub const COV_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "C = cov(integer_A_or_B, w, nanflag)",
        inputs: &COV_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat preserves exact integer differences while centering each variable before producing double covariance.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = cov(A, integer_w, nanflag)",
        inputs: &COV_INTEGER_NORMALIZATION_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed integer normalization is a RunMat-only control and must equal zero or one.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = cov(A, integer_vector_weights, nanflag)",
        inputs: &COV_INTEGER_VECTOR_WEIGHTS_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "RunMat-only observation-weight vectors are nonnegative and finite; integer weights enter the floating weighted covariance domain and the data input determines output precision.",
    },
];
