use super::integer_inputs::*;
use crate::*;
pub const DECOMPOSITION_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "dA = decomposition(integer_A,...)",
        inputs: &DECOMPOSITION_INTEGER_COEFFICIENT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Construction and structural metadata retain exact integer storage; resident coefficients gather through their owner into host object storage.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "dA = decomposition(A,...,'CheckCondition',integer_value)",
        inputs: &DECOMPOSITION_INTEGER_CHECK_CONDITION,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The compatibility gate runs before any resident control value is gathered.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "dA = decomposition(A,...,'RankTolerance',integer_value)",
        inputs: &DECOMPOSITION_INTEGER_RANK_TOLERANCE,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "RunMat-only typed-integer RankTolerance crosses an explicit exact binary64 boundary after the compatibility gate.",
    },
];

pub const DECOMPOSITION_CTRANSPOSE_INTEGER_CAPABILITIES:
    [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "out = ctranspose(integer_decomposition)",
    inputs: &DECOMPOSITION_INTEGER_OBJECT,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::PreserveInput,
    overflow: BuiltinIntegerOverflowRule::Saturate,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::FunctionSpecific,
    notes: "Transpose and conjugation retain the exact coefficient class; signed/unsigned integer conjugation uses the class assignment saturation rule.",
}];

pub const DECOMPOSITION_PROPERTY_INTEGER_CAPABILITIES:
    [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "property = integer_decomposition.Property",
    inputs: &DECOMPOSITION_INTEGER_OBJECT,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::NotApplicable,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::FunctionSpecific,
    notes: "Property access reads metadata from an exact RunMat-only integer decomposition object and remains gated whenever that persisted object is consumed.",
}];

pub const DECOMPOSITION_TRANSFORM_INTEGER_CAPABILITIES:
    [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "out = +integer_decomposition, -integer_decomposition, or scalar scaling",
    inputs: &DECOMPOSITION_INTEGER_OBJECT,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::PreserveInput,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::FunctionSpecific,
    notes: "Unary and scalar transforms retain the exact coefficient matrix and update only decomposition metadata; every persisted-object entry point rechecks the RunMat extension gate.",
}];

pub const DECOMPOSITION_SCALE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "out = integer_decomposition scaled by a floating scalar",
        inputs: &DECOMPOSITION_INTEGER_OBJECT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "The exact coefficient matrix is retained while decomposition scale metadata changes.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "out = dA .* integer_scale or integer_scale .* dA",
        inputs: &DECOMPOSITION_INTEGER_SCALE_FACTOR,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The RunMat-only real integer factor must be exactly representable at the binary64 scale-metadata boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "out = dA .* complex_integer_scale or complex_integer_scale .* dA",
        inputs: &DECOMPOSITION_COMPLEX_INTEGER_SCALE_FACTOR,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Both RunMat-only paired integer components must be exactly representable at the complex binary64 scale-metadata boundary.",
    },
];

pub const DECOMPOSITION_RDIVIDE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "out = integer_decomposition / scalar",
        inputs: &DECOMPOSITION_INTEGER_OBJECT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Scalar division retains the exact coefficient matrix and updates only decomposition metadata; every persisted-object entry point rechecks the RunMat extension gate.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "X = B / integer_decomposition",
        inputs: &DECOMPOSITION_INTEGER_OBJECT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Exact integer coefficients are explicitly promoted to binary64 at the solve boundary; the solution is double or complex double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "out = dA / integer_scale",
        inputs: &DECOMPOSITION_INTEGER_SCALE_FACTOR,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The RunMat-only real integer divisor must be exactly representable at the binary64 scale-metadata boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "out = dA / complex_integer_scale",
        inputs: &DECOMPOSITION_COMPLEX_INTEGER_SCALE_FACTOR,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Both RunMat-only paired integer divisor components must be exactly representable at the complex binary64 scale-metadata boundary.",
    },
];

pub const DECOMPOSITION_SOLVE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "X = integer_decomposition \\ B or X = B / integer_decomposition",
        inputs: &DECOMPOSITION_INTEGER_OBJECT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Exact integer coefficients are explicitly promoted to binary64 at the solve boundary; the solution is double or complex double.",
    }];
