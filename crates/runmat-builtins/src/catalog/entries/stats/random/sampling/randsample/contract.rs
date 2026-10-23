//! randsample declarations; shared sampling parameters remain family-owned.
use super::super::contract::{
    ERROR_INTERNAL, ERROR_INVALID_ARGUMENT, INPUTS_DATA_K, INPUTS_N_K, INPUTS_N_K_OPTIONS, OUTPUT_Y,
};
use crate::*;

pub const RANDSAMPLE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "y = randsample(n, k)",
        inputs: &INPUTS_N_K,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = randsample(population, k)",
        inputs: &INPUTS_DATA_K,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = randsample(___, replacement, w)",
        inputs: &INPUTS_N_K_OPTIONS,
        outputs: &OUTPUT_Y,
    },
];

pub const RANDSAMPLE_INTEGER_N_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "randsample-integer-range",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "randsample with a typed-integer range limit is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RandsampleIntegerRangeExtension"),
};

pub const RANDSAMPLE_INTEGER_POPULATION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "randsample-integer-population",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "randsample with a typed-integer population is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RandsampleIntegerPopulationExtension"),
    };

pub const RANDSAMPLE_INTEGER_K_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "randsample-integer-count",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "randsample with a typed-integer sample count is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RandsampleIntegerCountExtension"),
};

pub const RANDSAMPLE_INTEGER_REPLACEMENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "randsample-integer-replacement",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "randsample with a typed-integer replacement flag is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RandsampleIntegerReplacementExtension"),
    };

pub const RANDSAMPLE_INTEGER_WEIGHTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "randsample-integer-weights",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "randsample with typed-integer sampling weights is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RandsampleIntegerWeightsExtension"),
    };

pub const RANDSAMPLE_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    RANDSAMPLE_INTEGER_N_EXTENSION,
    RANDSAMPLE_INTEGER_POPULATION_EXTENSION,
    RANDSAMPLE_INTEGER_K_EXTENSION,
    RANDSAMPLE_INTEGER_REPLACEMENT_EXTENSION,
    RANDSAMPLE_INTEGER_WEIGHTS_EXTENSION,
];

pub const RANDSAMPLE_INTEGER_RANGE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents single/double n; RunMat typed range limits are exact structural controls and return double sampled indices.",
    }];

pub const RANDSAMPLE_INTEGER_POPULATION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "population",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target lists single/double/logical/text/categorical populations; RunMat preserves exact typed-integer population values, orientation, class, and supported residency.",
    }];

pub const RANDSAMPLE_INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "k/replacement/w",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed count, replacement, and weight roles are independently gated; only weights cross a checked binary64 probability boundary.",
    }];

pub const RANDSAMPLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "y = randsample(integer_n,k,___)",
        inputs: &RANDSAMPLE_INTEGER_RANGE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The range form samples 1:n and emits a double column/scalar result.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = randsample(integer_population,k,___)",
        inputs: &RANDSAMPLE_INTEGER_POPULATION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Sampling copies authoritative values by index and restores resident population output through the exact owner.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = randsample(...,integer_k,integer_replacement,integer_w)",
        inputs: &RANDSAMPLE_INTEGER_CONTROL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Counts and flags remain structural; weights require exact representability before floating probability normalization.",
    },
];

pub const RANDSAMPLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RANDSAMPLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        BuiltinErrorDescriptor {
            code: "RM.randsample.INVALID_ARGUMENT",
            identifier: Some("RunMat:randsample:InvalidArgument"),
            when: ERROR_INVALID_ARGUMENT.when,
            message: ERROR_INVALID_ARGUMENT.message,
        },
        BuiltinErrorDescriptor {
            code: "RM.randsample.INTERNAL",
            identifier: Some("RunMat:randsample:Internal"),
            when: ERROR_INTERNAL.when,
            message: ERROR_INTERNAL.message,
        },
    ],
};
