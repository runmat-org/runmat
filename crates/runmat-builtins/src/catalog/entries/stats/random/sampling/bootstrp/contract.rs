//! bootstrp declarations; shared sampling parameters remain family-owned.
use super::super::contract::{
    ERROR_INTERNAL, ERROR_INVALID_ARGUMENT, INPUTS_BOOTSTRP, OUTPUT_BOOTSTAT,
    OUTPUT_BOOTSTAT_BOOTSAM,
};
use crate::*;

pub const BOOTSTRP_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "bootstat = bootstrp(nboot, bootfun, d)",
        inputs: &INPUTS_BOOTSTRP,
        outputs: &OUTPUT_BOOTSTAT,
    },
    BuiltinSignatureDescriptor {
        label: "bootstat = bootstrp(nboot, bootfun, d1, ..., dN)",
        inputs: &INPUTS_BOOTSTRP,
        outputs: &OUTPUT_BOOTSTAT,
    },
    BuiltinSignatureDescriptor {
        label: "[bootstat, bootsam] = bootstrp(___)",
        inputs: &INPUTS_BOOTSTRP,
        outputs: &OUTPUT_BOOTSTAT_BOOTSAM,
    },
];

pub const BOOTSTRP_INTEGER_NBOOT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "bootstrp-integer-nboot",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "bootstrp with an integer nboot control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BootstrpIntegerNbootExtension"),
    };

pub const BOOTSTRP_LOGICAL_NBOOT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "bootstrp-logical-nboot",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "bootstrp with a logical nboot control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BootstrpLogicalNbootExtension"),
    };

pub const BOOTSTRP_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "bootstrp-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "bootstrp with integer sample data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BootstrpIntegerDataExtension"),
    };

pub const BOOTSTRP_INTEGER_WEIGHTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "bootstrp-integer-weights",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "bootstrp with integer observation weights is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BootstrpIntegerWeightsExtension"),
    };

pub const BOOTSTRP_LOGICAL_WEIGHTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "bootstrp-logical-weights",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "bootstrp with logical observation weights is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BootstrpLogicalWeightsExtension"),
    };

pub const BOOTSTRP_TEXT_CALLABLE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "bootstrp-text-callable",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "bootstrp with a text callback name is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BootstrpTextCallableExtension"),
    };

pub const BOOTSTRP_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bootstrp-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bootstrp with gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BootstrpGpuInputExtension"),
};

pub const BOOTSTRP_EXTENSIONS: [BuiltinExtensionDescriptor; 7] = [
    BOOTSTRP_INTEGER_NBOOT_EXTENSION,
    BOOTSTRP_LOGICAL_NBOOT_EXTENSION,
    BOOTSTRP_INTEGER_DATA_EXTENSION,
    BOOTSTRP_INTEGER_WEIGHTS_EXTENSION,
    BOOTSTRP_LOGICAL_WEIGHTS_EXTENSION,
    BOOTSTRP_TEXT_CALLABLE_EXTENSION,
    BOOTSTRP_GPU_INPUT_EXTENSION,
];

pub const BOOTSTRP_INTEGER_NBOOT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "nboot",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat accepts every integer class as an exact positive scalar bootstrap count; the public control domain is single or double.",
    }];

pub const BOOTSTRP_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "d1,...,dN",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat samples every integer data class by exact row or vector indexing; public nongrouping numeric data is single or double.",
    }];

pub const BOOTSTRP_INTEGER_WEIGHT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Weights",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat accepts every nonnegative integer weight class and converts it once at the floating multinomial-probability boundary; the public weight domain is single or double.",
    }];

pub const BOOTSTRP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "bootstat = bootstrp(nboot,bootfun,d1,...,dN) with integer nboot",
        inputs: &BOOTSTRP_INTEGER_NBOOT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only exact structural count; callback output class determines bootstat and bootsam remains double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "bootstat = bootstrp(nboot,bootfun,d1,...,dN) with integer data",
        inputs: &BOOTSTRP_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only resampling preserves authoritative integer storage into each callback; homogeneous numeric or logical callback output determines bootstat class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "bootstat = bootstrp(___,Weights=w) with integer weights",
        inputs: &BOOTSTRP_INTEGER_WEIGHT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only integer weights cross one deliberate binary64 probability boundary after exact nonnegative validation; sampled data and callback output retain their own classes.",
    },
];

pub const BOOTSTRP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BOOTSTRP_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        BuiltinErrorDescriptor {
            code: "RM.bootstrp.INVALID_ARGUMENT",
            identifier: Some("RunMat:bootstrp:InvalidArgument"),
            when: ERROR_INVALID_ARGUMENT.when,
            message: ERROR_INVALID_ARGUMENT.message,
        },
        BuiltinErrorDescriptor {
            code: "RM.bootstrp.INTERNAL",
            identifier: Some("RunMat:bootstrp:Internal"),
            when: ERROR_INTERNAL.when,
            message: ERROR_INTERNAL.message,
        },
    ],
};
