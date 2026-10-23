//! unidrnd declarations; shared sampling parameters remain family-owned.
use super::super::contract::{
    ERROR_INTERNAL, ERROR_INVALID_ARGUMENT, INPUTS_N, INPUTS_N_SZ, OUTPUT_R,
};
use crate::*;

pub const UNIDRND_INTEGER_LIMIT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "unidrnd-integer-limit",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "unidrnd with a typed-integer upper limit is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:UnidrndIntegerLimitExtension"),
    };

pub const UNIDRND_INTEGER_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "unidrnd-integer-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "unidrnd with typed-integer size arguments is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:UnidrndIntegerSizeExtension"),
};

pub const UNIDRND_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    UNIDRND_INTEGER_LIMIT_EXTENSION,
    UNIDRND_INTEGER_SIZE_EXTENSION,
];

pub const UNIDRND_INTEGER_LIMIT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double upper limits; RunMat mode accepts typed integers only when the sampling boundary can represent them exactly as binary64.",
    }];

pub const UNIDRND_INTEGER_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sz, sz1, ...",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double size controls; RunMat mode decodes typed integer extents exactly as structural values.",
    }];

pub const UNIDRND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "r = unidrnd(integer_n, ___)",
        inputs: &UNIDRND_INTEGER_LIMIT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The upper limit is extension-gated before gather and must cross the binary64 random-sampling boundary without rounding.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "r = unidrnd(n, integer_sz)",
        inputs: &UNIDRND_INTEGER_SIZE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed extents are extension-gated before provider access and parsed exactly as output shape.",
    },
];

pub const UNIDRND_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "r = unidrnd(n)",
        inputs: &INPUTS_N,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = unidrnd(n, sz)",
        inputs: &INPUTS_N_SZ,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = unidrnd(n, sz1, sz2, ...)",
        inputs: &INPUTS_N_SZ,
        outputs: &OUTPUT_R,
    },
];

pub const UNIDRND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UNIDRND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        BuiltinErrorDescriptor {
            code: "RM.unidrnd.INVALID_ARGUMENT",
            identifier: Some("RunMat:unidrnd:InvalidArgument"),
            when: ERROR_INVALID_ARGUMENT.when,
            message: ERROR_INVALID_ARGUMENT.message,
        },
        BuiltinErrorDescriptor {
            code: "RM.unidrnd.INTERNAL",
            identifier: Some("RunMat:unidrnd:Internal"),
            when: ERROR_INTERNAL.when,
            message: ERROR_INTERNAL.message,
        },
    ],
};
