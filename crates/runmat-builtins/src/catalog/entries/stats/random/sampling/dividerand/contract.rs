//! dividerand declarations; shared sampling parameters remain family-owned.
use super::super::contract::{
    ERROR_INTERNAL, ERROR_INVALID_ARGUMENT, INPUTS_DIVIDERAND_Q, INPUTS_DIVIDERAND_RATIOS,
    OUTPUT_DIVIDERAND,
};
use crate::*;

pub const DIVIDERAND_RESIDENT_ARGUMENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "dividerand-resident-argument",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "resident dividerand arguments are gathered as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DividerandResidentArgumentExtension"),
    };

pub const DIVIDERAND_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [DIVIDERAND_RESIDENT_ARGUMENT_EXTENSION];

pub const DIVIDERAND_INTEGER_Q_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Q",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Q is a nonnegative scalar target count and typed integer scalars are decoded from authoritative storage.",
    }];

pub const DIVIDERAND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[trainInd,valInd,testInd] = dividerand(integer_Q,___)",
        inputs: &DIVIDERAND_INTEGER_Q_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Q must fit usize and RunMat's bounded allocation policy; all returned index vectors are double row vectors.",
    }];

pub const DIVIDERAND_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[trainInd, valInd, testInd] = dividerand(Q)",
        inputs: &INPUTS_DIVIDERAND_Q,
        outputs: &OUTPUT_DIVIDERAND,
    },
    BuiltinSignatureDescriptor {
        label: "[trainInd, valInd, testInd] = dividerand(Q, trainRatio, valRatio, testRatio)",
        inputs: &INPUTS_DIVIDERAND_RATIOS,
        outputs: &OUTPUT_DIVIDERAND,
    },
];
pub const DIVIDERAND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DIVIDERAND_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        BuiltinErrorDescriptor {
            code: "RM.dividerand.INVALID_ARGUMENT",
            identifier: Some("RunMat:dividerand:InvalidArgument"),
            when: ERROR_INVALID_ARGUMENT.when,
            message: ERROR_INVALID_ARGUMENT.message,
        },
        BuiltinErrorDescriptor {
            code: "RM.dividerand.INTERNAL",
            identifier: Some("RunMat:dividerand:Internal"),
            when: ERROR_INTERNAL.when,
            message: ERROR_INTERNAL.message,
        },
    ],
};
