use crate::*;
pub const MUST_BE_NON_NAN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "mustBeNonNan(integer_A)",
        inputs: &super::super::support::VALIDATOR_INTEGER_VALUE,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes:
            "Native integer storage cannot contain NaN and passes from metadata without conversion.",
    },
];
pub const MUST_BE_NON_NAN_RESIDENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "mustBeNonNan.resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "explicit gpuArray validation is not documented for mustBeNonNan",
        error_identifier: Some("RunMat:compatibility:mustBeNonNanResidentInput"),
    };
pub const MUST_BE_NON_NAN_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [MUST_BE_NON_NAN_RESIDENT_EXTENSION];
pub const MUST_BE_NON_NAN_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NON_NAN.OUTPUTS",
    identifier: Some("RunMat:mustBeNonNan:TooManyOutputs"),
    when: "An output is explicitly requested.",
    message: "mustBeNonNan: too many output arguments",
};
pub const MUST_BE_NON_NAN_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NON_NAN.INVALID",
    identifier: Some("RunMat:mustBeNonNan:InvalidArgument"),
    when: "Input count, class, shape or control admission fails.",
    message: "mustBeNonNan: invalid argument",
};
pub const MUST_BE_NON_NAN_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NON_NAN.FAILED",
    identifier: Some("RunMat:mustBeNonNan:ValidationFailed"),
    when: "The supplied value does not satisfy this validator.",
    message: "mustBeNonNan: value does not satisfy validator",
};
pub const MUST_BE_NON_NAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeNonNan(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_NON_NAN_ERROR_OUTPUTS,
        MUST_BE_NON_NAN_ERROR_INVALID,
        MUST_BE_NON_NAN_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_TEXT_INPUT,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD_PAYLOAD,
    ],
};
