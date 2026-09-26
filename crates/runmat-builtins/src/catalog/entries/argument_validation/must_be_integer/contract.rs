use crate::*;
pub const MUST_BE_INTEGER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "mustBeInteger(integer_A)", inputs: &super::super::support::VALIDATOR_INTEGER_VALUE, computation_domain: BuiltinIntegerComputationDomain::Predicate, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving, notes: "Every native integer class is integral by construction; this validator checks value integrality rather than integer storage class." }];
pub const MUST_BE_INTEGER_RESIDENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "mustBeInteger.resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "explicit gpuArray validation is not documented for mustBeInteger",
        error_identifier: Some("RunMat:compatibility:mustBeIntegerResidentInput"),
    };
pub const MUST_BE_INTEGER_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [MUST_BE_INTEGER_RESIDENT_EXTENSION];
pub const MUST_BE_INTEGER_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_INTEGER.OUTPUTS",
    identifier: Some("RunMat:mustBeInteger:TooManyOutputs"),
    when: "An output is explicitly requested.",
    message: "mustBeInteger: too many output arguments",
};
pub const MUST_BE_INTEGER_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_INTEGER.INVALID",
    identifier: Some("RunMat:mustBeInteger:InvalidArgument"),
    when: "Input count, class, shape or control admission fails.",
    message: "mustBeInteger: invalid argument",
};
pub const MUST_BE_INTEGER_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_INTEGER.FAILED",
    identifier: Some("RunMat:mustBeInteger:ValidationFailed"),
    when: "The supplied value does not satisfy this validator.",
    message: "mustBeInteger: value does not satisfy validator",
};
pub const MUST_BE_INTEGER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeInteger(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_INTEGER_ERROR_OUTPUTS,
        MUST_BE_INTEGER_ERROR_INVALID,
        MUST_BE_INTEGER_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_TEXT_INPUT,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD_PAYLOAD,
    ],
};
