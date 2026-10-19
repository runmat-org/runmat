use crate::*;
pub const MUST_BE_FINITE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "mustBeFinite(integer_A)", inputs: &super::super::support::VALIDATOR_INTEGER_VALUE, computation_domain: BuiltinIntegerComputationDomain::Predicate, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving, notes: "Every native integer value is finite; compatibility is decided from exact class metadata without floating conversion." }];
pub const MUST_BE_FINITE_RESIDENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "mustBeFinite.resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "explicit gpuArray validation is not documented for mustBeFinite",
        error_identifier: Some("RunMat:compatibility:mustBeFiniteResidentInput"),
    };
pub const MUST_BE_FINITE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [MUST_BE_FINITE_RESIDENT_EXTENSION];
pub const MUST_BE_FINITE_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_FINITE.OUTPUTS",
    identifier: Some("RunMat:mustBeFinite:TooManyOutputs"),
    when: "An output is explicitly requested.",
    message: "mustBeFinite: too many output arguments",
};
pub const MUST_BE_FINITE_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_FINITE.INVALID",
    identifier: Some("RunMat:mustBeFinite:InvalidArgument"),
    when: "Input count, class, shape or control admission fails.",
    message: "mustBeFinite: invalid argument",
};
pub const MUST_BE_FINITE_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_FINITE.FAILED",
    identifier: Some("RunMat:mustBeFinite:ValidationFailed"),
    when: "The supplied value does not satisfy this validator.",
    message: "mustBeFinite: value does not satisfy validator",
};
pub const MUST_BE_FINITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeFinite(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_FINITE_ERROR_OUTPUTS,
        MUST_BE_FINITE_ERROR_INVALID,
        MUST_BE_FINITE_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_TEXT_INPUT,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD_PAYLOAD,
    ],
};
