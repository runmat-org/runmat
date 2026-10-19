use crate::*;
pub const MUST_BE_NONZERO_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "mustBeNonzero(integer_A)", inputs: &super::super::support::VALIDATOR_INTEGER_VALUE, computation_domain: BuiltinIntegerComputationDomain::Predicate, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving, notes: "All integer elements compare exactly against zero without a floating compatibility view." }];
pub const MUST_BE_NONZERO_RESIDENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "mustBeNonzero.resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "explicit gpuArray validation is not documented for mustBeNonzero",
        error_identifier: Some("RunMat:compatibility:mustBeNonzeroResidentInput"),
    };
pub const MUST_BE_NONZERO_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [MUST_BE_NONZERO_RESIDENT_EXTENSION];
pub const MUST_BE_NONZERO_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NONZERO.OUTPUTS",
    identifier: Some("RunMat:mustBeNonzero:TooManyOutputs"),
    when: "An output is explicitly requested.",
    message: "mustBeNonzero: too many output arguments",
};
pub const MUST_BE_NONZERO_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NONZERO.INVALID",
    identifier: Some("RunMat:mustBeNonzero:InvalidArgument"),
    when: "Input count, class, shape or control admission fails.",
    message: "mustBeNonzero: invalid argument",
};
pub const MUST_BE_NONZERO_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NONZERO.FAILED",
    identifier: Some("RunMat:mustBeNonzero:ValidationFailed"),
    when: "The supplied value does not satisfy this validator.",
    message: "mustBeNonzero: value does not satisfy validator",
};
pub const MUST_BE_NONZERO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeNonzero(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_NONZERO_ERROR_OUTPUTS,
        MUST_BE_NONZERO_ERROR_INVALID,
        MUST_BE_NONZERO_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_TEXT_INPUT,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD_PAYLOAD,
    ],
};
