use crate::*;
pub const MUST_BE_IN_RANGE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "mustBeInRange(integer_A, integer_lower, integer_upper, flags...)", inputs: &super::super::support::VALIDATOR_INTEGER_RANGE_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Predicate, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::BroadcastCompatible, notes: "Documented same-class bounds and selectable open/closed endpoints compare exactly without binary64 materialization." }];
pub const MUST_BE_IN_RANGE_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_IN_RANGE.OUTPUTS",
    identifier: Some("RunMat:mustBeInRange:TooManyOutputs"),
    when: "An output is explicitly requested.",
    message: "mustBeInRange: too many output arguments",
};
pub const MUST_BE_IN_RANGE_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_IN_RANGE.INVALID",
    identifier: Some("RunMat:mustBeInRange:InvalidArgument"),
    when: "Input count, class, shape or control admission fails.",
    message: "mustBeInRange: invalid argument",
};
pub const MUST_BE_IN_RANGE_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_IN_RANGE.FAILED",
    identifier: Some("RunMat:mustBeInRange:ValidationFailed"),
    when: "The supplied value does not satisfy this validator.",
    message: "mustBeInRange: value does not satisfy validator",
};
pub const MUST_BE_IN_RANGE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
 signatures: &[BuiltinSignatureDescriptor {
 label: "mustBeInRange(A, lower, upper, flags...)",
 inputs: &[super::super::support::VALUE,
BuiltinParamDescriptor {
 name: "lower",
 ty: BuiltinParamType::Any,
 arity: BuiltinParamArity::Required,
 default: None,
 description: "Comparison or membership operand; exact storage is preserved." },
BuiltinParamDescriptor {
 name: "upper",
 ty: BuiltinParamType::Any,
 arity: BuiltinParamArity::Required,
 default: None,
 description: "Comparison or membership operand; exact storage is preserved." },
BuiltinParamDescriptor {
 name: "lowerFlag",
 ty: BuiltinParamType::Any,
 arity: BuiltinParamArity::Optional,
 default: Some("inclusive"),
 description: "Existing text parser accepts inclusive/exclusive endpoints and one-flag open-left/open-right aliases." },
BuiltinParamDescriptor {
 name: "upperFlag",
 ty: BuiltinParamType::Any,
 arity: BuiltinParamArity::Optional,
 default: Some("inclusive"),
 description: "Existing text parser accepts inclusive/exclusive endpoints and one-flag open-left/open-right aliases." }],
 outputs: &[] }],
 output_mode: BuiltinOutputMode::Fixed,
 completion_policy: BuiltinCompletionPolicy::Public,
 errors: &[MUST_BE_IN_RANGE_ERROR_OUTPUTS,
 MUST_BE_IN_RANGE_ERROR_INVALID,
 MUST_BE_IN_RANGE_ERROR_FAILED,
 super::super::support::VALIDATION_ERROR_TEXT_INPUT,
 super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
 super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,

 super::super::support::VALIDATION_ERROR_DOWNLOAD,
 super::super::support::VALIDATION_ERROR_DOWNLOAD_PAYLOAD] };
