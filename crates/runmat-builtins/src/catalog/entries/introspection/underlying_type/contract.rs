use crate::*;
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "typename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host string scalar class name.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value inspected through class metadata without payload gathering.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "typename = underlyingType(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const UNDERLYING_TYPE_ERROR_PROVIDER_PAYLOAD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNDERLYING_TYPE.PROVIDER_PAYLOAD_MISMATCH",
    identifier: Some("RunMat:underlyingType:ProviderPayloadMismatch"),
    when: "A resident input carries contradictory physical class metadata or no owning provider.",
    message: "underlyingType: resident input has contradictory physical class metadata",
};

pub const UNDERLYING_TYPE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[UNDERLYING_TYPE_ERROR_PROVIDER_PAYLOAD],
};
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "X", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Host and resident integer storage preserves its exact signedness and width at this metadata-only query." }];
pub const UNDERLYING_TYPE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "typename = underlyingType(integer_X)", inputs: &INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Returns a host string scalar from coherent exact dtype metadata without provider download." }];
