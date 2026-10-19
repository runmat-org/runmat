use crate::*;
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host logical scalar type match.",
}];
const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value inspected through class metadata without payload gathering.",
    },
    BuiltinParamDescriptor {
        name: "typename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Exact underlying class name as string scalar or character row.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isUnderlyingType(X, typename)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const IS_UNDERLYING_TYPE_ERROR_PROVIDER_PAYLOAD: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.IS_UNDERLYING_TYPE.PROVIDER_PAYLOAD_MISMATCH",
        identifier: Some("RunMat:isUnderlyingType:ProviderPayloadMismatch"),
        when:
            "A resident input carries contradictory physical class metadata or no owning provider.",
        message: "isUnderlyingType: resident input has contradictory physical class metadata",
    };
pub const IS_UNDERLYING_TYPE_ERROR_TYPE_NAME_INVALID: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.IS_UNDERLYING_TYPE.TYPE_NAME_INVALID",
        identifier: Some("RunMat:isUnderlyingType:TypeNameInvalid"),
        when: "Second argument is not a string scalar or row character vector.",
        message: "isUnderlyingType: TYPENAME must be a string scalar or character vector",
    };
pub const IS_UNDERLYING_TYPE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        IS_UNDERLYING_TYPE_ERROR_TYPE_NAME_INVALID,
        IS_UNDERLYING_TYPE_ERROR_PROVIDER_PAYLOAD,
    ],
};
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "X", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Host and resident integer storage preserves its exact signedness and width at this metadata-only query." }];
pub const IS_UNDERLYING_TYPE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "tf = isUnderlyingType(integer_X, typename)", inputs: &INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Logical, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Returns a host logical scalar from coherent exact dtype metadata without provider download." }];
