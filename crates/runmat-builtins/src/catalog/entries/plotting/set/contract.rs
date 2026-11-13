use crate::*;

const OUTPUT_STATUS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Returns `ok` when property updates succeed.",
}];
const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Live figure, axes, plot, text, or legend handle.",
    },
    BuiltinParamDescriptor {
        name: "pairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "One or more property/value pairs.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "set(h, property, value, ...)",
        inputs: &INPUTS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "status = set(h, property, value, ...)",
        inputs: &INPUTS,
        outputs: &OUTPUT_STATUS,
    },
];

const INTEGER_PROPERTY_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "property value", classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Acceptance and conversion depend on the selected graphics property. Numeric widths, sizes, limits, positions, and counts validate authoritative integer values at their property boundary; text properties still require text.",
}];
pub const SET_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "set(h, property, integer_value, ...)",
        inputs: &INTEGER_PROPERTY_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each property owns class, range, and storage validation; the generic setter does not round integer payloads through a universal floating accessor.",
    }];

pub const SET_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SET.INVALID_ARGUMENT",
    identifier: Some("RunMat:set:InvalidArgument"),
    when: "A handle, property-name shape, property/value pair, or owner-validated value is invalid. Some legacy plot-child setters still ignore unsupported style names or values.",
    message: "set: invalid argument",
};
pub const SET_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SET.INTERNAL",
    identifier: Some("RunMat:set:Internal"),
    when: "Graphics update or source-data readback fails after admission.",
    message: "set: internal operation failed",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [SET_ERROR_INVALID_ARGUMENT, SET_ERROR_INTERNAL];
pub const SET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
