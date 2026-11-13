use crate::*;

const INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "u",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight real or complex integer classes cross the explicit floating convolution boundary before multiplication and accumulation.",
    },
    BuiltinIntegerInputCapability {
        name: "v",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Mixed integer classes are independently converted to the selected floating output domain.",
    },
];

pub const CONV_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "w = conv(integer_u, integer_v, shape?)",
        inputs: &INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The output is single when either numeric input is single and double otherwise; integer and logical inputs never produce integer output. Full orientation follows the both-columns rule; same and valid follow u. Resident integer inputs gather exactly before floating convolution, and eligible results return to the owning provider.",
    }];
