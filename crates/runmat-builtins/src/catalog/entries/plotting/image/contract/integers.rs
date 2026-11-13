use crate::*;

pub(super) const IMAGE_INTEGER_CDATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "CData",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Indexed and truecolor CData accept every built-in integer class; exact CData remains authoritative on the image object.",
}];

pub const IMAGE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "h = image(integer_CData, ...)",
        inputs: &IMAGE_INTEGER_CDATA,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Indexed values retain direct-mapping provenance, truecolor values normalize across their native class range, and renderer conversion is explicit.",
    }];
