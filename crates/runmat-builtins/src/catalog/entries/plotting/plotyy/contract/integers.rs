use crate::*;

pub(super) const PLOTYY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X1/Y1/X2/Y2",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Both coordinate pairs accept all eight integer classes and retain native source properties on their line objects.",
    }];

pub const PLOTYY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[ax,h1,h2] = plotyy(integer_X1, integer_Y1, integer_X2, integer_Y2, ...)",
        inputs: &PLOTYY_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each delegated line retains exact native source and persisted property data; dual-axis display and rendering are explicit floating boundaries.",
    }];
