use crate::*;

pub(super) const RIBBON_INTEGER_COORDINATE_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public numeric coordinate surface admits integer vectors; native source storage remains authoritative for XData.",
    },
    BuiltinIntegerInputCapability {
        name: "Y",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Integer ribbon heights retain their native storage and class as authoritative ZData.",
    },
];

pub(super) const RIBBON_INTEGER_WIDTH_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "width",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The numeric width scalar is a rendering control and is materialized only at the ribbon-geometry boundary.",
    }];

pub const RIBBON_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = ribbon(integer_Y) or ribbon(integer_X,integer_Y,...)",
        inputs: &RIBBON_INTEGER_COORDINATE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Rendering deliberately materializes floating geometry, while graphics properties reconstruct XData and ZData from duplicated exact native source storage; the returned graphics handles are double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = ribbon(X,Y,integer_width)",
        inputs: &RIBBON_INTEGER_WIDTH_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Width is validated as a positive finite scalar before floating strip geometry is generated.",
    },
];
