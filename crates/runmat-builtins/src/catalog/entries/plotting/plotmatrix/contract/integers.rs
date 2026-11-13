use crate::*;

pub(super) const PLOTMATRIX_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X/Y columns",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Matrix columns accept every integer class and are gathered by native storage before each scatter cell is constructed.",
    }];

pub const PLOTMATRIX_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "plotmatrix(integer_X [, integer_Y], ...)",
        inputs: &PLOTMATRIX_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Column extraction preserves authoritative native storage and scatter source properties; histogram binning and rendering are explicit floating computation boundaries.",
    }];
