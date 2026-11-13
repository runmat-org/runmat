use crate::*;

pub(super) const POLARSCATTER_INTEGER_COORDINATE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "theta and rho",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target explicitly lists all eight integer classes for theta and rho coordinate arrays and permits any numeric class in selected table variables.",
    }];

pub(super) const POLARSCATTER_INTEGER_STYLE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sz and numeric c",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target explicitly lists every integer class for marker areas and numeric color data; these values enter explicit renderer metadata boundaries.",
    }];

pub(super) const POLARSCATTER_INTEGER_SELECTOR_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "thetavar, rhovar, or ColorVariable indices",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Numeric table-variable indices are documented one-based structural selectors.",
    }];

pub const POLARSCATTER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "ps = polarscatter(integer_theta,integer_rho,___)",
        inputs: &POLARSCATTER_INTEGER_COORDINATE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Authoritative coordinates gather exactly; polar conversion and rendering are explicit floating geometry boundaries.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "ps = polarscatter(theta,rho,integer_sz,integer_c,___)",
        inputs: &POLARSCATTER_INTEGER_STYLE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Marker sizing, colormap normalization, and renderer buffers are intentional floating boundaries after typed values are parsed from authoritative storage.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "ps = polarscatter(tbl,integer_thetavar,integer_rhovar,___)",
        inputs: &POLARSCATTER_INTEGER_SELECTOR_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Variable selection remains exact; broader table-form implementation is a general plotting surface concern.",
    },
];
