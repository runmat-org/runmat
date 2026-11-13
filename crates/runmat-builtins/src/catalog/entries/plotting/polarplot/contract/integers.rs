use crate::*;

pub(super) const POLARPLOT_INTEGER_COORDINATE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "theta, rho, or selected real table data",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public plotting surface accepts real numeric coordinate data and explicitly permits any real numeric class in selected table variables.",
    }];

pub(super) const POLARPLOT_COMPLEX_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "complex Z components",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public evidence does not establish paired complex-integer Z admission. RunMat keeps this useful form behind an explicit compatibility gate and checked renderer boundary.",
    }];

pub(super) const POLARPLOT_INTEGER_SELECTOR_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "thetavar or rhovar variable indices",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents numeric scalar/vector table-variable indices; these are exact one-based structural selectors.",
    }];

pub const POLARPLOT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = polarplot(integer_theta,integer_rho,___) or polarplot(integer_rho,___)",
        inputs: &POLARPLOT_INTEGER_COORDINATE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Coordinate storage is read authoritatively; polar-to-Cartesian conversion and rendering are explicit floating geometry boundaries.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = polarplot(complex_integer_Z,___)",
        inputs: &POLARPLOT_COMPLEX_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Componentwise integer Z is gated before conversion to polar renderer geometry.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = polarplot(tbl,integer_thetavar,integer_rhovar,___)",
        inputs: &POLARPLOT_INTEGER_SELECTOR_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Variable selection remains exact; broader table-form implementation is tracked as a general plotting surface concern.",
    },
];
