use crate::*;

pub(super) const fn integer_input(
    name: &'static str,
    availability: BuiltinIntegerInputAvailability,
    notes: &'static str,
) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &ALL_INTEGER_CLASSES,
        availability,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes,
    }
}

pub(super) const INTEGER_DOMAIN_INPUT: [BuiltinIntegerInputCapability; 1] = [integer_input(
    "xyinterval",
    BuiltinIntegerInputAvailability::Documented,
    "The documented two- or four-element plotting interval accepts real numeric vectors; authoritative integer bounds are validated before the host graphics conversion boundary.",
)];

pub(super) const INTEGER_MESH_DENSITY_INPUT: [BuiltinIntegerInputCapability; 1] = [integer_input(
    "MeshDensity",
    BuiltinIntegerInputAvailability::Documented,
    "MeshDensity is an exact scalar count with a documented effective minimum of three and a RunMat resource ceiling of 400.",
)];

pub(super) const INTEGER_LEVEL_LIST_INPUT: [BuiltinIntegerInputCapability; 1] = [integer_input(
    "LevelList",
    BuiltinIntegerInputAvailability::Documented,
    "LevelList explicitly documents every built-in integer class; a scalar is one level and vector order is not constrained.",
)];

pub(super) const INTEGER_LEVEL_STEP_INPUT: [BuiltinIntegerInputCapability; 1] = [integer_input(
    "LevelStep",
    BuiltinIntegerInputAvailability::Documented,
    "LevelStep explicitly documents every built-in integer class; zero selects automatic levels and positive values select spacing.",
)];

pub(super) const INTEGER_LINE_WIDTH_INPUT: [BuiltinIntegerInputCapability; 1] = [integer_input(
    "LineWidth",
    BuiltinIntegerInputAvailability::Documented,
    "A positive integer line width crosses one checked f32 graphics-property boundary.",
)];

pub(super) const INTEGER_FILL_INPUT: [BuiltinIntegerInputCapability; 1] = [integer_input(
    "Fill",
    BuiltinIntegerInputAvailability::Documented,
    "The documented numeric on/off form accepts only the exact scalar values zero and one.",
)];

pub(super) const INTEGER_LINE_COLOR_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "LineColor",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat mode accepts a three-element typed-integer RGB vector containing only zero or one.",
    }];

pub(super) const INTEGER_POSITIONAL_LEVEL_INPUT: [BuiltinIntegerInputCapability; 1] = [integer_input(
    "levels",
    BuiltinIntegerInputAvailability::RunMatOnly,
    "RunMat mode accepts the contour-style positional level count or value-vector shorthand after the function handle.",
)];

pub const FCONTOUR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 8] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fcontour(f, integer_xyinterval)",
        inputs: &INTEGER_DOMAIN_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Exact bounds cross one deliberate binary64 graphics-domain boundary; the result is an opaque graphics handle.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fcontour(..., 'MeshDensity', integer_density)",
        inputs: &INTEGER_MESH_DENSITY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The exact count controls bounded host callback sampling and does not become integer output.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fcontour(..., 'LevelList', integer_levels)",
        inputs: &INTEGER_LEVEL_LIST_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Authoritative level values cross one deliberate contour-geometry conversion boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fcontour(..., 'LevelStep', integer_step)",
        inputs: &INTEGER_LEVEL_STEP_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The exact scalar step crosses one checked f32 contour-geometry boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fcontour(..., 'LineWidth', integer_width)",
        inputs: &INTEGER_LINE_WIDTH_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The exact positive scalar crosses one checked f32 renderer-property boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fcontour(..., 'Fill', integer_on_off)",
        inputs: &INTEGER_FILL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Zero disables and one enables filled contour bands; the result remains graphics state.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fcontour(..., 'LineColor', integer_rgb)",
        inputs: &INTEGER_LINE_COLOR_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "This independently gated RunMat extension validates exact RGB components before graphics conversion.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fcontour(f, integer_level_count_or_values)",
        inputs: &INTEGER_POSITIONAL_LEVEL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "This independently gated RunMat shorthand reuses contour positional-level semantics.",
    },
];
