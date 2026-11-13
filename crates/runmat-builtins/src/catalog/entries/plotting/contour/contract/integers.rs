use super::controls::{
    INTEGER_N_NOTES, INTEGER_V_NOTES, INTEGER_X_NOTES, INTEGER_Y_NOTES, INTEGER_Z_NOTES,
};
use crate::*;

pub(super) const fn documented_integer_input(
    name: &'static str,
    notes: &'static str,
) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes,
    }
}

pub(super) const INTEGER_Z_INPUT: [BuiltinIntegerInputCapability; 1] =
    [documented_integer_input("Z", INTEGER_Z_NOTES)];

pub(super) const INTEGER_Z_COUNT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    documented_integer_input("Z", INTEGER_Z_NOTES),
    documented_integer_input("N", INTEGER_N_NOTES),
];

pub(super) const INTEGER_Z_LEVELS_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    documented_integer_input("Z", INTEGER_Z_NOTES),
    documented_integer_input("V", INTEGER_V_NOTES),
];

pub(super) const INTEGER_X_Y_Z_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    documented_integer_input("X", INTEGER_X_NOTES),
    documented_integer_input("Y", INTEGER_Y_NOTES),
    documented_integer_input("Z", INTEGER_Z_NOTES),
];

pub(super) const INTEGER_X_Y_Z_COUNT_INPUTS: [BuiltinIntegerInputCapability; 4] = [
    documented_integer_input("X", INTEGER_X_NOTES),
    documented_integer_input("Y", INTEGER_Y_NOTES),
    documented_integer_input("Z", INTEGER_Z_NOTES),
    documented_integer_input("N", INTEGER_N_NOTES),
];

pub(super) const INTEGER_X_Y_Z_LEVELS_INPUTS: [BuiltinIntegerInputCapability; 4] = [
    documented_integer_input("X", INTEGER_X_NOTES),
    documented_integer_input("Y", INTEGER_Y_NOTES),
    documented_integer_input("Z", INTEGER_Z_NOTES),
    documented_integer_input("V", INTEGER_V_NOTES),
];

pub(super) const INTEGER_LEVEL_LIST_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "LevelList",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented property accepts every built-in integer class as an exact monotonically increasing level vector.",
    }];

pub(super) const INTEGER_LEVEL_STEP_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "LevelStep",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented property accepts every built-in integer class as a positive scalar spacing.",
    }];

pub(super) const INTEGER_LINE_WIDTH_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "LineWidth",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A positive scalar integer line width crosses the explicit graphics-property boundary after exact scalar validation.",
    }];

pub(super) const INTEGER_LINE_COLOR_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "LineColor",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat mode admits an integer RGB triplet only when every exact component is 0 or 1; the public color contract does not explicitly advertise integer RGB classes.",
    }];

pub const CONTOUR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 10] = [
    BuiltinIntegerCapabilityDescriptor { form: "h = contour(integer_Z)", inputs: &INTEGER_Z_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Authoritative integer heights are validated exactly and then cross the client contour-geometry boundary; resident integer grids gather through their owning provider rather than entering the floating WGPU path." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contour(integer_Z, integer_N)", inputs: &INTEGER_Z_COUNT_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Z follows the floating contour boundary while N remains exact through positive platform/allocation validation; the output is an opaque contour handle." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contour(integer_Z, integer_V)", inputs: &INTEGER_Z_LEVELS_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Exact integer level order is checked before levels and heights cross the client graphics boundary; resident integer inputs gather from the owning provider." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contour(integer_X, integer_Y, integer_Z)", inputs: &INTEGER_X_Y_Z_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Coordinate shape, meshgrid identity, and monotonicity are established from authoritative integer storage before X, Y, and Z enter client contour geometry." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contour(integer_X, integer_Y, integer_Z, integer_N)", inputs: &INTEGER_X_Y_Z_COUNT_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Coordinates and heights cross the graphics boundary only after exact validation; N is a bounded structural count." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contour(integer_X, integer_Y, integer_Z, integer_V)", inputs: &INTEGER_X_Y_Z_LEVELS_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Coordinate and level ordering are exact before the explicit client graphics conversion; all resident integer inputs gather." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contour(..., 'LevelList', integer_levels)", inputs: &INTEGER_LEVEL_LIST_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "LevelList uses authoritative integer order and then the same floating contour-level boundary as positional V." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contour(..., 'LevelStep', integer_step)", inputs: &INTEGER_LEVEL_STEP_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "LevelStep is read as one exact positive scalar and must remain finite at the f32 graphics-property boundary." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contour(..., 'LineWidth', integer_width)", inputs: &INTEGER_LINE_WIDTH_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "LineWidth is read as one exact positive scalar and must remain finite at the f32 graphics-property boundary." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contour(..., 'LineColor', integer_rgb)", inputs: &INTEGER_LINE_COLOR_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The RunMat-only compatibility gate precedes provider gather; exact 0/1 component validation follows owning-provider gather but precedes f32 color conversion." },
];
