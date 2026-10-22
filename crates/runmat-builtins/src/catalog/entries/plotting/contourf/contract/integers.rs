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

pub(super) const INTEGER_Z: [BuiltinIntegerInputCapability; 1] = [documented_integer_input(
    "Z",
    "The documented height matrix accepts every built-in integer class and must contain at least two exact distinct values.",
)];

pub(super) const INTEGER_Z_N: [BuiltinIntegerInputCapability; 2] = [
    documented_integer_input("Z", "The documented height matrix accepts every built-in integer class and is validated exactly before filled-contour geometry."),
    documented_integer_input("N", "A scalar integer selects a positive bounded contour-level count through exact structural validation."),
];

pub(super) const INTEGER_Z_V: [BuiltinIntegerInputCapability; 2] = [
    documented_integer_input("Z", "The documented height matrix accepts every built-in integer class and is validated exactly before filled-contour geometry."),
    documented_integer_input("V", "The documented integer level vector is checked for exact monotonic order before graphics conversion."),
];

pub(super) const INTEGER_X_Y_Z: [BuiltinIntegerInputCapability; 3] = [
    documented_integer_input("X", "The documented coordinate vector or matrix accepts every built-in integer class and is validated exactly for shape, orientation, and strict monotonicity."),
    documented_integer_input("Y", "The documented coordinate vector or matrix accepts every built-in integer class and is validated exactly for shape, orientation, and strict monotonicity."),
    documented_integer_input("Z", "The documented height matrix accepts every built-in integer class and is validated exactly before filled-contour geometry."),
];

pub(super) const INTEGER_X_Y_Z_N: [BuiltinIntegerInputCapability; 4] = [
    documented_integer_input("X", "The documented coordinate vector or matrix accepts every built-in integer class and is validated exactly before graphics conversion."),
    documented_integer_input("Y", "The documented coordinate vector or matrix accepts every built-in integer class and is validated exactly before graphics conversion."),
    documented_integer_input("Z", "The documented height matrix accepts every built-in integer class and is validated exactly before filled-contour geometry."),
    documented_integer_input("N", "A scalar integer selects a positive bounded contour-level count through exact structural validation."),
];

pub(super) const INTEGER_X_Y_Z_V: [BuiltinIntegerInputCapability; 4] = [
    documented_integer_input("X", "The documented coordinate vector or matrix accepts every built-in integer class and is validated exactly before graphics conversion."),
    documented_integer_input("Y", "The documented coordinate vector or matrix accepts every built-in integer class and is validated exactly before graphics conversion."),
    documented_integer_input("Z", "The documented height matrix accepts every built-in integer class and is validated exactly before filled-contour geometry."),
    documented_integer_input("V", "The documented integer level vector is checked for exact monotonic order before graphics conversion."),
];

pub(super) const INTEGER_LEVEL_LIST: [BuiltinIntegerInputCapability; 1] = [documented_integer_input(
    "LevelList",
    "The documented property accepts every built-in integer class as an exact monotonically increasing level vector.",
)];

pub(super) const INTEGER_LEVEL_STEP: [BuiltinIntegerInputCapability; 1] = [documented_integer_input(
    "LevelStep",
    "The documented property accepts every built-in integer class as a positive scalar spacing.",
)];

pub(super) const INTEGER_LINE_WIDTH: [BuiltinIntegerInputCapability; 1] = [documented_integer_input(
    "LineWidth",
    "A positive scalar integer line width crosses the explicit graphics-property boundary after scalar validation.",
)];

pub(super) const INTEGER_LINE_COLOR: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "LineColor",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes:
        "RunMat mode admits a typed-integer RGB triplet only when every exact component is 0 or 1.",
}];

pub const CONTOURF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 10] = [
    BuiltinIntegerCapabilityDescriptor { form: "h = contourf(integer_Z)", inputs: &INTEGER_Z, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Authoritative integer heights are validated exactly and then cross the client filled-contour boundary; resident integer grids gather through their owning provider instead of entering the floating WGPU path." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contourf(integer_Z, integer_N)", inputs: &INTEGER_Z_N, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Z crosses the graphics boundary after exact validation while N remains exact through positive allocation-bound validation." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contourf(integer_Z, integer_V)", inputs: &INTEGER_Z_V, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Integer level order and height distinctness are established before client graphics conversion." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contourf(integer_X, integer_Y, integer_Z)", inputs: &INTEGER_X_Y_Z, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Coordinate shape, matrix orientation, monotonicity, and height distinctness are checked from authoritative integer storage before filled-contour conversion." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contourf(integer_X, integer_Y, integer_Z, integer_N)", inputs: &INTEGER_X_Y_Z_N, computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Coordinates and heights cross the graphics boundary after exact validation; N is a bounded structural count." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contourf(integer_X, integer_Y, integer_Z, integer_V)", inputs: &INTEGER_X_Y_Z_V, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Coordinate and level order are exact before the explicit client graphics conversion; resident integer inputs gather." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contourf(..., 'LevelList', integer_levels)", inputs: &INTEGER_LEVEL_LIST, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "LevelList is ordered from authoritative integer storage and then crosses the floating contour-level boundary." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contourf(..., 'LevelStep', integer_step)", inputs: &INTEGER_LEVEL_STEP, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "LevelStep is validated as one positive scalar and then crosses the f32 graphics-property boundary." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contourf(..., 'LineWidth', integer_width)", inputs: &INTEGER_LINE_WIDTH, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "LineWidth is validated as one positive scalar and then crosses the f32 graphics-property boundary." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contourf(..., 'LineColor', integer_rgb)", inputs: &INTEGER_LINE_COLOR, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The shared RunMat-only integer RGB gate precedes owning-provider gather; exact 0/1 validation follows gather and precedes f32 conversion." },
];
