//! Documentation facts shared by every identity of this policy family.

use crate::fea_contract::policy_types::documentation::{faq, page, section, source, unit_test};
use crate::*;

pub(super) const ERRORS: BuiltinDocumentationSection = section(
    "Errors",
    &[
        "Invalid arguments raise `RunMat:fea:InvalidInput` before any object is created: an empty id, a nonfinite, complex, logical, or device-resident value, a wrong shape or sample count, an unordered time axis, or a policy object of the wrong class. A failure to encode the typed object raises `RunMat:fea:Internal`.",
    ],
);
pub(super) const USE_DISPLACEMENT_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. `fea.StructuralInitialCondition.displacement` samples it to set an initial displacement field before a structural step.",
        "The `constantMeters`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_ROTATION_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. No RunMat excitation, constraint, or initial-condition constructor consumes a rotation signal yet; the model retains it with its other signals.",
        "The `constantRadians`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_VELOCITY_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. Fluid inlets and moving walls (`fea.FluidExcitation.inletVelocity`, `.wallMotion`), acoustic normal velocity (`fea.AcousticExcitation.normalVelocity`), prescribed fluid velocity (`fea.FluidConstraint.prescribedVelocity`), and structural or fluid initial conditions consume it.",
        "The `constantMetersPerSecond`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_ACCELERATION_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. `fea.StructuralMotion.acceleration` turns a time-domain acceleration history (linear interpolation, reject extrapolation) into derivative-consistent support motion, and `fea.SpectrumConstruction.fromAccelerationHistoryExact` or `.fromAccelerationHistoryRecursive` derives a shock response spectrum from it.",
        "The `constantMetersPerSecondSquared`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_ANGULAR_VELOCITY_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. `fea.StructuralInitialCondition.angularVelocity` samples it to start a structural step with a rigid spin, for example a rotor already turning.",
        "The `constantRadiansPerSecond`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_ANGULAR_ACCELERATION_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. No RunMat excitation, constraint, or initial-condition constructor consumes an angular-acceleration signal yet; the model retains it with its other signals.",
        "The `constantRadiansPerSecondSquared`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_FORCE_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. `fea.StructuralExcitation.force(id, region, signal)` binds it to a structural boundary region; the total force is distributed over the region.",
        "The `constantNewtons`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_MOMENT_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. `fea.StructuralExcitation.moment(id, region, signal)` binds it to a structural boundary region as a resultant moment.",
        "The `constantNewtonMeters`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_TRACTION_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. `fea.StructuralExcitation.traction(id, region, signal)` applies it as a force per unit area on a structural boundary region, in global components.",
        "The `constantPascals`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_PRESSURE_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. `fea.StructuralExcitation.pressure` applies it normal to a structural boundary region; `fea.AcousticExcitation.pressure` prescribes acoustic pressure on a boundary; acoustic and fluid initial conditions sample it.",
        "The `constantPascals`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_BODY_FORCE_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. `fea.StructuralExcitation.bodyForce(id, region, signal)` applies it throughout a structural volume region, for example self-weight `rho*g`.",
        "The `constantNewtonsPerCubicMeter`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_TEMPERATURE_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. `fea.ThermalExcitation.ambientTemperature` sets the ambient temperature of a boundary, `fea.ThermalExcitation.temperatureField` imposes a temperature on a volume for coupling, and `fea.ThermalInitialCondition.temperature` samples it as an initial temperature.",
        "The `constantKelvin`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_TEMPERATURE_DIFFERENCE_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. No RunMat excitation, constraint, or initial-condition constructor consumes a temperature-difference signal yet; the model retains it with its other signals. Use `fea.TemperatureSignal` for absolute temperatures.",
        "The `constantKelvin`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_HEAT_FLUX_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. `fea.ThermalExcitation.heatFlux(id, region, signal)` applies it as an inward heat flux on a thermal boundary region.",
        "The `constantWattsPerSquareMeter`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_HEAT_GENERATION_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. `fea.ThermalExcitation.heatGeneration(id, region, signal)` applies it throughout a thermal volume region, for example Joule or reaction heating.",
        "The `constantWattsPerCubicMeter`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_ELECTRIC_POTENTIAL_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. `fea.ElectricalExcitation.terminalVoltage(id, region, signal)` drives a terminal region, and `fea.ElectricalInitialCondition.potential` samples it as an initial potential.",
        "The `constantVolts`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_ELECTRIC_CURRENT_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. `fea.ElectricalExcitation.current(id, region, signal)` drives a total current through a conducting volume region.",
        "The `constantAmperes`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_ELECTRIC_CHARGE_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. `fea.ElectricalExcitation.charge(id, region, signal)` places a total charge on a volume region for electrostatics.",
        "The `constantCoulombs`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_CURRENT_DENSITY_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. `fea.ElectricalExcitation.currentDensity(id, region, signal)` imposes it throughout a conducting volume region, for example a coil's winding current.",
        "The `constantAmperesPerSquareMeter`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_CHARGE_DENSITY_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. No RunMat excitation, constraint, or initial-condition constructor consumes a charge-density signal yet; the model retains it with its other signals. Use `fea.ElectricChargeSignal` with `fea.ElectricalExcitation.charge` for a total charge.",
        "The `constantCoulombsPerCubicMeter`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_MAGNETIC_VECTOR_POTENTIAL_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. `fea.MagneticExcitation.vectorPotential(id, region, signal)` imposes it on a magnetic volume region, and `fea.MagneticInitialCondition.vectorPotential` samples it as an initial field.",
        "The `constantWebersPerMeter`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_MASS_FLOW_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. `fea.FluidExcitation.massFlow(id, region, signal)` prescribes the total mass flow through a fluid boundary region.",
        "The `constantKilogramsPerSecond`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const USE_VOLUME_FLOW_SIGNAL: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "List the signal on the model with `fea.model(..., \"Signals\", {...})`. `fea.AcousticExcitation.volumeVelocity(id, region, signal)` prescribes the total outward volume flow of a finite acoustic boundary patch, such as a loudspeaker or vent.",
        "The `constantCubicMetersPerSecond`, `fromTimeSamples`, `fromFunction`, and `fromTimeKnots` constructors all produce the same class, so every consumer accepts any of them.",
    ],
);
pub(super) const UNITS_FAQ: BuiltinDocumentationFaq = faq(
    "Can I pass values in other units?",
    "No. The constructor fixes the canonical SI unit of the class; convert first, for example multiply kilonewtons by 1000 before calling a newton constructor. Typed quantities from `.fea` sources cross into RunMat in the same canonical units.",
);
pub(super) const INTEGER_FAQ: BuiltinDocumentationFaq = faq(
    "Can I pass integer arrays?",
    "Yes. All eight integer classes are accepted for times, values, and components; RunMat validates the exact values and converts them once to binary64 SI storage, so very wide integers can round.",
);
pub(super) const LINKS: &[BuiltinDocumentationLink] = &[
    page("Loads and signals", "/docs/simulation/loads"),
    page(
        "Initial conditions",
        "/docs/simulation/constraints/initial-conditions",
    ),
];
pub(super) const FUNCTION_LINKS: &[BuiltinDocumentationLink] = &[
    page(
        "RunMat function integration",
        "/docs/simulation/language/runmat-function-integration",
    ),
    page("Loads and signals", "/docs/simulation/loads"),
];
pub(super) const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[
    source(
        "Constant signal constructors",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/signals.rs",
    ),
    source(
        "Time-sampled signal constructors",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/signals/time_samples.rs",
    ),
];
pub(super) const FUNCTION_IMPLEMENTATION: &[BuiltinDocumentationLink] = &[
    source(
        "Callable signal constructors",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/signals/callable.rs",
    ),
    source(
        "Run-time signal materialization",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/signals/materialize.rs",
    ),
];
pub(super) const VERIFICATION: &[BuiltinEvidenceReference] = &[
    unit_test(
        "Typed constant signal and shape checks",
        "crates/runmat-runtime/src/builtins/fea/signals.rs::tests::force_signal_is_typed_portable_and_rejects_bad_shape",
    ),
    unit_test(
        "Signal class cannot carry another quantity",
        "crates/runmat-runtime/src/builtins/fea/signals.rs::tests::signal_class_cannot_lie_about_its_quantity",
    ),
    unit_test(
        "Time samples keep vector samples and policies",
        "crates/runmat-runtime/src/builtins/fea/signals.rs::tests::time_samples_preserve_vector_samples_and_closed_policies",
    ),
    unit_test(
        "Current-density time samples are vectors",
        "crates/runmat-runtime/src/builtins/fea/signals/tests/current_density.rs::current_density_time_samples_are_three_component_vectors",
    ),
    unit_test(
        "Time samples reject unordered axes",
        "crates/runmat-runtime/src/builtins/fea/signals.rs::tests::time_samples_reject_unsorted_axes_and_mismatched_values",
    ),
    unit_test(
        "Catalog contract conformance",
        "crates/runmat-builtins/src/catalog/entries/fea/policy_types/choices/signals/tests.rs::constant_signal_constructors_conform",
    ),
    unit_test(
        "Sampled signal inference",
        "crates/runmat-builtins/src/fea_contract/policy_choices/tests.rs::scalar_time_samples_check_counts_policies_and_residency",
    ),
];
pub(super) const FUNCTION_VERIFICATION: &[BuiltinEvidenceReference] = &[
    unit_test(
        "Callable freezes captures and exact arguments",
        "crates/runmat-runtime/src/builtins/fea/signals/callable.rs::tests::callable_constructor_freezes_captures_and_exact_argument_kinds",
    ),
    unit_test(
        "Workspace and nonportable callables are rejected",
        "crates/runmat-runtime/src/builtins/fea/signals/callable.rs::tests::callable_constructor_rejects_workspace_and_nonportable_captures",
    ),
    unit_test(
        "Callable signal inference",
        "crates/runmat-builtins/src/fea_contract/policy_choices/tests.rs::function_signals_check_callable_policies_and_name_value_options",
    ),
];
