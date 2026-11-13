//! Single typed class identities shared by FEA runtime values and catalog
//! inference. Public spellings are declared here once, not reconstituted by
//! identity-name dispatch.

use runmat_types::StaticClassIdentity;

pub const FEA_STUDY_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Study");
pub const FEA_SWEEP_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Sweep");
pub const FEA_VALIDATION_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Validation");
pub const FEA_SWEEP_VALIDATION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.SweepValidation");
pub const FEA_PLAN_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Plan");
pub const FEA_SWEEP_PLAN_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.SweepPlan");
pub const FEA_RUN_RESULT_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.RunResult");
pub const FEA_STEP_RUN_RESULT_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StepRunResult");
pub const FEA_SWEEP_RUN_RESULT_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.SweepRunResult");
pub const FEA_MODEL_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Model");
pub const FEA_MATERIAL_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Material");
pub const FEA_MATERIAL_ASSIGNMENT_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.MaterialAssignment");
pub const FEA_PROPERTY_REGION_TARGET_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.PropertyRegionTarget");
pub const FEA_EXECUTION_PLAN_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ExecutionPlan");
pub const FEA_RESULTS_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Results");
pub const FEA_FIELD_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Field");
pub const FEA_FIELD_PAGE_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.FieldPage");
pub const FEA_COMPARE_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Compare");
pub const FEA_TRENDS_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Trends");
pub const FEA_STRUCTURAL_FIELD_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StructuralField");
pub const FEA_RESPONSE_SPECTRUM_FIELD_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ResponseSpectrumField");
pub const FEA_RANDOM_RESPONSE_FIELD_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.RandomResponseField");
pub const FEA_OUTPUT_FIELD_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.OutputField");
pub const FEA_FREQUENCY_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Frequency");
pub const FEA_TIME_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Time");
pub const GEOMETRY_ASSET_CLASS: StaticClassIdentity = StaticClassIdentity::new("geometry.Asset");
pub const MESHING_MESH_CLASS: StaticClassIdentity = StaticClassIdentity::new("meshing.Mesh");

/// Every typed field selector accepted by `fea.field`.
pub const FEA_FIELD_SELECTOR_CLASSES: [StaticClassIdentity; 4] = [
    FEA_STRUCTURAL_FIELD_CLASS,
    FEA_RESPONSE_SPECTRUM_FIELD_CLASS,
    FEA_RANDOM_RESPONSE_FIELD_CLASS,
    FEA_OUTPUT_FIELD_CLASS,
];

/// Dimensioned value classes (`fea.<Quantity>.<unit>` constructors).
pub const FEA_DISPLACEMENT_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.Displacement");
pub const FEA_ROTATION_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Rotation");
pub const FEA_TEMPERATURE_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Temperature");
pub const FEA_TEMPERATURE_DIFFERENCE_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.TemperatureDifference");
pub const FEA_PRESSURE_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Pressure");
pub const FEA_VELOCITY_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Velocity");
pub const FEA_HEAT_FLUX_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.HeatFlux");
pub const FEA_HEAT_TRANSFER_COEFFICIENT_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.HeatTransferCoefficient");
pub const FEA_SPECIFIC_ACOUSTIC_IMPEDANCE_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.SpecificAcousticImpedance");
pub const FEA_ELECTRIC_POTENTIAL_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ElectricPotential");
pub const FEA_ELECTRIC_CURRENT_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ElectricCurrent");
pub const FEA_MAGNETIC_VECTOR_POTENTIAL_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.MagneticVectorPotential");
pub const FEA_LENGTH_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Length");
pub const FEA_POSITION_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Position");
pub const FEA_TRANSLATION_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Translation");
pub const FEA_COMPLEX_PHASE_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ComplexPhase");
pub const FEA_THERMAL_CONTACT_RESISTANCE_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ThermalContactResistance");
pub const FEA_FRICTION_COEFFICIENT_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.FrictionCoefficient");
pub const FEA_MASS_DENSITY_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.MassDensity");
pub const FEA_DYNAMIC_VISCOSITY_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.DynamicViscosity");
pub const FEA_TURBULENCE_INTENSITY_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.TurbulenceIntensity");
pub const FEA_RAYLEIGH_MASS_COEFFICIENT_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.RayleighMassCoefficient");
pub const FEA_RAYLEIGH_STIFFNESS_COEFFICIENT_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.RayleighStiffnessCoefficient");
