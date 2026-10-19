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
pub const GEOMETRY_ASSET_CLASS: StaticClassIdentity = StaticClassIdentity::new("geometry.Asset");
pub const MESHING_MESH_CLASS: StaticClassIdentity = StaticClassIdentity::new("meshing.Mesh");

/// Every typed field selector accepted by `fea.field`.
pub const FEA_FIELD_SELECTOR_CLASSES: [StaticClassIdentity; 4] = [
    FEA_STRUCTURAL_FIELD_CLASS,
    FEA_RESPONSE_SPECTRUM_FIELD_CLASS,
    FEA_RANDOM_RESPONSE_FIELD_CLASS,
    FEA_OUTPUT_FIELD_CLASS,
];
