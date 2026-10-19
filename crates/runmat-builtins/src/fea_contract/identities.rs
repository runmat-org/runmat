//! Single typed class identities shared by FEA runtime values and catalog
//! inference. Public spellings are declared here once, not reconstituted by
//! identity-name dispatch.

use runmat_types::StaticClassIdentity;

pub const FEA_STUDY_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Study");
pub const FEA_SWEEP_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Sweep");
pub const FEA_VALIDATION_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Validation");
pub const FEA_PLAN_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Plan");
pub const FEA_RUN_RESULT_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.RunResult");
pub const FEA_MODEL_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Model");
pub const FEA_MATERIAL_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Material");
pub const FEA_MATERIAL_ASSIGNMENT_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.MaterialAssignment");
pub const FEA_BOUNDARY_CONDITION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.BoundaryCondition");
pub const FEA_LOAD_CASE_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.LoadCase");
pub const FEA_STEP_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Step");
pub const FEA_DOMAIN_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Domain");
pub const FEA_INTERFACE_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Interface");
pub const FEA_RUN_OPTIONS_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.RunOptions");
pub const FEA_RESULTS_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Results");
pub const FEA_FIELD_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Field");
pub const FEA_COMPARE_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Compare");
pub const FEA_TRENDS_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Trends");
