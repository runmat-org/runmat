pub(in crate::builtins::fea) use runmat_builtins::fea_contract::identities::*;
pub(in crate::builtins::fea) const FEA_STUDY_SPEC_JSON_PROPERTY: &str =
    "__runmat_fea_study_spec_json";
pub(in crate::builtins::fea) const FEA_SWEEP_SPEC_JSON_PROPERTY: &str =
    "__runmat_fea_sweep_spec_json";
pub(in crate::builtins::fea) const FEA_PAYLOAD_JSON_PROPERTY: &str = "__runmat_fea_payload_json";
pub(in crate::builtins::fea) const FEA_STUDY_CONTEXT_JSON_PROPERTY: &str =
    "__runmat_fea_study_context_json";
pub(in crate::builtins::fea) const FEA_RUN_ID_CONTEXT_PROPERTY: &str = "__runmat_fea_run_id";

pub(in crate::builtins::fea) const LOAD_NAME: &str = "fea.load";
pub(in crate::builtins::fea) const STUDY_NAME: &str = "fea.study";
pub(in crate::builtins::fea) const AUTHOR_STUDY_NAME: &str = "fea.authorStudy";
pub(in crate::builtins::fea) const SWEEP_NAME: &str = "fea.sweep";
pub(in crate::builtins::fea) const MODEL_NAME: &str = "fea.model";
pub(in crate::builtins::fea) const MATERIAL_NAME: &str = "fea.material";
pub(in crate::builtins::fea) const MATERIAL_ASSIGNMENT_NAME: &str = "fea.materialAssignment";
pub(in crate::builtins::fea) const BOUNDARY_CONDITION_NAME: &str = "fea.boundaryCondition";
pub(in crate::builtins::fea) const LOAD_CASE_NAME: &str = "fea.loadCase";
pub(in crate::builtins::fea) const STEP_NAME: &str = "fea.step";
pub(in crate::builtins::fea) const DOMAIN_NAME: &str = "fea.domain";
pub(in crate::builtins::fea) const INTERFACE_NAME: &str = "fea.interface";
pub(in crate::builtins::fea) const RUN_OPTIONS_NAME: &str = "fea.runOptions";
pub(in crate::builtins::fea) const VALIDATE_NAME: &str = "fea.validate";
pub(in crate::builtins::fea) const PLAN_NAME: &str = "fea.plan";
pub(in crate::builtins::fea) const RUN_NAME: &str = "fea.run";
pub(in crate::builtins::fea) const RESULTS_NAME: &str = "fea.results";
pub(in crate::builtins::fea) const FIELD_NAME: &str = "fea.field";
pub(in crate::builtins::fea) const PLOT_NAME: &str = "fea.plot";
pub(in crate::builtins::fea) const COMPARE_NAME: &str = "fea.compare";
pub(in crate::builtins::fea) const TRENDS_NAME: &str = "fea.trends";
