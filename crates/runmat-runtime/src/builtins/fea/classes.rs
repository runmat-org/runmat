pub(super) const FEA_STUDY_CLASS: &str = "fea.Study";
pub(super) const FEA_SWEEP_CLASS: &str = "fea.Sweep";
pub(super) const FEA_VALIDATION_CLASS: &str = "fea.Validation";
pub(super) const FEA_PLAN_CLASS: &str = "fea.Plan";
pub(super) const FEA_RUN_RESULT_CLASS: &str = "fea.RunResult";
pub(super) const FEA_MODEL_CLASS: &str = "fea.Model";
pub(super) const FEA_MATERIAL_CLASS: &str = "fea.Material";
pub(super) const FEA_MATERIAL_ASSIGNMENT_CLASS: &str = "fea.MaterialAssignment";
pub(super) const FEA_BOUNDARY_CONDITION_CLASS: &str = "fea.BoundaryCondition";
pub(super) const FEA_STEP_CLASS: &str = "fea.Step";
pub(super) const FEA_DOMAIN_CLASS: &str = "fea.Domain";
pub(super) const FEA_INTERFACE_CLASS: &str = "fea.Interface";
pub(super) const FEA_RESULTS_CLASS: &str = "fea.Results";
pub(super) const FEA_FIELD_CLASS: &str = "fea.Field";
pub(super) const FEA_COMPARE_CLASS: &str = "fea.Compare";
pub(super) const FEA_TRENDS_CLASS: &str = "fea.Trends";
pub(super) const FEA_STUDY_CONTEXT_PROPERTY: &str = "study";

pub(super) static FEA_CLASSES_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(FEA_STUDY_CLASS);

pub(super) const STUDY_FIELDS: &[&str] = &[
    "study_id",
    "geometry",
    "create_model_intent",
    "model",
    "execution",
    "meshing_settings",
    "solver_mesh_artifact_path",
    "meshing_evidence_artifact_path",
];
pub(super) const SWEEP_FIELDS: &[&str] = &["sweep_id", "studies", "fail_fast"];
pub(super) const VALIDATION_FIELDS: &[&str] = &[
    "sweep_id",
    "valid",
    "issue_codes",
    "issues",
    "study_entries",
    "evidence_artifact_path",
];
pub(super) const PLAN_FIELDS: &[&str] = &[
    "study_id",
    "model_id",
    "procedure_step",
    "execution",
    "operation_sequence",
    "run_operation",
    "run_op_version",
    "study_fingerprint",
    "sweep_id",
    "study_count",
    "planned_count",
    "failed_count",
    "failure_entries",
    "plan_entries",
    "evidence_artifact_path",
];
pub(super) const RUN_RESULT_FIELDS: &[&str] = &[
    "study_id",
    "model_id",
    "model_template",
    "procedure_step",
    "execution",
    "identity",
    "solver_mesh_artifact_path",
    "meshing_evidence_artifact_path",
    "study_fingerprint",
    "operation_sequence",
    "run_operation",
    "run_op_version",
    "run_id",
    "run_status",
    "numerically_acceptable",
    "production_publishable",
    "solver_convergence",
    "result_quality",
    "quality_reasons",
    "production_publication_reasons",
    "provenance",
    "sweep_id",
    "study_count",
    "success_count",
    "failed_count",
    "failure_entries",
    "run_entries",
    "evidence_artifact_path",
    FEA_STUDY_CONTEXT_PROPERTY,
];
pub(super) const MODEL_FIELDS: &[&str] = &[
    "model_id",
    "geometry_id",
    "geometry_revision",
    "units",
    "frame",
    "materials",
    "material_assignments",
    "structural",
    "thermo_mechanical",
    "electro_thermal",
    "electromagnetic",
    "cfd",
    "interfaces",
    "constraints",
    "boundaries",
    "connectors",
    "signals",
    "excitations",
    "excitation_cases",
    "initial_conditions",
    "response_spectra",
    "sequence",
];
pub(super) const MATERIAL_FIELDS: &[&str] = &[
    "material_id",
    "name",
    "mechanical",
    "thermal",
    "acoustic",
    "electrical",
    "plastic",
];
pub(super) const MATERIAL_ASSIGNMENT_FIELDS: &[&str] = &[
    "region_id",
    "expected_material_id",
    "assigned_material_id",
    "confidence",
];
pub(super) const BOUNDARY_CONDITION_FIELDS: &[&str] = &["constraint", "boundary"];
pub(super) const STEP_FIELDS: &[&str] = &[
    "step_id",
    "family",
    "procedure",
    "solution_space",
    "independent_domain",
    "activations",
    "state_inputs",
    "state_outputs",
    "observations",
    "accuracy",
    "restart",
];
pub(super) const DOMAIN_FIELDS: &[&str] = &["kind", "data"];
pub(super) const INTERFACE_FIELDS: &[&str] = &[
    "interface_id",
    "primary_region_id",
    "secondary_region_id",
    "kind",
];
pub(super) const RESULTS_FIELDS: &[&str] = &[
    "identity",
    "field_descriptors",
    "fields",
    "modal_results",
    "thermal_results",
    "transient_results",
    "nonlinear_results",
    "electromagnetic_results",
    "diagnostics",
    "run_status",
    "numerically_acceptable",
    "production_publishable",
    "quality_reasons",
    "production_publication_reasons",
    "provenance",
    "summary",
    "run_id",
    FEA_STUDY_CONTEXT_PROPERTY,
];
pub(super) const FIELD_FIELDS: &[&str] = &[
    "field_id",
    "shape",
    "values",
    "unit",
    "location",
    "kind",
    "family",
    "quantity",
    "topology_id",
    "element_kind",
    "component_count",
    "element_count",
    "entity_count",
    "value_count",
    "storage",
    "run_id",
    FEA_STUDY_CONTEXT_PROPERTY,
];
pub(super) const COMPARE_FIELDS: &[&str] = &[
    "baseline_run_id",
    "candidate_run_id",
    "numerically_acceptable_changed",
    "production_publishable_changed",
    "run_status_changed",
    "quality_reason_count_delta",
    "failed_increment_delta",
    "max_iteration_delta",
    "nonlinear_spike_count_delta",
    "nonlinear_stall_count_delta",
    "solve_ms_delta",
];
pub(super) const TRENDS_FIELDS: &[&str] = &["window_size", "summaries"];
