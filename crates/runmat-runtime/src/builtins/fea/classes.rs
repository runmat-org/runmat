pub(super) const FEA_STUDY_CLASS: &str = "fea.Study";
pub(super) const FEA_SWEEP_CLASS: &str = "fea.Sweep";
pub(super) const FEA_VALIDATION_CLASS: &str = "fea.Validation";
pub(super) const FEA_SWEEP_VALIDATION_CLASS: &str = "fea.SweepValidation";
pub(super) const FEA_PLAN_CLASS: &str = "fea.Plan";
pub(super) const FEA_SWEEP_PLAN_CLASS: &str = "fea.SweepPlan";
pub(super) const FEA_RUN_RESULT_CLASS: &str = "fea.RunResult";
pub(super) const FEA_SWEEP_RUN_RESULT_CLASS: &str = "fea.SweepRunResult";
pub(super) const FEA_MODEL_CLASS: &str = "fea.Model";
pub(super) const FEA_MATERIAL_CLASS: &str = "fea.Material";
pub(super) const FEA_MATERIAL_ASSIGNMENT_CLASS: &str = "fea.MaterialAssignment";
pub(super) const FEA_BOUNDARY_CONDITION_CLASS: &str = "fea.BoundaryCondition";
pub(super) const FEA_STRUCTURAL_CONSTRAINT_CLASS: &str = "fea.StructuralConstraint";
pub(super) const FEA_THERMAL_CONSTRAINT_CLASS: &str = "fea.ThermalConstraint";
pub(super) const FEA_ELECTRICAL_CONSTRAINT_CLASS: &str = "fea.ElectricalConstraint";
pub(super) const FEA_MAGNETIC_CONSTRAINT_CLASS: &str = "fea.MagneticConstraint";
pub(super) const FEA_ACOUSTIC_CONSTRAINT_CLASS: &str = "fea.AcousticConstraint";
pub(super) const FEA_FLUID_CONSTRAINT_CLASS: &str = "fea.FluidConstraint";
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
pub(super) const VALIDATION_FIELDS: &[&str] =
    &["valid", "issue_codes", "issues", "evidence_artifact_path"];
pub(super) const SWEEP_VALIDATION_FIELDS: &[&str] = &[
    "sweep_id",
    "valid",
    "issue_codes",
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
    "evidence_artifact_path",
];
pub(super) const SWEEP_PLAN_FIELDS: &[&str] = &[
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
    "evidence_artifact_path",
    FEA_STUDY_CONTEXT_PROPERTY,
];
pub(super) const SWEEP_RUN_RESULT_FIELDS: &[&str] = &[
    "sweep_id",
    "study_count",
    "success_count",
    "failed_count",
    "failure_entries",
    "run_entries",
    "evidence_artifact_path",
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
pub(super) const STRUCTURAL_CONSTRAINT_FIELDS: &[&str] =
    &["geometry_id", "geometry_revision", "constraint"];
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

type FeaClassSchema = (
    &'static str,
    &'static str,
    &'static [&'static str],
    &'static [&'static str],
    crate::execution::value_codec::ImmutableValueClassValidator,
);

pub(crate) fn ensure_fea_classes_registered() {
    static PORTABLE_SCHEMAS: Once = Once::new();
    PORTABLE_SCHEMAS.call_once(|| {
        for (class_name, type_identity, fields, required, validate) in fea_class_schemas() {
            crate::execution::value_codec::register_immutable_value_class(
                crate::execution::value_codec::ImmutableValueClassSchema {
                    class_name,
                    type_identity,
                    schema_version: 1,
                    allowed_fields: fields,
                    required_fields: required,
                    validate,
                },
            )
            .unwrap_or_else(|error| {
                panic!("failed to register immutable FEA class {class_name}: {error}")
            });
        }
    });
    FEA_CLASSES_REGISTERED.ensure(|| {
        for (class_name, _, fields, _, _) in fea_class_schemas() {
            let methods = match class_name {
                FEA_STUDY_CLASS | FEA_SWEEP_CLASS => workflow_methods(),
                FEA_STRUCTURAL_CONSTRAINT_CLASS => structural_constraint_methods(),
                FEA_THERMAL_CONSTRAINT_CLASS => thermal_constraint_methods(),
                FEA_ELECTRICAL_CONSTRAINT_CLASS => electrical_constraint_methods(),
                FEA_MAGNETIC_CONSTRAINT_CLASS => magnetic_constraint_methods(),
                FEA_ACOUSTIC_CONSTRAINT_CLASS => acoustic_constraint_methods(),
                FEA_FLUID_CONSTRAINT_CLASS => fluid_constraint_methods(),
                FEA_RUN_RESULT_CLASS => run_result_methods(),
                FEA_RESULTS_CLASS => results_methods(),
                FEA_FIELD_CLASS => field_methods(),
                _ => HashMap::new(),
            };
            crate::class_registry::register_class(crate::class_registry::RuntimeClass {
                name: class_name.to_string(),
                parent: None,
                properties: immutable_runtime_properties(fields),
                methods,
            });
        }
    });
}

fn immutable_runtime_properties(
    fields: &[&str],
) -> HashMap<String, crate::class_registry::RuntimeProperty> {
    fields
        .iter()
        .map(|name| {
            (
                (*name).to_string(),
                crate::class_registry::RuntimeProperty {
                    name: (*name).to_string(),
                    is_static: false,
                    is_constant: true,
                    is_dependent: false,
                    get_access: MemberAccess::Public,
                    set_access: MemberAccess::Private,
                    default_value: None,
                },
            )
        })
        .collect()
}

fn validate_fea_object(object: &ObjectInstance) -> Result<(), String> {
    if object
        .properties
        .keys()
        .any(|name| name.starts_with("__runmat_"))
    {
        return Err(format!(
            "immutable FEA class `{}` contains an opaque runtime property",
            object.class_name
        ));
    }
    Ok(())
}

fn validate_condition_object(object: &ObjectInstance) -> Result<(), String> {
    validate_fea_object(object)?;
    let variants = ["constraint", "boundary"]
        .into_iter()
        .filter(|name| object.properties.contains_key(*name))
        .count();
    if variants != 1 {
        return Err(
            "fea.BoundaryCondition must contain exactly one typed condition variant".into(),
        );
    }
    Ok(())
}

fn validate_bound_constraint_object(object: &ObjectInstance) -> Result<(), String> {
    validate_fea_object(object)?;
    match object.properties.get("geometry_id") {
        Some(Value::String(value)) if !value.is_empty() => {}
        _ => {
            return Err(format!(
                "{}.geometry_id must be a non-empty string",
                object.class_name
            ))
        }
    }
    if !matches!(
        object.properties.get("geometry_revision"),
        Some(Value::Int(IntValue::U32(_)))
    ) {
        return Err(format!(
            "{}.geometry_revision must be uint32",
            object.class_name
        ));
    }
    if !matches!(object.properties.get("constraint"), Some(Value::Struct(_))) {
        return Err(format!(
            "{}.constraint must be a typed structure",
            object.class_name
        ));
    }
    Ok(())
}

fn structural_constraint_methods() -> HashMap<String, crate::class_registry::RuntimeMethod> {
    static_methods(&[
        ("fixed", "fea.StructuralConstraint.fixed"),
        (
            "normalDisplacement",
            "fea.StructuralConstraint.normalDisplacement",
        ),
        ("symmetry", "fea.StructuralConstraint.symmetry"),
        ("antisymmetry", "fea.StructuralConstraint.antisymmetry"),
    ])
}

fn thermal_constraint_methods() -> HashMap<String, crate::class_registry::RuntimeMethod> {
    static_methods(&[("prescribed", "fea.ThermalConstraint.prescribed")])
}

fn electrical_constraint_methods() -> HashMap<String, crate::class_registry::RuntimeMethod> {
    static_methods(&[
        (
            "prescribedPotential",
            "fea.ElectricalConstraint.prescribedPotential",
        ),
        ("equipotential", "fea.ElectricalConstraint.equipotential"),
        (
            "floatingConductor",
            "fea.ElectricalConstraint.floatingConductor",
        ),
        (
            "integralCurrent",
            "fea.ElectricalConstraint.integralCurrent",
        ),
        (
            "potentialReference",
            "fea.ElectricalConstraint.potentialReference",
        ),
    ])
}

fn magnetic_constraint_methods() -> HashMap<String, crate::class_registry::RuntimeMethod> {
    static_methods(&[
        (
            "prescribedVectorPotential",
            "fea.MagneticConstraint.prescribedVectorPotential",
        ),
        (
            "vectorPotentialGround",
            "fea.MagneticConstraint.vectorPotentialGround",
        ),
        ("coulombGauge", "fea.MagneticConstraint.coulombGauge"),
    ])
}

fn acoustic_constraint_methods() -> HashMap<String, crate::class_registry::RuntimeMethod> {
    static_methods(&[(
        "prescribedPressure",
        "fea.AcousticConstraint.prescribedPressure",
    )])
}

fn fluid_constraint_methods() -> HashMap<String, crate::class_registry::RuntimeMethod> {
    static_methods(&[
        (
            "prescribedVelocity",
            "fea.FluidConstraint.prescribedVelocity",
        ),
        ("movingWall", "fea.FluidConstraint.movingWall"),
        ("noSlipWall", "fea.FluidConstraint.noSlipWall"),
        ("slipWall", "fea.FluidConstraint.slipWall"),
        ("symmetry", "fea.FluidConstraint.symmetry"),
        ("pressureReference", "fea.FluidConstraint.pressureReference"),
    ])
}

fn static_methods(names: &[(&str, &str)]) -> HashMap<String, crate::class_registry::RuntimeMethod> {
    names
        .iter()
        .map(|(name, function_name)| {
            (
                (*name).to_string(),
                crate::class_registry::RuntimeMethod {
                    name: (*name).to_string(),
                    is_static: true,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: (*function_name).to_string(),
                    implicit_class_argument: None,
                },
            )
        })
        .collect()
}

fn fea_class_schemas() -> [FeaClassSchema; 25] {
    [
        (
            FEA_STUDY_CLASS,
            "runmat.fea.study",
            STUDY_FIELDS,
            &["study_id", "geometry", "create_model_intent", "execution"],
            validate_fea_object,
        ),
        (
            FEA_SWEEP_CLASS,
            "runmat.fea.sweep",
            SWEEP_FIELDS,
            SWEEP_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_VALIDATION_CLASS,
            "runmat.fea.validation",
            VALIDATION_FIELDS,
            VALIDATION_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_SWEEP_VALIDATION_CLASS,
            "runmat.fea.sweep-validation",
            SWEEP_VALIDATION_FIELDS,
            SWEEP_VALIDATION_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_PLAN_CLASS,
            "runmat.fea.plan",
            PLAN_FIELDS,
            PLAN_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_SWEEP_PLAN_CLASS,
            "runmat.fea.sweep-plan",
            SWEEP_PLAN_FIELDS,
            SWEEP_PLAN_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_RUN_RESULT_CLASS,
            "runmat.fea.run-result",
            RUN_RESULT_FIELDS,
            &[
                "study_id",
                "model_id",
                "model_template",
                "procedure_step",
                "execution",
                "identity",
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
                "evidence_artifact_path",
            ],
            validate_fea_object,
        ),
        (
            FEA_SWEEP_RUN_RESULT_CLASS,
            "runmat.fea.sweep-run-result",
            SWEEP_RUN_RESULT_FIELDS,
            SWEEP_RUN_RESULT_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_MODEL_CLASS,
            "runmat.fea.model",
            MODEL_FIELDS,
            &[
                "model_id",
                "geometry_id",
                "geometry_revision",
                "units",
                "frame",
                "materials",
                "material_assignments",
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
            ],
            validate_fea_object,
        ),
        (
            FEA_MATERIAL_CLASS,
            "runmat.fea.material",
            MATERIAL_FIELDS,
            &["material_id", "name", "mechanical", "thermal"],
            validate_fea_object,
        ),
        (
            FEA_MATERIAL_ASSIGNMENT_CLASS,
            "runmat.fea.material-assignment",
            MATERIAL_ASSIGNMENT_FIELDS,
            MATERIAL_ASSIGNMENT_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_BOUNDARY_CONDITION_CLASS,
            "runmat.fea.boundary-condition",
            BOUNDARY_CONDITION_FIELDS,
            &[],
            validate_condition_object,
        ),
        (
            FEA_STRUCTURAL_CONSTRAINT_CLASS,
            "runmat.fea.structural-constraint",
            STRUCTURAL_CONSTRAINT_FIELDS,
            STRUCTURAL_CONSTRAINT_FIELDS,
            validate_bound_constraint_object,
        ),
        (
            FEA_THERMAL_CONSTRAINT_CLASS,
            "runmat.fea.thermal-constraint",
            STRUCTURAL_CONSTRAINT_FIELDS,
            STRUCTURAL_CONSTRAINT_FIELDS,
            validate_bound_constraint_object,
        ),
        (
            FEA_ELECTRICAL_CONSTRAINT_CLASS,
            "runmat.fea.electrical-constraint",
            STRUCTURAL_CONSTRAINT_FIELDS,
            STRUCTURAL_CONSTRAINT_FIELDS,
            validate_bound_constraint_object,
        ),
        (
            FEA_MAGNETIC_CONSTRAINT_CLASS,
            "runmat.fea.magnetic-constraint",
            STRUCTURAL_CONSTRAINT_FIELDS,
            STRUCTURAL_CONSTRAINT_FIELDS,
            validate_bound_constraint_object,
        ),
        (
            FEA_ACOUSTIC_CONSTRAINT_CLASS,
            "runmat.fea.acoustic-constraint",
            STRUCTURAL_CONSTRAINT_FIELDS,
            STRUCTURAL_CONSTRAINT_FIELDS,
            validate_bound_constraint_object,
        ),
        (
            FEA_FLUID_CONSTRAINT_CLASS,
            "runmat.fea.fluid-constraint",
            STRUCTURAL_CONSTRAINT_FIELDS,
            STRUCTURAL_CONSTRAINT_FIELDS,
            validate_bound_constraint_object,
        ),
        (
            FEA_STEP_CLASS,
            "runmat.fea.step",
            STEP_FIELDS,
            STEP_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_DOMAIN_CLASS,
            "runmat.fea.domain",
            DOMAIN_FIELDS,
            DOMAIN_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_INTERFACE_CLASS,
            "runmat.fea.interface",
            INTERFACE_FIELDS,
            INTERFACE_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_RESULTS_CLASS,
            "runmat.fea.results",
            RESULTS_FIELDS,
            &[
                "identity",
                "field_descriptors",
                "fields",
                "run_status",
                "numerically_acceptable",
                "production_publishable",
                "quality_reasons",
                "production_publication_reasons",
                "provenance",
                "summary",
                "run_id",
            ],
            validate_fea_object,
        ),
        (
            FEA_FIELD_CLASS,
            "runmat.fea.field",
            FIELD_FIELDS,
            &[
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
            ],
            validate_fea_object,
        ),
        (
            FEA_COMPARE_CLASS,
            "runmat.fea.compare",
            COMPARE_FIELDS,
            &[
                "baseline_run_id",
                "candidate_run_id",
                "numerically_acceptable_changed",
                "production_publishable_changed",
                "run_status_changed",
                "quality_reason_count_delta",
            ],
            validate_fea_object,
        ),
        (
            FEA_TRENDS_CLASS,
            "runmat.fea.trends",
            TRENDS_FIELDS,
            TRENDS_FIELDS,
            validate_fea_object,
        ),
    ]
}
use std::collections::HashMap;
use std::sync::Once;

use runmat_types::MemberAccess;
use runmat_value::{IntValue, ObjectInstance, Value};

use super::{field_methods, results_methods, run_result_methods, workflow_methods};
