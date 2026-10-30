use runmat_builtins::fea_contract::identities;
use runmat_builtins::fea_contract::model_types::classes as model_classes;

pub(super) const FEA_STUDY_CLASS: &str = identities::FEA_STUDY_CLASS.display_name();
pub(super) const FEA_SWEEP_CLASS: &str = identities::FEA_SWEEP_CLASS.display_name();
pub(super) const FEA_VALIDATION_CLASS: &str = identities::FEA_VALIDATION_CLASS.display_name();
pub(super) const FEA_SWEEP_VALIDATION_CLASS: &str =
    identities::FEA_SWEEP_VALIDATION_CLASS.display_name();
pub(super) const FEA_PLAN_CLASS: &str = identities::FEA_PLAN_CLASS.display_name();
pub(super) const FEA_SWEEP_PLAN_CLASS: &str = identities::FEA_SWEEP_PLAN_CLASS.display_name();
pub(super) const FEA_RUN_RESULT_CLASS: &str = identities::FEA_RUN_RESULT_CLASS.display_name();
pub(super) const FEA_STEP_RUN_RESULT_CLASS: &str =
    identities::FEA_STEP_RUN_RESULT_CLASS.display_name();
pub(super) const FEA_SWEEP_RUN_RESULT_CLASS: &str =
    identities::FEA_SWEEP_RUN_RESULT_CLASS.display_name();
pub(super) const FEA_MODEL_CLASS: &str = identities::FEA_MODEL_CLASS.display_name();
pub(super) const FEA_MATERIAL_CLASS: &str = identities::FEA_MATERIAL_CLASS.display_name();
pub(super) const FEA_MATERIAL_ASSIGNMENT_CLASS: &str =
    identities::FEA_MATERIAL_ASSIGNMENT_CLASS.display_name();
pub(super) const FEA_PROPERTY_REGION_TARGET_CLASS: &str =
    identities::FEA_PROPERTY_REGION_TARGET_CLASS.display_name();
pub(super) const FEA_STRUCTURAL_CONSTRAINT_CLASS: &str =
    model_classes::FEA_STRUCTURAL_CONSTRAINT_CLASS.display_name();
pub(super) const FEA_THERMAL_CONSTRAINT_CLASS: &str =
    model_classes::FEA_THERMAL_CONSTRAINT_CLASS.display_name();
pub(super) const FEA_ELECTRICAL_CONSTRAINT_CLASS: &str =
    model_classes::FEA_ELECTRICAL_CONSTRAINT_CLASS.display_name();
pub(super) const FEA_MAGNETIC_CONSTRAINT_CLASS: &str =
    model_classes::FEA_MAGNETIC_CONSTRAINT_CLASS.display_name();
pub(super) const FEA_ACOUSTIC_CONSTRAINT_CLASS: &str =
    model_classes::FEA_ACOUSTIC_CONSTRAINT_CLASS.display_name();
pub(super) const FEA_FLUID_CONSTRAINT_CLASS: &str =
    model_classes::FEA_FLUID_CONSTRAINT_CLASS.display_name();
pub(super) const FEA_STRUCTURAL_BOUNDARY_CLASS: &str =
    model_classes::FEA_STRUCTURAL_BOUNDARY_CLASS.display_name();
pub(super) const FEA_THERMAL_BOUNDARY_CLASS: &str =
    model_classes::FEA_THERMAL_BOUNDARY_CLASS.display_name();
pub(super) const FEA_MAGNETIC_BOUNDARY_CLASS: &str =
    model_classes::FEA_MAGNETIC_BOUNDARY_CLASS.display_name();
pub(super) const FEA_ACOUSTIC_BOUNDARY_CLASS: &str =
    model_classes::FEA_ACOUSTIC_BOUNDARY_CLASS.display_name();
pub(super) const FEA_FLUID_BOUNDARY_CLASS: &str =
    model_classes::FEA_FLUID_BOUNDARY_CLASS.display_name();
pub(super) const FEA_STRUCTURAL_STEP_CLASS: &str =
    model_classes::FEA_STRUCTURAL_STEP_CLASS.display_name();
pub(super) const FEA_THERMAL_STEP_CLASS: &str =
    model_classes::FEA_THERMAL_STEP_CLASS.display_name();
pub(super) const FEA_ELECTROMAGNETIC_STEP_CLASS: &str =
    model_classes::FEA_ELECTROMAGNETIC_STEP_CLASS.display_name();
pub(super) const FEA_ACOUSTIC_STEP_CLASS: &str =
    model_classes::FEA_ACOUSTIC_STEP_CLASS.display_name();
pub(super) const FEA_FLUID_STEP_CLASS: &str = model_classes::FEA_FLUID_STEP_CLASS.display_name();
pub(super) const FEA_THERMOMECHANICAL_STEP_CLASS: &str =
    model_classes::FEA_THERMO_MECHANICAL_STEP_CLASS.display_name();
pub(super) const FEA_ELECTROTHERMAL_STEP_CLASS: &str =
    model_classes::FEA_ELECTRO_THERMAL_STEP_CLASS.display_name();
pub(super) const FEA_CONJUGATE_HEAT_TRANSFER_STEP_CLASS: &str =
    model_classes::FEA_CONJUGATE_HEAT_TRANSFER_STEP_CLASS.display_name();
pub(super) const FEA_FLUID_STRUCTURE_STEP_CLASS: &str =
    model_classes::FEA_FLUID_STRUCTURE_STEP_CLASS.display_name();
pub(super) const FEA_THERMOMECHANICAL_DOMAIN_CLASS: &str =
    model_classes::FEA_THERMO_MECHANICAL_DOMAIN_CLASS.display_name();
pub(super) const FEA_THERMAL_DOMAIN_CLASS: &str =
    model_classes::FEA_THERMAL_DOMAIN_CLASS.display_name();
pub(super) const FEA_ELECTROTHERMAL_DOMAIN_CLASS: &str =
    model_classes::FEA_ELECTRO_THERMAL_DOMAIN_CLASS.display_name();
pub(super) const FEA_ELECTROMAGNETIC_DOMAIN_CLASS: &str =
    model_classes::FEA_ELECTROMAGNETIC_DOMAIN_CLASS.display_name();
pub(super) const FEA_FLUID_DOMAIN_CLASS: &str =
    model_classes::FEA_FLUID_DOMAIN_CLASS.display_name();
pub(super) const FEA_CONTACT_INTERFACE_CLASS: &str =
    model_classes::FEA_CONTACT_INTERFACE_CLASS.display_name();
pub(super) const FEA_FLUID_STRUCTURE_INTERFACE_CLASS: &str =
    model_classes::FEA_FLUID_STRUCTURE_INTERFACE_CLASS.display_name();
pub(super) const FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_CLASS: &str =
    model_classes::FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_CLASS.display_name();
pub(super) const FEA_RESULTS_CLASS: &str = identities::FEA_RESULTS_CLASS.display_name();
pub(super) const FEA_FIELD_CLASS: &str = identities::FEA_FIELD_CLASS.display_name();
pub(super) const FEA_FIELD_PAGE_CLASS: &str = identities::FEA_FIELD_PAGE_CLASS.display_name();
pub(super) const FEA_COMPARE_CLASS: &str = identities::FEA_COMPARE_CLASS.display_name();
pub(super) const FEA_TRENDS_CLASS: &str = identities::FEA_TRENDS_CLASS.display_name();
pub(super) const FEA_STUDY_CONTEXT_PROPERTY: &str = "study";

pub(super) static FEA_CLASSES_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(StaticClassIdentity::new(FEA_STUDY_CLASS));

pub(super) const STUDY_FIELDS: &[&str] = &[
    "study_id",
    "geometry",
    "model",
    "execution",
    "meshing_settings",
    "immutable_values",
    "solver_mesh_artifact_path",
    "meshing_evidence_artifact_path",
    super::signals::bindings::FROZEN_GENERATORS_PROPERTY,
];
pub(super) const SWEEP_FIELDS: &[&str] = &[
    "sweep_id",
    "studies",
    "fail_fast",
    super::signals::bindings::FROZEN_STUDY_GENERATORS_PROPERTY,
];
pub(super) const VALIDATION_FIELDS: &[&str] =
    &["valid", "issue_codes", "issues", "evidence_artifact_path"];
pub(super) const SWEEP_VALIDATION_FIELDS: &[&str] = &[
    "sweep_id",
    "valid",
    "issue_codes",
    "study_entries",
    "evidence_artifact_path",
];
/// Mirrors `AnalysisStudyPlanData`; every field is always serialized.
pub(super) const PLAN_FIELDS: &[&str] = &[
    "study_id",
    "model_id",
    "sequence_id",
    "steps",
    "operation_sequence",
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
    "sequence_identity",
    "steps",
    "states",
    "solver_mesh_artifact_path",
    "meshing_evidence_artifact_path",
    "study_fingerprint",
    "operation_sequence",
    "run_status",
    "numerically_acceptable",
    "solver_convergence",
    "result_quality",
    "evidence_artifact_path",
    "result_manifest_artifact_path",
    FEA_STUDY_CONTEXT_PROPERTY,
];
pub(super) const STEP_RUN_RESULT_FIELDS: &[&str] = &[
    "ordinal",
    "procedure_step",
    "execution",
    "identity",
    "run_operation",
    "run_op_version",
    "run_id",
    "run_status",
    "numerically_acceptable",
    "solver_convergence",
    "result_quality",
    "quality_reasons",
    "provenance",
    "evidence_artifact_path",
    FEA_STUDY_CONTEXT_PROPERTY,
];
pub(super) const SWEEP_RUN_RESULT_FIELDS: &[&str] = &[
    "sweep_id",
    "identity",
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
    "random_excitations",
    "sequence",
    super::signals::bindings::FROZEN_GENERATORS_PROPERTY,
];
/// Mirrors `MaterialModel`. Absent optional capabilities are omitted when the
/// object is built, so only the identity and provenance fields are required.
pub(super) const MATERIAL_FIELDS: &[&str] = &[
    "material_id",
    "name",
    "provenance",
    "mass_density_kg_per_m3",
    "elasticity",
    "damping",
    "thermal",
    "acoustic",
    "fluid",
    "electrical",
    "plastic",
];
pub(super) const MATERIAL_REQUIRED_FIELDS: &[&str] = &["material_id", "name", "provenance"];
pub(super) const MATERIAL_ASSIGNMENT_FIELDS: &[&str] = &[
    "geometry_id",
    "geometry_revision",
    "target",
    "expected_material_id",
    "assigned_material_id",
    "confidence",
];
pub(super) const PROPERTY_REGION_TARGET_FIELDS: &[&str] =
    &["geometry_id", "geometry_revision", "target"];
pub(super) const STRUCTURAL_CONSTRAINT_FIELDS: &[&str] =
    &["geometry_id", "geometry_revision", "constraint"];
pub(super) const BOUNDARY_FIELDS: &[&str] = &["geometry_id", "geometry_revision", "boundary"];
pub(super) const STEP_FIELDS: &[&str] = &[
    "step_id",
    "family",
    "procedure",
    "solution_space",
    "settings",
    "excitation_case_selection",
    "independent_domain",
    "activations",
    "state_inputs",
    "state_outputs",
    "observations",
    "accuracy",
    "restart",
];
pub(super) const REFERENCE_TEMPERATURE_DOMAIN_FIELDS: &[&str] = &["reference_temperature"];
pub(super) const ELECTROMAGNETIC_DOMAIN_FIELDS: &[&str] = &["reference_frequency"];
pub(super) const FLUID_DOMAIN_FIELDS: &[&str] =
    &["density", "dynamic_viscosity", "turbulence_intensity"];
pub(super) const INTERFACE_FIELDS: &[&str] = &["geometry_id", "geometry_revision", "interface"];
pub(super) const RESULTS_FIELDS: &[&str] = &[
    "identity",
    "field_descriptors",
    "fields",
    "modal_results",
    "thermal_results",
    "transient_results",
    "nonlinear_results",
    "structural_frequency_results",
    "diagnostics",
    "run_status",
    "numerically_acceptable",
    "quality_reasons",
    "provenance",
    "summary",
    "run_id",
    FEA_STUDY_CONTEXT_PROPERTY,
];
pub(super) const FIELD_FIELDS: &[&str] = &[
    "field_id",
    "shape",
    "values",
    "materialized",
    "unit",
    "location",
    "kind",
    "family",
    "quantity",
    "statistic",
    "topology_id",
    "element_kind",
    "component_count",
    "element_count",
    "entity_count",
    "value_count",
    "storage",
    "sample",
    "numeric_representation",
    "run_id",
    FEA_STUDY_CONTEXT_PROPERTY,
];
pub(super) const FIELD_PAGE_FIELDS: &[&str] = &[
    "run_id",
    "field_id",
    "sample",
    "values",
    "start",
    "count",
    "total_count",
    "next_start",
    "has_more",
    "numeric_representation",
];
pub(super) const COMPARE_FIELDS: &[&str] = &[
    "baseline_run_id",
    "candidate_run_id",
    "numerically_acceptable_changed",
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
        static FIELDS: super::object_projection::ProjectionFields = std::sync::OnceLock::new();
        super::object_projection::register_schemas(
            &FIELDS,
            fea_class_schemas()
                .into_iter()
                .map(|(class_name, type_identity, fields, required, validate)| {
                    crate::execution::value_codec::ImmutableValueClassSchema {
                        class_name,
                        type_identity,
                        schema_version: 1,
                        allowed_fields: fields,
                        required_fields: required,
                        validate,
                    }
                })
                .collect(),
        );
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
                FEA_STRUCTURAL_BOUNDARY_CLASS => structural_boundary_methods(),
                FEA_THERMAL_BOUNDARY_CLASS => thermal_boundary_methods(),
                FEA_MAGNETIC_BOUNDARY_CLASS => magnetic_boundary_methods(),
                FEA_ACOUSTIC_BOUNDARY_CLASS => acoustic_boundary_methods(),
                FEA_FLUID_BOUNDARY_CLASS => fluid_boundary_methods(),
                FEA_STRUCTURAL_STEP_CLASS => structural_step_methods(),
                FEA_THERMAL_STEP_CLASS => thermal_step_methods(),
                FEA_ELECTROMAGNETIC_STEP_CLASS => electromagnetic_step_methods(),
                FEA_ACOUSTIC_STEP_CLASS => acoustic_step_methods(),
                FEA_FLUID_STEP_CLASS => fluid_step_methods(),
                FEA_THERMOMECHANICAL_STEP_CLASS => thermomechanical_step_methods(),
                FEA_ELECTROTHERMAL_STEP_CLASS => electrothermal_step_methods(),
                FEA_CONJUGATE_HEAT_TRANSFER_STEP_CLASS => conjugate_heat_transfer_step_methods(),
                FEA_FLUID_STRUCTURE_STEP_CLASS => fluid_structure_step_methods(),
                FEA_THERMAL_DOMAIN_CLASS => thermal_domain_methods(),
                FEA_THERMOMECHANICAL_DOMAIN_CLASS => thermomechanical_domain_methods(),
                FEA_ELECTROTHERMAL_DOMAIN_CLASS => electrothermal_domain_methods(),
                FEA_ELECTROMAGNETIC_DOMAIN_CLASS => electromagnetic_domain_methods(),
                FEA_FLUID_DOMAIN_CLASS => fluid_domain_methods(),
                FEA_CONTACT_INTERFACE_CLASS => contact_interface_methods(),
                FEA_FLUID_STRUCTURE_INTERFACE_CLASS => fluid_structure_interface_methods(),
                FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_CLASS => {
                    conjugate_heat_transfer_interface_methods()
                }
                FEA_RUN_RESULT_CLASS => run_result_methods(),
                FEA_STEP_RUN_RESULT_CLASS => step_run_result_methods(),
                FEA_RESULTS_CLASS => results_methods(),
                FEA_FIELD_CLASS => field_methods(),
                super::field_selectors::STRUCTURAL_FIELD_CLASS => structural_field_methods(),
                super::field_selectors::RESPONSE_SPECTRUM_FIELD_CLASS => {
                    response_spectrum_field_methods()
                }
                super::field_selectors::RANDOM_RESPONSE_FIELD_CLASS => {
                    random_response_field_methods()
                }
                super::field_selectors::OUTPUT_FIELD_CLASS => output_field_methods(),
                _ => HashMap::new(),
            };
            let properties = fields
                .iter()
                .copied()
                .chain(std::iter::once(
                    crate::builtins::common::json::typed_object_projection::PROPERTY,
                ))
                .map(|field| {
                    let get_access = if matches!(
                        field,
                        super::signals::bindings::FROZEN_GENERATORS_PROPERTY
                            | super::signals::bindings::FROZEN_STUDY_GENERATORS_PROPERTY
                            | crate::builtins::common::json::typed_object_projection::PROPERTY
                    ) {
                        MemberAccess::Private
                    } else {
                        MemberAccess::Public
                    };
                    immutable_property(field, get_access)
                })
                .collect();
            register_value_class(class_name, properties, methods);
        }
    });
}

fn validate_fea_object(object: &ObjectInstance) -> Result<(), String> {
    if object.properties().keys().any(|name| {
        name.starts_with("__runmat_")
            && name != crate::builtins::common::json::typed_object_projection::PROPERTY
    }) {
        return Err(format!(
            "immutable FEA class `{}` contains an opaque runtime property",
            object.class_name()
        ));
    }
    super::object_projection::validate(object)?;
    for field in [
        super::signals::bindings::FROZEN_GENERATORS_PROPERTY,
        super::signals::bindings::FROZEN_STUDY_GENERATORS_PROPERTY,
    ] {
        if object
            .properties()
            .get(field)
            .is_some_and(|value| !matches!(value, Value::Cell(_)))
        {
            return Err(format!(
                "immutable FEA class `{}` has malformed frozen generator data",
                object.class_name()
            ));
        }
    }
    Ok(())
}

fn validate_bound_constraint_object(object: &ObjectInstance) -> Result<(), String> {
    validate_fea_object(object)?;
    match object.properties().get("geometry_id") {
        Some(Value::String(runmat_value::StringElement::Text(value))) if !value.is_empty() => {}
        _ => {
            return Err(format!(
                "{}.geometry_id must be a non-empty string",
                object.class_name()
            ))
        }
    }
    if !matches!(
        object.properties().get("geometry_revision"),
        Some(Value::Int(IntValue::U32(_)))
    ) {
        return Err(format!(
            "{}.geometry_revision must be uint32",
            object.class_name()
        ));
    }
    if !matches!(
        object.properties().get("constraint"),
        Some(Value::Struct(_))
    ) {
        return Err(format!(
            "{}.constraint must be a typed structure",
            object.class_name()
        ));
    }
    Ok(())
}

fn validate_bound_boundary_object(object: &ObjectInstance) -> Result<(), String> {
    validate_fea_object(object)?;
    match object.properties().get("geometry_id") {
        Some(Value::String(runmat_value::StringElement::Text(value))) if !value.is_empty() => {}
        _ => {
            return Err(format!(
                "{}.geometry_id must be a non-empty string",
                object.class_name()
            ))
        }
    }
    if !matches!(
        object.properties().get("geometry_revision"),
        Some(Value::Int(IntValue::U32(_)))
    ) {
        return Err(format!(
            "{}.geometry_revision must be uint32",
            object.class_name()
        ));
    }
    if !matches!(object.properties().get("boundary"), Some(Value::Struct(_))) {
        return Err(format!(
            "{}.boundary must be a typed structure",
            object.class_name()
        ));
    }
    Ok(())
}

fn validate_bound_interface_object(object: &ObjectInstance) -> Result<(), String> {
    validate_fea_object(object)?;
    match object.properties().get("geometry_id") {
        Some(Value::String(runmat_value::StringElement::Text(value))) if !value.is_empty() => {}
        _ => {
            return Err(format!(
                "{}.geometry_id must be a non-empty string",
                object.class_name()
            ))
        }
    }
    if !matches!(
        object.properties().get("geometry_revision"),
        Some(Value::Int(IntValue::U32(_)))
    ) {
        return Err(format!(
            "{}.geometry_revision must be uint32",
            object.class_name()
        ));
    }
    if !matches!(object.properties().get("interface"), Some(Value::Struct(_))) {
        return Err(format!(
            "{}.interface must be a typed structure",
            object.class_name()
        ));
    }
    Ok(())
}

fn structural_constraint_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([
        ("fixed", "fea.StructuralConstraint.fixed"),
        ("prescribed", "fea.StructuralConstraint.prescribed"),
        (
            "prescribedMotion",
            "fea.StructuralConstraint.prescribedMotion",
        ),
        (
            "normalDisplacement",
            "fea.StructuralConstraint.normalDisplacement",
        ),
        ("symmetry", "fea.StructuralConstraint.symmetry"),
        ("antisymmetry", "fea.StructuralConstraint.antisymmetry"),
        ("rigidCoupling", "fea.StructuralConstraint.rigidCoupling"),
        (
            "distributingCoupling",
            "fea.StructuralConstraint.distributingCoupling",
        ),
        ("tie", "fea.StructuralConstraint.tie"),
        ("periodic", "fea.StructuralConstraint.periodic"),
    ])
}

fn thermal_constraint_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([
        ("prescribed", "fea.ThermalConstraint.prescribed"),
        ("tie", "fea.ThermalConstraint.tie"),
        ("periodic", "fea.ThermalConstraint.periodic"),
        ("affineJump", "fea.ThermalConstraint.affineJump"),
    ])
}

fn electrical_constraint_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([
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
        ("periodic", "fea.ElectricalConstraint.periodic"),
    ])
}

fn magnetic_constraint_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([
        (
            "prescribedVectorPotential",
            "fea.MagneticConstraint.prescribedVectorPotential",
        ),
        (
            "vectorPotentialGround",
            "fea.MagneticConstraint.vectorPotentialGround",
        ),
        ("coulombGauge", "fea.MagneticConstraint.coulombGauge"),
        ("periodic", "fea.MagneticConstraint.periodic"),
        (
            "phaseShiftedPeriodic",
            "fea.MagneticConstraint.phaseShiftedPeriodic",
        ),
    ])
}

fn acoustic_constraint_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([
        (
            "prescribedPressure",
            "fea.AcousticConstraint.prescribedPressure",
        ),
        ("periodic", "fea.AcousticConstraint.periodic"),
        (
            "phaseShiftedPeriodic",
            "fea.AcousticConstraint.phaseShiftedPeriodic",
        ),
    ])
}

fn fluid_constraint_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([
        (
            "prescribedVelocity",
            "fea.FluidConstraint.prescribedVelocity",
        ),
        ("movingWall", "fea.FluidConstraint.movingWall"),
        ("noSlipWall", "fea.FluidConstraint.noSlipWall"),
        ("slipWall", "fea.FluidConstraint.slipWall"),
        ("symmetry", "fea.FluidConstraint.symmetry"),
        ("pressureReference", "fea.FluidConstraint.pressureReference"),
        ("periodic", "fea.FluidConstraint.periodic"),
    ])
}

fn structural_boundary_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([("pressure", "fea.StructuralBoundary.pressure")])
}

fn thermal_boundary_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([
        ("heatFlux", "fea.ThermalBoundary.heatFlux"),
        ("convection", "fea.ThermalBoundary.convection"),
    ])
}

fn magnetic_boundary_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([("insulation", "fea.MagneticBoundary.insulation")])
}

fn acoustic_boundary_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([
        ("rigidWall", "fea.AcousticBoundary.rigidWall"),
        ("radiation", "fea.AcousticBoundary.radiation"),
        ("impedance", "fea.AcousticBoundary.impedance"),
    ])
}

fn fluid_boundary_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([("outletPressure", "fea.FluidBoundary.outletPressure")])
}

fn contact_interface_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([
        ("frictionless", "fea.ContactInterface.frictionless"),
        ("coulomb", "fea.ContactInterface.coulomb"),
    ])
}

fn fluid_structure_interface_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([("noSlip", "fea.FluidStructureInterface.noSlip")])
}

fn conjugate_heat_transfer_interface_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([
        ("perfect", "fea.ConjugateHeatTransferInterface.perfect"),
        (
            "conductance",
            "fea.ConjugateHeatTransferInterface.conductance",
        ),
        (
            "resistance",
            "fea.ConjugateHeatTransferInterface.resistance",
        ),
    ])
}

fn structural_step_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([
        ("static", "fea.StructuralStep.static"),
        ("modal", "fea.StructuralStep.modal"),
        ("harmonic", "fea.StructuralStep.harmonic"),
        ("transient", "fea.StructuralStep.transient"),
        ("nonlinear", "fea.StructuralStep.nonlinear"),
    ])
}

fn thermal_step_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([("transient", "fea.ThermalStep.transient")])
}

fn electromagnetic_step_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([("harmonic", "fea.ElectromagneticStep.harmonic")])
}

fn acoustic_step_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([("harmonic", "fea.AcousticStep.harmonic")])
}

fn fluid_step_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([
        ("steadyState", "fea.FluidStep.steadyState"),
        ("transient", "fea.FluidStep.transient"),
    ])
}

fn thermomechanical_step_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([("transient", "fea.ThermoMechanicalStep.transient")])
}

fn electrothermal_step_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([
        ("transient", "fea.ElectroThermalStep.transient"),
        ("nonlinear", "fea.ElectroThermalStep.nonlinear"),
    ])
}

fn conjugate_heat_transfer_step_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([
        ("steadyState", "fea.ConjugateHeatTransferStep.steadyState"),
        ("transient", "fea.ConjugateHeatTransferStep.transient"),
    ])
}

fn fluid_structure_step_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([("transient", "fea.FluidStructureStep.transient")])
}

fn thermomechanical_domain_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([(
        "atReferenceTemperature",
        "fea.ThermoMechanicalDomain.atReferenceTemperature",
    )])
}

fn thermal_domain_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([(
        "atReferenceTemperature",
        "fea.ThermalDomain.atReferenceTemperature",
    )])
}

fn electrothermal_domain_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([(
        "atReferenceTemperature",
        "fea.ElectroThermalDomain.atReferenceTemperature",
    )])
}

fn electromagnetic_domain_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([("atFrequency", "fea.ElectromagneticDomain.atFrequency")])
}

fn fluid_domain_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([("newtonian", "fea.FluidDomain.newtonian")])
}

fn structural_field_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([
        ("displacement", "fea.StructuralField.displacement"),
        ("rotation", "fea.StructuralField.rotation"),
        ("vonMises", "fea.StructuralField.vonMises"),
        ("nodalVonMises", "fea.StructuralField.nodalVonMises"),
        ("strain", "fea.StructuralField.strain"),
        ("stress", "fea.StructuralField.stress"),
        (
            "integrationPointStrain",
            "fea.StructuralField.integrationPointStrain",
        ),
        (
            "integrationPointStress",
            "fea.StructuralField.integrationPointStress",
        ),
        ("reactionForce", "fea.StructuralField.reactionForce"),
        ("reactionMoment", "fea.StructuralField.reactionMoment"),
        (
            "constraintReactionForce",
            "fea.StructuralField.constraintReactionForce",
        ),
        (
            "constraintReactionMomentAboutGlobalOrigin",
            "fea.StructuralField.constraintReactionMomentAboutGlobalOrigin",
        ),
        (
            "strainEnergyDensity",
            "fea.StructuralField.strainEnergyDensity",
        ),
        ("totalStrainEnergy", "fea.StructuralField.totalStrainEnergy"),
        ("residualNorm", "fea.StructuralField.residualNorm"),
        ("equationScale", "fea.StructuralField.equationScale"),
        ("beamAxialForce", "fea.StructuralField.beamAxialForce"),
        ("beamShearForce", "fea.StructuralField.beamShearForce"),
        ("beamTorsionMoment", "fea.StructuralField.beamTorsionMoment"),
        ("beamBendingMoment", "fea.StructuralField.beamBendingMoment"),
        ("beamBendingStress", "fea.StructuralField.beamBendingStress"),
        ("beamTorsionStress", "fea.StructuralField.beamTorsionStress"),
        (
            "shellMembraneForce",
            "fea.StructuralField.shellMembraneForce",
        ),
        (
            "shellBendingMoment",
            "fea.StructuralField.shellBendingMoment",
        ),
        (
            "shellTransverseShear",
            "fea.StructuralField.shellTransverseShear",
        ),
        ("shellVonMises", "fea.StructuralField.shellVonMises"),
    ])
}

fn output_field_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([("named", "fea.OutputField.named")])
}

fn response_spectrum_field_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([
        (
            "relativeDisplacement",
            "fea.ResponseSpectrumField.relativeDisplacement",
        ),
        ("rotation", "fea.ResponseSpectrumField.rotation"),
        (
            "relativeVelocity",
            "fea.ResponseSpectrumField.relativeVelocity",
        ),
        (
            "relativeAngularVelocity",
            "fea.ResponseSpectrumField.relativeAngularVelocity",
        ),
        (
            "relativeAcceleration",
            "fea.ResponseSpectrumField.relativeAcceleration",
        ),
        (
            "relativeAngularAcceleration",
            "fea.ResponseSpectrumField.relativeAngularAcceleration",
        ),
        (
            "absoluteAcceleration",
            "fea.ResponseSpectrumField.absoluteAcceleration",
        ),
        (
            "absoluteAngularAcceleration",
            "fea.ResponseSpectrumField.absoluteAngularAcceleration",
        ),
        ("pseudoVelocity", "fea.ResponseSpectrumField.pseudoVelocity"),
        (
            "pseudoAngularVelocity",
            "fea.ResponseSpectrumField.pseudoAngularVelocity",
        ),
        (
            "pseudoAcceleration",
            "fea.ResponseSpectrumField.pseudoAcceleration",
        ),
        (
            "pseudoAngularAcceleration",
            "fea.ResponseSpectrumField.pseudoAngularAcceleration",
        ),
        ("envelope", "fea.ResponseSpectrumField.envelope"),
    ])
}

fn random_response_field_methods() -> HashMap<MethodName, RuntimeMethod> {
    static_methods([
        ("rms", "fea.RandomResponseField.rms"),
        ("variance", "fea.RandomResponseField.variance"),
    ])
}

fn fea_class_schemas() -> [FeaClassSchema; 50] {
    [
        (
            FEA_STUDY_CLASS,
            "runmat.fea.study",
            STUDY_FIELDS,
            &["study_id", "geometry", "model", "execution"],
            validate_fea_object,
        ),
        (
            FEA_SWEEP_CLASS,
            "runmat.fea.sweep",
            SWEEP_FIELDS,
            &["sweep_id", "studies", "fail_fast"],
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
            RUN_RESULT_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_STEP_RUN_RESULT_CLASS,
            "runmat.fea.step-run-result",
            STEP_RUN_RESULT_FIELDS,
            STEP_RUN_RESULT_FIELDS,
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
            MATERIAL_REQUIRED_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_PROPERTY_REGION_TARGET_CLASS,
            "runmat.fea.property-region-target",
            PROPERTY_REGION_TARGET_FIELDS,
            PROPERTY_REGION_TARGET_FIELDS,
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
            FEA_STRUCTURAL_BOUNDARY_CLASS,
            "runmat.fea.structural-boundary",
            BOUNDARY_FIELDS,
            BOUNDARY_FIELDS,
            validate_bound_boundary_object,
        ),
        (
            FEA_THERMAL_BOUNDARY_CLASS,
            "runmat.fea.thermal-boundary",
            BOUNDARY_FIELDS,
            BOUNDARY_FIELDS,
            validate_bound_boundary_object,
        ),
        (
            FEA_MAGNETIC_BOUNDARY_CLASS,
            "runmat.fea.magnetic-boundary",
            BOUNDARY_FIELDS,
            BOUNDARY_FIELDS,
            validate_bound_boundary_object,
        ),
        (
            FEA_ACOUSTIC_BOUNDARY_CLASS,
            "runmat.fea.acoustic-boundary",
            BOUNDARY_FIELDS,
            BOUNDARY_FIELDS,
            validate_bound_boundary_object,
        ),
        (
            FEA_FLUID_BOUNDARY_CLASS,
            "runmat.fea.fluid-boundary",
            BOUNDARY_FIELDS,
            BOUNDARY_FIELDS,
            validate_bound_boundary_object,
        ),
        (
            FEA_STRUCTURAL_STEP_CLASS,
            "runmat.fea.structural-step",
            STEP_FIELDS,
            STEP_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_THERMAL_STEP_CLASS,
            "runmat.fea.thermal-step",
            STEP_FIELDS,
            STEP_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_ELECTROMAGNETIC_STEP_CLASS,
            "runmat.fea.electromagnetic-step",
            STEP_FIELDS,
            STEP_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_ACOUSTIC_STEP_CLASS,
            "runmat.fea.acoustic-step",
            STEP_FIELDS,
            STEP_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_FLUID_STEP_CLASS,
            "runmat.fea.fluid-step",
            STEP_FIELDS,
            STEP_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_THERMOMECHANICAL_STEP_CLASS,
            "runmat.fea.thermomechanical-step",
            STEP_FIELDS,
            STEP_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_ELECTROTHERMAL_STEP_CLASS,
            "runmat.fea.electrothermal-step",
            STEP_FIELDS,
            STEP_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_CONJUGATE_HEAT_TRANSFER_STEP_CLASS,
            "runmat.fea.conjugate-heat-transfer-step",
            STEP_FIELDS,
            STEP_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_FLUID_STRUCTURE_STEP_CLASS,
            "runmat.fea.fluid-structure-step",
            STEP_FIELDS,
            STEP_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_THERMAL_DOMAIN_CLASS,
            "runmat.fea.thermal-domain",
            REFERENCE_TEMPERATURE_DOMAIN_FIELDS,
            REFERENCE_TEMPERATURE_DOMAIN_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_THERMOMECHANICAL_DOMAIN_CLASS,
            "runmat.fea.thermomechanical-domain",
            REFERENCE_TEMPERATURE_DOMAIN_FIELDS,
            REFERENCE_TEMPERATURE_DOMAIN_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_ELECTROTHERMAL_DOMAIN_CLASS,
            "runmat.fea.electrothermal-domain",
            REFERENCE_TEMPERATURE_DOMAIN_FIELDS,
            REFERENCE_TEMPERATURE_DOMAIN_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_ELECTROMAGNETIC_DOMAIN_CLASS,
            "runmat.fea.electromagnetic-domain",
            ELECTROMAGNETIC_DOMAIN_FIELDS,
            ELECTROMAGNETIC_DOMAIN_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_FLUID_DOMAIN_CLASS,
            "runmat.fea.fluid-domain",
            FLUID_DOMAIN_FIELDS,
            FLUID_DOMAIN_FIELDS,
            validate_fea_object,
        ),
        (
            FEA_CONTACT_INTERFACE_CLASS,
            "runmat.fea.contact-interface",
            INTERFACE_FIELDS,
            INTERFACE_FIELDS,
            validate_bound_interface_object,
        ),
        (
            FEA_FLUID_STRUCTURE_INTERFACE_CLASS,
            "runmat.fea.fluid-structure-interface",
            INTERFACE_FIELDS,
            INTERFACE_FIELDS,
            validate_bound_interface_object,
        ),
        (
            FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_CLASS,
            "runmat.fea.conjugate-heat-transfer-interface",
            INTERFACE_FIELDS,
            INTERFACE_FIELDS,
            validate_bound_interface_object,
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
                "quality_reasons",
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
                "materialized",
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
            FEA_FIELD_PAGE_CLASS,
            "runmat.fea.field-page",
            FIELD_PAGE_FIELDS,
            FIELD_PAGE_FIELDS,
            validate_fea_object,
        ),
        (
            super::field_selectors::STRUCTURAL_FIELD_CLASS,
            "runmat.fea.structural-field",
            super::study_values::VALUE_FIELD,
            super::study_values::VALUE_FIELD,
            validate_fea_object,
        ),
        (
            super::field_selectors::RESPONSE_SPECTRUM_FIELD_CLASS,
            "runmat.fea.response-spectrum-field",
            super::study_values::VALUE_FIELD,
            super::study_values::VALUE_FIELD,
            validate_fea_object,
        ),
        (
            super::field_selectors::RANDOM_RESPONSE_FIELD_CLASS,
            "runmat.fea.random-response-field",
            super::study_values::VALUE_FIELD,
            super::study_values::VALUE_FIELD,
            validate_fea_object,
        ),
        (
            super::field_selectors::OUTPUT_FIELD_CLASS,
            "runmat.fea.output-field",
            super::study_values::VALUE_FIELD,
            super::study_values::VALUE_FIELD,
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

use crate::builtins::fea::object_projection::register_value_class;
use crate::class_registry::declaration::{immutable_property, static_methods};
use crate::class_registry::RuntimeMethod;
use runmat_types::{MemberAccess, MethodName, StaticClassIdentity};
use runmat_value::{IntValue, ObjectInstance, Value};

use super::{
    field_methods, results_methods, run_result_methods, step_run_result_methods, workflow_methods,
};
