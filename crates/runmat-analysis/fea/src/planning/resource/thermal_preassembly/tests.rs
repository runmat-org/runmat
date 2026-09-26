use super::*;
use crate::{
    execution::ThermalWorkload,
    fixtures::{fixture_model, FixtureId},
};
use runmat_analysis_core::{AnalysisSequence, Temperature, ThermalDomain};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn workload() -> ThermalWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.sequence =
        AnalysisSequence::from_steps("thermal", vec![ProcedureStep::thermal_transient("heat")]);
    ThermalWorkload {
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: crate::ComputeBackend::Cpu,
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        procedure: crate::ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
        residual_target: 1e-4,
    }
}

#[test]
fn assignment_diagnostics_are_admitted_and_checked_before_retention() {
    use runmat_analysis_core::EvidenceConfidence;
    let mut workload = workload();
    let step = workload.model.sequence.steps[0].clone();
    let base = ThermalMetadataResourceFacts::derive(&workload.model, &step).unwrap();
    let mut alternate = workload.model.materials[0].clone();
    alternate.material_id = "expected".repeat(512);
    workload.model.material_assignments[0].expected_material_id = alternate.material_id.clone();
    workload.model.material_assignments[0].confidence = EvidenceConfidence::Probable;
    workload.model.materials.push(alternate);
    let facts = ThermalMetadataResourceFacts::derive(&workload.model, &step).unwrap();
    assert!(facts.reservation_bytes().unwrap().0 > base.reservation_bytes().unwrap().0);
    let diagnostics = crate::diagnostics::builders::material_assignment_diagnostics(
        &workload.model.material_assignments,
    );
    facts.verify_assignment_diagnostics(&diagnostics).unwrap();
    assert!(base.verify_assignment_diagnostics(&diagnostics).is_err());
    let oversized_solver_diagnostic = crate::diagnostics::FeaDiagnostic {
        code: "FEA_SOLVER_METHOD".into(),
        severity: crate::diagnostics::FeaDiagnosticSeverity::Info,
        message: "x".repeat(facts.solver_diagnostics.encoded_bytes_upper_bound as usize + 1),
    };
    assert!(facts
        .verify_solver_diagnostics(&[oversized_solver_diagnostic])
        .is_err());
    let mut overflowing = facts;
    overflowing
        .solver_diagnostics
        .materialization_resident_bytes_estimate = u64::MAX;
    assert!(overflowing.reservation_bytes().is_err());
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let schedule = workload.solve_schedule().unwrap();
        let preassembly =
            ThermalPreassemblyResourceFacts::derive(&workload.model, &step, &mesh, &schedule)
                .unwrap();
        assert_eq!(
            preassembly.metadata.assignment_diagnostics,
            facts.assignment_diagnostics
        );
        let mut bounded = step.clone();
        bounded.accuracy.resources.maximum_resident_bytes =
            preassembly.reservation_bytes().unwrap().0 - 1;
        assert!(matches!(
            ThermalPreassemblyResourceFacts::derive(&workload.model, &bounded, &mesh, &schedule),
            Err(ResourcePlanningError::AccuracyCeiling { .. })
        ));
    }
}

#[test]
fn mesh_validation_is_admitted_before_projection_in_derivation_and_execution() {
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let mut workload = workload();
    // Raw field maps must be counted even when invalid. A stale seal makes validation fail
    // if either execution or projection reaches it before the resource gate.
    mesh.topology.field_topologies[0]
        .ordered_entity_ids
        .extend(10_000..110_000);
    mesh.canonical_digest = runmat_meshing_core::StableDigest::ZERO;
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = 1_000_000;
    let schedule = workload.solve_schedule().unwrap();
    assert!(matches!(
        ThermalPreassemblyResourceFacts::derive(
            &workload.model,
            &workload.model.sequence.steps[0],
            &mesh,
            &schedule
        ),
        Err(ResourcePlanningError::AccuracyCeiling { .. })
    ));
    assert!(matches!(
        workload.execute(&mesh),
        Err(crate::FeaRunError::Resource(
            ResourcePlanningError::AccuracyCeiling { .. }
        ))
    ));
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = u64::MAX;
    assert!(matches!(
        workload.execute(&mesh),
        Err(crate::FeaRunError::Resource(
            ResourcePlanningError::Invalid(_)
        ))
    ));
}

#[test]
fn producer_metadata_is_bounded_and_verified_against_actual_initial_state() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let mut workload = workload();
        let base = super::super::ThermalMetadataResourceFacts::derive(
            &workload.model,
            &workload.model.sequence.steps[0],
        )
        .unwrap();
        let long_id = "thermal_long_step".repeat(256);
        workload.step_id = runmat_analysis_core::ProcedureStepId(long_id.clone());
        workload.model.sequence.steps[0].step_id = workload.step_id.clone();
        let grown = super::super::ThermalMetadataResourceFacts::derive(
            &workload.model,
            &workload.model.sequence.steps[0],
        )
        .unwrap();
        assert!(
            grown.initial_state.encoded_bytes_upper_bound
                > base.initial_state.encoded_bytes_upper_bound
        );
        assert!(
            grown.physical_diagnostics.encoded_bytes_upper_bound
                > base.physical_diagnostics.encoded_bytes_upper_bound
        );
        let result = workload.execute(&mesh).unwrap();
        grown.verify_initial_state(&result.initial_state).unwrap();
        let mut substituted = result.initial_state.clone();
        substituted.procedure_step_id.push_str(&long_id);
        assert!(grown.verify_initial_state(&substituted).is_err());
        let mut too_many = result.initial_state.clone();
        too_many
            .applied_initial_condition_ids
            .push("undeclared".into());
        assert!(grown.verify_initial_state(&too_many).is_err());
    }
}

#[test]
fn model_materialization_is_composed_and_enforced_before_assembly() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let mut workload = workload();
        let schedule = workload.solve_schedule().unwrap();
        let base = ThermalPreassemblyResourceFacts::derive(
            &workload.model,
            &workload.model.sequence.steps[0],
            &mesh,
            &schedule,
        )
        .unwrap();
        workload.model.model_id.0 = "large_model_identifier".repeat(512);
        let grown = ThermalPreassemblyResourceFacts::derive(
            &workload.model,
            &workload.model.sequence.steps[0],
            &mesh,
            &schedule,
        )
        .unwrap();
        assert!(grown.model_materialization_bytes.0 > base.model_materialization_bytes.0);
        let required = grown.reservation_bytes().unwrap().0;
        assert_eq!(
            required,
            grown.model_materialization_bytes.0
                + grown
                    .operators
                    .numerical_reservation_bytes(&grown.history)
                    .unwrap()
                    .0
                + grown.sources.reservation_bytes().unwrap().0
                + grown.observations.reservation_bytes().unwrap().0
                + grown.realization.reservation_bytes().unwrap().0
                + grown.metadata.reservation_bytes().unwrap().0
                + grown.property_compilation_bytes.0
        );
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = required - 1;
        assert!(
            matches!(workload.execute(&mesh), Err(crate::FeaRunError::Resource(
            ResourcePlanningError::AccuracyCeiling { required: actual, maximum, .. }
        )) if actual == required && maximum == required - 1)
        );
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = required;
        let result = workload.execute(&mesh).unwrap();
        assert!(result
            .temperature_snapshots
            .iter()
            .flat_map(|field| field.as_host_f64().unwrap())
            .all(|value| (value - 293.15).abs() < 1e-9));
        let mut overflowing = grown;
        overflowing.model_materialization_bytes = ByteCount(u64::MAX);
        assert!(overflowing.reservation_bytes().is_err());
    }
}

#[test]
fn planning_ceiling_precedes_geometric_projection_at_exact_boundary() {
    use runmat_meshing_core::CanonicalMeshingContract;
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut mesh = canonical_tetrahedron_solver_mesh(order);
        // Canonical connectivity remains valid, but physical projection must reject zero volume.
        for node in &mut mesh.topology.nodes {
            node.coordinates_m = [0.0; 3];
        }
        mesh.seal_canonical_digest().unwrap();
        mesh.validate_canonical().unwrap();
        let mut workload = workload();
        let schedule = workload.solve_schedule().unwrap();
        let history = ThermalHistoryResourceFacts::derive(&mesh, &schedule).unwrap();
        let model_bytes =
            runmat_canonical_codec::serialized_materialization_resident_bytes_estimate(
                &workload.model,
                crate::execution::THERMAL_WORKLOAD_LIMITS,
            )
            .unwrap()
                * 2;
        let planning = ThermalOperatorResourceFacts::derivation_resident_bytes_estimate(&mesh)
            .unwrap()
            .0;
        let required = history.reservation_bytes().unwrap().0 + model_bytes + planning;
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = required - 1;
        assert!(
            matches!(workload.execute(&mesh), Err(crate::FeaRunError::Resource(
            ResourcePlanningError::AccuracyCeiling { required: actual, maximum, .. }
        )) if actual == required && maximum == required - 1)
        );
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = required;
        let expected = crate::topology::AnalysisProjectionError::InvalidElementGeometry {
            element_id: mesh.topology.volume_elements[0].element_id,
        }
        .to_string();
        assert!(matches!(
            workload.execute(&mesh),
            Err(crate::FeaRunError::Resource(
                ResourcePlanningError::Invalid(message)
            )) if message == expected
        ));
    }
}

#[test]
fn oversized_input_strings_fail_the_shared_codec_limit() {
    let mut workload = workload();
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let schedule = workload.solve_schedule().unwrap();
    workload.model.model_id.0 =
        "x".repeat(crate::execution::THERMAL_WORKLOAD_LIMITS.maximum_string_bytes + 1);
    assert!(ThermalPreassemblyResourceFacts::derive(
        &workload.model,
        &workload.model.sequence.steps[0],
        &mesh,
        &schedule
    )
    .is_err());
}

#[test]
fn retained_property_metadata_is_reserved_before_assembly() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut workload = workload();
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let schedule = workload.solve_schedule().unwrap();
        let base = ThermalPreassemblyResourceFacts::derive(
            &workload.model,
            &workload.model.sequence.steps[0],
            &mesh,
            &schedule,
        )
        .unwrap();
        workload.model.materials[0].name = "long_material_description".repeat(512);
        let facts = ThermalPreassemblyResourceFacts::derive(
            &workload.model,
            &workload.model.sequence.steps[0],
            &mesh,
            &schedule,
        )
        .unwrap();
        assert!(facts.property_compilation_bytes.0 > base.property_compilation_bytes.0);
        assert_eq!(
            facts.properties.solid_materials().len(),
            mesh.topology.volume_elements.len()
        );
        let required = facts.reservation_bytes().unwrap().0;
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = required - 1;
        assert!(
            matches!(workload.execute(&mesh), Err(crate::FeaRunError::Resource(ResourcePlanningError::AccuracyCeiling { required: actual, .. })) if actual == required)
        );
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = required;
        workload.execute(&mesh).unwrap();
    }
}
