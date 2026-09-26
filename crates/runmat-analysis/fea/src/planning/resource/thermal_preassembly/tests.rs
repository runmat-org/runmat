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
                + grown.property_materialization_bytes.0
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
        assert!(facts.property_materialization_bytes.0 > base.property_materialization_bytes.0);
        assert_eq!(
            facts.properties.solid_materials().len(),
            mesh.topology.volume_elements.len()
        );
        let required = facts.reservation_bytes().unwrap().0;
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = required - facts.property_materialization_bytes.0;
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
