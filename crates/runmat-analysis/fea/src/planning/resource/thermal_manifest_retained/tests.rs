use super::*;
use crate::{
    execution::ThermalWorkload,
    fixtures::{fixture_model, FixtureId},
    planning::resource::ThermalPreassemblyResourceFacts,
};
use runmat_analysis_core::*;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

#[test]
fn retained_metadata_is_explicit_and_composed_before_assembly() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.sequence =
        AnalysisSequence::from_steps("thermal", vec![ProcedureStep::thermal_transient("heat")]);
    let mut workload = ThermalWorkload {
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: crate::ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
        procedure: crate::ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
        residual_target: 1e-4,
    };
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = u64::MAX;
        let identity = crate::admission::admit_nonstructural(
            PhysicsFamily::Thermal,
            workload.procedure.procedure(),
            Some(&mesh),
            SourceImplementation::CanonicalVolumeProjection,
            BoundaryTransferImplementation::CanonicalBoundaryProjection,
            workload.formulation_requirement,
        )
        .unwrap();
        let facts = ThermalPreassemblyResourceFacts::derive(
            &workload.model,
            &workload.model.sequence.steps[0],
            &mesh,
            &workload.solve_schedule().unwrap(),
            &identity,
        )
        .unwrap();
        let retained = facts.manifest_retained;
        assert_eq!(
            retained.scalar_inventory_bytes.0,
            ((mesh.topology.nodes.len() + mesh.topology.volume_elements.len() + 3 + 2) * 8
                + 2 * std::mem::size_of::<ThermalEnergyBalance>()) as u64
        );
        assert_eq!(retained.source_bytes.0, 0);
        assert_eq!(
            retained.backend_bytes.0,
            (2 * std::mem::size_of::<AcceleratorSolveEvidence>()) as u64
        );
        assert!(retained.identity_heap_bytes.0 > 0);
        assert!(retained.diagnostic_bytes.0 > 0);
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
        let mut overflow = retained;
        overflow.inline_bytes = ByteCount(u64::MAX);
        assert!(overflow.reservation_bytes().is_err());
    }
    assert!(slots::<u64>(usize::MAX).is_err());
}
