use super::*;
use crate::{
    execution::ThermalWorkload,
    fixtures::{fixture_model, FixtureId},
    planning::resource::ThermalPreassemblyResourceFacts,
    ComputeBackend, ThermalProcedureControl,
};
use runmat_analysis_core::*;
use runmat_canonical_codec::{profile_serialized, CanonicalLimits};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn workload(procedure: ThermalProcedureControl) -> ThermalWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    let step_id = "thermal_metadata_α".repeat(64);
    let step = match procedure {
        ThermalProcedureControl::Steady => ProcedureStep::thermal_steady(step_id.clone()),
        ThermalProcedureControl::BackwardEuler { .. } => ProcedureStep::thermal_transient(step_id),
    };
    let mut alternate = model.materials[0].clone();
    alternate.material_id = "expected_material_β".repeat(64);
    model.material_assignments[0].expected_material_id = alternate.material_id.clone();
    model.material_assignments[0].confidence = EvidenceConfidence::Probable;
    model.materials.push(alternate);
    model.sequence = AnalysisSequence::from_steps("thermal", vec![step]);
    // The authored ceiling is part of the encoded model, so resident requirements are
    // measured under a non-binding authored ceiling (present, like the ceilings tests author).
    model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(runmat_types::wire_integer::MAX_WIRE_INTEGER);
    ThermalWorkload {
        initial_fields: Vec::new(),
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        procedure,
        residual_target: 1e-4,
    }
}

#[test]
fn complete_manifest_profile_bounds_real_steady_and_transient_headers() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for procedure in [
            ThermalProcedureControl::Steady,
            ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
            ThermalProcedureControl::BackwardEuler { snapshot_count: 12 },
        ] {
            let mut workload = workload(procedure);
            let mesh = canonical_tetrahedron_solver_mesh(order);
            let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
            let boundary = projection.boundary_faces[0].provenance[0]
                .analysis_selection_id()
                .unwrap();
            workload
                .model
                .constraints
                .push(AnalysisConstraint::Thermal(ThermalConstraint {
                    constraint_id: "thermal_anchor".into(),
                    kind: ThermalConstraintKind::Prescribed {
                        target: RegionTarget::boundary(
                            runmat_meshing_core::fixtures::fixture_boundary_selection(&(boundary)),
                        ),
                        value: PrescribedValue::Constant(Temperature::new(293.15)),
                    },
                }));
            let identity = crate::admission::admit_nonstructural(
                PhysicsFamily::Thermal,
                procedure.procedure(),
                Some(&mesh),
                SourceImplementation::CanonicalVolumeProjection,
                BoundaryTransferImplementation::CanonicalBoundaryProjection,
            )
            .unwrap();
            let schedule = workload.solve_schedule().unwrap();
            let facts = ThermalPreassemblyResourceFacts::derive(
                &[],
                &workload.model,
                &workload.model.sequence.steps[0],
                &mesh,
                &schedule,
                &identity,
            )
            .unwrap();
            let result = workload.execute(&mesh).unwrap();
            let manifest = ThermalSolverResultManifest::from_result(&result).unwrap();
            let actual = profile_serialized(
                &manifest,
                CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64),
            )
            .unwrap();
            let bound = facts.manifest_metadata.complete_header;
            assert!(actual.encoded_bytes_upper_bound <= bound.encoded_bytes_upper_bound);
            assert!(
                actual.materialization_resident_bytes_estimate
                    <= bound.materialization_resident_bytes_estimate
            );
            assert!(
                manifest.canonical_bytes().unwrap().len() as u64 <= bound.encoded_bytes_upper_bound
            );
            let required = facts.reservation_bytes().unwrap().0;
            workload.model.sequence.steps[0]
                .accuracy
                .resources
                .maximum_resident_bytes = Some(required - 1);
            assert!(
                matches!(workload.execute(&mesh), Err(crate::FeaRunError::Resource(ResourcePlanningError::AccuracyCeiling { required: value, .. })) if value == required)
            );
            workload.model.sequence.steps[0]
                .accuracy
                .resources
                .maximum_resident_bytes = Some(required);
            workload.execute(&mesh).unwrap();
        }
    }
}

#[test]
fn profile_composition_checks_overflow_without_partial_update() {
    let original = SerializedCanonicalProfile {
        encoded_bytes_upper_bound: 7,
        materialization_resident_bytes_estimate: 9,
    };
    for value in [
        SerializedCanonicalProfile {
            encoded_bytes_upper_bound: u64::MAX,
            materialization_resident_bytes_estimate: 1,
        },
        SerializedCanonicalProfile {
            encoded_bytes_upper_bound: 1,
            materialization_resident_bytes_estimate: u64::MAX,
        },
    ] {
        let mut actual = original;
        assert!(add(&mut actual, value, 1).is_err());
        assert_eq!(actual, original);
    }
}

#[test]
fn scalar_inventory_profiles_grow_with_all_retained_axes() {
    let base = ThermalSolverResultManifest::scalar_inventory_profile(4, 1, 3, 2, 0).unwrap();
    for (nodes, elements, samples, solves) in
        [(40, 1, 3, 2), (4, 10, 3, 2), (4, 1, 30, 2), (4, 1, 3, 20)]
    {
        let grown = ThermalSolverResultManifest::scalar_inventory_profile(
            nodes, elements, samples, solves, 0,
        )
        .unwrap();
        assert!(grown.encoded_bytes_upper_bound > base.encoded_bytes_upper_bound);
        assert!(
            grown.materialization_resident_bytes_estimate
                > base.materialization_resident_bytes_estimate
        );
    }
}

#[test]
fn planning_rejects_an_identity_for_a_different_procedure() {
    let workload = workload(ThermalProcedureControl::BackwardEuler { snapshot_count: 3 });
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let identity = crate::admission::admit_nonstructural(
        PhysicsFamily::Thermal,
        AnalysisProcedure::Static,
        Some(&mesh),
        SourceImplementation::CanonicalVolumeProjection,
        BoundaryTransferImplementation::CanonicalBoundaryProjection,
    )
    .unwrap();
    assert!(
        matches!(ThermalPreassemblyResourceFacts::derive(&[], &workload.model,
        &workload.model.sequence.steps[0], &mesh, &workload.solve_schedule().unwrap(), &identity),
        Err(ResourcePlanningError::Invalid(message)) if message.contains("identity differs"))
    );
}
