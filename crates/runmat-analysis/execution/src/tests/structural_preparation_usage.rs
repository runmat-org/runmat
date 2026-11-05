use super::*;
use runmat_analysis_fea::{
    execution::StructuralLinearWorkload,
    fixtures::{fixture_model, FixtureId},
    planning::resource::{LocalExecutionPlan, ResourcePlanningError, StaticSolveSelection},
    solve::{backend::kind::LinearAlgebraBackendKind, preconditioner::SpdPreconditionerKind},
    structural_linear_preparation_work_upper_bound, ComputeBackend,
};
use runmat_meshing_core::{ElementOrder, PersistentEntityId, PersistentEntityKind, StableDigest};
use runmat_simulation_usage::{LedgerError, OperationCount, UsageContext, UsageLedger};
use std::sync::Arc;

fn observed_solid_fixture(
    order: ElementOrder,
) -> (
    runmat_analysis_core::AnalysisModel,
    runmat_meshing_core::SolverMeshArtifact,
) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.sequence.steps[0].observations = vec![structural_displacement_observation()];
    let mut mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
    for (index, face) in mesh.topology.boundary_faces.iter_mut().enumerate() {
        face.provenance.push(PersistentEntityId::new(
            PersistentEntityKind::Face,
            if index == 0 { "root" } else { "tip" }.into(),
            vec!["root".into()],
        ));
        face.provenance.sort();
    }
    let region = runmat_meshing_core::fixtures::fixture_entity_for_selection(
        model.material_assignments[0]
            .selected_region_id()
            .expect("cantilever fixture uses selected material region"),
    )
    .unwrap();
    mesh.topology.volume_elements[0].region_id = region.clone();
    mesh.topology.regions[0].region_id = region;
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    (model, mesh)
}

#[test]
fn whole_structural_preparation_conserves_exact_composed_grant() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, mesh) = observed_solid_fixture(order);
        let quote = structural_linear_preparation_work_upper_bound(Some(&mesh)).unwrap();
        let short = OperationCount::from_operations(quote.operations().checked_sub(1).unwrap());

        // The forced tiny machine ceiling visits both Auto candidates. Its
        // resource refusal must not derive or charge the shared projection twice.
        for (grant, machine_memory_bytes) in [(quote, None), (short, None), (quote, Some(1))] {
            let ledger = Arc::new(UsageLedger::new(quote, grant).unwrap());
            let result = UsageContext::new(Arc::clone(&ledger)).run_sync(|| {
                prepare_structural_linear_program(StructuralLinearProgramPreparation {
                    authored_study_digest: digest("exact-whole-preparation"),
                    frozen_project_digest: None,
                    program_revision: revision(),
                    artifact_access: access(),
                    workload: StructuralLinearWorkload {
                        schema_version: 2,
                        step_id: model.sequence.steps[0].step_id.clone(),
                        model: model.clone(),
                        backend: ComputeBackend::Cpu,
                        preconditioner: SpdPreconditionerKind::Auto,
                        algebra_backend: LinearAlgebraBackendKind::CpuReference,
                        observation_execution: runmat_analysis_fea::results::observation::ObservationExecutionPolicy::serial(),
                        require_solver_mesh_for_solid: true,
                        state_input: None,
                    },
                    mesh: Some(mesh.clone()),
                    state_input: None,
                    chunk_bytes: 4096,
                    inventory_limits: ObjectInventoryLimits::default(),
                    required_capabilities: BTreeSet::new(),
                    accelerator_requests: Vec::new(),
                    local_execution: LocalExecutionPlan::serial(),
                    solve_selection: StaticSolveSelection::WithinCeiling { machine_memory_bytes },
                })
            });
            let snapshot = ledger.snapshot().unwrap();
            assert_eq!(snapshot.reserved, OperationCount::ZERO);
            if grant == short {
                let error = result.err().expect("one operation short must refuse");
                assert_eq!(
                    error.accounting_cause(),
                    Some(&LedgerError::MoreGrantRequired {
                        required: quote,
                        granted: short,
                    })
                );
                assert!(snapshot.completed.assembly < quote);
            } else {
                assert_eq!(snapshot.completed.assembly, quote);
                if machine_memory_bytes.is_some() {
                    assert!(matches!(
                        result,
                        Err(AnalysisBridgeError::Resource(
                            ResourcePlanningError::StaticSolveCeiling { maximum: 1, .. }
                        ))
                    ));
                } else {
                    result.expect("exact composed grant funds whole preparation");
                }
            }
        }
    }
}

#[test]
fn owned_static_observations_acquire_capacity_before_consuming_their_quote() {
    use runmat_analysis_fea::{
        algebra::NumericalCapacityPhase, constraints::CanonicalStructuralConstraintPlan,
        run_linear_static_with_prepared_constraint_plan,
        run_linear_static_with_prepared_constraint_plan_tet4_owned, LinearStaticSolveOptions,
    };
    use std::sync::Mutex;

    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, mesh) = observed_solid_fixture(order);
        let constraints =
            CanonicalStructuralConstraintPlan::compile_for_model(&model, Some(&mesh)).unwrap();
        let quote = runmat_analysis_fea::results::observation::structural_observation_derivation_work_upper_bound(Some(&mesh))
            .unwrap().checked_mul(2).unwrap();
        let options = LinearStaticSolveOptions {
            solver_mesh: Some(mesh),
            ..Default::default()
        };
        let inline = run_linear_static_with_prepared_constraint_plan(
            &model,
            ComputeBackend::Cpu,
            options.clone(),
            &constraints,
            None,
        )
        .unwrap();
        let ledger = Arc::new(
            UsageLedger::new(
                OperationCount::from_operations(1_000_000_000),
                OperationCount::ZERO,
            )
            .unwrap(),
        );
        let phases = Arc::new(Mutex::new(Vec::new()));
        let context = UsageContext::new(Arc::clone(&ledger));
        let owned = crate::serial::finish_inline(context.scope(
            run_linear_static_with_prepared_constraint_plan_tet4_owned(
                &model,
                ComputeBackend::Cpu,
                options,
                &constraints,
                None,
                usize::MAX,
                |phase, work| {
                    let ledger = Arc::clone(&ledger);
                    let phases = Arc::clone(&phases);
                    async move {
                        let before = ledger.snapshot().unwrap();
                        let required = before
                            .completed
                            .total()
                            .unwrap()
                            .checked_add(before.reserved)
                            .unwrap()
                            .checked_add(work)
                            .unwrap();
                        ledger.update_grant(required.max(before.granted)).unwrap();
                        phases.lock().unwrap().push((
                            phase,
                            work,
                            before.completed.total().unwrap(),
                        ));
                        Ok::<(), AnalysisBridgeError>(())
                    }
                },
            ),
        ))
        .unwrap();
        assert_eq!(owned.fields, inline.fields);
        assert_eq!(owned.observations, inline.observations);
        let observation_phases = phases
            .lock()
            .unwrap()
            .iter()
            .filter(|(phase, _, _)| *phase == NumericalCapacityPhase::StaticObservations)
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(observation_phases.len(), 1);
        assert_eq!(observation_phases[0].1, quote);
        let completed = ledger.snapshot().unwrap();
        assert_eq!(completed.reserved, OperationCount::ZERO);
        assert_eq!(
            completed
                .completed
                .total()
                .unwrap()
                .operations()
                .checked_sub(observation_phases[0].2.operations())
                .unwrap(),
            quote.operations(),
        );
        let cause = LedgerError::MoreGrantRequired {
            required: quote,
            granted: OperationCount::ZERO,
        };
        assert!(matches!(
            runmat_analysis_fea::FeaRunError::from(
                runmat_analysis_fea::results::observation::ObservationError::Accounting(cause.clone())
            ),
            runmat_analysis_fea::FeaRunError::Usage(actual) if actual == cause
        ));
    }
}
