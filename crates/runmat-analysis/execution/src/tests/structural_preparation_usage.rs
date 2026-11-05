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

#[test]
fn whole_structural_preparation_conserves_exact_composed_grant() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
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
