//! Original prepared program admission, before a numerical kernel is entered.
use super::*;
use runmat_execution::resource::ResourceRequest;
use std::sync::atomic::{AtomicUsize, Ordering};
struct Witness {
    entries: AtomicUsize,
    expected: ResourceRequest,
}
impl AnalysisKernel for Witness {
    fn execute(
        &self,
        _: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        panic!("reserved dispatch lost the original reservation")
    }
    fn execute_reserved(
        &self,
        invocation: ReservedAnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        assert_eq!(invocation.resources(), &self.expected);
        self.entries.fetch_add(1, Ordering::SeqCst);
        Err(AnalysisBridgeError::Invalid(
            "reserved callback reached".into(),
        ))
    }
}
#[test]
fn original_prepared_resources_must_fit_the_execution_reservation_before_kernel_entry() {
    use runmat_analysis_fea::{
        execution::StructuralLinearWorkload,
        fixtures::{fixture_model, FixtureId},
        solve::{backend::kind::LinearAlgebraBackendKind, preconditioner::SpdPreconditionerKind},
        ComputeBackend,
    };
    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let workload = StructuralLinearWorkload {
        schema_version: 2,
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::serial(),
        require_solver_mesh_for_solid: true,
        state_input: None,
    };
    let prepared = prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest: digest("reservation-study"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: access(),
        workload,
        mesh: None,
        state_input: None,
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
        accelerator_requests: vec![],
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        solve_selection:
            runmat_analysis_fea::planning::resource::StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
    })
    .unwrap();
    let mut store = MemoryObjects::default();
    prepared.program.write_to(&mut store).unwrap();
    let witness = Witness {
        entries: AtomicUsize::new(0),
        expected: prepared.resource_plan.request.clone(),
    };
    let original_digest = prepared.program.request.canonical_digest().unwrap();
    for dimension in 0..3 {
        let mut reservation = witness.expected.clone();
        match dimension {
            0 => reservation.memory_bytes -= 1,
            1 => reservation.cpu_millicores -= 1,
            2 => reservation.scratch_bytes -= 1,
            _ => unreachable!(),
        }
        assert!(execute_reserved_serial_analysis(
            &prepared.program.request,
            &mut store,
            &witness,
            4096,
            ObjectInventoryLimits::default(),
            &reservation
        )
        .is_err());
        assert_eq!(witness.entries.load(Ordering::SeqCst), 0);
    }
    let error = execute_reserved_serial_analysis(
        &prepared.program.request,
        &mut store,
        &witness,
        4096,
        ObjectInventoryLimits::default(),
        &witness.expected,
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("reserved callback reached"),
        "{error}"
    );
    assert_eq!(witness.entries.load(Ordering::SeqCst), 1);
    assert_eq!(
        prepared.program.request.canonical_digest().unwrap(),
        original_digest
    );
}

#[test]
fn absent_governing_plan_cannot_be_replaced_by_a_large_execution_reservation() {
    let (_, program, mut store) = super::host_transport::fixture();
    let resources = ResourceRequest {
        cpu_millicores: 1000,
        memory_bytes: 1024 * 1024 * 1024,
        scratch_bytes: 0,
        max_wall_millis: 1000,
        max_artifact_bytes: 1024 * 1024,
        max_egress_bytes: 1024 * 1024,
        max_relay_bytes: 0,
        accelerators: vec![],
        required_capabilities: BTreeSet::new(),
    };
    let witness = Witness {
        entries: AtomicUsize::new(0),
        expected: resources.clone(),
    };
    let error = execute_reserved_serial_analysis(
        program.request(),
        &mut store,
        &witness,
        6,
        ObjectInventoryLimits::default(),
        &resources,
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("current governing resource plan"),
        "{error}"
    );
    assert_eq!(witness.entries.load(Ordering::SeqCst), 0);
}
