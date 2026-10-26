//! Actual governed worker and original-field authority evidence.
use super::*;
use runmat_analysis_core::*;
use runmat_analysis_execution::{
    prepare_fluid_steady_task, AnalysisOutputPolicy, FluidSteadyTaskPreparation,
    PreparedFluidSteadyTask,
};
use runmat_analysis_fea::execution::FluidSteadySolverResult;
use runmat_execution::{ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ArtifactAccess};

mod chain;
mod closed_reference;
mod pressure_state;
#[path = "../../../tests/support/fluid.rs"]
mod source_fixture;

pub(crate) fn task(
    nonzero: bool,
    chunk: usize,
) -> (
    FluidSteadyWorkload,
    SolverMeshArtifact,
    PreparedFluidSteadyTask,
) {
    task_with_pressure(nonzero, chunk, 0.0)
}

fn task_with_pressure(
    nonzero: bool,
    chunk: usize,
    reference_pa: f64,
) -> (
    FluidSteadyWorkload,
    SolverMeshArtifact,
    PreparedFluidSteadyTask,
) {
    let (workload, mesh) = source_fixture::workload(nonzero, reference_pa);
    let digest = Digest::sha256(b"native fluid original authority fixture");
    let prepared = prepare_fluid_steady_task(FluidSteadyTaskPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        authored_study_digest: digest,
        frozen_project_digest: None,
        program_revision: ProgramRevision::new(
            digest,
            digest,
            ProgramEnvironment::new(1, 1, digest, digest, "matlab").unwrap(),
        )
        .unwrap(),
        artifact_access: ArtifactAccess {
            authorization_scope: "native-fluid".into(),
            encryption_context: digest,
        },
        workload: workload.clone(),
        mesh: mesh.clone(),
        output_policy: AnalysisOutputPolicy::new(chunk, ObjectInventoryLimits::default()).unwrap(),
        required_capabilities: Default::default(),
    })
    .unwrap();
    (workload, mesh, prepared)
}

fn invoke(task: &PreparedFluidSteadyTask) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    execute(AnalysisKernelInvocation {
        kind: AnalysisWorkKind::SolveFluidSteady,
        host: &task.host,
        inputs: &task.inputs,
        output_policy: task.output_policy,
    })
}

#[test]
fn native_original_zero_and_analytic_translation_are_chunk_policy_invariant() {
    for nonzero in [false, true] {
        let mut identity = None;
        for chunk in [1024, 8192] {
            let (workload, mesh, task) = task(nonzero, chunk);
            let outputs = invoke(&task).unwrap();
            assert_eq!(outputs.len(), 2);
            let mut outputs = outputs.into_iter();
            let original = outputs
                .next()
                .unwrap()
                .into_prepared(chunk, task.output_policy.inventory_limits())
                .unwrap();
            let bytes = original.payload();
            let result = FluidSteadySolverResult::from_canonical_bytes_for(
                &bytes,
                Digest::sha256(&bytes),
                &workload,
                &mesh,
            )
            .unwrap();
            if let Some(previous) = identity {
                assert_eq!(previous, result.identity().unwrap());
            }
            identity = Some(result.identity().unwrap());
            for velocity in result.original().velocity_m_per_s.canonical_values() {
                assert!((velocity[0] - if nonzero { 1.0 } else { 0.0 }).abs() < 1e-6);
                assert!(velocity[1].abs() < 1e-6 && velocity[2].abs() < 1e-6);
            }
            let receipt = outputs
                .next()
                .unwrap()
                .into_prepared(chunk, task.output_policy.inventory_limits())
                .unwrap();
            let receipt =
                FluidNumericalReceiptArtifact::from_canonical_bytes(&receipt.payload()).unwrap();
            receipt.validate_against(&workload, &result, &mesh).unwrap();
        }
    }
}

#[test]
fn native_rejects_missing_inputs_and_output_policy_substitution() {
    let (_, _, mut task) = task(false, 1024);
    task.output_policy = AnalysisOutputPolicy::new(8192, ObjectInventoryLimits::default()).unwrap();
    assert!(invoke(&task).is_err());
    task.output_policy = AnalysisOutputPolicy::new(1024, ObjectInventoryLimits::default()).unwrap();
    task.inputs.pop();
    assert!(invoke(&task).is_err());
}
