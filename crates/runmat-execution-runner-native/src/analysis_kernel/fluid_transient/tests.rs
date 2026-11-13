//! Original worker invocations across terminated producer processes, not task-pool placement.
use super::*;
use runmat_analysis_core::*;
use runmat_analysis_execution::{
    prepare_fluid_transient_task, AnalysisOutputPolicy, FluidTransientTaskPreparation,
    PreparedFluidTransientTask,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, FluidTransientBlockWorkload, FluidTransientCheckpoint,
    FluidTransientWorkload,
};
use runmat_analysis_fea::planning::resource::FeaResourceDemand;
use runmat_execution::Digest;
use runmat_execution::{ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ArtifactAccess};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use std::process::Command;
fn fixture() -> (FluidTransientWorkload, SolverMeshArtifact) {
    let (steady, mesh, _) = super::super::fluid_steady::tests::task(true, 8192);
    let mut model = steady.model;
    model.sequence = AnalysisSequence::from_steps(
        "native-transient",
        vec![ProcedureStep::fluid_transient("advance")],
    );
    model.sequence.steps[0].independent_domain = IndependentDomain::TimeSeconds {
        origin: 0.,
        start: 0.,
        end: 0.3,
        interval_count: 3,
    };
    model
        .initial_conditions
        .push(InitialCondition::FluidVelocity(InitialConditionValue {
            initial_condition_id: "original-u".into(),
            target: InitialConditionTarget::WholeModel,
            frame: CoordinateFrame::Global,
            value: InitialValue::UniformVector3([
                Velocity::new(1.),
                Velocity::new(0.),
                Velocity::new(0.),
            ]),
            applies_before_step_id: "advance".into(),
        }));
    (
        FluidTransientWorkload {
            model,
            step_id: ProcedureStepId("advance".into()),
            policy: steady.policy,
            geometry_policy: steady.geometry_policy,
            initial_fields: Vec::new(),
        },
        mesh,
    )
}
fn task(
    request: FluidTransientBlockWorkload,
    mesh: SolverMeshArtifact,
    chunk: usize,
) -> PreparedFluidTransientTask {
    let d = Digest::sha256("native-original-transient");
    prepare_fluid_transient_task(FluidTransientTaskPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        authored_study_digest: d,
        frozen_project_digest: None,
        program_revision: ProgramRevision::new(
            d,
            d,
            ProgramEnvironment::new(1, 1, d, d, "matlab").unwrap(),
        )
        .unwrap(),
        artifact_access: ArtifactAccess {
            authorization_scope: "fluid-transient-test".into(),
            encryption_context: d,
        },
        workload: request,
        mesh,
        output_policy: AnalysisOutputPolicy::new(chunk, ObjectInventoryLimits::default()).unwrap(),
        required_capabilities: Default::default(),
    })
    .unwrap()
}
fn invoke(task: &PreparedFluidTransientTask) -> AnalysisBridgeResult<FluidTransientCheckpoint> {
    let mut output = execute(AnalysisKernelInvocation {
        kind: AnalysisWorkKind::SolveFluidTransientBlock,
        host: &task.host,
        inputs: &task.inputs,
        output_policy: task.output_policy,
    })?;
    let original = output.remove(0).into_prepared(
        task.output_policy.chunk_bytes(),
        task.output_policy.inventory_limits(),
    )?;
    let checkpoint = FluidTransientCheckpoint::from_canonical_bytes(&original.payload())?;
    if checkpoint.identity()? != original.manifest.logical_identity {
        return Err(invalid("checkpoint output identity differs"));
    }
    let request_objects = task
        .inputs
        .iter()
        .find(|i| i.manifest.kind == AnalysisNumericalObjectKind::FluidTransientBlockWorkload)
        .unwrap();
    let mesh_objects = task
        .inputs
        .iter()
        .find(|i| i.manifest.kind == AnalysisNumericalObjectKind::Mesh)
        .unwrap();
    let request = FluidTransientBlockWorkload::from_canonical_bytes(&request_objects.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_objects.payload()).map_err(invalid)?;
    request.recover_output(&checkpoint, &mesh)?;
    Ok(checkpoint)
}
#[test]
fn native_transient_producer_child() {
    let Some(directory) = std::env::var_os("RUNMAT_FEA_TRANSIENT_CHILD_DIR") else {
        return;
    };
    let path = std::path::PathBuf::from(directory);
    let request = FluidTransientBlockWorkload::from_canonical_bytes(
        &std::fs::read(path.join("request.bin")).unwrap(),
    )
    .unwrap();
    let mesh = SolverMeshArtifact::canonical_decode(&std::fs::read(path.join("mesh.bin")).unwrap())
        .unwrap();
    let prepared = task(request, mesh, 1024);
    let checkpoint = invoke(&prepared).unwrap();
    std::fs::write(
        path.join("checkpoint.bin"),
        checkpoint.canonical_bytes().unwrap(),
    )
    .unwrap();
}
fn child(
    request: &FluidTransientBlockWorkload,
    mesh: &SolverMeshArtifact,
) -> FluidTransientCheckpoint {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("request.bin"),
        request.canonical_bytes().unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("mesh.bin"),
        mesh.canonical_encode().unwrap(),
    )
    .unwrap();
    let status = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "analysis_kernel::fluid_transient::tests::native_transient_producer_child",
            "--nocapture",
        ])
        .env("RUNMAT_FEA_TRANSIENT_CHILD_DIR", dir.path())
        .status()
        .unwrap();
    assert!(status.success());
    FluidTransientCheckpoint::from_canonical_bytes(
        &std::fs::read(dir.path().join("checkpoint.bin")).unwrap(),
    )
    .unwrap()
}
// witness: FEA-FLUID-TRANSIENT-001/producer_exit @
//   incompressible_laminar_transient/canonical_tetrahedra/native_process;
//   boundary: Child actually exits before new consumers resume; uniform translation and
//   substitution/budget rejection only.
#[test]
fn terminated_native_producers_preserve_original_history_and_admission_rejects_substitution() {
    let (workload, mesh) = fixture();
    let initial = workload.initial_checkpoint(&mesh).unwrap();
    let full_request = FluidTransientBlockWorkload {
        workload: workload.clone(),
        predecessor: initial.clone(),
        end_interval: 3,
    };
    let full = child(&full_request, &mesh);
    let first_request = FluidTransientBlockWorkload {
        workload: workload.clone(),
        predecessor: initial,
        end_interval: 1,
    };
    let first = child(&first_request, &mesh);
    let request = FluidTransientBlockWorkload {
        workload: workload.clone(),
        predecessor: first,
        end_interval: 3,
    };
    let split = child(&request, &mesh);
    split.recover(&workload, &mesh).unwrap();
    assert_eq!(
        split.intervals().collect::<Vec<_>>(),
        full.intervals().collect::<Vec<_>>()
    );
    assert_eq!(
        split.retained_results().last(),
        full.retained_results().last()
    );
    for (_, original) in split.retained_results() {
        // Uniform translation is reproduced to the solver's rounding, not bit-exactly.
        for velocity in original.velocity_m_per_s.canonical_values() {
            let error = (0..3)
                .map(|k| (velocity[k] - [1., 0., 0.][k]).abs())
                .fold(0.0, f64::max);
            assert!(error < 1e-12, "{velocity:?}");
        }
    }
    let prepared = task(request.clone(), mesh.clone(), 8192);
    let other_chunks = invoke(&prepared).unwrap();
    assert_eq!(other_chunks, split);
    for mutation in 0..3 {
        let mut inputs = prepared.inputs.clone();
        match mutation {
            0 => {
                inputs.remove(0);
            }
            1 => {
                inputs[0].manifest.logical_identity = Digest::sha256("foreign");
            }
            _ => {
                inputs[0].objects.last_mut().unwrap().bytes[0] ^= 1;
            }
        }
        assert!(execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::SolveFluidTransientBlock,
            host: &prepared.host,
            inputs: &inputs,
            output_policy: prepared.output_policy
        })
        .is_err());
    }
    let mut altered = prepared.resource_plan.input.clone();
    if let FeaResourceDemand::FluidTransientBlock(demand) = &mut altered.demand {
        demand.projection.phase_resident_bytes.0 -= 1;
    }
    let plan = runmat_analysis_fea::planning::resource::build_resource_plan(altered).unwrap();
    assert!(request.execute_with_resource_plan(&mesh, &plan).is_err());
}
