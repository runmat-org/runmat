//! Independent homogeneous J2 stress/strain reference through actual native tasks.
use super::*;
use runmat_analysis_core::*;
use runmat_analysis_execution::{
    prepare_nonlinear_solid_block_task, AnalysisKernel, AnalysisOutputPolicy,
    NonlinearSolidBlockPreparation, PreparedNonlinearSolidBlockTask,
};
use runmat_analysis_fea::execution::NonlinearSolidBlockReference;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ArtifactAccess};
use runmat_meshing_core::fixtures::slab_mesh;
mod cancellation;
mod chain;
mod coordinator;
mod durable;
mod durable_tamper;
mod layered;
mod observations;
mod planning;
mod process;
mod publication;
mod tamper;

#[path = "../../../tests/support/nonlinear.rs"]
mod support;
use support::fixture;

fn prepare(
    workload: &NonlinearSolidWorkload,
    mesh: &SolverMeshArtifact,
    request: NonlinearSolidBlockRequest,
    predecessor: Option<runmat_analysis_execution::PreparedAnalysisObjects>,
) -> PreparedNonlinearSolidBlockTask {
    let digest = Digest::sha256(b"native J2 original history");
    prepare_nonlinear_solid_block_task(NonlinearSolidBlockPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        authored_study_digest: digest,
        frozen_project_digest: None,
        program_revision: ProgramRevision::new(
            digest,
            digest,
            // The task runs on a pool worker whose host identity is this binary's program
            // environment (test_support::host_inventory); the program uses the same one.
            runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        )
        .unwrap(),
        artifact_access: ArtifactAccess {
            authorization_scope: "native-j2".into(),
            encryption_context: digest,
        },
        workload: workload.clone(),
        mesh: mesh.clone(),
        request,
        predecessor,
        output_policy: AnalysisOutputPolicy::new(4096, ObjectInventoryLimits::default()).unwrap(),
    })
    .unwrap()
}

#[test]
fn original_plastic_history_survives_native_task_boundaries_for_both_orders() {
    for quadratic in [false, true] {
        let (workload, mesh) = fixture(quadratic);
        let request = NonlinearSolidBlockRequest {
            workload: workload.identity().unwrap(),
            topology: Digest::from_bytes(*mesh.canonical_digest.bytes()),
            block_ordinal: 0,
            first_output_ordinal: 1,
            accepted_block_limit: 9,
            predecessor: None,
        };
        let first = prepare(&workload, &mesh, request.clone(), None);
        let kernel = super::super::NativeAnalysisKernel;
        let outputs = kernel
            .execute(AnalysisKernelInvocation {
                host: &first.host,
                kind: AnalysisWorkKind::SolveNonlinearSolidBlock,
                inputs: &first.inputs,
                output_policy: first.output_policy,
            })
            .unwrap();
        let runmat_analysis_execution::AnalysisKernelPayload::Prepared(objects) =
            &outputs[0].payload
        else {
            panic!("expected original prepared block");
        };
        let objects = (**objects).clone();
        let partial = NonlinearSolidBlockArtifact::from_canonical_bytes(
            &objects.payload(),
            objects.manifest.logical_identity,
        )
        .unwrap();
        assert!(!partial.complete);
        assert!(partial
            .result
            .equivalent_plastic_strain_snapshots
            .last()
            .unwrap()
            .iter()
            .all(|v| *v > 0.0));
        let next = NonlinearSolidBlockRequest {
            block_ordinal: 1,
            first_output_ordinal: partial.last_output_ordinal().unwrap() + 1,
            accepted_block_limit: 3,
            predecessor: Some(NonlinearSolidBlockReference {
                artifact_digest: partial.identity,
                controller_digest: partial.checkpoint.content_digest(),
            }),
            ..request
        };
        drop(outputs);
        drop(first);
        let resumed = prepare(&workload, &mesh, next, Some(objects));
        cancellation::reject_cancelled(&resumed);
        let outputs = kernel
            .execute(AnalysisKernelInvocation {
                host: &resumed.host,
                kind: AnalysisWorkKind::SolveNonlinearSolidBlock,
                inputs: &resumed.inputs,
                output_policy: resumed.output_policy,
            })
            .unwrap();
        let runmat_analysis_execution::AnalysisKernelPayload::Prepared(original) =
            &outputs[0].payload
        else {
            panic!("expected original prepared block");
        };
        let terminal = NonlinearSolidBlockArtifact::from_canonical_bytes(
            &original.payload(),
            original.manifest.logical_identity,
        )
        .unwrap();
        assert!(terminal.complete);
        for (_, _, p) in terminal.result.accepted_state.point_history() {
            assert!((p.equivalent_plastic_strain - 0.02).abs() < 2e-9);
            for (actual, exact) in p
                .engineering_plastic_strain
                .iter()
                .zip([0.02, -0.01, -0.01, 0.0, 0.0, 0.0])
            {
                assert!((actual - exact).abs() < 2e-9);
            }
        }
        for (node, u) in mesh.topology.nodes.iter().zip(
            terminal
                .result
                .accepted_state
                .displacement()
                .chunks_exact(3),
        ) {
            for (axis, strain) in [0.0256, -0.0114, -0.0114].into_iter().enumerate() {
                assert!(
                    (u[axis] - strain * node.coordinates_m.canonical_value()[axis]).abs() < 2e-9
                );
            }
        }
        for tensor in terminal
            .result
            .stress_snapshots
            .last()
            .unwrap()
            .chunks_exact(6)
        {
            for (actual, exact) in tensor.iter().zip([5.6, 0.0, 0.0, 0.0, 0.0, 0.0]) {
                assert!((actual - exact).abs() < 2e-9);
            }
        }
        assert_eq!(
            terminal.result.initial_equivalent_plastic_strain,
            *partial
                .result
                .equivalent_plastic_strain_snapshots
                .last()
                .unwrap()
        );
    }
}
