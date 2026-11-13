//! Reopened object-store chain: original field publication is a separate copy-only task.
use super::*;
use crate::NativeAnalysisKernel;
use runmat_analysis_execution::*;
use runmat_analysis_fea::results::field::{reconstruct_recovered_field, RecoveredFieldPartition};
use runmat_execution_artifact::{cache::FilesystemObjectStore, ProgramExecutionResponse};

fn run(
    store: &mut FilesystemObjectStore,
    program: &PreparedAnalysisProgram,
    host: &AnalysisHostWorkload,
    policy: AnalysisOutputPolicy,
) -> ProgramExecutionResponse {
    program.write_to(store).unwrap();
    let done = execute_serial_analysis(
        &program.request,
        store,
        &NativeAnalysisKernel,
        policy.chunk_bytes(),
        policy.inventory_limits(),
    )
    .unwrap();
    AnalysisHostResponse::completed(host, &done)
        .unwrap()
        .program_response()
}

mod publication;

#[test]
fn native_solve_receipt_and_copy_only_all_face_recovery_survive_cold_store() {
    for (workload, mesh, task) in [
        task(false, 8192),
        task(true, 8192),
        super::closed_reference::closed_task(),
    ] {
        let policy = task.output_policy;
        let limits = policy.inventory_limits();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cold-fluid");
        let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        let response = run(&mut store, &task.program, &task.host, policy);
        drop(store);
        let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        let imported = import_fluid_steady_result(&store, response, &task, limits).unwrap();
        imported
            .numerical_receipt
            .validate_against(&workload, &imported.result, &mesh)
            .unwrap();
        let source = |kind| {
            task.inputs
                .iter()
                .find(|i| i.manifest.kind == kind)
                .unwrap()
                .clone()
        };
        let recovery = prepare_fluid_field_recovery(FluidFieldRecoveryPreparation {
            source_host: task.host.clone(),
            workload: source(AnalysisNumericalObjectKind::FluidSteadyWorkload),
            mesh: source(AnalysisNumericalObjectKind::Mesh),
            source_resource_plan: source(AnalysisNumericalObjectKind::ResourcePlan),
            source_result: imported.objects.clone(),
            program_revision: task.program.request.recipe.program_revision.clone(),
            output_policy: policy,
        })
        .unwrap();
        let response = run(&mut store, &recovery.program, &recovery.host, policy);
        drop(store);
        let store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        let cold = import_fluid_recovered_fields(&store, response, &recovery, limits).unwrap();
        let expected = imported
            .result
            .recover_native_fields(&workload, &mesh, &recovery.request)
            .unwrap();
        // Velocity, pressure, face flux, vorticity, wall shear stress and face pressure.
        assert_eq!(cold.manifest.fields.len(), 6);
        for (entry, original) in cold.manifest.fields.iter().zip(expected.fields()) {
            let partitions = entry
                .partitions
                .iter()
                .map(|descriptor| {
                    let objects = import_analysis_partition(
                        &store,
                        AnalysisPartitionImport {
                            root: &cold.output.root,
                            expected_kind: AnalysisNumericalObjectKind::Field,
                            expected_logical_identity: cold.output.logical_identity,
                            expected_payload_digest: cold.output.payload_digest,
                            partition_identity: descriptor.partition.digest(),
                            access: &recovery.host.artifact_access,
                            limits,
                        },
                    )
                    .unwrap();
                    RecoveredFieldPartition::from_canonical_bytes(&objects.payload()).unwrap()
                })
                .collect();
            let field = reconstruct_recovered_field(
                entry.field,
                entry.definition.clone(),
                entry.sample,
                partitions,
                entry
                    .definition
                    .stored_entities(cold.manifest.entity_counts),
            )
            .unwrap();
            assert_eq!(&field, original);
        }
        let mut missing_mesh = recovery.inputs.clone();
        missing_mesh.retain(|input| input.manifest.kind != AnalysisNumericalObjectKind::Mesh);
        assert!(NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::RecoverField,
                host: &recovery.host,
                inputs: &missing_mesh,
                output_policy: policy,
            })
            .is_err());
        let mut corrupt_source = recovery.inputs.clone();
        let source = corrupt_source
            .iter_mut()
            .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::SolverResult)
            .unwrap();
        source.objects[1].bytes[0] ^= 1;
        assert!(NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::RecoverField,
                host: &recovery.host,
                inputs: &corrupt_source,
                output_policy: policy,
            })
            .is_err());
    }
}
