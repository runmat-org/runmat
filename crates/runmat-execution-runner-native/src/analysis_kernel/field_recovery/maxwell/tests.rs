//! Original worker solve -> retained publication -> cold partition evidence.
use super::*;
use crate::analysis_kernel::{maxwell_harmonic::tests::task, NativeAnalysisKernel};
use runmat_analysis_execution::{
    execute_serial_analysis, import_analysis_partition, import_maxwell_harmonic_result,
    import_maxwell_recovered_fields, prepare_maxwell_field_recovery, AnalysisHostResponse,
    AnalysisOutputPolicy, AnalysisPartitionImport, MaxwellFieldRecoveryPreparation,
    PreparedAnalysisProgram, PreparedMaxwellFieldRecovery,
};
use runmat_analysis_fea::results::field::RecoveredFieldPartition;
use runmat_execution_artifact::{cache::FilesystemObjectStore, ProgramExecutionResponse};

fn run(
    store: &mut FilesystemObjectStore,
    program: &PreparedAnalysisProgram,
    host: &runmat_analysis_execution::AnalysisHostWorkload,
    policy: AnalysisOutputPolicy,
) -> ProgramExecutionResponse {
    program.write_to(store).unwrap();
    let completed = execute_serial_analysis(
        &program.request,
        store,
        &NativeAnalysisKernel,
        policy.chunk_bytes(),
        policy.inventory_limits(),
    )
    .unwrap();
    AnalysisHostResponse::completed(host, &completed)
        .unwrap()
        .program_response()
}

fn invoke(
    prepared: &PreparedMaxwellFieldRecovery,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    execute(AnalysisKernelInvocation {
        kind: AnalysisWorkKind::RecoverField,
        host: &prepared.host,
        inputs: &prepared.inputs,
        output_policy: prepared.output_policy,
    })
}

#[test]
fn both_orders_publish_all_original_native_fields_and_reopen_cold() {
    for quadratic in [false, true] {
        for chunk in [1024, 8192] {
            let (workload, mesh, task) = task(quadratic, chunk);
            let limits = task.output_policy.inventory_limits();
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("fields");
            let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
            let response = run(&mut store, &task.program, &task.host, task.output_policy);
            let solved = import_maxwell_harmonic_result(&store, response, &task, limits).unwrap();
            let source_input = |kind| {
                task.inputs
                    .iter()
                    .find(|input| input.manifest.kind == kind)
                    .unwrap()
                    .clone()
            };
            let prepared = prepare_maxwell_field_recovery(MaxwellFieldRecoveryPreparation {
                source_host: task.host.clone(),
                workload: source_input(AnalysisNumericalObjectKind::MaxwellHarmonicWorkload),
                mesh: source_input(AnalysisNumericalObjectKind::Mesh),
                source_result: solved.objects.clone(),
                program_revision: task.program.request.recipe.program_revision.clone(),
                output_policy: task.output_policy,
                local_execution:
                    runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            })
            .unwrap();
            let original = solved
                .result
                .recover_native_fields(&workload, &mesh, &prepared.request)
                .unwrap();
            assert_eq!(original.fields().len(), 11);
            // Negative admission evidence uses freshly valid closures, not only missing inputs.
            let mut wrong_policy = prepared.output_policy.inventory_limits();
            wrong_policy.max_objects = 1;
            assert!(build_maxwell_recovery_metadata_resource_plan(
                &prepared.request,
                &prepared.inputs,
                AnalysisOutputPolicy::new(chunk, wrong_policy).unwrap(),
                runmat_analysis_fea::planning::resource::ByteCount(0),
                runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            )
            .is_err());
            let mut missing = prepared.inputs.clone();
            missing
                .retain(|input| input.manifest.kind != AnalysisNumericalObjectKind::SolverResult);
            assert!(execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::RecoverField,
                host: &prepared.host,
                inputs: &missing,
                output_policy: prepared.output_policy,
            })
            .is_err());
            let mut foreign_host = prepared.host.clone();
            foreign_host.workload.execution_plan = task.host.workload.execution_plan;
            assert!(execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::RecoverField,
                host: &foreign_host,
                inputs: &prepared.inputs,
                output_policy: prepared.output_policy,
            })
            .is_err());
            assert!(invoke(&prepared).is_ok());
            let response = run(
                &mut store,
                &prepared.program,
                &prepared.host,
                prepared.output_policy,
            );
            drop(store);
            let cold = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
            let imported =
                import_maxwell_recovered_fields(&cold, response, &prepared, limits).unwrap();
            assert_eq!(imported.manifest.fields.len(), 11);
            assert_eq!(
                imported.manifest.recovery_request,
                prepared.request.identity()
            );
            for (entry, field) in imported.manifest.fields.iter().zip(original.fields()) {
                assert_eq!(entry.definition, *field.definition());
                let expected = field.storage().as_host_f64().unwrap();
                // Phasor fields interleave real and imaginary parts; cycle-averaged losses are real.
                let width = field.definition().value_shape().component_count().unwrap()
                    * match field.definition().numeric_representation() {
                        runmat_analysis_fea::results::field::RecoveredFieldNumericRepresentation::ComplexInterleaved => 2,
                        _ => 1,
                    };
                let mut count = 0;
                for descriptor in &entry.partitions {
                    let object = import_analysis_partition(
                        &cold,
                        AnalysisPartitionImport {
                            root: &imported.output.root,
                            expected_kind: AnalysisNumericalObjectKind::Field,
                            expected_logical_identity: imported.output.logical_identity,
                            expected_payload_digest: imported.output.payload_digest,
                            partition_identity: descriptor.partition.digest(),
                            access: &prepared.host.artifact_access,
                            limits,
                        },
                    )
                    .unwrap();
                    let partition =
                        RecoveredFieldPartition::from_canonical_bytes(&object.payload()).unwrap();
                    for (offset, &index) in partition.source_indices.iter().enumerate() {
                        assert_eq!(
                            &partition.storage.as_host_f64().unwrap()
                                [offset * width..(offset + 1) * width],
                            &expected[index * width..(index + 1) * width],
                        );
                        count += 1;
                    }
                }
                assert_eq!(count * width, expected.len());
            }
        }
    }
}
