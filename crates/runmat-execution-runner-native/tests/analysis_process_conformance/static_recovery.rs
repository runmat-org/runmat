//! Original static solve -> separately admitted publication, through real native scheduling.
use super::*;
use runmat_analysis_execution::{
    import_analysis_partition, import_structural_recovered_fields,
    prepare_structural_field_recovery, AnalysisPartitionImport, StructuralFieldRecoveryPreparation,
};
use runmat_analysis_fea::results::field::{reconstruct_recovered_field, RecoveredFieldPartition};

pub(super) async fn run() {
    for branch in 0..3 {
        let directory = tempfile::tempdir().unwrap();
        let (host, program, inputs, resource_plan) = match branch {
            0 => fixture(revision(), "static-recovery-process"),
            1 => explicit_fixture(revision(), "static-recovery-process"),
            _ => mixed_fixture(revision(), "static-recovery-process"),
        };
        let mut config = NativeExecutionConfig::for_current_executable().unwrap();
        config.max_workers = 1;
        config.store_root = directory.path().join("session");
        config.worker_capabilities = capabilities();
        let session = NativeProgramSession::new(config).unwrap();
        let mut store = session.object_store();
        program.write_to(&mut store).unwrap();
        let task = session
            .submit_graph(vec![(
                program.request.clone(),
                analysis_submission(
                    session.scope_id(),
                    session.pool_id(),
                    &host,
                    &program.request,
                    Some(&resource_plan),
                    resource_plan.request.clone(),
                ),
            )])
            .unwrap()
            .pop()
            .unwrap();
        let solved = wait_for_native_task(&task).await;
        let response = success_response(&solved);
        let output = import_analysis_output_set(&store, response, &host).unwrap();
        let result = &output.manifest.outputs[0];
        let source_result = import_analysis_objects(
            &store,
            &result.root,
            AnalysisNumericalObjectKind::SolverResult,
            result.logical_identity,
            result.payload_digest,
            &host.artifact_access,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let mut original =
            StructuralLinearWorkloadResult::from_canonical_bytes(&source_result.payload()).unwrap();
        let fields = prepare_structural_field_recovery(StructuralFieldRecoveryPreparation {
            source_host: host.clone(),
            source_invocation: inputs
                .iter()
                .find(|i| i.manifest.kind == AnalysisNumericalObjectKind::SolverInvocation)
                .unwrap()
                .clone(),
            source_result,
            mesh: inputs
                .iter()
                .find(|i| i.manifest.kind == AnalysisNumericalObjectKind::Mesh)
                .cloned(),
            program_revision: revision(),
            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: ObjectInventoryLimits::default(),
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        })
        .unwrap();
        assert_eq!(
            fields.resource_plan.input.accuracy_ceiling,
            resource_plan.input.accuracy_ceiling
        );
        fields.program.write_to(&mut store).unwrap();
        let task = session
            .submit_graph(vec![(
                fields.program.request.clone(),
                analysis_submission(
                    session.scope_id(),
                    session.pool_id(),
                    &fields.host,
                    &fields.program.request,
                    Some(&fields.resource_plan),
                    fields.resource_plan.request.clone(),
                ),
            )])
            .unwrap()
            .pop()
            .unwrap();
        let published = wait_for_native_task(&task).await;
        let response = success_response(&published);
        // The execution owner retains disk lifetime through its existing source contract.
        // This guard owns no decoded model or field values.
        let _retained_source = session.object_source();
        let stored_root = store.root().to_path_buf();
        drop(task);
        drop(session);
        drop(store);
        let store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
            stored_root,
            ObjectInventoryLimits::default().max_object_bytes,
        )
        .unwrap();
        // Reopen and authenticate the retained output after the producing session is gone.
        let imported = import_structural_recovered_fields(
            &store,
            response,
            &fields,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let mut expected = runmat_analysis_fea::post::fields::take_structural_observable_fields(
            &mut original.run.fields,
        )
        .unwrap();
        expected.extend(
            runmat_analysis_fea::post::recover_constraint_group_observable_fields(
                &original.constraint_result,
            )
            .unwrap(),
        );
        assert_eq!(imported.manifest.fields.len(), expected.len());
        assert!(
            expected.len() > fields.request.plan.requirements.len(),
            "complete retained field inventory exceeds observation-only shape"
        );
        for entry in &imported.manifest.fields {
            let partitions = entry
                .partitions
                .iter()
                .map(|descriptor| {
                    let partition = import_analysis_partition(
                        &store,
                        AnalysisPartitionImport {
                            root: &imported.output.root,
                            expected_kind: AnalysisNumericalObjectKind::Field,
                            expected_logical_identity: imported.output.logical_identity,
                            expected_payload_digest: imported.output.payload_digest,
                            partition_identity: descriptor.partition.digest(),
                            access: &fields.host.artifact_access,
                            limits: ObjectInventoryLimits::default(),
                        },
                    )
                    .unwrap();
                    RecoveredFieldPartition::from_canonical_bytes(&partition.payload()).unwrap()
                })
                .collect();
            let actual = reconstruct_recovered_field(
                entry.field,
                entry.definition.clone(),
                entry.sample,
                partitions,
                entry
                    .definition
                    .stored_entities(imported.manifest.entity_counts),
            )
            .unwrap();
            let original = expected
                .iter()
                .find(|f| f.definition().field_id() == entry.definition.field_id())
                .unwrap();
            assert_eq!(
                &actual, original,
                "original field preserved on branch {branch}"
            );
        }
        eprintln!("static recovery scheduled branch {branch} PASS: original field values, source identity and unchanged resource ceiling");
    }
}
