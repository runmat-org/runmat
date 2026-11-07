use super::*;
use crate::remote::{checkpoint::RemoteDriverCheckpoint, recovery};
use runmat_execution::identity::{PoolId, TaskId};
use runmat_execution::value::ValueRefKind;
use runmat_execution::ExecutionScopeId;
use runmat_execution_runner::PoolSpec;

fn latest_catalog(
    control: &FakeControl,
    key: &RunKeyMaterial,
    run: &str,
) -> ProgramExecutionObjectLocation {
    let state = control.state.lock().unwrap();
    let checkpoint = state.stored.last().unwrap();
    assert_eq!(checkpoint.kind, DriverArtifactKind::Checkpoint);
    RemoteDriverCheckpoint::from_canonical_bytes(
        &open_object(
            key,
            &checkpoint.ciphertext,
            run,
            EncryptionPurpose::Checkpoint,
        )
        .unwrap(),
    )
    .unwrap()
    .closure_catalog
    .unwrap()
}

#[tokio::test]
async fn mixed_recovery_uses_exact_encryption_purposes_and_reencrypts_alias_changes() {
    let run = "mixed-recovery";
    let key = RunKeyMaterial::from_entropy([79; 32]).unwrap();
    let authority = DriverAuthority {
        server_url: "https://control.invalid".into(),
        run_id: run.into(),
        org_id: super::test_org_id(),
        project_id: "project".into(),
        allocation_lease_id: "allocation".into(),
        driver_lease_id: super::test_driver_lease_id(7),
        fencing_token: 7,
        credential: "credential".into(),
    };
    let revision = ProgramRevision::new(
        Digest::sha256("graph"),
        Digest::sha256("source"),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap();
    let effectful_recipe = recipe(revision.clone());
    let effectful_request = ProgramExecutionRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        artifact: ProgramArtifact::materialize(
            &effectful_recipe,
            ExecutableForm::InterpreterBytecodeV1,
            vec![1],
        )
        .unwrap(),
        recipe: effectful_recipe,
        callable: runmat_execution::ProgramCallable::semantic(
            runmat_execution::ProgramFunctionId(0),
            None,
        ),
        context: runmat_execution::ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        arguments: Vec::new(),
        requested_outputs: 1,
    };
    effectful_request.validate().unwrap();
    let mut recipe = recipe(revision);
    recipe.entrypoint = "analysis_workload".into();
    recipe.execution_mode = "recovery-publication-test".into();
    let request = ProgramExecutionRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        artifact: ProgramArtifact::materialize(&recipe, ExecutableForm::AnalysisWorkload, vec![1])
            .unwrap(),
        recipe,
        callable: runmat_execution::ProgramCallable::semantic(
            runmat_execution::ProgramFunctionId(0),
            Some("analysis_workload".into()),
        ),
        context: runmat_execution::ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        arguments: Vec::new(),
        requested_outputs: 1,
    };
    let control = FakeControl {
        bootstrap: unused_bootstrap(),
        state: Mutex::new(FakeState::default()),
    };
    assert!(recovery::RemoteRecoveryPublisher::new(
        &control,
        &key,
        &authority,
        "revision",
        &effectful_request,
        3,
        None,
    )
    .is_err());
    let root = result_object(run, b"manifest");
    let mut input = result_object(run, b"input bytes");
    input.reference.kind = ValueRefKind::DriverObject;
    let mut publisher = recovery::RemoteRecoveryPublisher::new(
        &control, &key, &authority, "revision", &request, 3, None,
    )
    .unwrap();
    publisher
        .publish_objects(root.reference.clone(), vec![root.clone(), input.clone()])
        .await
        .unwrap();
    let catalog = latest_catalog(&control, &key, run);
    let restored = recovery::restore(&control, &key, &authority, &catalog)
        .await
        .unwrap();
    assert!(restored.objects.iter().any(|object| object == &input));
    let input_location = restored
        .locations
        .iter()
        .find(|location| location.reference == input.reference)
        .unwrap();
    let input_ciphertext =
        control.state.lock().unwrap().downloads[&input_location.artifact_id].clone();
    assert_eq!(
        open_object(&key, &input_ciphertext, run, EncryptionPurpose::Input).unwrap(),
        input.bytes
    );
    assert!(open_object(&key, &input_ciphertext, run, EncryptionPurpose::Result).is_err());
    assert_eq!(control.state.lock().unwrap().stored.len(), 4);

    // Same content and same typed reference reuses its original encrypted envelope.
    publisher
        .publish_objects(root.reference.clone(), vec![input.clone(), root.clone()])
        .await
        .unwrap();
    assert_eq!(control.state.lock().unwrap().stored.len(), 6);
    input.reference.kind = ValueRefKind::ResultObject;
    publisher
        .publish_objects(root.reference.clone(), vec![root.clone(), input.clone()])
        .await
        .unwrap();
    assert_eq!(control.state.lock().unwrap().stored.len(), 9);
    let next_catalog = latest_catalog(&control, &key, run);
    let next = recovery::restore(&control, &key, &authority, &next_catalog)
        .await
        .unwrap();
    let new_location = next
        .locations
        .iter()
        .find(|location| location.reference == input.reference)
        .unwrap();
    assert_ne!(new_location.artifact_id, input_location.artifact_id);
    assert!(next.objects.iter().any(|object| object == &input));

    // A valid input envelope substituted at the result alias must fail purpose authentication.
    control
        .state
        .lock()
        .unwrap()
        .downloads
        .insert(new_location.artifact_id.clone(), input_ciphertext);
    assert!(recovery::restore(&control, &key, &authority, &next_catalog)
        .await
        .is_err());
    assert_eq!(publisher.checkpoint_sequence(), 6);
}

#[tokio::test]
async fn scheduler_and_physical_roots_survive_one_serialized_publication_sequence() {
    use crate::remote::pool_tests::{executable_bundle, inventory, submission};

    let run = "combined-recovery";
    let key = RunKeyMaterial::from_entropy([80; 32]).unwrap();
    let authority = DriverAuthority {
        server_url: "https://control.invalid".into(),
        run_id: run.into(),
        org_id: super::test_org_id(),
        project_id: "project".into(),
        allocation_lease_id: "allocation".into(),
        driver_lease_id: super::test_driver_lease_id(7),
        fencing_token: 7,
        credential: "credential".into(),
    };
    let revision = ProgramRevision::new(
        Digest::sha256("combined-graph"),
        Digest::sha256("combined-source"),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap();
    let mut recipe = recipe(revision);
    recipe.entrypoint = "analysis_workload".into();
    recipe.execution_mode = "recovery-publication-test".into();
    let request = ProgramExecutionRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        artifact: ProgramArtifact::materialize(&recipe, ExecutableForm::AnalysisWorkload, vec![1])
            .unwrap(),
        recipe,
        callable: runmat_execution::ProgramCallable::semantic(
            runmat_execution::ProgramFunctionId(0),
            Some("analysis_workload".into()),
        ),
        context: runmat_execution::ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        arguments: Vec::new(),
        requested_outputs: 1,
    };
    let control = FakeControl {
        bootstrap: unused_bootstrap(),
        state: Mutex::new(FakeState::default()),
    };
    let (program, artifact, bundle) = executable_bundle().await;
    let scope = ExecutionScopeId::derive(&[run.as_bytes(), b"remote-root"]);
    let pool_id = PoolId::derive(&[run.as_bytes(), b"remote-pool"]);
    let spec = PoolSpec {
        id: pool_id,
        min_workers: 0,
        max_workers: 1,
        max_in_flight: 1,
        resource_limit: inventory(1000),
    };
    let pool = crate::remote::RemotePoolDriver::new_with_value_scope(
        scope,
        spec.clone(),
        authority.fencing_token,
        bundle.clone(),
        run,
    )
    .unwrap();
    let task = submission(
        scope,
        pool_id,
        TaskId::derive(&[b"combined-checkpoint-task"]),
        artifact,
    );
    let _waiter = pool.submit(task.clone(), program.clone()).unwrap();
    let mut publisher = recovery::RemoteRecoveryPublisher::new(
        &control, &key, &authority, "revision", &request, 3, None,
    )
    .unwrap();
    publisher
        .bind_pool(Arc::clone(&pool), None, BTreeMap::new())
        .unwrap();
    let physical = result_object(run, b"physical state");
    publisher
        .publish_objects(physical.reference.clone(), vec![physical])
        .await
        .unwrap();
    let current_watermark = pool
        .capture_checkpoint(5)
        .unwrap()
        .scheduler_event_watermark();
    let stored_before_rejection = control.state.lock().unwrap().stored.len();
    let unavailable_event = publisher
        .publish_pool_through(current_watermark + 1)
        .await
        .expect_err("a checkpoint cannot claim an event absent from its scheduler snapshot");
    assert!(unavailable_event
        .to_string()
        .contains("did not include its committed scheduler event"));
    assert_eq!(publisher.checkpoint_sequence(), 4);
    assert_eq!(
        control.state.lock().unwrap().stored.len(),
        stored_before_rejection
    );
    let published_watermark = publisher.publish_pool().await.unwrap();
    assert_eq!(published_watermark, current_watermark);
    assert_eq!(publisher.checkpoint_sequence(), 5);
    let checkpoint = {
        let state = control.state.lock().unwrap();
        RemoteDriverCheckpoint::from_canonical_bytes(
            &open_object(
                &key,
                &state.stored.last().unwrap().ciphertext,
                run,
                EncryptionPurpose::Checkpoint,
            )
            .unwrap(),
        )
        .unwrap()
    };
    assert!(checkpoint.closure_catalog.is_some());
    let pool_catalog = checkpoint.pool_catalog.expect("scheduler root retained");
    let staged = crate::remote::pool_checkpoint::restore(&control, &key, &authority, &pool_catalog)
        .await
        .unwrap();
    let restored = staged
        .into_pool(
            crate::remote::RemotePoolRestoreAuthority {
                scope_id: scope,
                pool: spec,
                value_scope: run.into(),
                driver_fence: 8,
            },
            Arc::from(bundle),
        )
        .unwrap();
    let _reattached = restored.submit(task, program).unwrap();
    assert_eq!(
        restored
            .capture_checkpoint(6)
            .unwrap()
            .object_references()
            .unwrap()
            .len(),
        0
    );
}
