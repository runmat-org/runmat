use std::collections::BTreeSet;

use runmat_execution::handle::OutputContract;
use runmat_execution::identity::{ArtifactId, WorkerId};
use runmat_execution::resource::{ResourceInventory, ResourceRequest};
use runmat_execution::state::{PoolState, TaskState};
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::{Digest, ExecutionScopeId, PoolId, TaskId};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_runner::driver::{DriverAction, DriverCommand, DriverConfig};
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{AttemptReport, Driver, PoolSpec, TaskSubmission, WorkerSpec};

use crate::tests::{fixture, MemoryCache};
use crate::{
    import_result_publication, prepare_result_publication, prepare_stage_objects,
    MeshingArtifactAccess,
};

#[test]
fn publication_binds_authorization_without_changing_logical_identity() {
    let objects = fixture(vec![vec![1; 200]], 1024);
    let first = prepare_result_publication(
        objects.clone(),
        access("run-one", 1),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let second = prepare_result_publication(
        objects,
        access("run-two", 2),
        ObjectInventoryLimits::default(),
    )
    .unwrap();

    assert_eq!(
        first.root_output().logical_digest().unwrap(),
        second.root_output().logical_digest().unwrap()
    );
    assert_ne!(first.result_objects()[0].id, second.result_objects()[0].id);
    assert_ne!(
        first.result_objects()[0].authorization_scope,
        second.result_objects()[0].authorization_scope
    );
    assert_ne!(
        first.result_objects()[0].encryption_context,
        second.result_objects()[0].encryption_context
    );
}

#[test]
fn published_root_is_reauthorized_and_revalidated_on_import() {
    let publication = publication();
    let mut cache = MemoryCache::default();
    cache.insert_all(&publication.stage_objects().objects);
    let run_access = access("run", 7);
    let run_root = match publication.root_output() {
        runmat_execution::value::ValuePayload::Object(reference) => reference.as_ref(),
        _ => panic!("publication root must be an object"),
    };
    let imported = import_result_publication(
        &cache,
        run_root,
        run_access,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(imported, publication);

    let error = import_result_publication(
        &cache,
        run_root,
        access("other-run", 7),
        ObjectInventoryLimits::default(),
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("outside meshing artifact authority"));
}

#[test]
fn incomplete_object_set_cannot_become_an_attempt_success() {
    let mut objects = fixture(vec![vec![1; 200], vec![2; 300]], 512);
    let chunk = objects
        .objects
        .iter()
        .position(|object| object.descriptor.logical_name.contains("/chunks/"))
        .unwrap();
    objects.objects.remove(chunk);

    let error =
        prepare_result_publication(objects, access("run", 1), ObjectInventoryLimits::default())
            .unwrap_err();

    assert!(error.to_string().contains("unavailable"));
}

#[test]
fn diagnostic_only_manifest_cannot_satisfy_a_dependency() {
    let mut objects = fixture(vec![vec![1; 200]], 1024);
    objects.manifest.disposition =
        runmat_meshing_core::MeshingManifestDispositionV2::DiagnosticOnly;
    let manifest = objects.manifest.clone();
    let identity = objects.result_identity.clone();
    let chunks = manifest
        .chunks
        .iter()
        .map(|descriptor| {
            let object = objects
                .objects
                .iter()
                .find(|object| object.descriptor.digest.bytes() == descriptor.digest.bytes())
                .unwrap();
            runmat_meshing_core::EncodedMeshingChunkV2 {
                descriptor: descriptor.clone(),
                bytes: object.bytes.clone(),
            }
        })
        .collect();
    objects = prepare_stage_objects(identity, manifest, chunks, ObjectInventoryLimits::default())
        .unwrap();

    let error =
        prepare_result_publication(objects, access("run", 1), ObjectInventoryLimits::default())
            .unwrap_err();

    assert!(error.to_string().contains("diagnostic-only"));
}

#[test]
fn runner_commits_root_and_complete_inventory_atomically() {
    let publication = publication();
    let (mut driver, request, task_id) = scheduled_task();
    assert!(driver.snapshot().tasks[&task_id].committed.is_none());

    driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::Succeeded {
                result: publication.attempt_success(),
            },
        )))
        .unwrap();

    let snapshot = driver.snapshot();
    let task = &snapshot.tasks[&task_id];
    assert_eq!(task.state, TaskState::Succeeded);
    let committed = task.committed.as_ref().unwrap();
    assert_eq!(committed.outputs, vec![publication.root_output().clone()]);
    assert_eq!(committed.result_objects, publication.result_objects());
}

#[test]
fn stale_and_duplicate_publications_are_not_visible_and_are_collected() {
    let publication = publication();
    let (mut driver, request, task_id) = scheduled_task();
    let mut stale = BackendReport::for_request(
        &request,
        AttemptReport::Succeeded {
            result: publication.attempt_success(),
        },
    );
    stale.driver_fence = stale.driver_fence.saturating_sub(1);
    let stale_actions = driver.handle(DriverCommand::BackendReport(stale)).unwrap();
    assert!(driver.snapshot().tasks[&task_id].committed.is_none());
    assert_gc_contains(&stale_actions, publication.result_objects());

    let success = BackendReport::for_request(
        &request,
        AttemptReport::Succeeded {
            result: publication.attempt_success(),
        },
    );
    driver
        .handle(DriverCommand::BackendReport(success.clone()))
        .unwrap();
    let committed = driver.snapshot().tasks[&task_id].committed.clone();
    let duplicate_actions = driver
        .handle(DriverCommand::BackendReport(success))
        .unwrap();
    assert_eq!(driver.snapshot().tasks[&task_id].committed, committed);
    assert_gc_contains(&duplicate_actions, publication.result_objects());
}

fn publication() -> crate::PreparedMeshingResultPublication {
    prepare_result_publication(
        fixture(vec![vec![1; 200], vec![2; 300]], 512),
        access("run", 7),
        ObjectInventoryLimits::default(),
    )
    .unwrap()
}

fn access(scope: &str, seed: u8) -> MeshingArtifactAccess {
    MeshingArtifactAccess {
        authorization_scope: scope.into(),
        encryption_context: Digest::sha256([seed]),
    }
}

fn scheduled_task() -> (Driver, runmat_execution_runner::AttemptRequest, TaskId) {
    let scope = ExecutionScopeId::derive(&[b"meshing-scope"]);
    let pool = PoolId::derive(&[b"meshing-pool"]);
    let worker = WorkerId::derive(&[b"meshing-worker"]);
    let inventory = ResourceInventory {
        cpu_millicores: 2_000,
        memory_bytes: 4 * 1024 * 1024,
        scratch_bytes: 4 * 1024 * 1024,
        accelerators: Vec::new(),
        capabilities: BTreeSet::new(),
    };
    let mut driver = Driver::new(DriverConfig::default(), 11).unwrap();
    driver
        .handle(DriverCommand::RegisterScope {
            scope_id: scope,
            parent: None,
        })
        .unwrap();
    driver
        .handle(DriverCommand::CreatePool(PoolSpec {
            id: pool,
            min_workers: 1,
            max_workers: 1,
            max_in_flight: 1,
            resource_limit: inventory.clone(),
        }))
        .unwrap();
    driver
        .handle(DriverCommand::SetPoolState {
            pool_id: pool,
            state: PoolState::Ready,
        })
        .unwrap();
    driver
        .handle(DriverCommand::RegisterWorker(WorkerSpec {
            id: worker,
            pool_id: pool,
            resources: inventory,
        }))
        .unwrap();
    let task_id = TaskId::derive(&[b"meshing-publication"]);
    let actions = driver
        .handle(DriverCommand::Submit(Box::new(TaskSubmission {
            request: TaskRequest {
                id: task_id,
                scope_id: scope,
                pool_id: pool,
                program_artifact_id: ArtifactId::derive(&[b"meshing-host"]),
                callable: Callable {
                    owner_identity: "runmat.meshing".into(),
                    qualified_name: "publish-stage".into(),
                    entrypoint_digest: Digest::sha256(b"publish-stage"),
                },
                inputs: Vec::new(),
                outputs: OutputContract {
                    requested_outputs: 1,
                },
                resources: ResourceRequest {
                    cpu_millicores: 1_000,
                    memory_bytes: 1024,
                    scratch_bytes: 1024,
                    max_wall_millis: 10_000,
                    max_artifact_bytes: 8 * 1024 * 1024,
                    max_egress_bytes: 0,
                    max_relay_bytes: 0,
                    accelerators: Vec::new(),
                    required_capabilities: BTreeSet::new(),
                },
                retry: RetryPolicy::Never,
                deadline_unix_millis: None,
            },
            dependencies: BTreeSet::new(),
            priority: 0,
        })))
        .unwrap();
    let request = actions
        .into_iter()
        .find_map(|action| match action {
            DriverAction::Launch(request) => Some(request),
            _ => None,
        })
        .unwrap();
    (driver, request, task_id)
}

fn assert_gc_contains(actions: &[DriverAction], expected: &[runmat_execution::value::ValueRef]) {
    assert!(actions.iter().any(|action| matches!(
        action,
        DriverAction::GarbageCollectResults { objects, .. } if objects == expected
    )));
}
