use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use runmat_execution_transport_native::control::{
    DriverAuthority, DriverControlPlane, ResourceRequest as AllocationResources,
};

use super::pool_execution::RemotePoolExecutionOutcome;
use crate::{NativeExecutionError, NativeExecutionResult};

const WORKER_RELEASE_TIMEOUT: Duration = Duration::from_secs(10);
const ALLOCATION_RECONCILE_TIMEOUT: Duration = Duration::from_secs(10);

pub(super) async fn reconcile_ambiguous_generation(
    control: &dyn DriverControlPlane,
    authority: &DriverAuthority,
    desired_workers: u32,
    resources: &AllocationResources,
    generation: &AtomicU64,
) -> Result<(), String> {
    let observed = tokio::time::timeout(
        ALLOCATION_RECONCILE_TIMEOUT,
        control.list_workers(authority),
    )
    .await;
    match observed {
        Ok(Ok(worker_pool))
            if worker_pool.generation > 0
                && worker_pool.desired_workers == desired_workers
                && worker_pool.resources == *resources =>
        {
            generation.store(worker_pool.generation, Ordering::Release);
            Ok(())
        }
        Ok(Ok(_)) => Err(
            "the Server reported a substituted pool intent after the control response was lost"
                .into(),
        ),
        Ok(Err(error)) => Err(format!(
            "worker allocation ownership could not be observed after the control response was lost: {error}"
        )),
        Err(_) => Err(
            "worker allocation ownership observation timed out after the control response was lost"
                .into(),
        ),
    }
}

pub(super) async fn reconcile_or_indeterminate(
    control: &dyn DriverControlPlane,
    authority: &DriverAuthority,
    desired_workers: u32,
    resources: &AllocationResources,
    generation: &AtomicU64,
    primary: impl std::fmt::Display,
) -> Result<(), RemotePoolExecutionOutcome> {
    reconcile_ambiguous_generation(control, authority, desired_workers, resources, generation)
        .await
        .map_err(|ownership| {
            RemotePoolExecutionOutcome::Indeterminate(format!("{primary}; {ownership}"))
        })
}

pub(super) async fn resolve_failed_initial(
    control: &dyn DriverControlPlane,
    authority: &DriverAuthority,
    desired_workers: u32,
    requested_resources: &AllocationResources,
    primary: String,
    cancelled: bool,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    let observed = tokio::time::timeout(
        ALLOCATION_RECONCILE_TIMEOUT,
        control.list_workers(authority),
    )
    .await;
    match observed {
        Ok(Ok(worker_pool)) if worker_pool.generation > 0 && worker_pool.desired_workers > 0 => {
            let diagnostic = if worker_pool.desired_workers == desired_workers
                && worker_pool.resources == *requested_resources
            {
                format!("initial worker allocation failed: {primary}")
            } else {
                format!(
                    "initial worker allocation failed and the observed allocation intent was substituted: {primary}"
                )
            };
            let outcome = if cancelled {
                Ok(RemotePoolExecutionOutcome::Cancelled)
            } else {
                Err(protocol(diagnostic))
            };
            finalize(
                control,
                authority,
                &worker_pool.resources,
                worker_pool.generation,
                outcome,
            )
            .await
        }
        Ok(Ok(worker_pool)) if worker_pool.desired_workers == 0 => {
            let outcome = if cancelled {
                Ok(RemotePoolExecutionOutcome::Cancelled)
            } else {
                Err(protocol(primary))
            };
            finalize(
                control,
                authority,
                &worker_pool.resources,
                worker_pool.generation,
                outcome,
            )
            .await
        }
        Ok(Ok(_)) | Ok(Err(_)) | Err(_) => Ok(RemotePoolExecutionOutcome::Indeterminate(format!(
            "initial worker allocation failed and allocation ownership could not be proven: {primary}"
        ))),
    }
}

pub(super) async fn finalize(
    control: &dyn DriverControlPlane,
    authority: &DriverAuthority,
    resources: &AllocationResources,
    generation: u64,
    outcome: NativeExecutionResult<RemotePoolExecutionOutcome>,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    let released = tokio::time::timeout(
        WORKER_RELEASE_TIMEOUT,
        control.resize_workers(authority, generation, 0, resources.clone()),
    )
    .await;
    match released {
        Ok(Ok(released))
            if released.desired_workers == 0
                && released.generation == generation.saturating_add(1)
                && released.resources == *resources
                && released.workers.is_empty() =>
        {
            outcome
        }
        Ok(Ok(_)) => Ok(indeterminate(
            outcome,
            "the Server substituted worker-release intent",
        )),
        Ok(Err(error)) => Ok(indeterminate(
            outcome,
            &format!("worker allocation release was not acknowledged: {error}"),
        )),
        Err(_) => Ok(indeterminate(
            outcome,
            "worker allocation release timed out",
        )),
    }
}

fn indeterminate(
    outcome: NativeExecutionResult<RemotePoolExecutionOutcome>,
    release: &str,
) -> RemotePoolExecutionOutcome {
    let primary = match outcome {
        Ok(RemotePoolExecutionOutcome::Indeterminate(message)) => message,
        Ok(RemotePoolExecutionOutcome::Completed(_)) => "remote execution completed".into(),
        Ok(RemotePoolExecutionOutcome::Cancelled) => "remote execution was cancelled".into(),
        Err(error) => format!("remote execution failed: {error}"),
    };
    RemotePoolExecutionOutcome::Indeterminate(format!(
        "{primary}; release ownership is indeterminate because {release}"
    ))
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use runmat_execution_transport_native::control::{
        DriverArtifactDownload, DriverBootstrap, DriverHeartbeat, DriverRunTransition,
        DriverWorkerPool, StoreDriverArtifact, StoredDriverArtifact,
    };
    use runmat_execution_transport_native::{TransportError, TransportResult};

    use super::*;

    struct ReleaseControl {
        fail_release: bool,
        expected_release_generation: u64,
        listed: Option<DriverWorkerPool>,
        releases: Arc<Mutex<Vec<u64>>>,
    }

    #[async_trait]
    impl DriverControlPlane for ReleaseControl {
        async fn bootstrap(
            &self,
            _authority: &DriverAuthority,
        ) -> TransportResult<DriverBootstrap> {
            unreachable!("release fixture does not bootstrap")
        }

        async fn heartbeat(
            &self,
            _authority: &DriverAuthority,
            _ttl_seconds: u64,
        ) -> TransportResult<DriverHeartbeat> {
            unreachable!("release fixture does not heartbeat")
        }

        async fn record_usage(
            &self,
            _authority: &DriverAuthority,
            _source_sequence: u64,
        ) -> TransportResult<bool> {
            unreachable!("release fixture does not report usage")
        }

        async fn download(&self, _artifact: &DriverArtifactDownload) -> TransportResult<Vec<u8>> {
            unreachable!("release fixture does not download")
        }

        async fn store_artifact(
            &self,
            _authority: &DriverAuthority,
            _artifact: StoreDriverArtifact<'_>,
        ) -> TransportResult<StoredDriverArtifact> {
            unreachable!("release fixture does not store artifacts")
        }

        async fn transition(
            &self,
            _authority: &DriverAuthority,
            _transition: DriverRunTransition,
        ) -> TransportResult<()> {
            unreachable!("release fixture does not transition")
        }

        async fn resize_workers(
            &self,
            _authority: &DriverAuthority,
            expected_generation: u64,
            desired_workers: u32,
            resources: AllocationResources,
        ) -> TransportResult<DriverWorkerPool> {
            assert_eq!(expected_generation, self.expected_release_generation);
            assert_eq!(desired_workers, 0);
            self.releases.lock().unwrap().push(expected_generation);
            if self.fail_release {
                Err(TransportError::Unavailable("fixture release failed".into()))
            } else {
                Ok(DriverWorkerPool {
                    generation: expected_generation + 1,
                    desired_workers: 0,
                    resources,
                    workers: Vec::new(),
                })
            }
        }

        async fn list_workers(
            &self,
            _authority: &DriverAuthority,
        ) -> TransportResult<DriverWorkerPool> {
            self.listed
                .clone()
                .ok_or_else(|| TransportError::Unavailable("fixture has no observation".into()))
        }
    }

    fn authority() -> DriverAuthority {
        DriverAuthority {
            server_url: "https://control.invalid".into(),
            run_id: "run-release-fixture".into(),
            org_id: crate::remote::test_ids::test_org_id(),
            project_id: "project-release-fixture".into(),
            allocation_lease_id: "allocation-release-fixture".into(),
            driver_lease_id: crate::remote::test_ids::test_driver_lease_id(1),
            fencing_token: 1,
            credential: "credential-release-fixture".into(),
        }
    }

    fn resources() -> AllocationResources {
        AllocationResources {
            cpu_millicores: 1_000,
            memory_bytes: 1024,
            scratch_bytes: 1024,
            accelerators: Vec::new(),
            maximum_wall_millis: 60_000,
        }
    }

    #[tokio::test]
    async fn terminal_outcome_is_returned_only_after_worker_release_is_proven() {
        let outcome = finalize(
            &ReleaseControl {
                fail_release: false,
                expected_release_generation: 7,
                listed: None,
                releases: Arc::new(Mutex::new(Vec::new())),
            },
            &authority(),
            &resources(),
            7,
            Ok(RemotePoolExecutionOutcome::Cancelled),
        )
        .await
        .unwrap();

        assert!(matches!(outcome, RemotePoolExecutionOutcome::Cancelled));
    }

    #[tokio::test]
    async fn unproven_worker_release_makes_the_outcome_indeterminate() {
        let outcome = finalize(
            &ReleaseControl {
                fail_release: true,
                expected_release_generation: 7,
                listed: None,
                releases: Arc::new(Mutex::new(Vec::new())),
            },
            &authority(),
            &resources(),
            7,
            Err(NativeExecutionError::Protocol(
                "primary execution failure".into(),
            )),
        )
        .await
        .unwrap();

        let RemotePoolExecutionOutcome::Indeterminate(message) = outcome else {
            panic!("release failure must be indeterminate")
        };
        assert!(message.contains("primary execution failure"));
        assert!(message.contains("release was not acknowledged"));
    }

    #[tokio::test]
    async fn response_lost_after_commit_releases_the_observed_generation() {
        let resources = resources();
        let releases = Arc::new(Mutex::new(Vec::new()));
        let control = ReleaseControl {
            fail_release: false,
            expected_release_generation: 9,
            listed: Some(DriverWorkerPool {
                generation: 9,
                desired_workers: 2,
                resources: resources.clone(),
                workers: Vec::new(),
            }),
            releases: Arc::clone(&releases),
        };
        let generation = AtomicU64::new(8);

        reconcile_ambiguous_generation(&control, &authority(), 2, &resources, &generation)
            .await
            .unwrap();
        let outcome = finalize(
            &control,
            &authority(),
            &resources,
            generation.load(Ordering::Acquire),
            Ok(RemotePoolExecutionOutcome::Cancelled),
        )
        .await
        .unwrap();

        assert!(matches!(outcome, RemotePoolExecutionOutcome::Cancelled));
        assert_eq!(releases.lock().unwrap().as_slice(), &[9]);
    }
}
