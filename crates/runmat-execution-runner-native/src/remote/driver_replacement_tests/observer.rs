//! The test process is the original OS parent of the real worker child. It
//! observes with the production monotonic clock, signs with the production
//! identity primitive, and fsyncs each receipt before the control port sees it.

use super::*;
use runmat_execution::security::{
    EndpointTrustPolicy, NodeClockEndV1, NodeClockOwnerV1, NodeClockStartRequestV1,
    NodeClockStartV1, WorkerAttemptJournalEntryV1, MAX_CLOCK_OBSERVATION_WIRE_BYTES,
};
use runmat_execution_transport_native::control::{
    ClockDriverHeartbeatRequestV1, ClockDriverHeartbeatResponseV1, ClockEndDemandV1,
    ClockObservationReleaseManifestV1,
};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};

const NODE_SECRET: [u8; 32] = [47; 32];

#[derive(Default)]
struct ClockState {
    reserved: BTreeSet<[u8; 16]>,
    starts: BTreeMap<[u8; 16], NodeClockStartV1>,
    ends: BTreeMap<[u8; 16], NodeClockEndV1>,
    demands: BTreeMap<[u8; 16], ClockEndDemandV1>,
    latest_release: Option<ClockObservationReleaseManifestV1>,
    completed: Vec<(NodeClockStartV1, NodeClockEndV1)>,
    predispatch_checkpoint_ticks: Option<u64>,
}

pub(super) struct ClockControl {
    root: PathBuf,
    journal: PathBuf,
    domain_nonce: [u8; 16],
    fingerprint: String,
    parents: BTreeMap<u64, super::super::config::DriverClockParent>,
    state: Mutex<ClockState>,
}

pub(super) struct OwnerGuard(tokio::task::JoinHandle<()>);

impl Drop for OwnerGuard {
    fn drop(&mut self) {
        self.0.abort();
    }
}

impl ClockControl {
    pub(super) fn new(root: &Path, journal: PathBuf) -> Arc<Self> {
        let signer = EndpointIdentitySigner::from_secret(NODE_SECRET).unwrap();
        let domain_nonce = [61; 16];
        let mut parents = BTreeMap::new();
        for fence in [1_u64, 2, 3] {
            let request_directory = root.join(format!("clock-request-{fence}"));
            let receipt_directory = root.join(format!("clock-receipt-{fence}"));
            std::fs::create_dir(&request_directory).unwrap();
            std::fs::create_dir(&receipt_directory).unwrap();
            parents.insert(
                fence,
                super::super::config::DriverClockParent {
                    allocation_fence: fence,
                    process_launch_nonce: [u8::try_from(70 + fence).unwrap(); 16],
                    clock_domain_nonce: domain_nonce,
                    node_fingerprint: signer.fingerprint(),
                    request_directory,
                    receipt_directory,
                },
            );
        }
        Arc::new(Self {
            root: root.into(),
            journal,
            domain_nonce,
            fingerprint: signer.fingerprint(),
            parents,
            state: Mutex::new(ClockState::default()),
        })
    }

    pub(super) fn parent(&self, fence: u64) -> super::super::config::DriverClockParent {
        self.parents[&fence].clone()
    }

    pub(super) fn spawn_owner(self: &Arc<Self>) -> OwnerGuard {
        let owner = Arc::clone(self);
        OwnerGuard(tokio::spawn(async move {
            let mut tick = tokio::time::interval(std::time::Duration::from_millis(5));
            loop {
                tick.tick().await;
                owner.observe_once();
            }
        }))
    }

    fn policy(&self) -> EndpointTrustPolicy {
        EndpointTrustPolicy {
            permitted_tiers: Default::default(),
            trusted_identity_fingerprints: [self.fingerprint.clone()].into_iter().collect(),
            allowed_attestation_classes: Default::default(),
            require_pinned_identity: true,
            now_unix_millis: 0,
            maximum_clock_skew_millis: 0,
        }
    }

    fn observe_once(&self) {
        let signer = EndpointIdentitySigner::from_secret(NODE_SECRET).unwrap();
        for parent in self.parents.values() {
            for entry in std::fs::read_dir(&parent.request_directory).unwrap() {
                let entry = entry.unwrap();
                if entry.file_name() == ".pending" {
                    continue;
                }
                assert!(entry.file_type().unwrap().is_file());
                let request: NodeClockStartRequestV1 =
                    serde_json::from_slice(&std::fs::read(entry.path()).unwrap()).unwrap();
                request.validate_shape().unwrap();
                assert_eq!(request.run_id, RUN_ID);
                assert_eq!(request.driver_allocation_lease_id, DRIVER_ALLOCATION);
                assert_eq!(request.driver_allocation_fence, parent.allocation_fence);
                assert_eq!(
                    request.driver_process_launch_nonce,
                    parent.process_launch_nonce
                );
                assert_eq!(request.clock_domain_nonce, self.domain_nonce);
                assert!(self
                    .state
                    .lock()
                    .unwrap()
                    .reserved
                    .contains(&request.observation_id));
                let reading = runmat_process_host::clock::observe_monotonic().unwrap();
                let mut start = NodeClockStartV1 {
                    schema_version: 1,
                    owner: NodeClockOwnerV1 {
                        org_id: crate::remote::test_ids::TEST_ORG_ID.into(),
                        cluster_id: "cluster-clock".into(),
                        node_id: "node-clock".into(),
                        run_id: RUN_ID.into(),
                        driver_allocation_lease_id: DRIVER_ALLOCATION.into(),
                        driver_allocation_fence: parent.allocation_fence,
                        driver_process_launch_nonce: parent.process_launch_nonce,
                        observation_id: request.observation_id,
                        clock_domain_nonce: self.domain_nonce,
                        ticks_per_second: reading.ticks_per_second,
                        identity_public_key: signer.public_key().to_vec(),
                        identity_fingerprint: signer.fingerprint(),
                    },
                    started_monotonic_ticks: reading.ticks,
                    signature: Vec::new(),
                };
                signer.sign_node_clock_start(&mut start).unwrap();
                persist(
                    &self.root.join("signed-clock-starts").join(format!(
                        "{}.json",
                        URL_SAFE_NO_PAD.encode(request.observation_id)
                    )),
                    &start,
                );
                persist(&parent.receipt_directory.join(entry.file_name()), &start);
                std::fs::remove_file(entry.path()).unwrap();
            }
        }
        let demands = {
            let state = self.state.lock().unwrap();
            state
                .demands
                .iter()
                .filter(|(id, _)| !state.ends.contains_key(*id))
                .map(|(id, demand)| (*id, demand.clone()))
                .collect::<Vec<_>>()
        };
        for (id, demand) in demands {
            let start = self.state.lock().unwrap().starts[&id].clone();
            let reading = runmat_process_host::clock::observe_monotonic().unwrap();
            if reading.ticks <= start.started_monotonic_ticks {
                continue;
            }
            let mut end = NodeClockEndV1 {
                schema_version: 1,
                owner: start.owner.clone(),
                start_receipt_digest: start.receipt_digest().unwrap(),
                demand_challenge: demand.demand_challenge,
                ended_monotonic_ticks: reading.ticks,
                signature: Vec::new(),
            };
            signer.sign_node_clock_end(&mut end).unwrap();
            persist(
                &self
                    .root
                    .join("signed-clock-ends")
                    .join(format!("{}.json", URL_SAFE_NO_PAD.encode(id))),
                &end,
            );
            let mut state = self.state.lock().unwrap();
            assert!(state.ends.insert(id, end).is_none());
        }
    }

    pub(super) fn heartbeat(
        &self,
        control: &DurableControl,
        authority: &DriverAuthority,
        request: ClockDriverHeartbeatRequestV1,
    ) -> runmat_execution_transport_native::TransportResult<ClockDriverHeartbeatResponseV1> {
        if authority.run_id != RUN_ID || !(1..=3).contains(&authority.fencing_token) {
            return Err(unavailable("foreign clock run or driver fence"));
        }
        let mut state = self.state.lock().unwrap();
        let mut reserved = Vec::new();
        for id in request.start_reservations {
            state.reserved.insert(id);
            reserved.push(id);
        }
        let mut ids = request.lookup_ids.into_iter().collect::<BTreeSet<_>>();
        for start in request.start_receipts {
            self.policy()
                .verify_node_clock_start(&start, &self.fingerprint)
                .map_err(|error| unavailable(&error.to_string()))?;
            let id = start.owner.observation_id;
            if !state.reserved.contains(&id) || start.owner.run_id != RUN_ID {
                return Err(unavailable("unreserved clock start"));
            }
            if state.starts.get(&id).is_some_and(|prior| prior != &start) {
                return Err(unavailable("clock start changed after registration"));
            }
            state.starts.insert(id, start);
            ids.insert(id);
        }
        for demand in request.end_demands {
            let Some(start) = state.starts.get(&demand.observation_id) else {
                return Err(unavailable("clock demand has no original start"));
            };
            if start.receipt_digest().unwrap() != demand.start_receipt_digest {
                return Err(unavailable("clock demand start digest changed"));
            }
            if state
                .demands
                .get(&demand.observation_id)
                .is_some_and(|prior| prior != &demand)
            {
                return Err(unavailable("clock challenge changed after first demand"));
            }
            state.demands.insert(demand.observation_id, demand.clone());
            ids.insert(demand.observation_id);
        }
        let mut release_accepted = false;
        if let Some(manifest) = request.release_manifest {
            if state.latest_release.as_ref() == Some(&manifest) {
                release_accepted = true;
            } else {
                let durable = control.state.lock().unwrap();
                let latest = durable
                    .latest_checkpoint
                    .as_ref()
                    .ok_or_else(|| unavailable("release has no durable checkpoint"))?;
                let checkpoint = stored_driver_checkpoint(control, &durable, latest);
                let artifact = &durable.artifacts[latest];
                if manifest.checkpoint_artifact_id != *latest
                    || manifest.checkpoint_ciphertext_digest
                        != ciphertext_digest(&artifact.ciphertext)
                    || manifest.checkpoint_sequence != checkpoint.sequence
                    || manifest.driver_fence != checkpoint.source_driver_fence()
                    || manifest.entries.is_empty()
                {
                    return Err(unavailable(
                        "release differs from latest accepted checkpoint",
                    ));
                }
                for entry in &manifest.entries {
                    let start = state
                        .starts
                        .get(&entry.observation_id)
                        .ok_or_else(|| unavailable("release start is absent"))?;
                    let end = state
                        .ends
                        .get(&entry.observation_id)
                        .ok_or_else(|| unavailable("release end is absent"))?;
                    if start.receipt_digest().unwrap() != entry.start_receipt_digest
                        || end.receipt_digest().unwrap() != entry.end_receipt_digest
                    {
                        return Err(unavailable("release receipt digest differs"));
                    }
                }
                for entry in &manifest.entries {
                    let id = entry.observation_id;
                    let start = state.starts.remove(&id).unwrap();
                    let end = state.ends.remove(&id).unwrap();
                    state.completed.push((start, end));
                    state.demands.remove(&id);
                    state.reserved.remove(&id);
                }
                state.latest_release = Some(manifest);
                release_accepted = true;
            }
        }
        Ok(ClockDriverHeartbeatResponseV1 {
            run: DriverHeartbeat {
                run_state: "running".into(),
                cancellation_requested: false,
                expires_at_millis: i64::MAX,
            },
            reserved_start_ids: reserved,
            acked_start_ids: Vec::new(),
            starts: ids
                .iter()
                .filter_map(|id| state.starts.get(id).cloned())
                .collect(),
            ends: ids
                .iter()
                .filter_map(|id| state.ends.get(id).cloned())
                .collect(),
            pending_end_demands: ids
                .iter()
                .filter_map(|id| state.demands.get(id).cloned())
                .collect(),
            release_accepted,
            latest_accepted_release_manifest: state.latest_release.clone(),
        })
    }

    pub(super) fn worker_admitted(&self) -> bool {
        std::fs::read_dir(&self.journal).unwrap().any(|entry| {
            entry
                .unwrap()
                .path()
                .extension()
                .is_some_and(|ext| ext == "receipt")
        })
    }

    pub(super) async fn before_checkpoint_ack(&self) {
        let first = {
            let state = self.state.lock().unwrap();
            !state.starts.is_empty() && state.predispatch_checkpoint_ticks.is_none()
        };
        if first {
            assert!(
                !self.worker_admitted(),
                "worker launched before first observed checkpoint ACK"
            );
            tokio::time::sleep(std::time::Duration::from_millis(40)).await;
            assert!(
                !self.worker_admitted(),
                "worker launched while checkpoint ACK was held"
            );
            let tick = runmat_process_host::clock::observe_monotonic().unwrap();
            self.state.lock().unwrap().predispatch_checkpoint_ticks = Some(tick.ticks);
        }
    }

    pub(super) fn assert_preexecution_worker_not_admitted(&self) {
        assert!(
            self.state
                .lock()
                .unwrap()
                .predispatch_checkpoint_ticks
                .is_none(),
            "preexecution replacement unexpectedly acknowledged a checkpoint"
        );
        assert!(
            !self.worker_admitted(),
            "preexecution replacement admitted a first-generation worker attempt"
        );
    }

    pub(super) fn assert_worker_started_after_barrier(&self, run_key: &RunKeyMaterial) {
        let barrier = self
            .state
            .lock()
            .unwrap()
            .predispatch_checkpoint_ticks
            .expect("original observed checkpoint ACK");
        let starts = std::fs::read_dir(&self.journal)
            .unwrap()
            .filter_map(|entry| {
                let path = entry.unwrap().path();
                (path.extension().is_some_and(|ext| ext == "receipt")).then(|| {
                    let bytes = std::fs::read(path).unwrap();
                    let plain = open_object(
                        run_key,
                        &bytes,
                        RUN_ID,
                        EncryptionPurpose::RetirementReceipt,
                    )
                    .unwrap();
                    let entry: WorkerAttemptJournalEntryV1 =
                        serde_json::from_slice(&plain[32..]).unwrap();
                    entry.started_monotonic_ticks
                })
            })
            .collect::<Vec<_>>();
        assert!(
            !starts.is_empty(),
            "actual OS worker never admitted an attempt"
        );
        assert!(starts.into_iter().all(|start| start > barrier));
    }

    pub(super) fn assert_released_before_success(&self) {
        let state = self.state.lock().unwrap();
        assert!(state.predispatch_checkpoint_ticks.is_some());
        assert!(
            !state.completed.is_empty(),
            "node clock end was not released"
        );
        assert!(state.starts.is_empty() && state.ends.is_empty());
        assert!(
            state
                .completed
                .iter()
                .any(|(start, end)| end.measured_wall_millis(start).unwrap() >= 250),
            "original owner interval omitted the driver outage"
        );
        let (original, completed) = state.completed.first().unwrap();
        let mut restarted = completed.clone();
        restarted.owner.clock_domain_nonce = [99; 16];
        EndpointIdentitySigner::from_secret(NODE_SECRET)
            .unwrap()
            .sign_node_clock_end(&mut restarted)
            .unwrap();
        assert!(self
            .policy()
            .verify_node_clock_end(&restarted, &self.fingerprint)
            .is_ok());
        assert!(
            restarted.measured_wall_millis(original).is_err(),
            "a correctly signed end from a restarted owner cannot close the old domain"
        );
    }
}

fn persist<T: serde::Serialize>(destination: &Path, value: &T) {
    std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
    let bytes = serde_json::to_vec(value).unwrap();
    assert!(bytes.len() <= MAX_CLOCK_OBSERVATION_WIRE_BYTES);
    if destination.exists() {
        assert_eq!(std::fs::read(destination).unwrap(), bytes);
        return;
    }
    let staged = destination.parent().unwrap().join(".pending");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staged)
        .unwrap();
    file.write_all(&bytes).unwrap();
    file.sync_all().unwrap();
    runmat_process_host::durable_file::publish_synced_file(&staged, destination).unwrap();
}

#[test]
fn original_parent_observer_survives_actual_worker_loss_and_driver_replacement() {
    let case = thermal::Case::matrix()
        .next()
        .expect("one finite authored thermal case");
    let result = run_actual_driver_replacement_with_observer(
        ExactStudyFamily::ThermalPrestress(case),
        ReplacementInterruption::WorkloadAttemptActive,
        true,
    );
    assert!(result.is_some());
}

#[test]
fn lost_original_parent_cannot_supply_a_replacement_clock_end() {
    let directory = tempfile::tempdir().unwrap();
    let journal = directory.path().join("unused-worker-journal");
    std::fs::create_dir(&journal).unwrap();
    let clock = ClockControl::new(directory.path(), journal);
    let control = Arc::new(exact_study_fixture(ExactStudyFamily::StructuralTransient).control);
    *control.clock_control.lock().unwrap() = Some(Arc::clone(&clock));
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let owner = clock.spawn_owner();
            let authority = driver_authority(1, "http://fixture.invalid".into());
            let start = clock
                .parent(1)
                .begin(&authority, control.as_ref())
                .await
                .unwrap();
            drop(owner);
            let missing = tokio::time::timeout(
                std::time::Duration::from_millis(150),
                super::super::clock_observer::observe_end(control.as_ref(), &authority, &start),
            )
            .await;
            assert!(
                missing.is_err(),
                "a missing original parent produced an end receipt"
            );
            let state = clock.state.lock().unwrap();
            assert!(state.ends.is_empty() && state.latest_release.is_none());
        });
}
