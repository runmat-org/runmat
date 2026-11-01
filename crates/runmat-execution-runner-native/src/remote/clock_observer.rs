//! Driver-parent node clock operations serialized by the existing recovery owner.

use std::sync::Arc;

use runmat_execution::security::{EndpointTrustPolicy, NodeClockEndV1, NodeClockStartV1};
use runmat_execution::{Digest, DigestStream, ExecutionScopeId, TaskId};
use runmat_execution_transport_native::control::{
    ClockDriverHeartbeatRequestV1, ClockEndDemandV1, DriverAuthority, DriverControlPlane,
};
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::{NativeExecutionError, NativeExecutionResult};

/// One typed orchestration scope for every replayable remote admission route.
/// The pool holds physical Launch actions until checkpoint() acknowledges the
/// exact task-to-original-observer binding. close() is required before a
/// successful result can escape its route.
pub(crate) struct RemoteObservedWave {
    pool: Arc<super::RemotePoolDriver>,
    start: Option<NodeClockStartV1>,
    cached_wall: Option<u64>,
    cached_needs_release: bool,
    identity: RemoteWaveIdentity,
    planned_tasks: Vec<TaskId>,
    observed_tasks: Vec<TaskId>,
    armed: bool,
}

/// Encrypted-checkpoint identity of an exact, immutable task inventory.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub(super) struct RemoteWaveIdentity(Digest);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(super) struct CertifiedWave {
    pub(super) tasks: Vec<TaskId>,
    pub(super) measured_wall_millis: u64,
}

impl RemoteWaveIdentity {
    pub(super) fn new(scope: ExecutionScopeId, tasks: &[TaskId]) -> NativeExecutionResult<Self> {
        let mut ids = tasks.to_vec();
        ids.sort_unstable();
        if ids.is_empty() || ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(protocol(
                "remote wave needs a nonempty unique task inventory",
            ));
        }
        let mut digest = DigestStream::new();
        digest.update(b"runmat-remote-wave-inventory-v1");
        digest.update(scope.bytes());
        digest.update(&(ids.len() as u64).to_be_bytes());
        for id in ids {
            digest.update(id.bytes());
        }
        Ok(Self(digest.finish()))
    }
}

impl RemoteObservedWave {
    pub(crate) async fn begin(
        pool: Arc<super::RemotePoolDriver>,
        planned_tasks: &[TaskId],
    ) -> NativeExecutionResult<Self> {
        let identity = RemoteWaveIdentity::new(pool.scope_id(), planned_tasks)?;
        let (start, cached_wall, observed_tasks, cached_needs_release) =
            pool.begin_clock_wave(identity, planned_tasks).await?;
        Ok(Self {
            pool,
            start,
            cached_wall,
            cached_needs_release,
            identity,
            planned_tasks: planned_tasks.to_vec(),
            observed_tasks,
            armed: false,
        })
    }

    pub(crate) fn arm(&mut self, tasks: &[TaskId]) -> NativeExecutionResult<()> {
        if !self.armed {
            if RemoteWaveIdentity::new(self.pool.scope_id(), tasks)? != self.identity {
                return Err(protocol(
                    "remote wave task inventory changed after original start",
                ));
            }
            if let Some(start) = &self.start {
                self.pool.arm_clock_wave(start, &self.observed_tasks)?;
            }
            self.armed = true;
            Ok(())
        } else {
            Err(protocol("remote observed wave was armed twice"))
        }
    }

    pub(crate) fn observed_tasks(&self) -> &[TaskId] {
        &self.observed_tasks
    }

    pub(crate) async fn checkpoint(&self) -> NativeExecutionResult<()> {
        if self.start.is_some() {
            self.pool
                .acknowledge_clock_wave(&self.observed_tasks)
                .await?;
        }
        Ok(())
    }

    pub(crate) async fn close(self) -> NativeExecutionResult<Option<u64>> {
        if self.cached_needs_release {
            if !self.pool.completed_tasks_certified(&self.planned_tasks) {
                let result = self.pool.checkpoint_closed_wave().await;
                if let Err(error) = result {
                    if !self.pool.completed_tasks_certified(&self.planned_tasks) {
                        return Err(error);
                    }
                }
            }
            if !self.pool.completed_tasks_certified(&self.planned_tasks) {
                return Err(protocol(
                    "closed wave release lacks certified checkpoint acknowledgement",
                ));
            }
        }
        if let Some(wall) = self.cached_wall {
            return Ok(Some(wall));
        }
        if let Some(start) = &self.start {
            let wall = self
                .pool
                .end_clock_wave(start, &self.observed_tasks)
                .await?
                .ok_or_else(|| protocol("remote wave lost its original node observer"))?;
            self.pool.checkpoint_closed_wave().await?;
            Ok(Some(wall))
        } else {
            Ok(None)
        }
    }
}

pub(super) enum ClockCommand {
    Begin(oneshot::Sender<NativeExecutionResult<NodeClockStartV1>>),
    Checkpoint(oneshot::Sender<NativeExecutionResult<()>>),
    Close(oneshot::Sender<NativeExecutionResult<()>>),
    End {
        start: Box<NodeClockStartV1>,
        respond: oneshot::Sender<NativeExecutionResult<NodeClockEndV1>>,
    },
}

// A queued command holds its tag and ownership handles, not an inline receipt.
const _: () = assert!(std::mem::size_of::<ClockCommand>() <= 3 * std::mem::size_of::<usize>());

pub(super) async fn observe_end(
    control: &dyn DriverControlPlane,
    authority: &DriverAuthority,
    start: &NodeClockStartV1,
) -> NativeExecutionResult<NodeClockEndV1> {
    let fresh_challenge: [u8; 16] = loop {
        let value: [u8; 16] = rand::random();
        if value.iter().any(|byte| *byte != 0) {
            break value;
        }
    };
    let digest = start.receipt_digest().map_err(protocol)?;
    let initial = control
        .clock_heartbeat(
            authority,
            30,
            ClockDriverHeartbeatRequestV1 {
                lookup_ids: vec![start.owner.observation_id],
                ..Default::default()
            },
        )
        .await
        .map_err(protocol)?;
    if !initial.starts.iter().any(|saved| saved == start) {
        return Err(protocol(
            "Server does not retain the exact original node-signed clock start",
        ));
    }
    let mut challenge = initial
        .pending_end_demands
        .iter()
        .find(|pending| pending.observation_id == start.owner.observation_id)
        .map(|pending| {
            if pending.start_receipt_digest != digest {
                return Err(protocol("pending clock demand differs from original start"));
            }
            Ok(pending.demand_challenge)
        })
        .transpose()?
        .unwrap_or(fresh_challenge);
    if let Some(end) = initial
        .ends
        .into_iter()
        .find(|end| end.owner.observation_id == start.owner.observation_id)
    {
        verify_end(start, &end)?;
        return Ok(end);
    }
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(90);
    loop {
        if tokio::time::Instant::now() >= deadline {
            return Err(protocol(
                "original node clock owner did not close wave before deadline",
            ));
        }
        let response = control
            .clock_heartbeat(
                authority,
                30,
                ClockDriverHeartbeatRequestV1 {
                    end_demands: vec![ClockEndDemandV1 {
                        observation_id: start.owner.observation_id,
                        start_receipt_digest: digest,
                        demand_challenge: challenge,
                    }],
                    lookup_ids: vec![start.owner.observation_id],
                    ..Default::default()
                },
            )
            .await
            .map_err(protocol)?;
        if let Some(pending) = response
            .pending_end_demands
            .iter()
            .find(|pending| pending.observation_id == start.owner.observation_id)
        {
            if pending.start_receipt_digest != digest {
                return Err(protocol("pending clock demand differs from original start"));
            }
            challenge = pending.demand_challenge;
        }
        if let Some(end) = response
            .ends
            .into_iter()
            .find(|end| end.owner.observation_id == start.owner.observation_id)
        {
            verify_end(start, &end)?;
            if end.demand_challenge != challenge {
                return Err(protocol(
                    "node clock end differs from recorded demand challenge",
                ));
            }
            return Ok(end);
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
}

fn verify_end(start: &NodeClockStartV1, end: &NodeClockEndV1) -> NativeExecutionResult<()> {
    let policy = EndpointTrustPolicy {
        permitted_tiers: Default::default(),
        trusted_identity_fingerprints: [start.owner.identity_fingerprint.clone()]
            .into_iter()
            .collect(),
        allowed_attestation_classes: Default::default(),
        require_pinned_identity: true,
        now_unix_millis: 0,
        maximum_clock_skew_millis: 0,
    };
    policy
        .verify_node_clock_end(end, &start.owner.identity_fingerprint)
        .map_err(protocol)?;
    end.measured_wall_millis(start).map_err(protocol)?;
    Ok(())
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
