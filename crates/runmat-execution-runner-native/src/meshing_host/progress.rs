//! Progress frames from a meshing worker to its parent.
//!
//! Two streams share the worker's progress channel: stage checkpoints ([`MeshingProgress`]) and
//! live work counts ([`MeshingWorkCount`]). The parent requires strictly increasing frame
//! sequences, so [`ProgressFrames`] numbers every frame under one lock. Work counts leave at most
//! every [`WORK_COUNT_INTERVAL`] per unit; a count that reaches its total always leaves.
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use runmat_meshing_core::work_count::{MeshingWorkCount, MeshingWorkObserver, MeshingWorkUnit};
use runmat_meshing_core::MeshingProgress;
use runmat_meshing_execution::MeshingProgressSink;

use crate::protocol::{ProgramProgress, NATIVE_WORKER_MESSAGE_SCHEMA_V2};

pub(crate) const MESHING_WORK_MEDIA_TYPE: &str = "application/vnd.runmat.meshing-work-count+json";
pub(crate) const MESHING_WORK_VALUE_SCHEMA: &str = "runmat.meshing.work_count.v1";
const WORK_COUNT_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Default)]
struct FrameState {
    sequence: u64,
    work_sent_at: BTreeMap<MeshingWorkUnit, Instant>,
}

pub(super) struct ProgressFrames {
    sender: tokio::sync::mpsc::Sender<ProgramProgress>,
    error: Arc<Mutex<Option<String>>>,
    state: Mutex<FrameState>,
}

impl ProgressFrames {
    pub(super) fn new(
        sender: tokio::sync::mpsc::Sender<ProgramProgress>,
        error: Arc<Mutex<Option<String>>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            sender,
            error,
            state: Mutex::new(FrameState::default()),
        })
    }

    fn send(&self, state: &mut FrameState, mut progress: ProgramProgress) {
        state.sequence += 1;
        progress.sequence = state.sequence;
        let sent = progress.validate().and_then(|()| {
            self.sender
                .blocking_send(progress)
                .map_err(|_| "native progress receiver closed".to_string())
        });
        if let Err(error) = sent {
            *self.error.lock().expect("meshing progress error poisoned") = Some(error);
        }
    }

    fn send_work(&self, count: MeshingWorkCount) {
        let mut state = self.state.lock().expect("meshing progress frames poisoned");
        let now = Instant::now();
        let finished = count.total == Some(count.current);
        if !finished
            && state
                .work_sent_at
                .get(&count.unit)
                .is_some_and(|at| now.duration_since(*at) < WORK_COUNT_INTERVAL)
        {
            return;
        }
        let Ok(payload) = serde_json::to_vec(&count) else {
            return;
        };
        state.work_sent_at.insert(count.unit, now);
        self.send(
            &mut state,
            ProgramProgress {
                schema_version: NATIVE_WORKER_MESSAGE_SCHEMA_V2,
                sequence: 0,
                media_type: MESHING_WORK_MEDIA_TYPE.into(),
                value_schema: MESHING_WORK_VALUE_SCHEMA.into(),
                payload,
            },
        );
    }

    /// The observer this worker installs for kernel work counts.
    pub(super) fn work_observer(self: &Arc<Self>) -> MeshingWorkObserver {
        let frames = Arc::clone(self);
        Arc::new(move |count| frames.send_work(count))
    }
}

pub(super) struct ChannelProgress(pub(super) Arc<ProgressFrames>);

impl MeshingProgressSink for ChannelProgress {
    fn record(&mut self, progress: &MeshingProgress) {
        match super::encode_meshing_progress(progress) {
            Ok(progress) => {
                let mut state = self
                    .0
                    .state
                    .lock()
                    .expect("meshing progress frames poisoned");
                self.0.send(&mut state, progress);
            }
            Err(error) => {
                *self
                    .0
                    .error
                    .lock()
                    .expect("meshing progress error poisoned") = Some(error);
            }
        }
    }
}

/// The work count a worker frame carries, or `None` for other progress media.
pub(crate) fn decode_work_count(progress: &ProgramProgress) -> Option<MeshingWorkCount> {
    (progress.media_type == MESHING_WORK_MEDIA_TYPE
        && progress.value_schema == MESHING_WORK_VALUE_SCHEMA)
        .then(|| serde_json::from_slice(&progress.payload).ok())
        .flatten()
}
