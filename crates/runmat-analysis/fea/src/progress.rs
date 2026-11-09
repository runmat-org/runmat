use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::{atomic::AtomicBool, Arc};

use runmat_execution::identity::AttemptId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaProgressPhase {
    GeometryLoad,
    RegionResolution,
    MeshPrep,
    ModelAssembly,
    Solve,
    Postprocess,
    ArtifactPersistence,
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaProgressStatus {
    Started,
    Advanced,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FeaProgressEvent {
    pub operation: String,
    pub phase: FeaProgressPhase,
    pub status: FeaProgressStatus,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(with = "runmat_types::wire_integer::option")]
    pub current: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(with = "runmat_types::wire_integer::option")]
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fraction: Option<f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

pub type FeaProgressHandler = Arc<dyn Fn(FeaProgressEvent) + Send + Sync + 'static>;

#[derive(Clone, Debug, Default)]
pub struct FeaCancellationToken {
    cancelled: runmat_execution::CancellationToken,
}

impl FeaCancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_shared(cancelled: Arc<AtomicBool>) -> Self {
        Self {
            cancelled: runmat_execution::CancellationToken::from_shared(cancelled),
        }
    }

    pub fn from_token(cancelled: runmat_execution::CancellationToken) -> Self {
        Self { cancelled }
    }

    pub fn cancel(&self) {
        self.cancelled.cancel();
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.is_cancelled()
    }

    pub(crate) fn shared_token(&self) -> runmat_execution::CancellationToken {
        self.cancelled.clone()
    }
}

#[derive(Clone)]
pub struct FeaProgressContext {
    pub handler: Option<FeaProgressHandler>,
    pub cancellation: FeaCancellationToken,
    pub attempt_id: Option<AttemptId>,
}

impl Default for FeaProgressContext {
    fn default() -> Self {
        Self {
            handler: None,
            cancellation: FeaCancellationToken::new(),
            attempt_id: None,
        }
    }
}

thread_local! {
    static FEA_PROGRESS_CONTEXT: RefCell<Option<FeaProgressContext>> = const { RefCell::new(None) };
}

/// Restores the previous solver context on the thread that installed it.
///
/// Install a separate context inside each worker; share the cancellation token
/// and handler rather than moving this guard between threads.
///
/// ```compile_fail
/// use runmat_analysis_fea::{replace_fea_progress_context, FeaProgressContext};
/// let guard = replace_fea_progress_context(FeaProgressContext::default());
/// std::thread::spawn(move || drop(guard));
/// ```
pub struct FeaProgressContextGuard {
    previous: Option<FeaProgressContext>,
    // Zero storage, but prevents Send/Sync for a thread-local restoration owner.
    thread_affinity: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl Drop for FeaProgressContextGuard {
    fn drop(&mut self) {
        FEA_PROGRESS_CONTEXT.with(|slot| {
            slot.replace(self.previous.take());
        });
    }
}

/// Capture shared handler, cancellation and attempt identity for a CPU worker.
/// Install the clone on its destination thread; never move the thread-bound guard.
pub fn capture_fea_progress_context() -> FeaProgressContext {
    FEA_PROGRESS_CONTEXT.with(|slot| slot.borrow().clone().unwrap_or_default())
}

pub fn replace_fea_progress_context(context: FeaProgressContext) -> FeaProgressContextGuard {
    let previous = FEA_PROGRESS_CONTEXT.with(|slot| slot.replace(Some(context)));
    FeaProgressContextGuard {
        previous,
        thread_affinity: std::marker::PhantomData,
    }
}

pub fn emit_progress(event: FeaProgressEvent) {
    let handler = FEA_PROGRESS_CONTEXT.with(|slot| {
        slot.borrow()
            .as_ref()
            .and_then(|context| context.handler.clone())
    });
    if let Some(handler) = handler {
        handler(event);
    }
}

pub fn emit_phase(
    operation: impl Into<String>,
    phase: FeaProgressPhase,
    status: FeaProgressStatus,
    message: impl Into<String>,
    current: Option<u64>,
    total: Option<u64>,
) {
    let fraction = match (current, total) {
        (Some(current), Some(total)) if total > 0 => Some((current as f64 / total as f64).min(1.0)),
        _ => None,
    };
    emit_progress(FeaProgressEvent {
        operation: operation.into(),
        phase,
        status,
        message: message.into(),
        current,
        total,
        fraction,
        metadata: BTreeMap::new(),
    });
}

/// Reports one completed step of a marching solve as a [`FeaProgressStatus::Advanced`] solve
/// event: `step` of `total`, the step's end time and its wall time. A host that logs progress
/// (the native worker does, at info) then shows how far the march is and what a step costs.
pub(crate) fn report_step(
    operation: &str,
    step: usize,
    total: usize,
    time_s: f64,
    started: runmat_time::Instant,
    detail: std::fmt::Arguments<'_>,
) {
    emit_phase(
        operation,
        FeaProgressPhase::Solve,
        FeaProgressStatus::Advanced,
        format!(
            "step {step} of {total} to t = {time_s:e} s in {:.2} s{detail}",
            started.elapsed().as_secs_f64()
        ),
        Some(step as u64),
        Some(total as u64),
    );
}

pub fn is_cancelled() -> bool {
    FEA_PROGRESS_CONTEXT.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|context| context.cancellation.is_cancelled())
            .unwrap_or(false)
    })
}

pub(crate) fn cancellation_token() -> FeaCancellationToken {
    FEA_PROGRESS_CONTEXT.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|context| context.cancellation.clone())
            .unwrap_or_default()
    })
}

pub(crate) fn attempt_id() -> Option<AttemptId> {
    FEA_PROGRESS_CONTEXT.with(|slot| {
        slot.borrow()
            .as_ref()
            .and_then(|context| context.attempt_id)
    })
}

pub fn check_cancelled(operation: impl Into<String>) -> Result<(), crate::contracts::FeaRunError> {
    if is_cancelled() {
        emit_phase(
            operation,
            FeaProgressPhase::Complete,
            FeaProgressStatus::Cancelled,
            "FEA solve cancelled",
            None,
            None,
        );
        return Err(crate::contracts::FeaRunError::Cancelled);
    }
    Ok(())
}

impl runmat_canonical_codec::CanonicalCodecControl for FeaCancellationToken {
    fn is_cancelled(&self) -> bool {
        FeaCancellationToken::is_cancelled(self)
    }
}

#[cfg(test)]
mod control_tests;

/// Wall-clock phase measurement exported as one tracing completion event
/// (`target = "runmat_fea::phase"`, fields `phase`, `elapsed_ms`, `threads`).
pub struct PhaseTimer {
    phase: &'static str,
    started: runmat_time::Instant,
}

pub fn phase_timer(phase: &'static str) -> PhaseTimer {
    PhaseTimer {
        phase,
        started: runmat_time::Instant::now(),
    }
}

impl Drop for PhaseTimer {
    fn drop(&mut self) {
        tracing::info!(
            target: "runmat_fea::phase",
            phase = self.phase,
            elapsed_ms = self.started.elapsed().as_secs_f64() * 1e3,
            threads = runmat_cpu_execution::current_threads(),
            "fea phase complete"
        );
    }
}
