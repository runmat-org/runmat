use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fraction: Option<f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

pub type FeaProgressHandler = Arc<dyn Fn(FeaProgressEvent) + Send + Sync + 'static>;

#[derive(Clone, Debug, Default)]
pub struct FeaCancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl FeaCancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_shared(cancelled: Arc<AtomicBool>) -> Self {
        Self { cancelled }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub(crate) fn shared_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancelled)
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

pub struct FeaProgressContextGuard {
    previous: Option<FeaProgressContext>,
}

impl Drop for FeaProgressContextGuard {
    fn drop(&mut self) {
        FEA_PROGRESS_CONTEXT.with(|slot| {
            slot.replace(self.previous.take());
        });
    }
}

pub fn replace_fea_progress_context(context: FeaProgressContext) -> FeaProgressContextGuard {
    let previous = FEA_PROGRESS_CONTEXT.with(|slot| slot.replace(Some(context)));
    FeaProgressContextGuard { previous }
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
