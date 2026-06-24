use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Arc;

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
pub type FeaCancellationPredicate = Arc<dyn Fn() -> bool + Send + Sync + 'static>;

#[derive(Clone)]
struct FeaProgressContext {
    handler: Option<FeaProgressHandler>,
    cancellation: Option<FeaCancellationPredicate>,
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

pub fn replace_fea_progress_context(
    handler: Option<FeaProgressHandler>,
    cancellation: Option<FeaCancellationPredicate>,
) -> FeaProgressContextGuard {
    let previous = FEA_PROGRESS_CONTEXT.with(|slot| {
        slot.replace(Some(FeaProgressContext {
            handler,
            cancellation,
        }))
    });
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
            .and_then(|context| context.cancellation.clone())
            .map(|is_cancelled| is_cancelled())
            .unwrap_or(false)
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
