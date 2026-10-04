//! One complete-call selection and receipt for a dataset path effect.

use super::*;
use crate::context::host_effect::{
    select_host_effect, DatasetPathOperation, HostEffectOutput, HostEffectReceipt,
    HostEffectRequest, HostEffectWorkload,
};
use crate::context::RuntimeContext;
use crate::execution::ExecutionServiceError;
use runmat_execution::{Digest, WholeWorkObservationOutcome};
use std::future::Future;
use std::path::Path;

pub(super) fn facts(value: &impl serde::Serialize) -> Digest {
    Digest::sha256(serde_json::to_vec(value).expect("typed dataset domain facts serialize"))
}

pub(super) fn select(
    runtime: Option<&RuntimeContext>,
    authority: &DataPathAuthority,
    operation: DatasetPathOperation,
    source: &Path,
    destination: Option<&Path>,
    domain_facts: Digest,
) -> BuiltinResult<Option<HostEffectReceipt>> {
    let workload = HostEffectWorkload::DatasetPath {
        operation,
        source,
        destination,
        // The path bindings still construct one Value when it is discarded by
        // the caller. Its storage must remain covered by the retained bound.
        output: HostEffectOutput::ReturnedValue,
    };
    select_host_effect(
        runtime,
        HostEffectRequest {
            operation: operation_name(operation),
            workload,
            domain_facts,
            filesystem: authority.provider_arc(),
        },
    )
}

pub(super) async fn execute<T>(
    runtime: Option<&RuntimeContext>,
    receipt: Option<HostEffectReceipt>,
    effect: impl Future<Output = BuiltinResult<T>>,
) -> BuiltinResult<T> {
    if let Some(receipt) = receipt.as_ref() {
        receipt.checkpoint()?;
    }
    let mut cancelled = false;
    let result = if let Some(runtime) = runtime {
        let mut budget = runtime.cpu_cooperation();
        match budget.checkpoint(0).await {
            Ok(()) => match budget.await_cancellable(effect).await {
                Ok(result) => result,
                Err(ExecutionServiceError::Cancelled) => {
                    cancelled = true;
                    Err(cancelled_error())
                }
                Err(error) => Err(error.into_runtime_error()),
            },
            Err(ExecutionServiceError::Cancelled) => {
                cancelled = true;
                Err(cancelled_error())
            }
            Err(error) => Err(error.into_runtime_error()),
        }
    } else {
        effect.await
    };
    let result = if result.is_ok() {
        if let Some(receipt) = receipt.as_ref() {
            receipt.checkpoint()?;
        }
        if let Some(runtime) = runtime {
            if runtime.is_cancelled() {
                cancelled = true;
                Err(cancelled_error())
            } else {
                result
            }
        } else {
            result
        }
    } else {
        result
    };
    if let Some(receipt) = receipt {
        receipt.finish(if cancelled {
            WholeWorkObservationOutcome::Cancelled
        } else if result.is_ok() {
            WholeWorkObservationOutcome::Success
        } else {
            WholeWorkObservationOutcome::OperationalFailure
        });
    }
    result
}

fn operation_name(operation: DatasetPathOperation) -> &'static str {
    match operation {
        DatasetPathOperation::Create => "data.create",
        DatasetPathOperation::Open => "data.open",
        DatasetPathOperation::Exists => "data.exists",
        DatasetPathOperation::Delete => "data.delete",
        DatasetPathOperation::Copy => "data.copy",
        DatasetPathOperation::Move => "data.move",
        DatasetPathOperation::Import => "data.import",
        DatasetPathOperation::Export => "data.export",
        DatasetPathOperation::List => "data.list",
        DatasetPathOperation::Inspect => "data.inspect",
    }
}

fn cancelled_error() -> crate::RuntimeError {
    crate::runtime_error::semantic_error("RunMat:Cancelled", "data invocation was cancelled")
}
