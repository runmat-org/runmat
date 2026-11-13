//! One complete-call selection and receipt for a dataset path effect.

use super::*;
use crate::context::host_effect::{
    select_host_effect, DatasetPathOperation, HostEffectOutput, HostEffectReceipt,
    HostEffectRequest, HostEffectWorkload,
};
use crate::context::RuntimeContext;
use runmat_execution::Digest;
use std::path::Path;

pub(super) use super::super::execution::execute;

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
