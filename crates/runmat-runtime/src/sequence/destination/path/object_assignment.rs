use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;

use runmat_value::Value;

use super::{AssignmentStepSpec, PreparedAssignmentStep};
use crate::object::dispatch::resolve_object_index_protocol;
use crate::object::indexing::{ObjectIndexOp, ObjectSubscriptPath};
use crate::object::protocol::ProtocolResolution;
use crate::runtime_error::semantic_error;
use crate::sequence::destination::{PreparedSequenceEndpoint, SequenceEndpointSpec};
use crate::RuntimeError;

pub(super) fn assign_object_path<'a>(
    current: Value,
    mut prefix: VecDeque<PreparedAssignmentStep>,
    expected_class: runmat_types::ClassIdentity,
    suffix_steps: Vec<AssignmentStepSpec>,
    endpoint: SequenceEndpointSpec,
    path: ObjectSubscriptPath,
    values: Vec<Value>,
    caller: Option<&'a str>,
) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'a>> {
    Box::pin(async move {
        if prefix.is_empty() {
            let actual =
                crate::object::indexing::class_name_from_base(&current).ok_or_else(|| {
                    semantic_error(
                        "PreparedDestinationReceiverChanged",
                        "prepared object destination no longer refers to an object receiver",
                    )
                })?;
            if actual != &expected_class {
                return Err(semantic_error(
                    "PreparedDestinationReceiverChanged",
                    format!("prepared object destination changed class from {expected_class} to {actual}"),
                ));
            }
            return match resolve_object_index_protocol(&current, ObjectIndexOp::Subsasgn, caller)? {
                ProtocolResolution::Method(method) => {
                    crate::object::protocol::invoke_resolved_object_assignment(
                        &method, current, path, values,
                    )
                    .await
                }
                ProtocolResolution::DefaultIndexing => {
                    assign_unprepared_path(current, suffix_steps.into(), endpoint, values, caller)
                        .await
                }
            };
        }
        let step = prefix.pop_front().expect("nonempty prefix checked above");
        let child = super::structural::read_step(current.clone(), &step, caller).await?;
        let updated = assign_object_path(
            child,
            prefix,
            expected_class,
            suffix_steps,
            endpoint,
            path,
            values,
            caller,
        )
        .await?;
        super::structural::write_step(current, step, updated, caller).await
    })
}

fn assign_unprepared_path<'a>(
    current: Value,
    mut steps: VecDeque<AssignmentStepSpec>,
    endpoint: SequenceEndpointSpec,
    values: Vec<Value>,
    caller: Option<&'a str>,
) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'a>> {
    Box::pin(async move {
        let Some(spec) = steps.pop_front() else {
            return PreparedSequenceEndpoint::prepare(&current, endpoint)
                .await?
                .assign(current, values, caller)
                .await;
        };
        let step = super::structural::prepare_step(&current, spec).await?;
        let child = super::structural::read_step(current.clone(), &step, caller).await?;
        let updated = assign_unprepared_path(child, steps, endpoint, values, caller).await?;
        super::structural::write_step(current, step, updated, caller).await
    })
}
