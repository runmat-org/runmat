use super::super::control::{ControlIter, ControlVec};
use crate::execution::RuntimeContinuationFuture;

use crate::execution::RootedValue;
use runmat_value::Value;

use super::{AssignmentStepSpec, PreparedAssignmentStep};
use crate::object::dispatch::resolve_object_index_protocol;
use crate::object::indexing::{ObjectIndexOp, ObjectSubscriptPath};
use crate::runtime_error::semantic_error;
use crate::sequence::destination::{PreparedSequenceEndpoint, SequenceEndpointSpec};

pub(super) fn assign_object_path<'a>(
    context: Option<&'a crate::context::RuntimeContext>,
    current: Value,
    mut prefix: ControlIter<PreparedAssignmentStep>,
    expected_class: runmat_value::SharedClassIdentity,
    suffix_steps: ControlVec<AssignmentStepSpec>,
    endpoint: SequenceEndpointSpec,
    path: ObjectSubscriptPath,
    cardinality: crate::builtins::introspection::object_indexing::PreparedObjectAssignmentCardinality,
    values: runmat_value::ValueSequence,
    caller: Option<&'a str>,
) -> RuntimeContinuationFuture<'a, RootedValue> {
    super::super::control::admit_future(context, async move {
        check_cardinality(&cardinality)?;
        if prefix.len() == 0 {
            let actual =
                crate::object::indexing::class_name_from_base(&current).ok_or_else(|| {
                    semantic_error(
                        "PreparedDestinationReceiverChanged",
                        "prepared object destination no longer refers to an object receiver",
                    )
                })?;
            if actual != expected_class.as_identity() {
                return Err(semantic_error(
                    "PreparedDestinationReceiverChanged",
                    format!(
                        "prepared object destination changed class from {} to {actual}",
                        expected_class.as_identity()
                    ),
                ));
            }
            return match resolve_object_index_protocol(&current, ObjectIndexOp::Subsasgn, caller)? {
                Some(method) => match context {
                    Some(context) => {
                        crate::object::protocol::invoke_resolved_object_assignment_sequence_in_context(
                            context, &method, current, path, values,
                        )
                        .await
                    }
                    None => {
                        crate::object::protocol::invoke_resolved_object_assignment_sequence(
                            &method, current, path, values,
                        )
                        .await
                    }
                },
                None => {
                    assign_unprepared_path(
                        context,
                        current,
                        suffix_steps.into_iter(),
                        endpoint,
                        cardinality,
                        values,
                        caller,
                    )
                    .await
                }
            };
        }
        let step = prefix.next().expect("nonempty prefix checked above");
        let child = super::structural::read_step(context, current.clone(), &step, caller).await?;
        let updated = child
            .with_owned_async(|child| {
                assign_object_path(
                    context,
                    child,
                    prefix,
                    expected_class,
                    suffix_steps,
                    endpoint,
                    path,
                    cardinality,
                    values,
                    caller,
                )
            })
            .await?;
        updated
            .with_owned_async(|updated| {
                super::structural::write_step(context, current, step, updated, caller)
            })
            .await
    })
}

fn assign_unprepared_path<'a>(
    context: Option<&'a crate::context::RuntimeContext>,
    current: Value,
    mut steps: ControlIter<AssignmentStepSpec>,
    endpoint: SequenceEndpointSpec,
    cardinality: crate::builtins::introspection::object_indexing::PreparedObjectAssignmentCardinality,
    values: runmat_value::ValueSequence,
    caller: Option<&'a str>,
) -> RuntimeContinuationFuture<'a, RootedValue> {
    super::super::control::admit_future(context, async move {
        check_cardinality(&cardinality)?;
        let Some(spec) = steps.next() else {
            let prepared = PreparedSequenceEndpoint::prepare_with_cardinality(
                context,
                &current,
                endpoint,
                Some(cardinality),
            )
            .await?;
            return match context {
                Some(context) => {
                    prepared
                        .assign_sequence_in_context(context, current, values, caller)
                        .await
                }
                None => prepared.assign_sequence(current, values, caller).await,
            };
        };
        let step =
            super::structural::prepare_step_with_initialization(context, &current, spec, false)
                .await?;
        let child = super::structural::read_step(context, current.clone(), &step, caller).await?;
        let updated = child
            .with_owned_async(|child| {
                assign_unprepared_path(context, child, steps, endpoint, cardinality, values, caller)
            })
            .await?;
        updated
            .with_owned_async(|updated| {
                super::structural::write_step(context, current, step, updated, caller)
            })
            .await
    })
}

fn check_cardinality(
    cardinality: &crate::builtins::introspection::object_indexing::PreparedObjectAssignmentCardinality,
) -> Result<(), crate::RuntimeError> {
    if cardinality
        .method
        .as_ref()
        .is_some_and(|method| !crate::object::protocol::resolved_method_is_current(method))
    {
        return Err(semantic_error(
            "StaleObjectProtocolResolution",
            "object protocol binding changed after the destination was prepared",
        ));
    }
    Ok(())
}
