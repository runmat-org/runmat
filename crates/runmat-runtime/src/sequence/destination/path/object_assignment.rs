use super::super::control::{ControlIter, ControlVec};
use crate::execution::RuntimeContinuationFuture;

use crate::execution::RootedValue;
use runmat_value::Value;

use super::{AssignmentStepSpec, PreparedAssignmentStep};
use crate::object::dispatch::resolve_object_index_protocol_with_access;
use crate::object::indexing::{ObjectIndexOp, ObjectSubscriptPath};
use crate::object::resolve::{ContextualMemberScope, MemberStoreScope, OrdinaryMemberScope};
use crate::runtime_error::semantic_error;
use crate::sequence::destination::{PreparedSequenceEndpoint, SequenceEndpointSpec};

pub(super) fn assign_object_path<'a>(
    context: Option<&'a crate::context::RuntimeContext>,
    mut current: Value,
    prefix: ControlIter<PreparedAssignmentStep>,
    expected_class: runmat_value::SharedClassIdentity,
    suffix_steps: ControlVec<AssignmentStepSpec>,
    endpoint: SequenceEndpointSpec,
    path: ObjectSubscriptPath,
    cardinality: crate::builtins::introspection::object_indexing::PreparedObjectAssignmentCardinality,
    values: runmat_value::ValueSequence,
    caller: Option<&'a str>,
    access: Option<&'a crate::object::protocol::ObjectAccessContext>,
    mut roots: crate::execution::RootedValueLease,
) -> RuntimeContinuationFuture<'a, RootedValue> {
    super::super::control::admit_future(context, move || async move {
        check_cardinality(&cardinality)?;
        if prefix.len() == 0 {
            return assign_object_leaf(
                context,
                current,
                expected_class,
                suffix_steps,
                endpoint,
                path,
                cardinality,
                values,
                caller,
                access,
                roots,
            )
            .await;
        }
        let mut ancestors = super::structural::Writeback::new(context, prefix.len())?;
        for step in prefix {
            ancestors.check(context)?;
            let child =
                super::structural::read_step(context, current.clone(), &step, caller, access)
                    .await?;
            ancestors.push(context, current, step, roots)?;
            (current, roots) = child.into_value_and_lease();
            // This is the next recursive node's original pre-effect check.
            check_cardinality(&cardinality)?;
        }
        let updated = assign_object_leaf(
            context,
            current,
            expected_class,
            suffix_steps,
            endpoint,
            path,
            cardinality,
            values,
            caller,
            access,
            roots,
        )
        .await?;
        ancestors.finish(context, updated, caller, access).await
    })
}

async fn assign_object_leaf(
    context: Option<&crate::context::RuntimeContext>,
    current: Value,
    expected_class: runmat_value::SharedClassIdentity,
    suffix_steps: ControlVec<AssignmentStepSpec>,
    endpoint: SequenceEndpointSpec,
    path: ObjectSubscriptPath,
    cardinality: crate::builtins::introspection::object_indexing::PreparedObjectAssignmentCardinality,
    values: runmat_value::ValueSequence,
    caller: Option<&str>,
    access: Option<&crate::object::protocol::ObjectAccessContext>,
    roots: crate::execution::RootedValueLease,
) -> Result<RootedValue, crate::RuntimeError> {
    let actual = crate::object::indexing::class_name_from_base(&current).ok_or_else(|| {
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
    match resolve_object_index_protocol_with_access(
        &current,
        ObjectIndexOp::Subsasgn,
        caller,
        access,
    )? {
        Some(method) => {
            let result = match context {
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
            };
            drop(roots);
            result
        }
        None => {
            assign_unprepared_path(
                context,
                current,
                suffix_steps.into_iter(),
                endpoint,
                cardinality,
                values,
                caller,
                access,
                roots,
            )
            .await
        }
    }
}
fn assign_unprepared_path<'a>(
    context: Option<&'a crate::context::RuntimeContext>,
    current: Value,
    steps: ControlIter<AssignmentStepSpec>,
    endpoint: SequenceEndpointSpec,
    cardinality: crate::builtins::introspection::object_indexing::PreparedObjectAssignmentCardinality,
    values: runmat_value::ValueSequence,
    caller: Option<&'a str>,
    access: Option<&'a crate::object::protocol::ObjectAccessContext>,
    roots: crate::execution::RootedValueLease,
) -> RuntimeContinuationFuture<'a, RootedValue> {
    match context {
        Some(context) => assign_unprepared_path_with_scope(
            ContextualMemberScope(context),
            current,
            steps,
            endpoint,
            cardinality,
            values,
            caller,
            access,
            roots,
        ),
        None => assign_unprepared_path_with_scope(
            OrdinaryMemberScope,
            current,
            steps,
            endpoint,
            cardinality,
            values,
            caller,
            access,
            roots,
        ),
    }
}

fn assign_unprepared_path_with_scope<'a, M: MemberStoreScope<'a> + 'a>(
    scope: M,
    mut current: Value,
    steps: ControlIter<AssignmentStepSpec>,
    endpoint: SequenceEndpointSpec,
    cardinality: crate::builtins::introspection::object_indexing::PreparedObjectAssignmentCardinality,
    values: runmat_value::ValueSequence,
    caller: Option<&'a str>,
    access: Option<&'a crate::object::protocol::ObjectAccessContext>,
    mut roots: crate::execution::RootedValueLease,
) -> RuntimeContinuationFuture<'a, RootedValue> {
    let context = scope.runtime();
    super::super::control::admit_future(context, move || async move {
        check_cardinality(&cardinality)?;
        if steps.len() == 0 {
            let prepared = PreparedSequenceEndpoint::prepare_with_cardinality(
                context,
                &current,
                endpoint,
                Some(cardinality),
                access,
            )
            .await?;
            let result = prepared
                .assign_sequence_with_scope(scope, current, values, caller, access)
                .await;
            drop(roots);
            return result;
        }
        let mut ancestors = super::structural::Writeback::new(context, steps.len())?;
        for spec in steps {
            ancestors.check(context)?;
            let step = super::structural::prepare_step_with_initialization(
                context,
                &current,
                spec,
                runmat_types::AssignmentChildCreation::ExistingOnly,
                super::ReceiverCreationOrigin::Existing,
            )
            .await?;
            let child =
                super::structural::read_step(context, current.clone(), &step, caller, access)
                    .await?;
            ancestors.push(context, current, step, roots)?;
            (current, roots) = child.into_value_and_lease();
            check_cardinality(&cardinality)?;
        }
        let prepared = PreparedSequenceEndpoint::prepare_with_cardinality(
            context,
            &current,
            endpoint,
            Some(cardinality),
            access,
        )
        .await?;
        let updated = prepared
            .assign_sequence_with_scope(scope, current, values, caller, access)
            .await?;
        drop(roots);
        ancestors.finish(context, updated, caller, access).await
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
