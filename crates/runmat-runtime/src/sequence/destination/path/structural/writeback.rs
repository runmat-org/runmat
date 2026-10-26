//! The prepared path's original forward-read / reverse-write transaction.
//!
//! Keep ancestors as admitted metadata rather than recursive async poll frames.
//! Every Value and root lease is moved from the original traversal; selectors
//! are consumed once and no destination or payload generation is mirrored.

use crate::execution::{RootedValue, RootedValueLease, RuntimeContinuationFuture};
use crate::object::resolve::{ContextualMemberScope, MemberStoreScope, OrdinaryMemberScope};
use crate::sequence::destination::control::{ControlIter, ControlVec, DestinationControl};
use crate::sequence::destination::PreparedSequenceEndpoint;
use crate::RuntimeError;
use runmat_value::{Value, ValueSequence};

use super::super::PreparedAssignmentStep;

struct Ancestor {
    base: Value,
    step: PreparedAssignmentStep,
    // Preserve the registration until after its Values/selector owners retire.
    roots: RootedValueLease,
}

/// Original parent generations waiting for reverse writeback, not a payload
/// workspace. The allocator's exact backing claims retire after the elements.
pub(in crate::sequence::destination::path) struct Writeback {
    ancestors: ControlVec<Ancestor>,
    control: DestinationControl,
}

impl Writeback {
    pub(in crate::sequence::destination::path) fn check(
        &self,
        context: Option<&crate::context::RuntimeContext>,
    ) -> Result<(), RuntimeError> {
        self.control.check_owner(context)
    }

    pub(in crate::sequence::destination::path) fn new(
        context: Option<&crate::context::RuntimeContext>,
        depth: usize,
    ) -> Result<Self, RuntimeError> {
        let control = DestinationControl::new(context)?;
        let mut ancestors = control.vector();
        control.reserve(context, &mut ancestors, depth)?;
        Ok(Self { ancestors, control })
    }

    pub(in crate::sequence::destination::path) fn push(
        &mut self,
        context: Option<&crate::context::RuntimeContext>,
        base: Value,
        step: PreparedAssignmentStep,
        roots: RootedValueLease,
    ) -> Result<(), RuntimeError> {
        self.control
            .push(context, &mut self.ancestors, Ancestor { base, step, roots })
    }

    pub(in crate::sequence::destination::path) async fn finish(
        mut self,
        context: Option<&crate::context::RuntimeContext>,
        mut updated: RootedValue,
        caller: Option<&str>,
        access: Option<&crate::object::protocol::ObjectAccessContext>,
    ) -> Result<RootedValue, RuntimeError> {
        self.control.check_owner(context)?;
        while let Some(Ancestor { base, step, roots }) = self.ancestors.pop() {
            self.control.check_owner(context)?;
            let (rhs, rhs_roots) = updated.into_value_and_lease();
            updated = super::write_step(context, base, step, rhs, caller, access).await?;
            self.control.check_owner(context)?;
            // The successor is rooted before either original generation may
            // retire, exactly as the old with_owned_async writeback.
            drop(rhs_roots);
            drop(roots);
        }
        Ok(updated)
    }
}

pub(in crate::sequence::destination::path) fn assign_path<'a>(
    context: Option<&'a crate::context::RuntimeContext>,
    current: Value,
    steps: ControlIter<PreparedAssignmentStep>,
    endpoint: PreparedSequenceEndpoint,
    values: ValueSequence,
    caller: Option<&'a str>,
    access: Option<&'a crate::object::protocol::ObjectAccessContext>,
    roots: RootedValueLease,
) -> RuntimeContinuationFuture<'a, RootedValue> {
    match context {
        Some(context) => assign_path_with_scope(
            ContextualMemberScope(context),
            current,
            steps,
            endpoint,
            values,
            caller,
            access,
            roots,
        ),
        None => assign_path_with_scope(
            OrdinaryMemberScope,
            current,
            steps,
            endpoint,
            values,
            caller,
            access,
            roots,
        ),
    }
}

fn assign_path_with_scope<'a, M: MemberStoreScope<'a> + 'a>(
    scope: M,
    mut current: Value,
    steps: ControlIter<PreparedAssignmentStep>,
    endpoint: PreparedSequenceEndpoint,
    values: ValueSequence,
    caller: Option<&'a str>,
    access: Option<&'a crate::object::protocol::ObjectAccessContext>,
    mut roots: RootedValueLease,
) -> RuntimeContinuationFuture<'a, RootedValue> {
    let context = scope.runtime();
    crate::sequence::destination::control::admit_future(context, move || async move {
        if steps.len() == 0 {
            let result = endpoint
                .assign_sequence_with_scope(scope, current, values, caller, access)
                .await;
            drop(roots);
            return result;
        }

        // Quote the actual ancestor collection before the first read/callback.
        // Its allocator retains backing claims through physical deallocation.
        let mut ancestors = Writeback::new(context, steps.len())?;
        for step in steps {
            ancestors.check(context)?;
            let child = super::read_step(context, current.clone(), &step, caller, access).await?;
            // Reserved capacity makes this a move, not another allocation.
            ancestors.push(context, current, step, roots)?;
            (current, roots) = child.into_value_and_lease();
        }

        let updated = endpoint
            .assign_sequence_with_scope(scope, current, values, caller, access)
            .await?;
        drop(roots);
        ancestors.finish(context, updated, caller, access).await
    })
}
