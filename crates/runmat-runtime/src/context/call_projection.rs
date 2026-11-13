//! Exact invocation-local call overlays. Session loader configuration stays in `CallState`.

use crate::callsite::CallsiteInfo;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt, HostOutputPermit,
};
use crate::context::overlay_storage::{
    AdmittedOverlayVec, CheckedOverlayGrowth, OverlayInstallError, OverlayPushSnapshot,
    OverlayRetirement,
};
use crate::context::poll_projection::PollProjection;
use crate::context::RuntimeContext;
use crate::dispatcher::ClassAccessIdentity;
use crate::user_functions::{
    BoundFunctionInvoker, ExecutionSourceResolver, FunctionInvoker, FunctionResolver,
    ScopedNameTargetResolver, SemanticNameCandidates, SourceFunctionInfo,
};
use std::rc::Rc;
use std::sync::Arc;

struct Selection<T> {
    baseline: Option<T>,
    overlays: AdmittedOverlayVec<(u64, T)>,
}

type SelectionInstallResult<T> = Result<OverlayRetirement<(u64, T)>, OverlayInstallError<(u64, T)>>;

impl<T> Default for Selection<T> {
    fn default() -> Self {
        Self {
            baseline: None,
            overlays: AdmittedOverlayVec::unbounded(),
        }
    }
}

impl<T: Clone> Selection<T> {
    fn current(&self) -> Option<T> {
        self.overlays
            .as_slice()
            .last()
            .map(|(_, value)| value)
            .or(self.baseline.as_ref())
            .cloned()
    }

    fn snapshot(&self) -> Self {
        Self {
            baseline: self.current(),
            overlays: AdmittedOverlayVec::unbounded(),
        }
    }

    fn admitted_snapshot(
        &self,
        fallback: Option<T>,
        context: &RuntimeContext,
        permit: HostOutputPermit,
    ) -> Result<Self, HostAllocationError> {
        Ok(Self {
            baseline: self.current().or(fallback),
            overlays: AdmittedOverlayVec::managed_with_permit(context, permit)?,
        })
    }
}

impl<T> Selection<T> {
    fn current_ref(&self) -> Option<&T> {
        self.overlays
            .as_slice()
            .last()
            .map(|(_, value)| value)
            .or(self.baseline.as_ref())
    }

    fn snapshot_push(&self) -> Result<OverlayPushSnapshot, HostAllocationError> {
        self.overlays.snapshot_push()
    }

    fn install_push(
        &mut self,
        prepared: CheckedOverlayGrowth<(u64, T)>,
        token: u64,
        value: T,
    ) -> SelectionInstallResult<T> {
        self.overlays.install_push(prepared, (token, value))
    }

    fn push_unbounded(&mut self, token: u64, value: T) -> Result<(), T> {
        self.overlays
            .push_unbounded((token, value))
            .map_err(|(_, value)| value)
    }

    fn remove(&mut self, token: u64) -> Option<T> {
        if let Some(index) = self
            .overlays
            .as_slice()
            .iter()
            .position(|(candidate, _)| *candidate == token)
        {
            return self.overlays.retire(index).map(|(_, value)| value);
        }
        None
    }
}

#[derive(Clone, Copy)]
enum Field {
    SemanticInvoker,
    BoundInvoker,
    SemanticResolver,
    ScopedNameResolver,
    SemanticCandidates,
    SourceFunctions,
    ExecutionSourceResolver,
    ActiveFunction,
    Callsite,
    FunctionInputCallsite,
    ClassAccess,
}

/// A projection is inherited by value when a child invocation is constructed.
/// A guard then retires only its own token, even if sibling guards finish out of order.
#[derive(Default)]
pub(crate) struct CallProjection {
    next_token: u64,
    semantic_invoker: Selection<Option<Rc<FunctionInvoker>>>,
    bound_invoker: Selection<Option<Rc<BoundFunctionInvoker>>>,
    semantic_resolver: Selection<Option<Arc<FunctionResolver>>>,
    scoped_name_resolver: Selection<Option<Rc<ScopedNameTargetResolver>>>,
    semantic_candidates: Selection<Option<Arc<SemanticNameCandidates>>>,
    source_functions: Selection<Option<Arc<Vec<SourceFunctionInfo>>>>,
    execution_source_resolver: Selection<Option<Rc<ExecutionSourceResolver>>>,
    active_function: Selection<usize>,
    callsite: Selection<CallsiteInfo>,
    function_input_callsite: Selection<CallsiteInfo>,
    class_access: Selection<Option<ClassAccessIdentity>>,
}

impl CallProjection {
    pub(crate) fn poll_snapshot_retained_quote(
        &self,
        state: &crate::context::state::CallState,
    ) -> Option<(u64, usize)> {
        let callsite = self.callsite.current_ref().or(state.callsites.last());
        let input_callsite = self
            .function_input_callsite
            .current_ref()
            .or(state.function_input_callsites.last());
        let (span_bytes, span_claims) = [callsite, input_callsite].into_iter().flatten().try_fold(
            (0_u64, 0_usize),
            |(bytes, claims), callsite| {
                let (next_bytes, next_claims) = callsite.retained_quote().ok()?;
                Some((
                    bytes.checked_add(next_bytes)?,
                    claims.checked_add(next_claims)?,
                ))
            },
        )?;
        let (class_bytes, class_claims) = match self.class_access.current_ref() {
            Some(Some(identity)) => identity.retained_quote().ok()?,
            Some(None) => (0, 0),
            None => match state.class_access.as_ref() {
                Some(identity) => ClassAccessIdentity::retained_quote_for(identity).ok()?,
                None => (0, 0),
            },
        };
        Some((
            span_bytes.checked_add(class_bytes)?,
            span_claims.checked_add(class_claims)?,
        ))
    }

    pub(crate) fn poll_snapshot(
        &self,
        state: &crate::context::state::CallState,
        context: &RuntimeContext,
        permit: HostOutputPermit,
    ) -> Result<Self, HostAllocationError> {
        // Construct variable-sized fields once in their exact retained owners;
        // subsequent poll-local inheritance is O(1).
        Ok(Self {
            next_token: 0,
            semantic_invoker: self.semantic_invoker.admitted_snapshot(
                Some(state.semantic_invoker.clone()),
                context,
                permit.clone(),
            )?,
            bound_invoker: self.bound_invoker.admitted_snapshot(
                Some(state.bound_invoker.clone()),
                context,
                permit.clone(),
            )?,
            semantic_resolver: self.semantic_resolver.admitted_snapshot(
                Some(state.semantic_resolver.clone()),
                context,
                permit.clone(),
            )?,
            scoped_name_resolver: self.scoped_name_resolver.admitted_snapshot(
                Some(state.scoped_name_target_resolver.clone()),
                context,
                permit.clone(),
            )?,
            semantic_candidates: self.semantic_candidates.admitted_snapshot(
                Some(state.semantic_name_candidates.clone()),
                context,
                permit.clone(),
            )?,
            source_functions: self.source_functions.admitted_snapshot(
                Some(state.source_functions.clone()),
                context,
                permit.clone(),
            )?,
            execution_source_resolver: self.execution_source_resolver.admitted_snapshot(
                Some(state.execution_source_resolver.clone()),
                context,
                permit.clone(),
            )?,
            active_function: self.active_function.admitted_snapshot(
                state.active_functions.last().copied(),
                context,
                permit.clone(),
            )?,
            callsite: snapshot_callsite(
                &self.callsite,
                state.callsites.last(),
                context,
                permit.clone(),
            )?,
            function_input_callsite: snapshot_callsite(
                &self.function_input_callsite,
                state.function_input_callsites.last(),
                context,
                permit.clone(),
            )?,
            class_access: snapshot_class_access(
                &self.class_access,
                Some(state.class_access.as_ref()),
                context,
                permit,
            )?,
        })
    }

    pub(crate) fn snapshot(&self) -> Self {
        // A child inherits selected facts, not the parent's guard history.
        // Baselines have no retirement token and require no overlay allocation.
        Self {
            next_token: 0,
            semantic_invoker: self.semantic_invoker.snapshot(),
            bound_invoker: self.bound_invoker.snapshot(),
            semantic_resolver: self.semantic_resolver.snapshot(),
            scoped_name_resolver: self.scoped_name_resolver.snapshot(),
            semantic_candidates: self.semantic_candidates.snapshot(),
            source_functions: self.source_functions.snapshot(),
            execution_source_resolver: self.execution_source_resolver.snapshot(),
            active_function: self.active_function.snapshot(),
            callsite: self.callsite.snapshot(),
            function_input_callsite: self.function_input_callsite.snapshot(),
            class_access: self.class_access.snapshot(),
        }
    }

    fn token_snapshot(&self) -> Result<(u64, u64), HostAllocationError> {
        let next = self
            .next_token
            .checked_add(1)
            .ok_or(HostAllocationError::Overflow)?;
        Ok((self.next_token, next))
    }

    pub(crate) fn semantic_invoker(&self) -> Option<Option<Rc<FunctionInvoker>>> {
        self.semantic_invoker.current()
    }
    pub(crate) fn bound_invoker(&self) -> Option<Option<Rc<BoundFunctionInvoker>>> {
        self.bound_invoker.current()
    }
    pub(crate) fn semantic_resolver(&self) -> Option<Option<Arc<FunctionResolver>>> {
        self.semantic_resolver.current()
    }
    pub(crate) fn scoped_name_resolver(&self) -> Option<Option<Rc<ScopedNameTargetResolver>>> {
        self.scoped_name_resolver.current()
    }
    pub(crate) fn semantic_candidates(&self) -> Option<Option<Arc<SemanticNameCandidates>>> {
        self.semantic_candidates.current()
    }
    pub(crate) fn source_functions(&self) -> Option<Option<Arc<Vec<SourceFunctionInfo>>>> {
        self.source_functions.current()
    }
    pub(crate) fn execution_source_resolver(&self) -> Option<Option<Rc<ExecutionSourceResolver>>> {
        self.execution_source_resolver.current()
    }
    pub(crate) fn active_function(&self) -> Option<usize> {
        self.active_function.current()
    }
    pub(crate) fn callsite(&self) -> Option<CallsiteInfo> {
        self.callsite.current()
    }
    pub(crate) fn function_input_callsite(&self) -> Option<CallsiteInfo> {
        self.function_input_callsite.current()
    }
    pub(crate) fn class_access(&self) -> Option<Option<ClassAccessIdentity>> {
        self.class_access.current()
    }

    fn retire(&mut self, field: Field, token: u64) -> Option<CallProjectionRetirement> {
        match field {
            Field::SemanticInvoker => self
                .semantic_invoker
                .remove(token)
                .map(CallProjectionRetirement::SemanticInvoker),
            Field::BoundInvoker => self
                .bound_invoker
                .remove(token)
                .map(CallProjectionRetirement::BoundInvoker),
            Field::SemanticResolver => self
                .semantic_resolver
                .remove(token)
                .map(CallProjectionRetirement::SemanticResolver),
            Field::ScopedNameResolver => self
                .scoped_name_resolver
                .remove(token)
                .map(CallProjectionRetirement::ScopedNameResolver),
            Field::SemanticCandidates => self
                .semantic_candidates
                .remove(token)
                .map(CallProjectionRetirement::SemanticCandidates),
            Field::SourceFunctions => self
                .source_functions
                .remove(token)
                .map(CallProjectionRetirement::SourceFunctions),
            Field::ExecutionSourceResolver => self
                .execution_source_resolver
                .remove(token)
                .map(CallProjectionRetirement::ExecutionSourceResolver),
            Field::ActiveFunction => self
                .active_function
                .remove(token)
                .map(CallProjectionRetirement::ActiveFunction),
            Field::Callsite => self
                .callsite
                .remove(token)
                .map(CallProjectionRetirement::Callsite),
            Field::FunctionInputCallsite => self
                .function_input_callsite
                .remove(token)
                .map(CallProjectionRetirement::FunctionInputCallsite),
            Field::ClassAccess => self
                .class_access
                .remove(token)
                .map(CallProjectionRetirement::ClassAccess),
        }
    }
}

enum CallProjectionRetirement {
    SemanticInvoker(Option<Rc<FunctionInvoker>>),
    BoundInvoker(Option<Rc<BoundFunctionInvoker>>),
    SemanticResolver(Option<Arc<FunctionResolver>>),
    ScopedNameResolver(Option<Rc<ScopedNameTargetResolver>>),
    SemanticCandidates(Option<Arc<SemanticNameCandidates>>),
    SourceFunctions(Option<Arc<Vec<SourceFunctionInfo>>>),
    ExecutionSourceResolver(Option<Rc<ExecutionSourceResolver>>),
    ActiveFunction(usize),
    Callsite(CallsiteInfo),
    FunctionInputCallsite(CallsiteInfo),
    ClassAccess(Option<ClassAccessIdentity>),
}

impl Drop for CallProjectionRetirement {
    fn drop(&mut self) {
        match self {
            Self::SemanticInvoker(value) => _ = value,
            Self::BoundInvoker(value) => _ = value,
            Self::SemanticResolver(value) => _ = value,
            Self::ScopedNameResolver(value) => _ = value,
            Self::SemanticCandidates(value) => _ = value,
            Self::SourceFunctions(value) => _ = value,
            Self::ExecutionSourceResolver(value) => _ = value,
            Self::ActiveFunction(value) => _ = value,
            Self::Callsite(value) => _ = value,
            Self::FunctionInputCallsite(value) => _ = value,
            Self::ClassAccess(value) => _ = value,
        }
    }
}

fn snapshot_callsite(
    source: &Selection<CallsiteInfo>,
    fallback: Option<&CallsiteInfo>,
    context: &RuntimeContext,
    permit: HostOutputPermit,
) -> Result<Selection<CallsiteInfo>, HostAllocationError> {
    let baseline = source
        .current_ref()
        .or(fallback)
        .map(|callsite| callsite.admitted_clone_with_permit(permit.clone()))
        .transpose()?;
    Ok(Selection {
        baseline,
        overlays: AdmittedOverlayVec::managed_with_permit(context, permit)?,
    })
}

fn snapshot_class_access(
    source: &Selection<Option<ClassAccessIdentity>>,
    fallback: Option<Option<&runmat_types::ClassIdentity>>,
    context: &RuntimeContext,
    permit: HostOutputPermit,
) -> Result<Selection<Option<ClassAccessIdentity>>, HostAllocationError> {
    let baseline = match source.current_ref() {
        Some(Some(identity)) => Some(Some(identity.admitted_clone_with_permit(permit.clone())?)),
        Some(None) => Some(None),
        None => fallback
            .map(|identity| {
                identity
                    .map(|identity| {
                        ClassAccessIdentity::admitted_from_identity(identity, permit.clone())
                    })
                    .transpose()
            })
            .transpose()?,
    };
    Ok(Selection {
        baseline,
        overlays: AdmittedOverlayVec::managed_with_permit(context, permit)?,
    })
}

pub(crate) struct CallProjectionGuard {
    owner: PollProjection<CallProjection>,
    field: Field,
    token: u64,
}

impl Drop for CallProjectionGuard {
    fn drop(&mut self) {
        let retired = self.owner.borrow_mut().retire(self.field, self.token);
        drop(retired);
    }
}

fn ensure_selection_policy<T>(
    owner: &PollProjection<CallProjection>,
    context: &RuntimeContext,
    select: impl Fn(&CallProjection) -> &Selection<T>,
    select_mut: impl Fn(&mut CallProjection) -> &mut Selection<T>,
) -> Result<(), HostAllocationError> {
    if HostAllocationReceipt::policy(Some(context))? != HostAllocationPolicy::Managed {
        return Ok(());
    }
    let needs_owner = {
        let projection = owner.borrow();
        select(&projection).overlays.requires_managed_owner()?
    };
    if !needs_owner {
        return Ok(());
    }
    let prepared = AdmittedOverlayVec::prepare_managed_owner(context)?;
    let installed = {
        let mut projection = owner.borrow_mut();
        select_mut(&mut projection)
            .overlays
            .install_managed_owner(prepared)
    };
    match installed {
        Ok(retirement) => {
            drop(retirement);
            Ok(())
        }
        Err(failure) => {
            let cause = failure.cause;
            drop(failure);
            Err(cause)
        }
    }
}

macro_rules! push_overlay {
    ($name:ident, $field:ident, $kind:ident, $value:ty) => {
        pub(crate) fn $name(
            owner: &PollProjection<CallProjection>,
            value: $value,
        ) -> Result<CallProjectionGuard, crate::RuntimeError> {
            if let Some(runtime) = crate::context::legacy::active() {
                ensure_selection_policy(
                    owner,
                    &runtime,
                    |projection| &projection.$field,
                    |projection| &mut projection.$field,
                )
                .map_err(|source| call_projection_error(Some(&runtime), source))?;
                let (token, next_token, snapshot) = {
                    let projection = owner.borrow();
                    let (token, next_token) = projection
                        .token_snapshot()
                        .map_err(|source| call_projection_error(Some(&runtime), source))?;
                    let snapshot = projection
                        .$field
                        .snapshot_push()
                        .map_err(|source| call_projection_error(Some(&runtime), source))?;
                    (token, next_token, snapshot)
                };
                let prepared =
                    AdmittedOverlayVec::<(u64, $value)>::prepare_push(&runtime, snapshot)
                        .map_err(|source| call_projection_error(Some(&runtime), source))?;
                let prepared = prepared
                    .checkpoint(&runtime)
                    .map_err(|(source, _)| call_projection_error(Some(&runtime), source))?;
                let installed = {
                    let mut projection = owner.borrow_mut();
                    if projection.next_token != token {
                        Err((HostAllocationError::StaleAuthority, value, prepared))
                    } else {
                        match projection.$field.install_push(prepared, token, value) {
                            Ok(retirement) => {
                                projection.next_token = next_token;
                                Ok(retirement)
                            }
                            Err(OverlayInstallError {
                                cause,
                                value: (_, value),
                                prepared,
                            }) => Err((cause, value, prepared)),
                        }
                    }
                };
                match installed {
                    Ok(retirement) => drop(retirement),
                    Err((cause, value, prepared)) => {
                        drop((value, prepared));
                        return Err(call_projection_error(Some(&runtime), cause));
                    }
                }
                Ok(CallProjectionGuard {
                    owner: owner.clone(),
                    field: Field::$kind,
                    token,
                })
            } else {
                let installed = {
                    let mut projection = owner.borrow_mut();
                    let (token, next_token) = projection
                        .token_snapshot()
                        .map_err(|source| call_projection_error(None, source))?;
                    match projection.$field.push_unbounded(token, value) {
                        Ok(()) => {
                            projection.next_token = next_token;
                            Ok(token)
                        }
                        Err(value) => Err(value),
                    }
                };
                let token = installed.map_err(|value| {
                    let _retired = value;
                    call_projection_error(None, HostAllocationError::StaleAuthority)
                })?;
                Ok(CallProjectionGuard {
                    owner: owner.clone(),
                    field: Field::$kind,
                    token,
                })
            }
        }
    };
}

fn call_projection_error(
    context: Option<&RuntimeContext>,
    source: HostAllocationError,
) -> crate::RuntimeError {
    crate::context::host_allocation::host_allocation_runtime_error(
        context,
        "RunMat:Runtime:CallProjection",
        "call projection growth failed",
        source,
    )
}

push_overlay!(
    push_semantic_invoker,
    semantic_invoker,
    SemanticInvoker,
    Option<Rc<FunctionInvoker>>
);
push_overlay!(
    push_bound_invoker,
    bound_invoker,
    BoundInvoker,
    Option<Rc<BoundFunctionInvoker>>
);
push_overlay!(
    push_semantic_resolver,
    semantic_resolver,
    SemanticResolver,
    Option<Arc<FunctionResolver>>
);
push_overlay!(
    push_scoped_name_resolver,
    scoped_name_resolver,
    ScopedNameResolver,
    Option<Rc<ScopedNameTargetResolver>>
);
push_overlay!(
    push_semantic_candidates,
    semantic_candidates,
    SemanticCandidates,
    Option<Arc<SemanticNameCandidates>>
);
push_overlay!(
    push_source_functions,
    source_functions,
    SourceFunctions,
    Option<Arc<Vec<SourceFunctionInfo>>>
);
push_overlay!(
    push_execution_source_resolver,
    execution_source_resolver,
    ExecutionSourceResolver,
    Option<Rc<ExecutionSourceResolver>>
);
push_overlay!(push_active_function, active_function, ActiveFunction, usize);
push_overlay!(push_callsite, callsite, Callsite, CallsiteInfo);
push_overlay!(
    push_function_input_callsite,
    function_input_callsite,
    FunctionInputCallsite,
    CallsiteInfo
);
push_overlay!(
    push_class_access_owner,
    class_access,
    ClassAccess,
    Option<ClassAccessIdentity>
);

pub(crate) fn push_class_access(
    owner: &PollProjection<CallProjection>,
    value: Option<runmat_types::ClassIdentity>,
) -> Result<CallProjectionGuard, crate::RuntimeError> {
    let value = value.map(ClassAccessIdentity::from);
    let value = if let Some(runtime) = crate::context::legacy::active() {
        value
            .map(|value| value.admit(&runtime))
            .transpose()
            .map_err(|source| call_projection_error(Some(&runtime), source))?
    } else {
        value
    };
    push_class_access_owner(owner, value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::RuntimeContext;
    use crate::execution::RuntimeExecutionService;
    use crate::user_functions::{self, DynamicFunctionLoadPhase, DynamicFunctionLoader};
    use runmat_types::{ClassIdentity, FunctionSourceRole, SourceId, Span};
    use runmat_value::{Value, ValueSequence};
    use std::future::Future;
    use std::task::{Context, Poll};

    fn runtime() -> RuntimeContext {
        RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
    }

    #[test]
    fn exact_overlay_tokens_survive_out_of_order_retirement() {
        let owner = PollProjection::legacy(CallProjection::default());
        let first = push_class_access(&owner, Some(ClassIdentity::from("First"))).unwrap();
        let second = push_class_access(&owner, Some(ClassIdentity::from("Second"))).unwrap();
        drop(first);
        assert_eq!(
            owner.borrow().class_access(),
            Some(Some(ClassAccessIdentity::from(ClassIdentity::from(
                "Second"
            ))))
        );
        drop(second);
        assert_eq!(owner.borrow().class_access(), None);
    }

    #[test]
    fn inherited_snapshot_keeps_only_effective_fact() {
        let owner = PollProjection::legacy(CallProjection::default());
        let mut guards = Vec::new();
        for ordinal in 0..128 {
            guards.push(push_active_function(&owner, ordinal).unwrap());
        }
        let child = PollProjection::legacy(owner.borrow().snapshot());
        assert!(child.borrow().active_function.overlays.is_empty());
        assert_eq!(child.borrow().active_function.baseline, Some(127));
        assert_eq!(child.borrow().active_function(), Some(127));
        drop(guards);
        assert_eq!(child.borrow().active_function(), Some(127));
        let child_guard = push_active_function(&child, 129).unwrap();
        assert_eq!(child.borrow().active_function(), Some(129));
        drop(child_guard);
        assert_eq!(child.borrow().active_function(), Some(127));
    }

    #[test]
    fn synchronous_context_masks_and_restores_session_call_fallback() {
        let runtime = runtime();
        runtime.state().call.borrow_mut().semantic_resolver = Some(Arc::new(|_| Some(41)));
        runtime.state().call.borrow_mut().class_access = Some(ClassIdentity::from("SessionClass"));
        let _entered = runtime.enter();
        assert_eq!(
            user_functions::resolve_semantic_function_by_name("name"),
            Some(41)
        );
        let clear = user_functions::install_semantic_function_resolver(None).unwrap();
        let class = crate::dispatcher::push_class_access_context(None).unwrap();
        assert_eq!(
            user_functions::resolve_semantic_function_by_name("name"),
            None
        );
        assert_eq!(crate::dispatcher::class_access_context(), None);
        drop(clear);
        drop(class);
        assert_eq!(
            user_functions::resolve_semantic_function_by_name("name"),
            Some(41)
        );
        assert_eq!(
            crate::dispatcher::class_access_context(),
            Some(ClassAccessIdentity::from(ClassIdentity::from(
                "SessionClass"
            )))
        );
    }

    #[test]
    fn independent_session_and_parallel_lab_reentry_preserve_their_own_selection() {
        let parent = runtime();
        let _entered = parent.enter();
        let _parent_function = user_functions::push_active_semantic_function(11).unwrap();
        let lab = parent
            .fork_parallel_lab(parent.service_ports().clone())
            .expect("idle session fork");
        let independent = runtime();
        for (child, inherited) in [(lab, Some(11)), (independent, None)] {
            let entered = child.enter();
            assert_eq!(
                user_functions::current_active_semantic_function(),
                inherited
            );
            let selected = user_functions::push_active_semantic_function(22).unwrap();
            let nested = parent.enter();
            assert_eq!(user_functions::current_active_semantic_function(), Some(11));
            drop(nested);
            assert_eq!(user_functions::current_active_semantic_function(), Some(22));
            drop(selected);
            assert_eq!(
                user_functions::current_active_semantic_function(),
                inherited
            );
            drop(entered);
            assert_eq!(user_functions::current_active_semantic_function(), Some(11));
        }
    }

    #[test]
    fn two_pending_frames_keep_selected_calls_and_live_session_loader() {
        let runtime = runtime();
        let make = |ordinal: usize,
                    runtime: RuntimeContext,
                    release: futures::channel::oneshot::Receiver<()>| {
            let selected_runtime = runtime.clone();
            runtime.scope(async move {
                let invoker: Rc<user_functions::FunctionInvoker> =
                    Rc::new(move |selected_runtime, _, _, _| {
                        crate::user_functions::admit_user_function_with(
                            selected_runtime.as_ref(),
                            move || async move {
                                crate::execution::RootedValueSequence::new(
                                    ValueSequence::single(Value::Num(ordinal as f64)).unwrap(),
                                )
                                .map_err(|error| error.into_runtime_error())
                            },
                        )
                    });
                let bound: Rc<user_functions::BoundFunctionInvoker> =
                    Rc::new(move |runtime, _, _, _, _lexical_environment| {
                        crate::execution::admit_executable_sequence_with(
                            &runtime,
                            move || async move {
                                crate::execution::RootedValueSequence::new(
                                    ValueSequence::single(Value::Num(ordinal as f64)).unwrap(),
                                )
                                .map_err(|error| error.into_runtime_error())
                            },
                        )
                    });
                let _invoker =
                    user_functions::install_local_semantic_function_invoker(Rc::clone(&invoker))
                        .unwrap();
                let _bound =
                    user_functions::install_local_bound_function_invoker(Rc::clone(&bound))
                        .unwrap();
                let _resolver =
                    user_functions::install_semantic_function_resolver(Some(Arc::new(move |_| {
                        Some(ordinal)
                    })))
                    .unwrap();
                let _private =
                    user_functions::install_scoped_name_target_resolver(Some(Rc::new(move |_| {
                        Some(user_functions::ScopedSemanticNameTarget::Published(
                            runmat_types::FunctionId(ordinal),
                        ))
                    })))
                    .unwrap();
                let _source =
                    user_functions::install_source_function_catalog(Some(Arc::new(vec![
                        user_functions::SourceFunctionInfo {
                            source_id: SourceId(ordinal),
                            name: format!("frame_{ordinal}"),
                            function: ordinal,
                            role: FunctionSourceRole::Primary,
                            binding_origin: None,
                            source_file: None,
                            definition_owners: None,
                        },
                    ])))
                    .unwrap();
                assert_eq!(
                    user_functions::source_function_for(ordinal).unwrap().name,
                    format!("frame_{ordinal}")
                );
                let candidates: Arc<user_functions::SemanticNameCandidates> = Arc::new(move |_| {
                    vec![user_functions::SemanticNameCandidate {
                        function: ordinal,
                        source: None,
                        definition_owner: None,
                    }]
                });
                let _candidates =
                    user_functions::install_semantic_name_candidates(Some(Arc::clone(&candidates)))
                        .unwrap();
                let fact = user_functions::SourceFunctionInfo {
                    source_id: SourceId(ordinal),
                    name: format!("fact_{ordinal}"),
                    function: ordinal,
                    role: FunctionSourceRole::Primary,
                    binding_origin: None,
                    source_file: None,
                    definition_owners: None,
                };
                let _facts = user_functions::install_execution_source_resolver(Some(Rc::new(
                    user_functions::ExecutionSourceFacts::new(
                        Rc::new(move |function| (function == ordinal).then(|| fact.clone())),
                        Rc::new(|_| Vec::new()),
                        Rc::new(|_| None),
                        Rc::new(|_| None),
                    ),
                )))
                .unwrap();
                let _active = user_functions::push_active_semantic_function(ordinal).unwrap();
                let _callsite = crate::callsite::push_callsite(
                    None,
                    Some(vec![Span {
                        start: ordinal,
                        end: ordinal + 1,
                    }]),
                )
                .unwrap();
                let _class = crate::dispatcher::push_class_access_context(Some(
                    ClassIdentity::from(format!("Frame{ordinal}")),
                ))
                .unwrap();
                let verify = || {
                    assert!(Rc::ptr_eq(
                        &user_functions::current_semantic_function_invoker().unwrap(),
                        &invoker
                    ));
                    assert!(Rc::ptr_eq(
                        &user_functions::current_bound_function_invoker().unwrap(),
                        &bound
                    ));
                    assert!(Rc::ptr_eq(
                        &selected_runtime
                            .selected_call_projection()
                            .borrow()
                            .bound_invoker()
                            .unwrap()
                            .unwrap(),
                        &bound
                    ));
                    assert_eq!(
                        user_functions::resolve_semantic_function_by_name("selected"),
                        Some(ordinal)
                    );
                    assert_eq!(
                        user_functions::current_scoped_name_target_resolver().unwrap()("private")
                            .unwrap()
                            .function()
                            .0,
                        ordinal
                    );
                    assert_eq!(
                        user_functions::source_function_for(ordinal).unwrap().name,
                        format!("fact_{ordinal}")
                    );
                    assert!(Arc::ptr_eq(
                        &user_functions::current_semantic_name_candidates().unwrap(),
                        &candidates
                    ));
                    assert_eq!(
                        user_functions::current_active_semantic_function(),
                        Some(ordinal)
                    );
                    assert_eq!(
                        crate::callsite::current_callsite().unwrap().arg_spans[0].start,
                        ordinal
                    );
                    assert_eq!(
                        crate::dispatcher::class_access_context(),
                        Some(ClassAccessIdentity::from(ClassIdentity::from(format!(
                            "Frame{ordinal}"
                        ))))
                    );
                };
                verify();
                let nested = selected_runtime.enter();
                verify();
                drop(nested);
                verify();
                release.await.unwrap();
                verify();
                let nested = selected_runtime.enter();
                verify();
                drop(nested);
                verify();
                let output = user_functions::try_load_and_call_dynamic_function(
                    "late".into(),
                    crate::call::RootedFunctionArguments::ordinary(
                        crate::execution::RootedValues::in_context(Vec::new(), &selected_runtime)
                            .expect("root the selected loader arguments"),
                    ),
                    1,
                    DynamicFunctionLoadPhase::AfterSemantic,
                    None,
                )
                .await
                .break_value()
                .unwrap()
                .unwrap();
                assert_eq!(output.value().as_slice(), &[Value::Num(99.0)]);
            })
        };
        let (first_tx, first_rx) = futures::channel::oneshot::channel();
        let (second_tx, second_rx) = futures::channel::oneshot::channel();
        let mut first = Box::pin(make(1, runtime.clone(), first_rx));
        let mut second = Box::pin(make(2, runtime.clone(), second_rx));
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        assert!(matches!(first.as_mut().poll(&mut cx), Poll::Pending));
        assert!(matches!(second.as_mut().poll(&mut cx), Poll::Pending));
        let loader: Rc<DynamicFunctionLoader> = Rc::new(|runtime, _, _, _, _, _| {
            crate::execution::admit_dynamic_function_load_with(&runtime, || async {
                std::ops::ControlFlow::Break(
                    crate::execution::RootedValueSequence::new(
                        ValueSequence::single(Value::Num(99.0)).unwrap(),
                    )
                    .map_err(|error| error.into_runtime_error()),
                )
            })
        });
        runtime.set_dynamic_function_loader(Some(loader));
        first_tx.send(()).unwrap();
        assert!(matches!(first.as_mut().poll(&mut cx), Poll::Ready(())));
        second_tx.send(()).unwrap();
        assert!(matches!(second.as_mut().poll(&mut cx), Poll::Ready(())));
    }
}
