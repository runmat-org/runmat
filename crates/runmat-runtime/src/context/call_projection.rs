//! Exact invocation-local call overlays. Session loader configuration stays in `CallState`.

use crate::callsite::CallsiteInfo;
use crate::context::poll_projection::PollProjection;
use crate::user_functions::{
    BoundFunctionInvoker, ExecutionSourceResolver, FunctionInvoker, FunctionResolver,
    ScopedNameTargetResolver, SemanticNameCandidates, SourceFunctionInfo,
};
use runmat_types::ClassIdentity;
use runmat_value::{exact_capacity_vec, ExactHostNumericAllocationError};
use std::rc::Rc;
use std::sync::Arc;

struct Selection<T> {
    baseline: Option<T>,
    overlays: Vec<(u64, T)>,
}

impl<T> Default for Selection<T> {
    fn default() -> Self {
        Self {
            baseline: None,
            overlays: Vec::new(),
        }
    }
}

impl<T: Clone> Selection<T> {
    fn current(&self) -> Option<T> {
        self.overlays
            .last()
            .map(|(_, value)| value)
            .or(self.baseline.as_ref())
            .cloned()
    }

    fn snapshot(&self) -> Self {
        Self {
            baseline: self.current(),
            overlays: Vec::new(),
        }
    }
}

impl<T> Selection<T> {
    fn current_ref(&self) -> Option<&T> {
        self.overlays
            .last()
            .map(|(_, value)| value)
            .or(self.baseline.as_ref())
    }

    fn push(&mut self, token: u64, value: T) {
        self.overlays.push((token, value));
    }

    fn remove(&mut self, token: u64) -> Option<T> {
        if let Some(index) = self
            .overlays
            .iter()
            .position(|(candidate, _)| *candidate == token)
        {
            return Some(self.overlays.remove(index).1);
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
    class_access: Selection<Option<ClassIdentity>>,
}

impl CallProjection {
    pub(crate) fn poll_snapshot_scratch_bytes(&self) -> Option<u64> {
        let span_bytes = [&self.callsite, &self.function_input_callsite]
            .into_iter()
            .filter_map(|selection| selection.current_ref())
            .map(|callsite| callsite.arg_spans.len())
            .try_fold(0_usize, usize::checked_add)?
            .checked_mul(std::mem::size_of::<runmat_types::Span>())?;
        let class_bytes = self
            .class_access
            .current_ref()
            .and_then(Option::as_ref)
            .map_or(0, |identity| identity.display_name().len());
        span_bytes
            .checked_add(class_bytes)
            .and_then(|bytes| u64::try_from(bytes).ok())
    }

    pub(crate) fn poll_snapshot(&self) -> Result<Self, ExactHostNumericAllocationError> {
        // Construct variable-sized fields once after their exact scratch has
        // been admitted; `snapshot` would clone both callsite vectors first.
        Ok(Self {
            next_token: 0,
            semantic_invoker: self.semantic_invoker.snapshot(),
            bound_invoker: self.bound_invoker.snapshot(),
            semantic_resolver: self.semantic_resolver.snapshot(),
            scoped_name_resolver: self.scoped_name_resolver.snapshot(),
            semantic_candidates: self.semantic_candidates.snapshot(),
            source_functions: self.source_functions.snapshot(),
            execution_source_resolver: self.execution_source_resolver.snapshot(),
            active_function: self.active_function.snapshot(),
            callsite: snapshot_callsite(&self.callsite)?,
            function_input_callsite: snapshot_callsite(&self.function_input_callsite)?,
            class_access: snapshot_class_access(&self.class_access)?,
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

    fn token(&mut self) -> u64 {
        let token = self.next_token;
        self.next_token = token
            .checked_add(1)
            .expect("call overlay identity exhausted");
        token
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
    pub(crate) fn class_access(&self) -> Option<Option<ClassIdentity>> {
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
    ClassAccess(Option<ClassIdentity>),
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
) -> Result<Selection<CallsiteInfo>, ExactHostNumericAllocationError> {
    let baseline = source
        .current_ref()
        .map(|callsite| {
            let mut arg_spans = exact_capacity_vec(callsite.arg_spans.len())?;
            arg_spans.extend_from_slice(&callsite.arg_spans);
            Ok(CallsiteInfo {
                source_id: callsite.source_id,
                arg_spans,
            })
        })
        .transpose()?;
    Ok(Selection {
        baseline,
        overlays: Vec::new(),
    })
}

fn snapshot_class_access(
    source: &Selection<Option<ClassIdentity>>,
) -> Result<Selection<Option<ClassIdentity>>, ExactHostNumericAllocationError> {
    let baseline = source
        .current_ref()
        .map(|identity| {
            identity
                .as_ref()
                .map(|identity| {
                    let bytes = identity.display_name().as_bytes();
                    let mut owned = exact_capacity_vec(bytes.len())?;
                    owned.extend_from_slice(bytes);
                    let name = String::from_utf8(owned)
                        .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?;
                    ClassIdentity::new(name)
                        .map(Some)
                        .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
                })
                .unwrap_or(Ok(None))
        })
        .transpose()?;
    Ok(Selection {
        baseline,
        overlays: Vec::new(),
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

macro_rules! push_overlay {
    ($name:ident, $field:ident, $kind:ident, $value:ty) => {
        pub(crate) fn $name(
            owner: &PollProjection<CallProjection>,
            value: $value,
        ) -> CallProjectionGuard {
            let token = {
                let mut projection = owner.borrow_mut();
                let token = projection.token();
                projection.$field.push(token, value);
                token
            };
            CallProjectionGuard {
                owner: owner.clone(),
                field: Field::$kind,
                token,
            }
        }
    };
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
    push_class_access,
    class_access,
    ClassAccess,
    Option<ClassIdentity>
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::RuntimeContext;
    use crate::execution::RuntimeExecutionService;
    use crate::user_functions::{self, DynamicFunctionLoadPhase, DynamicFunctionLoader};
    use runmat_types::{FunctionSourceRole, SourceId, Span};
    use runmat_value::{Value, ValueSequence};
    use std::future::Future;
    use std::task::{Context, Poll};

    fn runtime() -> RuntimeContext {
        RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
    }

    #[test]
    fn exact_overlay_tokens_survive_out_of_order_retirement() {
        let owner = PollProjection::legacy(CallProjection::default());
        let first = push_class_access(&owner, Some(ClassIdentity::from("First")));
        let second = push_class_access(&owner, Some(ClassIdentity::from("Second")));
        drop(first);
        assert_eq!(
            owner.borrow().class_access(),
            Some(Some(ClassIdentity::from("Second")))
        );
        drop(second);
        assert_eq!(owner.borrow().class_access(), None);
    }

    #[test]
    fn inherited_snapshot_keeps_only_effective_fact() {
        let owner = PollProjection::legacy(CallProjection::default());
        let mut guards = Vec::new();
        for ordinal in 0..128 {
            guards.push(push_active_function(&owner, ordinal));
        }
        let child = PollProjection::legacy(owner.borrow().snapshot());
        assert!(child.borrow().active_function.overlays.is_empty());
        assert_eq!(child.borrow().active_function.baseline, Some(127));
        assert_eq!(child.borrow().active_function(), Some(127));
        drop(guards);
        assert_eq!(child.borrow().active_function(), Some(127));
        let child_guard = push_active_function(&child, 129);
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
        let clear = user_functions::install_semantic_function_resolver(None);
        let class = crate::dispatcher::push_class_access_context(None);
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
            Some(ClassIdentity::from("SessionClass"))
        );
    }

    #[test]
    fn independent_session_and_parallel_lab_reentry_preserve_their_own_selection() {
        let parent = runtime();
        let _entered = parent.enter();
        let _parent_function = user_functions::push_active_semantic_function(11);
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
            let selected = user_functions::push_active_semantic_function(22);
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
                let invoker: Rc<user_functions::FunctionInvoker> = Rc::new(move |_, _, _, _| {
                    Box::pin(async move {
                        crate::execution::RootedValueSequence::new(
                            ValueSequence::single(Value::Num(ordinal as f64)).unwrap(),
                        )
                        .map_err(|error| error.into_runtime_error())
                    })
                });
                let bound: Rc<user_functions::BoundFunctionInvoker> =
                    Rc::new(move |_, _, _, _, _lexical_environment| {
                        Box::pin(async move {
                            crate::execution::RootedValueSequence::new(
                                ValueSequence::single(Value::Num(ordinal as f64)).unwrap(),
                            )
                            .map_err(|error| error.into_runtime_error())
                        })
                    });
                let _invoker =
                    user_functions::install_local_semantic_function_invoker(Rc::clone(&invoker));
                let _bound =
                    user_functions::install_local_bound_function_invoker(Rc::clone(&bound));
                let _resolver =
                    user_functions::install_semantic_function_resolver(Some(Arc::new(move |_| {
                        Some(ordinal)
                    })));
                let _private =
                    user_functions::install_scoped_name_target_resolver(Some(Rc::new(move |_| {
                        Some(user_functions::ScopedSemanticNameTarget::Published(
                            runmat_types::FunctionId(ordinal),
                        ))
                    })));
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
                    ])));
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
                    user_functions::install_semantic_name_candidates(Some(Arc::clone(&candidates)));
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
                )));
                let _active = user_functions::push_active_semantic_function(ordinal);
                let _callsite = crate::callsite::push_callsite(
                    None,
                    Some(vec![Span {
                        start: ordinal,
                        end: ordinal + 1,
                    }]),
                );
                let _class = crate::dispatcher::push_class_access_context(Some(
                    ClassIdentity::from(format!("Frame{ordinal}")),
                ));
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
                        Some(ClassIdentity::from(format!("Frame{ordinal}")))
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
                    Vec::new(),
                    1,
                    DynamicFunctionLoadPhase::AfterSemantic,
                    None,
                )
                .await
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
        let loader: Rc<DynamicFunctionLoader> = Rc::new(|_, _, _, _, _, _| {
            Box::pin(async {
                Some(
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
