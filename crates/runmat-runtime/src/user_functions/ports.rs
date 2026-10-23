use super::*;
use crate::context::call_projection::{self, CallProjectionGuard};

fn active_projection() -> Option<
    crate::context::poll_projection::PollProjection<
        crate::context::call_projection::CallProjection,
    >,
> {
    crate::context::legacy::active().map(|runtime| runtime.call_projection().clone())
}

pub struct FunctionInvokerGuard {
    projection: Option<CallProjectionGuard>,
    previous: Option<Rc<FunctionInvoker>>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

pub struct BoundFunctionInvokerGuard {
    projection: Option<CallProjectionGuard>,
    previous: Option<Rc<BoundFunctionInvoker>>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

pub struct ExternalFunctionInvokerGuard {
    previous: Option<Rc<ExternalFunctionInvoker>>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

pub struct LexicalFunctionInvokerGuard {
    previous: Option<Rc<LexicalFunctionInvoker>>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

pub struct FunctionResolverGuard {
    projection: Option<CallProjectionGuard>,
    previous: Option<Arc<FunctionResolver>>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

pub struct ScopedNameTargetResolverGuard {
    projection: Option<CallProjectionGuard>,
    previous: Option<Rc<ScopedNameTargetResolver>>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

pub struct SemanticNameCandidatesGuard {
    projection: Option<CallProjectionGuard>,
    previous: Option<Arc<SemanticNameCandidates>>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

pub struct SourceFunctionCatalogGuard {
    projection: Option<CallProjectionGuard>,
    previous: Option<Arc<Vec<SourceFunctionInfo>>>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

pub struct ExecutionSourceResolverGuard {
    projection: Option<CallProjectionGuard>,
    previous: Option<Rc<ExecutionSourceResolver>>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

pub struct ActiveSemanticFunctionGuard {
    projection: Option<CallProjectionGuard>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

impl Drop for FunctionInvokerGuard {
    fn drop(&mut self) {
        if self.projection.take().is_some() {
            return;
        }
        let previous = self.previous.take();
        if let Some(state) = &self.state {
            state.call.borrow_mut().semantic_invoker = previous;
        } else {
            SEMANTIC_FUNCTION_INVOKER.with(|slot| {
                *slot.borrow_mut() = previous;
            });
        }
    }
}

impl Drop for BoundFunctionInvokerGuard {
    fn drop(&mut self) {
        if self.projection.take().is_some() {
            return;
        }
        let previous = self.previous.take();
        if let Some(state) = &self.state {
            state.call.borrow_mut().bound_invoker = previous;
        } else {
            BOUND_FUNCTION_INVOKER.with(|slot| {
                *slot.borrow_mut() = previous;
            });
        }
    }
}

impl Drop for ExternalFunctionInvokerGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        if let Some(state) = &self.state {
            state.call.borrow_mut().external_invoker = previous;
        } else {
            EXTERNAL_FUNCTION_INVOKER.with(|slot| {
                *slot.borrow_mut() = previous;
            });
        }
    }
}

impl Drop for LexicalFunctionInvokerGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        if let Some(state) = &self.state {
            state.call.borrow_mut().lexical_invoker = previous;
        } else {
            LEXICAL_FUNCTION_INVOKER.with(|slot| {
                *slot.borrow_mut() = previous;
            });
        }
    }
}

impl Drop for FunctionResolverGuard {
    fn drop(&mut self) {
        if self.projection.take().is_some() {
            return;
        }
        let previous = self.previous.take();
        if let Some(state) = &self.state {
            state.call.borrow_mut().semantic_resolver = previous;
        } else {
            SEMANTIC_FUNCTION_RESOLVER.with(|slot| {
                *slot.borrow_mut() = previous;
            });
        }
    }
}

impl Drop for ScopedNameTargetResolverGuard {
    fn drop(&mut self) {
        if self.projection.take().is_some() {
            return;
        }
        let previous = self.previous.take();
        if let Some(state) = &self.state {
            state.call.borrow_mut().scoped_name_target_resolver = previous;
        } else {
            SCOPED_NAME_TARGET_RESOLVER.with(|slot| *slot.borrow_mut() = previous);
        }
    }
}

impl Drop for SemanticNameCandidatesGuard {
    fn drop(&mut self) {
        if self.projection.take().is_some() {
            return;
        }
        let previous = self.previous.take();
        if let Some(state) = &self.state {
            state.call.borrow_mut().semantic_name_candidates = previous;
        } else {
            SEMANTIC_NAME_CANDIDATES.with(|slot| *slot.borrow_mut() = previous);
        }
    }
}

impl Drop for SourceFunctionCatalogGuard {
    fn drop(&mut self) {
        if self.projection.take().is_some() {
            return;
        }
        let previous = self.previous.take();
        if let Some(state) = &self.state {
            state.call.borrow_mut().source_functions = previous;
        } else {
            SOURCE_FUNCTION_CATALOG.with(|slot| {
                *slot.borrow_mut() = previous;
            });
        }
    }
}

impl Drop for ExecutionSourceResolverGuard {
    fn drop(&mut self) {
        if self.projection.take().is_some() {
            return;
        }
        let previous = self.previous.take();
        if let Some(state) = &self.state {
            state.call.borrow_mut().execution_source_resolver = previous;
        } else {
            EXECUTION_SOURCE_RESOLVER.with(|slot| *slot.borrow_mut() = previous);
        }
    }
}

impl Drop for ActiveSemanticFunctionGuard {
    fn drop(&mut self) {
        if self.projection.take().is_some() {
            return;
        }
        if let Some(state) = &self.state {
            state.call.borrow_mut().active_functions.pop();
        } else {
            ACTIVE_SEMANTIC_FUNCTION_STACK.with(|slot| {
                slot.borrow_mut().pop();
            });
        }
    }
}

pub fn install_semantic_function_invoker(
    invoker: Option<Arc<UnscopedFunctionInvoker>>,
) -> Result<FunctionInvokerGuard, crate::RuntimeError> {
    replace_semantic_function_invoker(invoker.map(|invoker| {
        Rc::new(
            move |runtime: Option<crate::context::RuntimeContext>,
                  function,
                  arguments: crate::call::RootedFunctionArguments,
                  requested_outputs|
                  -> Result<UserFunctionFuture, crate::RuntimeError> {
                let invoker = invoker.clone();
                crate::execution::rooted_continuation::adopt_unbounded_legacy_rooted_future(
                    runtime.as_ref(),
                    move || {
                        Box::pin(async move {
                            let arguments = arguments.materialize_legacy_contiguous(None)?;
                            invoker(function, &arguments, requested_outputs).await
                        })
                    },
                )
            },
        ) as Rc<FunctionInvoker>
    }))
}

pub fn install_local_semantic_function_invoker(
    invoker: Rc<FunctionInvoker>,
) -> Result<FunctionInvokerGuard, crate::RuntimeError> {
    replace_semantic_function_invoker(Some(invoker))
}

/// Installs the exact-definition route. The ordinary numeric invoker never
/// receives a bound value and cannot confer authority on one.
pub fn install_local_bound_function_invoker(
    invoker: Rc<BoundFunctionInvoker>,
) -> Result<BoundFunctionInvokerGuard, crate::RuntimeError> {
    if let Some(owner) = active_projection() {
        return Ok(BoundFunctionInvokerGuard {
            projection: Some(call_projection::push_bound_invoker(&owner, Some(invoker))?),
            previous: None,
            state: None,
        });
    }
    if let Some(state) = active_state() {
        let previous = state.call.borrow_mut().bound_invoker.replace(invoker);
        return Ok(BoundFunctionInvokerGuard {
            projection: None,
            previous,
            state: Some(state),
        });
    }
    let previous = BOUND_FUNCTION_INVOKER.with(|slot| slot.borrow_mut().replace(invoker));
    Ok(BoundFunctionInvokerGuard {
        projection: None,
        previous,
        state: None,
    })
}

pub fn clear_semantic_function_invoker() -> Result<FunctionInvokerGuard, crate::RuntimeError> {
    replace_semantic_function_invoker(None)
}

fn replace_semantic_function_invoker(
    invoker: Option<Rc<FunctionInvoker>>,
) -> Result<FunctionInvokerGuard, crate::RuntimeError> {
    if let Some(owner) = active_projection() {
        return Ok(FunctionInvokerGuard {
            projection: Some(call_projection::push_semantic_invoker(&owner, invoker)?),
            previous: None,
            state: None,
        });
    }
    if let Some(state) = active_state() {
        let previous = std::mem::replace(&mut state.call.borrow_mut().semantic_invoker, invoker);
        return Ok(FunctionInvokerGuard {
            projection: None,
            previous,
            state: Some(state),
        });
    }
    let previous =
        SEMANTIC_FUNCTION_INVOKER.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), invoker));
    Ok(FunctionInvokerGuard {
        projection: None,
        previous,
        state: None,
    })
}

pub fn install_external_function_invoker(
    invoker: Option<Arc<UnscopedExternalFunctionInvoker>>,
) -> ExternalFunctionInvokerGuard {
    replace_external_function_invoker(invoker.map(|invoker| {
        Rc::new(
            move |runtime: Option<crate::context::RuntimeContext>, call| {
                crate::execution::rooted_continuation::adopt_unbounded_legacy_rooted_future(
                    runtime.as_ref(),
                    || invoker(call),
                )
            },
        ) as Rc<ExternalFunctionInvoker>
    }))
}

pub fn install_local_external_function_invoker(
    invoker: Rc<ExternalFunctionInvoker>,
) -> ExternalFunctionInvokerGuard {
    replace_external_function_invoker(Some(invoker))
}

fn replace_external_function_invoker(
    invoker: Option<Rc<ExternalFunctionInvoker>>,
) -> ExternalFunctionInvokerGuard {
    if let Some(state) = active_state() {
        let previous = std::mem::replace(&mut state.call.borrow_mut().external_invoker, invoker);
        return ExternalFunctionInvokerGuard {
            previous,
            state: Some(state),
        };
    }
    let previous =
        EXTERNAL_FUNCTION_INVOKER.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), invoker));
    ExternalFunctionInvokerGuard {
        previous,
        state: None,
    }
}

pub fn install_lexical_function_invoker(
    invoker: Option<Arc<UnscopedLexicalFunctionInvoker>>,
) -> LexicalFunctionInvokerGuard {
    replace_lexical_function_invoker(invoker.map(|invoker| {
        Rc::new(
            move |runtime: Option<crate::context::RuntimeContext>, call| {
                crate::execution::rooted_continuation::adopt_unbounded_legacy_rooted_future(
                    runtime.as_ref(),
                    || invoker(call),
                )
            },
        ) as Rc<LexicalFunctionInvoker>
    }))
}

pub fn install_local_lexical_function_invoker(
    invoker: Rc<LexicalFunctionInvoker>,
) -> LexicalFunctionInvokerGuard {
    replace_lexical_function_invoker(Some(invoker))
}

fn replace_lexical_function_invoker(
    invoker: Option<Rc<LexicalFunctionInvoker>>,
) -> LexicalFunctionInvokerGuard {
    if let Some(state) = active_state() {
        let previous = std::mem::replace(&mut state.call.borrow_mut().lexical_invoker, invoker);
        return LexicalFunctionInvokerGuard {
            previous,
            state: Some(state),
        };
    }
    let previous =
        LEXICAL_FUNCTION_INVOKER.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), invoker));
    LexicalFunctionInvokerGuard {
        previous,
        state: None,
    }
}

pub fn install_semantic_function_resolver(
    resolver: Option<Arc<FunctionResolver>>,
) -> Result<FunctionResolverGuard, crate::RuntimeError> {
    if let Some(owner) = active_projection() {
        return Ok(FunctionResolverGuard {
            projection: Some(call_projection::push_semantic_resolver(&owner, resolver)?),
            previous: None,
            state: None,
        });
    }
    if let Some(state) = active_state() {
        let previous = std::mem::replace(&mut state.call.borrow_mut().semantic_resolver, resolver);
        return Ok(FunctionResolverGuard {
            projection: None,
            previous,
            state: Some(state),
        });
    }
    let previous = SEMANTIC_FUNCTION_RESOLVER
        .with(|slot| std::mem::replace(&mut *slot.borrow_mut(), resolver));
    Ok(FunctionResolverGuard {
        projection: None,
        previous,
        state: None,
    })
}

pub fn install_scoped_name_target_resolver(
    resolver: Option<Rc<ScopedNameTargetResolver>>,
) -> Result<ScopedNameTargetResolverGuard, crate::RuntimeError> {
    if let Some(owner) = active_projection() {
        return Ok(ScopedNameTargetResolverGuard {
            projection: Some(call_projection::push_scoped_name_resolver(
                &owner, resolver,
            )?),
            previous: None,
            state: None,
        });
    }
    if let Some(state) = active_state() {
        let previous = std::mem::replace(
            &mut state.call.borrow_mut().scoped_name_target_resolver,
            resolver,
        );
        return Ok(ScopedNameTargetResolverGuard {
            projection: None,
            previous,
            state: Some(state),
        });
    }
    let previous = SCOPED_NAME_TARGET_RESOLVER
        .with(|slot| std::mem::replace(&mut *slot.borrow_mut(), resolver));
    Ok(ScopedNameTargetResolverGuard {
        projection: None,
        previous,
        state: None,
    })
}

pub fn install_semantic_name_candidates(
    observer: Option<Arc<SemanticNameCandidates>>,
) -> Result<SemanticNameCandidatesGuard, crate::RuntimeError> {
    if let Some(owner) = active_projection() {
        return Ok(SemanticNameCandidatesGuard {
            projection: Some(call_projection::push_semantic_candidates(&owner, observer)?),
            previous: None,
            state: None,
        });
    }
    if let Some(state) = active_state() {
        let previous = std::mem::replace(
            &mut state.call.borrow_mut().semantic_name_candidates,
            observer,
        );
        return Ok(SemanticNameCandidatesGuard {
            projection: None,
            previous,
            state: Some(state),
        });
    }
    let previous =
        SEMANTIC_NAME_CANDIDATES.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), observer));
    Ok(SemanticNameCandidatesGuard {
        projection: None,
        previous,
        state: None,
    })
}

pub fn install_source_function_catalog(
    catalog: Option<Arc<Vec<SourceFunctionInfo>>>,
) -> Result<SourceFunctionCatalogGuard, crate::RuntimeError> {
    if let Some(owner) = active_projection() {
        return Ok(SourceFunctionCatalogGuard {
            projection: Some(call_projection::push_source_functions(&owner, catalog)?),
            previous: None,
            state: None,
        });
    }
    if let Some(state) = active_state() {
        let previous = std::mem::replace(&mut state.call.borrow_mut().source_functions, catalog);
        return Ok(SourceFunctionCatalogGuard {
            projection: None,
            previous,
            state: Some(state),
        });
    }
    let previous =
        SOURCE_FUNCTION_CATALOG.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), catalog));
    Ok(SourceFunctionCatalogGuard {
        projection: None,
        previous,
        state: None,
    })
}

pub fn install_execution_source_resolver(
    resolver: Option<Rc<ExecutionSourceResolver>>,
) -> Result<ExecutionSourceResolverGuard, crate::RuntimeError> {
    if let Some(owner) = active_projection() {
        return Ok(ExecutionSourceResolverGuard {
            projection: Some(call_projection::push_execution_source_resolver(
                &owner, resolver,
            )?),
            previous: None,
            state: None,
        });
    }
    if let Some(state) = active_state() {
        let previous = std::mem::replace(
            &mut state.call.borrow_mut().execution_source_resolver,
            resolver,
        );
        return Ok(ExecutionSourceResolverGuard {
            projection: None,
            previous,
            state: Some(state),
        });
    }
    let previous =
        EXECUTION_SOURCE_RESOLVER.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), resolver));
    Ok(ExecutionSourceResolverGuard {
        projection: None,
        previous,
        state: None,
    })
}

pub fn push_active_semantic_function(
    function: usize,
) -> Result<ActiveSemanticFunctionGuard, crate::RuntimeError> {
    if let Some(owner) = active_projection() {
        return Ok(ActiveSemanticFunctionGuard {
            projection: Some(call_projection::push_active_function(&owner, function)?),
            state: None,
        });
    }
    if let Some(state) = active_state() {
        state.call.borrow_mut().active_functions.push(function);
        return Ok(ActiveSemanticFunctionGuard {
            projection: None,
            state: Some(state),
        });
    }
    ACTIVE_SEMANTIC_FUNCTION_STACK.with(|slot| {
        slot.borrow_mut().push(function);
    });
    Ok(ActiveSemanticFunctionGuard {
        projection: None,
        state: None,
    })
}

pub fn current_semantic_function_invoker() -> Option<Rc<FunctionInvoker>> {
    if let Some(runtime) = crate::context::legacy::active() {
        return runtime
            .call_projection()
            .borrow()
            .semantic_invoker()
            .unwrap_or_else(|| runtime.state().call.borrow().semantic_invoker.clone());
    }
    SEMANTIC_FUNCTION_INVOKER.with(|slot| slot.borrow().clone())
}

pub fn current_bound_function_invoker() -> Option<Rc<BoundFunctionInvoker>> {
    if let Some(runtime) = crate::context::legacy::active() {
        return runtime
            .call_projection()
            .borrow()
            .bound_invoker()
            .unwrap_or_else(|| runtime.state().call.borrow().bound_invoker.clone());
    }
    BOUND_FUNCTION_INVOKER.with(|slot| slot.borrow().clone())
}

pub fn current_external_function_invoker() -> Option<Rc<ExternalFunctionInvoker>> {
    if let Some(state) = active_state() {
        return state.call.borrow().external_invoker.clone();
    }
    EXTERNAL_FUNCTION_INVOKER.with(|slot| slot.borrow().clone())
}

pub fn current_lexical_function_invoker() -> Option<Rc<LexicalFunctionInvoker>> {
    if let Some(state) = active_state() {
        return state.call.borrow().lexical_invoker.clone();
    }
    LEXICAL_FUNCTION_INVOKER.with(|slot| slot.borrow().clone())
}

pub fn current_semantic_function_resolver() -> Option<Arc<FunctionResolver>> {
    if let Some(runtime) = crate::context::legacy::active() {
        return runtime
            .call_projection()
            .borrow()
            .semantic_resolver()
            .unwrap_or_else(|| runtime.state().call.borrow().semantic_resolver.clone());
    }
    SEMANTIC_FUNCTION_RESOLVER.with(|slot| slot.borrow().clone())
}

pub fn current_scoped_name_target_resolver() -> Option<Rc<ScopedNameTargetResolver>> {
    if let Some(runtime) = crate::context::legacy::active() {
        return runtime
            .call_projection()
            .borrow()
            .scoped_name_resolver()
            .unwrap_or_else(|| {
                runtime
                    .state()
                    .call
                    .borrow()
                    .scoped_name_target_resolver
                    .clone()
            });
    }
    SCOPED_NAME_TARGET_RESOLVER.with(|slot| slot.borrow().clone())
}

pub fn current_semantic_name_candidates() -> Option<Arc<SemanticNameCandidates>> {
    if let Some(runtime) = crate::context::legacy::active() {
        return runtime
            .call_projection()
            .borrow()
            .semantic_candidates()
            .unwrap_or_else(|| {
                runtime
                    .state()
                    .call
                    .borrow()
                    .semantic_name_candidates
                    .clone()
            });
    }
    SEMANTIC_NAME_CANDIDATES.with(|slot| slot.borrow().clone())
}

pub fn current_active_semantic_function() -> Option<usize> {
    if let Some(runtime) = crate::context::legacy::active() {
        return runtime
            .call_projection()
            .borrow()
            .active_function()
            .or_else(|| {
                runtime
                    .state()
                    .call
                    .borrow()
                    .active_functions
                    .last()
                    .copied()
            });
    }
    ACTIVE_SEMANTIC_FUNCTION_STACK.with(|slot| slot.borrow().last().copied())
}

/// Seed a newly-created standalone runtime context from the legacy
/// thread-local call environment before entering its scope.
///
/// Embedders that already provide an explicit `RuntimeContext` must configure
/// that context directly. This bridge exists for the standalone VM entrypoints
/// that historically accepted resolver/invoker guards without a context.
pub fn inherit_legacy_call_environment(context: &crate::context::RuntimeContext) {
    let mut call = context.state().call.borrow_mut();
    call.semantic_invoker = SEMANTIC_FUNCTION_INVOKER.with(|slot| slot.borrow().clone());
    call.bound_invoker = BOUND_FUNCTION_INVOKER.with(|slot| slot.borrow().clone());
    call.external_invoker = EXTERNAL_FUNCTION_INVOKER.with(|slot| slot.borrow().clone());
    call.lexical_invoker = LEXICAL_FUNCTION_INVOKER.with(|slot| slot.borrow().clone());
    call.semantic_resolver = SEMANTIC_FUNCTION_RESOLVER.with(|slot| slot.borrow().clone());
    call.semantic_name_candidates = SEMANTIC_NAME_CANDIDATES.with(|slot| slot.borrow().clone());
    call.source_functions = SOURCE_FUNCTION_CATALOG.with(|slot| slot.borrow().clone());
    call.execution_source_resolver = EXECUTION_SOURCE_RESOLVER.with(|slot| slot.borrow().clone());
    call.active_functions = ACTIVE_SEMANTIC_FUNCTION_STACK.with(|slot| slot.borrow().clone());
}
