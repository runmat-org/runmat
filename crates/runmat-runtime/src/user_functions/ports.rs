use super::*;

pub struct FunctionInvokerGuard {
    previous: Option<Rc<FunctionInvoker>>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

pub struct BoundFunctionInvokerGuard {
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
    previous: Option<Arc<FunctionResolver>>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

pub struct SourceFunctionCatalogGuard {
    previous: Option<Arc<Vec<SourceFunctionInfo>>>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

pub struct ExecutionSourceResolverGuard {
    previous: Option<Rc<ExecutionSourceResolver>>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

pub struct ActiveSemanticFunctionGuard {
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

impl Drop for FunctionInvokerGuard {
    fn drop(&mut self) {
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

impl Drop for SourceFunctionCatalogGuard {
    fn drop(&mut self) {
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
    invoker: Option<Arc<FunctionInvoker>>,
) -> FunctionInvokerGuard {
    replace_semantic_function_invoker(invoker.map(|invoker| {
        Rc::new(move |function, arguments: &[Value], requested_outputs| {
            invoker(function, arguments, requested_outputs)
        }) as Rc<FunctionInvoker>
    }))
}

pub fn install_local_semantic_function_invoker(
    invoker: Rc<FunctionInvoker>,
) -> FunctionInvokerGuard {
    replace_semantic_function_invoker(Some(invoker))
}

/// Installs the exact-definition route. The ordinary numeric invoker never
/// receives a bound value and cannot confer authority on one.
pub fn install_local_bound_function_invoker(
    invoker: Rc<BoundFunctionInvoker>,
) -> BoundFunctionInvokerGuard {
    if let Some(state) = active_state() {
        let previous = state.call.borrow_mut().bound_invoker.replace(invoker);
        return BoundFunctionInvokerGuard {
            previous,
            state: Some(state),
        };
    }
    let previous = BOUND_FUNCTION_INVOKER.with(|slot| slot.borrow_mut().replace(invoker));
    BoundFunctionInvokerGuard {
        previous,
        state: None,
    }
}

pub fn clear_semantic_function_invoker() -> FunctionInvokerGuard {
    replace_semantic_function_invoker(None)
}

fn replace_semantic_function_invoker(invoker: Option<Rc<FunctionInvoker>>) -> FunctionInvokerGuard {
    if let Some(state) = active_state() {
        let previous = std::mem::replace(&mut state.call.borrow_mut().semantic_invoker, invoker);
        return FunctionInvokerGuard {
            previous,
            state: Some(state),
        };
    }
    let previous =
        SEMANTIC_FUNCTION_INVOKER.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), invoker));
    FunctionInvokerGuard {
        previous,
        state: None,
    }
}

pub fn install_external_function_invoker(
    invoker: Option<Arc<ExternalFunctionInvoker>>,
) -> ExternalFunctionInvokerGuard {
    replace_external_function_invoker(
        invoker.map(|invoker| Rc::new(move |call| invoker(call)) as Rc<ExternalFunctionInvoker>),
    )
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
    invoker: Option<Arc<LexicalFunctionInvoker>>,
) -> LexicalFunctionInvokerGuard {
    replace_lexical_function_invoker(
        invoker.map(|invoker| Rc::new(move |call| invoker(call)) as Rc<LexicalFunctionInvoker>),
    )
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
) -> FunctionResolverGuard {
    if let Some(state) = active_state() {
        let previous = std::mem::replace(&mut state.call.borrow_mut().semantic_resolver, resolver);
        return FunctionResolverGuard {
            previous,
            state: Some(state),
        };
    }
    let previous = SEMANTIC_FUNCTION_RESOLVER
        .with(|slot| std::mem::replace(&mut *slot.borrow_mut(), resolver));
    FunctionResolverGuard {
        previous,
        state: None,
    }
}

pub fn install_source_function_catalog(
    catalog: Option<Arc<Vec<SourceFunctionInfo>>>,
) -> SourceFunctionCatalogGuard {
    if let Some(state) = active_state() {
        let previous = std::mem::replace(&mut state.call.borrow_mut().source_functions, catalog);
        return SourceFunctionCatalogGuard {
            previous,
            state: Some(state),
        };
    }
    let previous =
        SOURCE_FUNCTION_CATALOG.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), catalog));
    SourceFunctionCatalogGuard {
        previous,
        state: None,
    }
}

pub fn install_execution_source_resolver(
    resolver: Option<Rc<ExecutionSourceResolver>>,
) -> ExecutionSourceResolverGuard {
    if let Some(state) = active_state() {
        let previous = std::mem::replace(
            &mut state.call.borrow_mut().execution_source_resolver,
            resolver,
        );
        return ExecutionSourceResolverGuard {
            previous,
            state: Some(state),
        };
    }
    let previous =
        EXECUTION_SOURCE_RESOLVER.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), resolver));
    ExecutionSourceResolverGuard {
        previous,
        state: None,
    }
}

pub fn push_active_semantic_function(function: usize) -> ActiveSemanticFunctionGuard {
    if let Some(state) = active_state() {
        state.call.borrow_mut().active_functions.push(function);
        return ActiveSemanticFunctionGuard { state: Some(state) };
    }
    ACTIVE_SEMANTIC_FUNCTION_STACK.with(|slot| {
        slot.borrow_mut().push(function);
    });
    ActiveSemanticFunctionGuard { state: None }
}

pub fn current_semantic_function_invoker() -> Option<Rc<FunctionInvoker>> {
    if let Some(state) = active_state() {
        return state.call.borrow().semantic_invoker.clone();
    }
    SEMANTIC_FUNCTION_INVOKER.with(|slot| slot.borrow().clone())
}

pub fn current_bound_function_invoker() -> Option<Rc<BoundFunctionInvoker>> {
    if let Some(state) = active_state() {
        return state.call.borrow().bound_invoker.clone();
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
    if let Some(state) = active_state() {
        return state.call.borrow().semantic_resolver.clone();
    }
    SEMANTIC_FUNCTION_RESOLVER.with(|slot| slot.borrow().clone())
}

pub fn current_active_semantic_function() -> Option<usize> {
    if let Some(state) = active_state() {
        return state.call.borrow().active_functions.last().copied();
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
    call.source_functions = SOURCE_FUNCTION_CATALOG.with(|slot| slot.borrow().clone());
    call.execution_source_resolver = EXECUTION_SOURCE_RESOLVER.with(|slot| slot.borrow().clone());
    call.active_functions = ACTIVE_SEMANTIC_FUNCTION_STACK.with(|slot| slot.borrow().clone());
}
