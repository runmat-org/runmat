use crate::RuntimeError;
use runmat_thread_local::runmat_thread_local;
use runmat_types::{CallableFallbackPolicy, CallableIdentity, SourceId};
use runmat_value::Value;
use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;

pub type UserFunctionFuture = Pin<Box<dyn Future<Output = Result<Value, RuntimeError>>>>;
pub type DynamicFunctionLoadFuture =
    Pin<Box<dyn Future<Output = Option<Result<Value, RuntimeError>>>>>;
pub type FunctionInvoker = dyn Fn(usize, &[Value], usize) -> UserFunctionFuture;
#[derive(Debug, Clone)]
pub struct ExternalFunctionCall {
    pub function: usize,
    pub display_name: String,
    pub arguments: Vec<Value>,
    pub requested_outputs: usize,
}
pub type ExternalFunctionInvoker = dyn Fn(ExternalFunctionCall) -> UserFunctionFuture;
pub type LexicalFunctionFuture =
    Pin<Box<dyn Future<Output = Result<crate::call::lexical::LexicalCallResult, RuntimeError>>>>;
pub type LexicalFunctionInvoker =
    dyn Fn(crate::call::lexical::LexicalCall) -> LexicalFunctionFuture;
pub type FunctionResolver = dyn Fn(&str) -> Option<usize> + Send + Sync;
pub type DynamicFunctionLoader =
    dyn Fn(String, Vec<Value>, usize) -> DynamicFunctionLoadFuture + Send + Sync;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFunctionInfo {
    pub source_id: SourceId,
    pub name: String,
    pub function: usize,
}

#[derive(Debug, Clone)]
pub struct CallableRequest {
    identity: CallableIdentity,
    fallback_policy: CallableFallbackPolicy,
    args: Vec<Value>,
    requested_outputs: usize,
}

impl CallableRequest {
    pub fn semantic(function: usize, args: Vec<Value>, requested_outputs: usize) -> Self {
        Self {
            identity: CallableIdentity::BoundFunction(runmat_types::FunctionId(function)),
            fallback_policy: CallableFallbackPolicy::None,
            args,
            requested_outputs,
        }
    }

    pub fn resolved(
        identity: CallableIdentity,
        fallback_policy: CallableFallbackPolicy,
        args: Vec<Value>,
        requested_outputs: usize,
    ) -> Self {
        Self {
            identity,
            fallback_policy,
            args,
            requested_outputs,
        }
    }
}

runmat_thread_local! {
    static SEMANTIC_FUNCTION_INVOKER: RefCell<Option<Rc<FunctionInvoker>>> =
        const { RefCell::new(None) };
    static EXTERNAL_FUNCTION_INVOKER: RefCell<Option<Rc<ExternalFunctionInvoker>>> =
        const { RefCell::new(None) };
    static LEXICAL_FUNCTION_INVOKER: RefCell<Option<Rc<LexicalFunctionInvoker>>> =
        const { RefCell::new(None) };
    static SEMANTIC_FUNCTION_RESOLVER: RefCell<Option<Arc<FunctionResolver>>> =
        const { RefCell::new(None) };
    static SOURCE_FUNCTION_CATALOG: RefCell<Option<Arc<Vec<SourceFunctionInfo>>>> =
        const { RefCell::new(None) };
    static ACTIVE_SEMANTIC_FUNCTION_STACK: RefCell<Vec<usize>> =
        const { RefCell::new(Vec::new()) };
}

pub struct FunctionInvokerGuard {
    previous: Option<Rc<FunctionInvoker>>,
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

pub fn source_functions_for(source_id: SourceId) -> Vec<SourceFunctionInfo> {
    if let Some(state) = active_state() {
        return source_functions_in_catalog(
            state.call.borrow().source_functions.as_deref(),
            source_id,
        );
    }
    SOURCE_FUNCTION_CATALOG
        .with(|slot| source_functions_in_catalog(slot.borrow().as_deref(), source_id))
}

pub async fn try_call_semantic_function(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
) -> Option<Result<Value, RuntimeError>> {
    let invoker = current_semantic_function_invoker();
    let invoker = invoker?;
    Some(invoker(function, args, requested_outputs).await)
}

pub async fn try_call_external_function(
    call: ExternalFunctionCall,
) -> Option<Result<Value, RuntimeError>> {
    let invoker = current_external_function_invoker()?;
    Some(invoker(call).await)
}

pub async fn try_call_lexical_function(
    call: crate::call::lexical::LexicalCall,
) -> Option<Result<crate::call::lexical::LexicalCallResult, RuntimeError>> {
    let invoker = current_lexical_function_invoker()?;
    Some(invoker(call).await)
}

pub async fn try_call_semantic_function_by_name(
    name: &str,
    args: &[Value],
    requested_outputs: usize,
) -> Option<Result<Value, RuntimeError>> {
    let function = resolve_semantic_function_by_name(name)?;
    try_call_semantic_function(function, args, requested_outputs).await
}

pub fn resolve_semantic_function_by_name(name: &str) -> Option<usize> {
    let resolver = current_semantic_function_resolver()?;
    resolver(name)
}

pub async fn try_load_and_call_dynamic_function(
    name: String,
    args: Vec<Value>,
    requested_outputs: usize,
) -> Option<Result<Value, RuntimeError>> {
    let loader = crate::context::legacy::active()?
        .state()
        .call
        .borrow()
        .dynamic_loader
        .clone()?;
    loader(name, args, requested_outputs).await
}

fn source_functions_in_catalog(
    catalog: Option<&Vec<SourceFunctionInfo>>,
    source_id: SourceId,
) -> Vec<SourceFunctionInfo> {
    catalog
        .map(|catalog| {
            catalog
                .iter()
                .filter(|info| info.source_id == source_id)
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

fn active_state() -> Option<std::rc::Rc<crate::context::RuntimeContextState>> {
    crate::context::legacy::active().map(|context| std::rc::Rc::clone(context.state()))
}

pub async fn try_call_semantic_descriptor(
    request: CallableRequest,
) -> Option<Result<Value, RuntimeError>> {
    let CallableRequest {
        identity,
        fallback_policy,
        args,
        requested_outputs,
    } = request;
    if let CallableIdentity::BoundFunction(function) = identity {
        return try_call_semantic_function(function.0, &args, requested_outputs).await;
    }
    if !fallback_policy.allows_semantic_name_resolution_for(&identity) {
        return None;
    }
    let name = fallback_policy.resolution_name_for(&identity)?;
    if matches!(identity, CallableIdentity::DynamicName(_))
        && crate::class_registry::get_class(&name).is_some()
    {
        // Constructor calls for class names must flow through runtime constructor dispatch,
        // not generic semantic name resolution.
        return None;
    }
    if let Some(result) = try_call_semantic_function_by_name(&name, &args, requested_outputs).await
    {
        return Some(result);
    }
    if matches!(
        identity,
        CallableIdentity::DynamicName(_)
            | CallableIdentity::Imported(_)
            | CallableIdentity::ExternalName(_)
    ) {
        return try_load_and_call_dynamic_function(name, args, requested_outputs).await;
    }
    None
}

#[cfg(test)]
mod lexical_tests {
    use super::*;
    use crate::call::lexical::{LexicalCall, LexicalCallResult, LexicalCapture};

    #[test]
    fn lexical_invoker_preserves_binding_identity_and_restores_scope() {
        assert!(current_lexical_function_invoker().is_none());
        let guard = install_lexical_function_invoker(Some(Arc::new(|mut call| {
            Box::pin(async move {
                assert_eq!(call.function, 7);
                assert_eq!(call.arguments, vec![Value::Num(2.0)]);
                call.captures[0].value = Value::Num(5.0);
                Ok(LexicalCallResult {
                    value: Value::Num(7.0),
                    captures: call.captures,
                })
            })
        })));
        let result = futures::executor::block_on(try_call_lexical_function(LexicalCall {
            function: 7,
            captures: vec![LexicalCapture {
                binding: runmat_types::BindingId(3),
                value: Value::Num(1.0),
            }],
            arguments: vec![Value::Num(2.0)],
            requested_outputs: 1,
        }))
        .expect("lexical invoker is installed")
        .expect("lexical call succeeds");
        assert_eq!(result.value, Value::Num(7.0));
        assert_eq!(result.captures[0].value, Value::Num(5.0));
        drop(guard);
        assert!(current_lexical_function_invoker().is_none());
    }

    #[test]
    fn external_invoker_preserves_identity_kind_and_restores_scope() {
        assert!(current_external_function_invoker().is_none());
        let guard = install_external_function_invoker(Some(Arc::new(|call| {
            Box::pin(async move {
                assert_eq!(call.function, 0);
                assert_eq!(call.display_name, "published");
                assert_eq!(call.arguments, vec![Value::Num(2.0)]);
                assert_eq!(call.requested_outputs, 1);
                Ok(Value::Num(12.0))
            })
        })));
        let result =
            futures::executor::block_on(try_call_external_function(ExternalFunctionCall {
                function: 0,
                display_name: "published".into(),
                arguments: vec![Value::Num(2.0)],
                requested_outputs: 1,
            }))
            .expect("external invoker is installed")
            .expect("external call succeeds");
        assert_eq!(result, Value::Num(12.0));
        drop(guard);
        assert!(current_external_function_invoker().is_none());
    }
}
