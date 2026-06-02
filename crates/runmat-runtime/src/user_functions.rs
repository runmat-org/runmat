use crate::RuntimeError;
use runmat_builtins::Value;
use runmat_hir::{CallableFallbackPolicy, CallableIdentity, SourceId};
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub type UserFunctionFuture = Pin<Box<dyn Future<Output = Result<Value, RuntimeError>>>>;
pub type FunctionInvoker = dyn Fn(usize, &[Value], usize) -> UserFunctionFuture + Send + Sync;
pub type FunctionResolver = dyn Fn(&str) -> Option<usize> + Send + Sync;

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
            identity: CallableIdentity::BoundFunction(runmat_hir::FunctionId(function)),
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
    static SEMANTIC_FUNCTION_INVOKER: RefCell<Option<Arc<FunctionInvoker>>> =
        const { RefCell::new(None) };
    static SEMANTIC_FUNCTION_RESOLVER: RefCell<Option<Arc<FunctionResolver>>> =
        const { RefCell::new(None) };
    static SOURCE_FUNCTION_CATALOG: RefCell<Option<Arc<Vec<SourceFunctionInfo>>>> =
        const { RefCell::new(None) };
}

pub struct FunctionInvokerGuard {
    previous: Option<Arc<FunctionInvoker>>,
}

pub struct FunctionResolverGuard {
    previous: Option<Arc<FunctionResolver>>,
}

pub struct SourceFunctionCatalogGuard {
    previous: Option<Arc<Vec<SourceFunctionInfo>>>,
}

impl Drop for FunctionInvokerGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        SEMANTIC_FUNCTION_INVOKER.with(|slot| {
            *slot.borrow_mut() = previous;
        });
    }
}

impl Drop for FunctionResolverGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        SEMANTIC_FUNCTION_RESOLVER.with(|slot| {
            *slot.borrow_mut() = previous;
        });
    }
}

impl Drop for SourceFunctionCatalogGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        SOURCE_FUNCTION_CATALOG.with(|slot| {
            *slot.borrow_mut() = previous;
        });
    }
}

pub fn install_semantic_function_invoker(
    invoker: Option<Arc<FunctionInvoker>>,
) -> FunctionInvokerGuard {
    let previous =
        SEMANTIC_FUNCTION_INVOKER.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), invoker));
    FunctionInvokerGuard { previous }
}

pub fn install_semantic_function_resolver(
    resolver: Option<Arc<FunctionResolver>>,
) -> FunctionResolverGuard {
    let previous = SEMANTIC_FUNCTION_RESOLVER
        .with(|slot| std::mem::replace(&mut *slot.borrow_mut(), resolver));
    FunctionResolverGuard { previous }
}

pub fn install_source_function_catalog(
    catalog: Option<Arc<Vec<SourceFunctionInfo>>>,
) -> SourceFunctionCatalogGuard {
    let previous =
        SOURCE_FUNCTION_CATALOG.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), catalog));
    SourceFunctionCatalogGuard { previous }
}

pub fn current_semantic_function_invoker() -> Option<Arc<FunctionInvoker>> {
    SEMANTIC_FUNCTION_INVOKER.with(|slot| slot.borrow().clone())
}

pub fn current_semantic_function_resolver() -> Option<Arc<FunctionResolver>> {
    SEMANTIC_FUNCTION_RESOLVER.with(|slot| slot.borrow().clone())
}

pub fn source_functions_for(source_id: SourceId) -> Vec<SourceFunctionInfo> {
    SOURCE_FUNCTION_CATALOG.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|catalog| {
                catalog
                    .iter()
                    .filter(|info| info.source_id == source_id)
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    })
}

pub async fn try_call_semantic_function(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
) -> Option<Result<Value, RuntimeError>> {
    let invoker = SEMANTIC_FUNCTION_INVOKER.with(|slot| slot.borrow().clone());
    let invoker = invoker?;
    Some(invoker(function, args, requested_outputs).await)
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
    let resolver = SEMANTIC_FUNCTION_RESOLVER.with(|slot| slot.borrow().clone())?;
    resolver(name)
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
        && runmat_builtins::get_class(&name).is_some()
    {
        // Constructor calls for class names must flow through runtime constructor dispatch,
        // not generic semantic name resolution.
        return None;
    }
    try_call_semantic_function_by_name(&name, &args, requested_outputs).await
}
