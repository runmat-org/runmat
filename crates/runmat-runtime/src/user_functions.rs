use crate::RuntimeError;
use runmat_builtins::Value;
use runmat_hir::{CallableFallbackPolicy, CallableIdentity};
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub type UserFunctionFuture = Pin<Box<dyn Future<Output = Result<Value, RuntimeError>>>>;
pub type FunctionInvoker = dyn Fn(usize, &[Value], usize) -> UserFunctionFuture + Send + Sync;
pub type FunctionResolver = dyn Fn(&str) -> Option<usize> + Send + Sync;

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
            identity: CallableIdentity::SemanticFunction(runmat_hir::FunctionId(function)),
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
}

pub struct FunctionInvokerGuard {
    previous: Option<Arc<FunctionInvoker>>,
}

pub struct FunctionResolverGuard {
    previous: Option<Arc<FunctionResolver>>,
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

pub fn current_semantic_function_invoker() -> Option<Arc<FunctionInvoker>> {
    SEMANTIC_FUNCTION_INVOKER.with(|slot| slot.borrow().clone())
}

pub fn current_semantic_function_resolver() -> Option<Arc<FunctionResolver>> {
    SEMANTIC_FUNCTION_RESOLVER.with(|slot| slot.borrow().clone())
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
    if let CallableIdentity::SemanticFunction(function) = identity {
        return try_call_semantic_function(function.0, &args, requested_outputs).await;
    }
    if !fallback_policy.allows_semantic_name_resolution_for(&identity) {
        return None;
    }
    let name = fallback_policy.resolution_name_for(&identity)?;
    try_call_semantic_function_by_name(&name, &args, requested_outputs).await
}
