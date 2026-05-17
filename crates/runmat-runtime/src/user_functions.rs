use crate::RuntimeError;
use runmat_builtins::Value;
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub type UserFunctionFuture = Pin<Box<dyn Future<Output = Result<Value, RuntimeError>>>>;
pub type SemanticFunctionInvoker =
    dyn Fn(usize, &[Value], usize) -> UserFunctionFuture + Send + Sync;
pub type SemanticFunctionResolver = dyn Fn(&str) -> Option<usize> + Send + Sync;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticCallableKind {
    Feval,
    Cellfun,
    Arrayfun,
    Other,
}

#[derive(Debug, Clone)]
pub struct SemanticCallableRequest {
    function: Option<usize>,
    name: Option<String>,
    args: Vec<Value>,
    requested_outputs: usize,
    kind: SemanticCallableKind,
}

impl SemanticCallableRequest {
    pub fn named(
        name: String,
        args: Vec<Value>,
        requested_outputs: usize,
        kind: SemanticCallableKind,
    ) -> Self {
        Self {
            function: None,
            name: Some(name),
            args,
            requested_outputs,
            kind,
        }
    }

    pub fn semantic(
        function: usize,
        name: String,
        args: Vec<Value>,
        requested_outputs: usize,
        kind: SemanticCallableKind,
    ) -> Self {
        Self {
            function: Some(function),
            name: Some(name),
            args,
            requested_outputs,
            kind,
        }
    }
}

runmat_thread_local! {
    static SEMANTIC_FUNCTION_INVOKER: RefCell<Option<Arc<SemanticFunctionInvoker>>> =
        const { RefCell::new(None) };
    static SEMANTIC_FUNCTION_RESOLVER: RefCell<Option<Arc<SemanticFunctionResolver>>> =
        const { RefCell::new(None) };
}

pub struct SemanticFunctionInvokerGuard {
    previous: Option<Arc<SemanticFunctionInvoker>>,
}

pub struct SemanticFunctionResolverGuard {
    previous: Option<Arc<SemanticFunctionResolver>>,
}

impl Drop for SemanticFunctionInvokerGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        SEMANTIC_FUNCTION_INVOKER.with(|slot| {
            *slot.borrow_mut() = previous;
        });
    }
}

impl Drop for SemanticFunctionResolverGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        SEMANTIC_FUNCTION_RESOLVER.with(|slot| {
            *slot.borrow_mut() = previous;
        });
    }
}

pub fn install_semantic_function_invoker(
    invoker: Option<Arc<SemanticFunctionInvoker>>,
) -> SemanticFunctionInvokerGuard {
    let previous =
        SEMANTIC_FUNCTION_INVOKER.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), invoker));
    SemanticFunctionInvokerGuard { previous }
}

pub fn install_semantic_function_resolver(
    resolver: Option<Arc<SemanticFunctionResolver>>,
) -> SemanticFunctionResolverGuard {
    let previous = SEMANTIC_FUNCTION_RESOLVER
        .with(|slot| std::mem::replace(&mut *slot.borrow_mut(), resolver));
    SemanticFunctionResolverGuard { previous }
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
    let resolver = SEMANTIC_FUNCTION_RESOLVER.with(|slot| slot.borrow().clone())?;
    let function = resolver(name)?;
    try_call_semantic_function(function, args, requested_outputs).await
}

pub async fn try_call_semantic_descriptor(
    request: SemanticCallableRequest,
) -> Option<Result<Value, RuntimeError>> {
    let _kind = request.kind;
    if let Some(function) = request.function {
        return try_call_semantic_function(function, &request.args, request.requested_outputs)
            .await;
    }
    let name = request.name.as_deref()?;
    try_call_semantic_function_by_name(name, &request.args, request.requested_outputs).await
}
