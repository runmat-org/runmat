use crate::RuntimeError;
use runmat_builtins::Value;
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub type UserFunctionFuture = Pin<Box<dyn Future<Output = Result<Value, RuntimeError>>>>;
pub type UserFunctionInvoker = dyn Fn(&str, &[Value]) -> UserFunctionFuture + Send + Sync;
pub type SemanticFunctionInvoker =
    dyn Fn(usize, &[Value], usize) -> UserFunctionFuture + Send + Sync;

runmat_thread_local! {
    static USER_FUNCTION_INVOKER: RefCell<Option<Arc<UserFunctionInvoker>>> =
        const { RefCell::new(None) };
    static SEMANTIC_FUNCTION_INVOKER: RefCell<Option<Arc<SemanticFunctionInvoker>>> =
        const { RefCell::new(None) };
}

pub struct UserFunctionInvokerGuard {
    previous: Option<Arc<UserFunctionInvoker>>,
}

pub struct SemanticFunctionInvokerGuard {
    previous: Option<Arc<SemanticFunctionInvoker>>,
}

impl Drop for UserFunctionInvokerGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        USER_FUNCTION_INVOKER.with(|slot| {
            *slot.borrow_mut() = previous;
        });
    }
}

impl Drop for SemanticFunctionInvokerGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        SEMANTIC_FUNCTION_INVOKER.with(|slot| {
            *slot.borrow_mut() = previous;
        });
    }
}

pub fn install_user_function_invoker(
    invoker: Option<Arc<UserFunctionInvoker>>,
) -> UserFunctionInvokerGuard {
    let previous =
        USER_FUNCTION_INVOKER.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), invoker));
    UserFunctionInvokerGuard { previous }
}

pub fn install_semantic_function_invoker(
    invoker: Option<Arc<SemanticFunctionInvoker>>,
) -> SemanticFunctionInvokerGuard {
    let previous =
        SEMANTIC_FUNCTION_INVOKER.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), invoker));
    SemanticFunctionInvokerGuard { previous }
}

pub async fn try_call_user_function(
    name: &str,
    args: &[Value],
) -> Option<Result<Value, RuntimeError>> {
    let invoker = USER_FUNCTION_INVOKER.with(|slot| slot.borrow().clone());
    let invoker = invoker?;
    let result = invoker(name, args).await;
    match result {
        Err(err)
            if err
                .identifier()
                .map(|id| id.ends_with("UndefinedFunction"))
                .unwrap_or(false)
                && err
                    .message()
                    .contains(&format!("Undefined function: {name}")) =>
        {
            None
        }
        other => Some(other),
    }
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
