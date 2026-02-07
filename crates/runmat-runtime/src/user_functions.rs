use crate::RuntimeError;
use runmat_builtins::Value;
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub type UserFunctionFuture = Pin<Box<dyn Future<Output = Result<Value, RuntimeError>>>>;
pub type UserFunctionInvoker = dyn Fn(&str, &[Value]) -> UserFunctionFuture + Send + Sync;

runmat_thread_local! {
    static USER_FUNCTION_INVOKER: RefCell<Option<Arc<UserFunctionInvoker>>> =
        const { RefCell::new(None) };
}

pub struct UserFunctionInvokerGuard {
    previous: Option<Arc<UserFunctionInvoker>>,
}

impl Drop for UserFunctionInvokerGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        USER_FUNCTION_INVOKER.with(|slot| {
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

pub async fn try_call_user_function(
    name: &str,
    args: &[Value],
) -> Option<Result<Value, RuntimeError>> {
    let invoker = USER_FUNCTION_INVOKER.with(|slot| slot.borrow().clone());
    let invoker = invoker?;
    Some(invoker(name, args).await)
}
