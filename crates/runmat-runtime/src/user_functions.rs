use crate::RuntimeError;
use runmat_thread_local::runmat_thread_local;
use runmat_types::{CallableFallbackPolicy, CallableIdentity, SourceId};
use runmat_value::{Value, ValueSequence};
use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;

#[cfg(test)]
#[path = "user_functions/test_support.rs"]
pub mod test_support;

pub type UserFunctionFuture = Pin<Box<dyn Future<Output = Result<ValueSequence, RuntimeError>>>>;
pub type DynamicFunctionLoadFuture =
    Pin<Box<dyn Future<Output = Option<Result<ValueSequence, RuntimeError>>>>>;
pub type DynamicFunctionClearFuture = Pin<Box<dyn Future<Output = Result<(), RuntimeError>>>>;
pub type FunctionInvoker = dyn Fn(usize, &[Value], usize) -> UserFunctionFuture;
pub type BoundFunctionInvoker =
    dyn Fn(runmat_value::BoundFunctionBinding, &[Value], usize) -> UserFunctionFuture;
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
#[path = "user_functions/source_facts.rs"]
mod source_facts;
pub use source_facts::*;

/// The point in callable resolution at which a session-backed loader runs.
///
/// Native callable artifacts such as MEX modules have language-defined
/// precedence over source functions. Ordinary dynamically discovered source
/// files are consulted only after the analyzed semantic registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicFunctionLoadPhase {
    BeforeSemantic,
    AfterSemantic,
}

pub type DynamicFunctionLoader = dyn Fn(
    crate::context::RuntimeContext,
    String,
    Vec<Value>,
    usize,
    DynamicFunctionLoadPhase,
    Option<crate::callsite::CallsiteInfo>,
) -> DynamicFunctionLoadFuture;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicFunctionClearRequest {
    All,
    NativeExtensions,
    Named(String),
}

/// Refresh the source lookup inventory without clearing live native modules or workspace values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicFunctionRefreshRequest {
    Default,
    Path,
    Toolbox,
    ToolboxCache,
}

pub type DynamicFunctionRefresher = dyn Fn(
    crate::context::RuntimeContext,
    DynamicFunctionRefreshRequest,
) -> DynamicFunctionClearFuture;

pub type DynamicFunctionClearer = dyn Fn(
    crate::context::RuntimeContext,
    DynamicFunctionClearRequest,
) -> DynamicFunctionClearFuture;

#[path = "user_functions/call_dispatch.rs"]
mod call_dispatch;
pub use call_dispatch::*;

runmat_thread_local! {
    static SEMANTIC_FUNCTION_INVOKER: RefCell<Option<Rc<FunctionInvoker>>> =
        const { RefCell::new(None) };
    static BOUND_FUNCTION_INVOKER: RefCell<Option<Rc<BoundFunctionInvoker>>> =
        const { RefCell::new(None) };
    static EXTERNAL_FUNCTION_INVOKER: RefCell<Option<Rc<ExternalFunctionInvoker>>> =
        const { RefCell::new(None) };
    static LEXICAL_FUNCTION_INVOKER: RefCell<Option<Rc<LexicalFunctionInvoker>>> =
        const { RefCell::new(None) };
    static SEMANTIC_FUNCTION_RESOLVER: RefCell<Option<Arc<FunctionResolver>>> =
        const { RefCell::new(None) };
    static SOURCE_FUNCTION_CATALOG: RefCell<Option<Arc<Vec<SourceFunctionInfo>>>> =
        const { RefCell::new(None) };
    static EXECUTION_SOURCE_RESOLVER: RefCell<Option<Rc<ExecutionSourceResolver>>> =
        const { RefCell::new(None) };
    static ACTIVE_SEMANTIC_FUNCTION_STACK: RefCell<Vec<usize>> =
        const { RefCell::new(Vec::new()) };
}

#[path = "user_functions/ports.rs"]
mod ports;
pub use ports::*;

fn active_state() -> Option<std::rc::Rc<crate::context::RuntimeContextState>> {
    crate::context::legacy::active().map(|context| std::rc::Rc::clone(context.state()))
}

pub(crate) fn current_session_lease() -> std::sync::Weak<()> {
    active_state().map_or_else(std::sync::Weak::new, |state| {
        Arc::downgrade(&state.session_lease)
    })
}

#[cfg(test)]
#[path = "user_functions/input_callsite_tests.rs"]
mod input_callsite_tests;

#[cfg(test)]
#[path = "user_functions/binding_origin_tests.rs"]
mod binding_origin_tests;

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
                    outputs: runmat_value::ValueSequence::single(Value::Num(7.0)).unwrap(),
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
        assert_eq!(result.outputs.into_values(), vec![Value::Num(7.0)]);
        assert_eq!(result.captures[0].value, Value::Num(5.0));
        drop(guard);
        assert!(current_lexical_function_invoker().is_none());
    }

    #[test]
    fn external_invoker_preserves_identity_kind_and_restores_scope() {
        assert!(current_external_function_invoker().is_none());
        let guard = install_external_function_invoker(Some(Arc::new(|call| {
            crate::sequence::single_value_future(async move {
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
        assert_eq!(result.into_values(), vec![Value::Num(12.0)]);
        drop(guard);
        assert!(current_external_function_invoker().is_none());
    }
}
