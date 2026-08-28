use super::{foreign_error, ForeignErrorKind};
use std::future::Future;
use std::pin::Pin;

use crate::context::{RuntimeCallRequest, RuntimeContext};
use crate::RuntimeError;
use runmat_types::CallableIdentity;
use runmat_value::Value;

#[derive(Debug, Clone)]
pub struct ForeignCallbackRequest {
    pub callable: CallableIdentity,
    pub arguments: Vec<Value>,
    pub requested_outputs: usize,
}

/// Normalize every supported RunMat callable value into the executor-neutral
/// callback request shared by foreign adapters. Closure captures are prepended
/// exactly once before the request crosses the adapter boundary.
pub fn foreign_callback_request(
    callback: &Value,
    mut arguments: Vec<Value>,
    requested_outputs: usize,
) -> Result<ForeignCallbackRequest, RuntimeError> {
    use runmat_types::{FunctionId, MethodId};

    let callable = match callback {
        Value::FunctionHandle(name) => crate::callable_identity_for_handle_name(name).0,
        Value::ExternalFunctionHandle(name) => crate::external_callable_identity_for_name(name),
        Value::MethodFunctionHandle(name) => CallableIdentity::Method(MethodId(name.clone())),
        Value::BoundFunctionHandle { function, .. } => {
            CallableIdentity::BoundFunction(FunctionId(*function))
        }
        Value::Closure(closure) => {
            let mut captured = closure.captures.clone();
            captured.append(&mut arguments);
            arguments = captured;
            closure.bound_function.map_or_else(
                || crate::callable_identity_for_handle_name(&closure.function_name).0,
                |function| CallableIdentity::AnonymousFunction(FunctionId(function)),
            )
        }
        _ => {
            return Err(foreign_error(
                ForeignErrorKind::CallbackFailed,
                "foreign callback target is not a callable RunMat value",
            ));
        }
    };
    Ok(ForeignCallbackRequest {
        callable,
        arguments,
        requested_outputs,
    })
}

pub fn invoke_foreign_callback(
    context: RuntimeContext,
    request: ForeignCallbackRequest,
) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'static>> {
    Box::pin(async move {
        if context
            .cancellation()
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            return Err(foreign_error(
                ForeignErrorKind::CallbackFailed,
                "foreign callback was cancelled before invocation",
            ));
        }
        let service = context
            .service_ports()
            .require_call("foreign callback")
            .map_err(|error| error.into_runtime_error())?
            .clone();
        context
            .scope(service.invoke(RuntimeCallRequest {
                identity: request.callable,
                arguments: request.arguments,
                requested_outputs: request.requested_outputs,
            }))
            .await
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{RuntimeCallRequest, RuntimeCallService, RuntimeServicePorts};
    use crate::execution::RuntimeExecutionService;
    use runmat_types::SymbolName;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[derive(Default)]
    struct RecordingCallService {
        requests: RefCell<Vec<RuntimeCallRequest>>,
    }

    impl RuntimeCallService for RecordingCallService {
        fn resolve(&self, _name: &str) -> Option<usize> {
            None
        }

        fn invoke(
            &self,
            request: RuntimeCallRequest,
        ) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'static>> {
            self.requests.borrow_mut().push(request);
            Box::pin(async { Ok(Value::Num(42.0)) })
        }
    }

    #[test]
    fn callback_reenters_through_the_originating_runtime_context() {
        let service = Rc::new(RecordingCallService::default());
        let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(RuntimeServicePorts::default().with_call(service.clone()));
        let result = futures::executor::block_on(invoke_foreign_callback(
            context,
            ForeignCallbackRequest {
                callable: CallableIdentity::DynamicName(SymbolName("callback".into())),
                arguments: vec![Value::Num(7.0)],
                requested_outputs: 1,
            },
        ))
        .unwrap();

        assert_eq!(result, Value::Num(42.0));
        let requests = service.requests.borrow();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].requested_outputs, 1);
        assert_eq!(requests[0].arguments, vec![Value::Num(7.0)]);
    }

    #[test]
    fn callback_observes_originating_session_cancellation() {
        let service = Rc::new(RecordingCallService::default());
        let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(RuntimeServicePorts::default().with_call(service.clone()));
        context
            .cancellation()
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let error = futures::executor::block_on(invoke_foreign_callback(
            context,
            ForeignCallbackRequest {
                callable: CallableIdentity::DynamicName(SymbolName("callback".into())),
                arguments: Vec::new(),
                requested_outputs: 0,
            },
        ))
        .unwrap_err();

        assert_eq!(error.identifier(), Some("RunMat:Foreign:CallbackFailed"));
        assert!(service.requests.borrow().is_empty());
    }
}
