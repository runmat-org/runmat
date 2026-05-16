use crate::bytecode::SemanticFunctionRegistry;
use crate::call::feval::{forward_builtin_feval, try_closure_builtin_fallback};
use runmat_builtins::{Closure, Value};
use runmat_runtime::RuntimeError;

#[derive(Debug, Clone)]
pub(crate) enum CallableTarget {
    Semantic {
        function: usize,
        name: Option<String>,
        name_fallback: bool,
    },
    NameOnlyBuiltinFallback(String),
    FevalForward(Value),
}

#[derive(Debug, Clone)]
pub(crate) struct CallableDescriptor {
    pub(crate) target: CallableTarget,
    pub(crate) args: Vec<Value>,
    pub(crate) requested_outputs: usize,
}

impl CallableDescriptor {
    pub(crate) fn semantic(
        function: runmat_hir::FunctionId,
        args: Vec<Value>,
        requested_outputs: usize,
    ) -> Self {
        Self {
            target: CallableTarget::Semantic {
                function: function.0,
                name: None,
                name_fallback: false,
            },
            args,
            requested_outputs,
        }
    }

    pub(crate) fn semantic_named(
        function: runmat_hir::FunctionId,
        name: String,
        args: Vec<Value>,
        requested_outputs: usize,
    ) -> Self {
        Self {
            target: CallableTarget::Semantic {
                function: function.0,
                name: Some(name),
                name_fallback: false,
            },
            args,
            requested_outputs,
        }
    }

    pub(crate) fn from_feval_value(
        func_val: Value,
        args: Vec<Value>,
        requested_outputs: usize,
        semantic_registry: &SemanticFunctionRegistry,
    ) -> Self {
        match func_val {
            Value::Closure(closure) => {
                Self::from_closure(closure, args, requested_outputs, semantic_registry)
            }
            Value::FunctionHandle(name) => {
                if let Some(function) = semantic_registry.resolve_name(&name) {
                    return Self {
                        target: CallableTarget::Semantic {
                            function: function.0,
                            name: Some(name),
                            name_fallback: true,
                        },
                        args,
                        requested_outputs,
                    };
                }
                Self {
                    target: CallableTarget::FevalForward(Value::FunctionHandle(name)),
                    args,
                    requested_outputs,
                }
            }
            Value::SemanticFunctionHandle { name, function } => Self {
                target: CallableTarget::Semantic {
                    function,
                    name: Some(name),
                    name_fallback: true,
                },
                args,
                requested_outputs,
            },
            other => Self {
                target: CallableTarget::FevalForward(other),
                args,
                requested_outputs,
            },
        }
    }

    fn from_closure(
        closure: Closure,
        args: Vec<Value>,
        requested_outputs: usize,
        semantic_registry: &SemanticFunctionRegistry,
    ) -> Self {
        let name = closure.function_name;
        let mut call_args = closure.captures;
        call_args.extend(args);
        if let Some(function) = closure.semantic_function {
            return Self {
                target: CallableTarget::Semantic {
                    function,
                    name: Some(name),
                    name_fallback: false,
                },
                args: call_args,
                requested_outputs,
            };
        }
        if let Some(function) = semantic_registry.resolve_name(&name) {
            return Self {
                target: CallableTarget::Semantic {
                    function: function.0,
                    name: Some(name),
                    name_fallback: false,
                },
                args: call_args,
                requested_outputs,
            };
        }
        Self {
            target: CallableTarget::NameOnlyBuiltinFallback(name),
            args: call_args,
            requested_outputs,
        }
    }
}

pub(crate) async fn execute_callable_descriptor(
    descriptor: CallableDescriptor,
) -> Result<Value, RuntimeError> {
    match descriptor.target {
        CallableTarget::Semantic {
            function,
            name,
            name_fallback,
        } => {
            if let Some(result) = runmat_runtime::user_functions::try_call_semantic_function(
                function,
                &descriptor.args,
                descriptor.requested_outputs,
            )
            .await
            {
                return result;
            }
            if name_fallback {
                if let Some(name) = name {
                    return forward_builtin_feval(Value::FunctionHandle(name), descriptor.args)
                        .await;
                }
            }
            Err(crate::interpreter::errors::mex(
                "UndefinedSemanticFunction",
                &format!("semantic function invoker unavailable for {function}"),
            ))
        }
        CallableTarget::NameOnlyBuiltinFallback(name) => {
            if let Some(result) = try_closure_builtin_fallback(&name, &descriptor.args).await? {
                return Ok(result);
            }
            Err(crate::interpreter::errors::mex(
                "UndefinedFunction",
                &format!("Undefined function: {name}"),
            ))
        }
        CallableTarget::FevalForward(func_value) => {
            forward_builtin_feval(func_value, descriptor.args).await
        }
    }
}

pub(crate) async fn try_execute_callable_descriptor(
    descriptor: CallableDescriptor,
) -> Result<Option<Value>, RuntimeError> {
    match descriptor.target {
        CallableTarget::Semantic {
            function,
            name,
            name_fallback,
        } => {
            if let Some(result) = runmat_runtime::user_functions::try_call_semantic_function(
                function,
                &descriptor.args,
                descriptor.requested_outputs,
            )
            .await
            {
                return result.map(Some);
            }
            if name_fallback {
                if let Some(name) = name {
                    return forward_builtin_feval(Value::FunctionHandle(name), descriptor.args)
                        .await
                        .map(Some);
                }
            }
            Ok(None)
        }
        CallableTarget::NameOnlyBuiltinFallback(name) => {
            if let Some(result) = try_closure_builtin_fallback(&name, &descriptor.args).await? {
                return Ok(Some(result));
            }
            Err(crate::interpreter::errors::mex(
                "UndefinedFunction",
                &format!("Undefined function: {name}"),
            ))
        }
        CallableTarget::FevalForward(func_value) => {
            forward_builtin_feval(func_value, descriptor.args)
                .await
                .map(Some)
        }
    }
}
