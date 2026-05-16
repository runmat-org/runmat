use crate::bytecode::SemanticFunctionRegistry;
use crate::call::feval::{forward_builtin_feval, try_closure_builtin_fallback};
use runmat_builtins::{Closure, Value};
use runmat_runtime::RuntimeError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CallableCallKind {
    Direct,
    Feval,
    EndExpr,
}

impl CallableCallKind {
    fn label(self) -> &'static str {
        match self {
            CallableCallKind::Direct => "direct call",
            CallableCallKind::Feval => "feval call",
            CallableCallKind::EndExpr => "end-expression call",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CallableMetadata {
    pub(crate) call_kind: CallableCallKind,
    pub(crate) display_name: Option<String>,
    pub(crate) source_id: Option<runmat_hir::SourceId>,
    pub(crate) span: Option<runmat_hir::Span>,
}

impl Default for CallableMetadata {
    fn default() -> Self {
        Self {
            call_kind: CallableCallKind::Direct,
            display_name: None,
            source_id: None,
            span: None,
        }
    }
}

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
    pub(crate) metadata: CallableMetadata,
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
            metadata: CallableMetadata::default(),
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
                name: Some(name.clone()),
                name_fallback: false,
            },
            args,
            requested_outputs,
            metadata: CallableMetadata {
                display_name: Some(name),
                ..CallableMetadata::default()
            },
        }
    }

    pub(crate) fn with_call_kind(mut self, call_kind: CallableCallKind) -> Self {
        self.metadata.call_kind = call_kind;
        self
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
                            name: Some(name.clone()),
                            name_fallback: true,
                        },
                        args,
                        requested_outputs,
                        metadata: CallableMetadata {
                            call_kind: CallableCallKind::Feval,
                            display_name: Some(name),
                            source_id: None,
                            span: None,
                        },
                    };
                }
                Self {
                    target: CallableTarget::FevalForward(Value::FunctionHandle(name)),
                    args,
                    requested_outputs,
                    metadata: CallableMetadata {
                        call_kind: CallableCallKind::Feval,
                        ..CallableMetadata::default()
                    },
                }
            }
            Value::SemanticFunctionHandle { name, function } => Self {
                target: CallableTarget::Semantic {
                    function,
                    name: Some(name.clone()),
                    name_fallback: true,
                },
                args,
                requested_outputs,
                metadata: CallableMetadata {
                    call_kind: CallableCallKind::Feval,
                    display_name: Some(name),
                    source_id: None,
                    span: None,
                },
            },
            other => Self {
                target: CallableTarget::FevalForward(other),
                args,
                requested_outputs,
                metadata: CallableMetadata {
                    call_kind: CallableCallKind::Feval,
                    ..CallableMetadata::default()
                },
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
                    name: Some(name.clone()),
                    name_fallback: false,
                },
                args: call_args,
                requested_outputs,
                metadata: CallableMetadata {
                    call_kind: CallableCallKind::Feval,
                    display_name: Some(name),
                    source_id: None,
                    span: None,
                },
            };
        }
        if let Some(function) = semantic_registry.resolve_name(&name) {
            return Self {
                target: CallableTarget::Semantic {
                    function: function.0,
                    name: Some(name.clone()),
                    name_fallback: false,
                },
                args: call_args,
                requested_outputs,
                metadata: CallableMetadata {
                    call_kind: CallableCallKind::Feval,
                    display_name: Some(name),
                    source_id: None,
                    span: None,
                },
            };
        }
        Self {
            target: CallableTarget::NameOnlyBuiltinFallback(name.clone()),
            args: call_args,
            requested_outputs,
            metadata: CallableMetadata {
                call_kind: CallableCallKind::Feval,
                display_name: Some(name),
                source_id: None,
                span: None,
            },
        }
    }
}

fn semantic_unavailable_error(function: usize, metadata: &CallableMetadata) -> RuntimeError {
    let display = metadata
        .display_name
        .as_deref()
        .map(|name| format!(" '{name}'"))
        .unwrap_or_default();
    let location = match (metadata.source_id, metadata.span) {
        (Some(source_id), Some(span)) => {
            format!(
                " at source {:?} span {}..{}",
                source_id, span.start, span.end
            )
        }
        (Some(source_id), None) => format!(" at source {:?}", source_id),
        (None, Some(span)) => format!(" at span {}..{}", span.start, span.end),
        (None, None) => String::new(),
    };
    crate::interpreter::errors::mex(
        "UndefinedSemanticFunction",
        &format!(
            "{}{} could not invoke semantic function {function}{location}",
            metadata.call_kind.label(),
            display,
        ),
    )
}

fn undefined_name_error(name: &str, metadata: &CallableMetadata) -> RuntimeError {
    let location = match (metadata.source_id, metadata.span) {
        (Some(source_id), Some(span)) => {
            format!(
                " at source {:?} span {}..{}",
                source_id, span.start, span.end
            )
        }
        (Some(source_id), None) => format!(" at source {:?}", source_id),
        (None, Some(span)) => format!(" at span {}..{}", span.start, span.end),
        (None, None) => String::new(),
    };
    crate::interpreter::errors::mex(
        "UndefinedFunction",
        &format!(
            "Undefined function in {}: {name}{location}",
            metadata.call_kind.label()
        ),
    )
}

pub(crate) async fn execute_callable_descriptor(
    descriptor: CallableDescriptor,
) -> Result<Value, RuntimeError> {
    let CallableDescriptor {
        target,
        args,
        requested_outputs,
        metadata,
    } = descriptor;
    match target {
        CallableTarget::Semantic {
            function,
            name,
            name_fallback,
        } => {
            if let Some(result) = runmat_runtime::user_functions::try_call_semantic_function(
                function,
                &args,
                requested_outputs,
            )
            .await
            {
                return result;
            }
            if name_fallback {
                if let Some(name) = name {
                    return forward_builtin_feval(Value::FunctionHandle(name), args).await;
                }
            }
            Err(semantic_unavailable_error(function, &metadata))
        }
        CallableTarget::NameOnlyBuiltinFallback(name) => {
            if let Some(result) = try_closure_builtin_fallback(&name, &args).await? {
                return Ok(result);
            }
            Err(undefined_name_error(&name, &metadata))
        }
        CallableTarget::FevalForward(func_value) => forward_builtin_feval(func_value, args).await,
    }
}

pub(crate) async fn try_execute_callable_descriptor(
    descriptor: CallableDescriptor,
) -> Result<Option<Value>, RuntimeError> {
    let CallableDescriptor {
        target,
        args,
        requested_outputs,
        metadata,
    } = descriptor;
    match target {
        CallableTarget::Semantic {
            function,
            name,
            name_fallback,
        } => {
            if let Some(result) = runmat_runtime::user_functions::try_call_semantic_function(
                function,
                &args,
                requested_outputs,
            )
            .await
            {
                return result.map(Some);
            }
            if name_fallback {
                if let Some(name) = name {
                    return forward_builtin_feval(Value::FunctionHandle(name), args)
                        .await
                        .map(Some);
                }
            }
            Ok(None)
        }
        CallableTarget::NameOnlyBuiltinFallback(name) => {
            if let Some(result) = try_closure_builtin_fallback(&name, &args).await? {
                return Ok(Some(result));
            }
            Err(undefined_name_error(&name, &metadata))
        }
        CallableTarget::FevalForward(func_value) => {
            forward_builtin_feval(func_value, args).await.map(Some)
        }
    }
}
