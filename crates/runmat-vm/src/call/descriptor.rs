use crate::bytecode::SemanticFunctionRegistry;
use crate::call::feval::forward_builtin_feval;
use runmat_builtins::{Closure, Value};
use runmat_hir::{CallableFallbackPolicy, CallableIdentity, FunctionId, SymbolName};
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

impl CallableMetadata {
    fn feval(display_name: Option<String>) -> Self {
        Self {
            call_kind: CallableCallKind::Feval,
            display_name,
            source_id: None,
            span: None,
        }
    }
}

#[derive(Debug, Clone)]
enum CallableTarget {
    Resolved {
        identity: CallableIdentity,
        fallback_policy: CallableFallbackPolicy,
    },
    FevalForward(Value),
}

#[derive(Debug, Clone)]
pub(crate) struct CallableDescriptor {
    target: CallableTarget,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
}

impl CallableDescriptor {
    fn parse_at_handle_name(text: &str) -> Option<String> {
        let handle = text.trim().strip_prefix('@')?.trim();
        if handle.is_empty() {
            None
        } else {
            Some(handle.to_string())
        }
    }

    fn semantic_inner(
        function: usize,
        name: Option<String>,
        fallback_policy: CallableFallbackPolicy,
        args: Vec<Value>,
        requested_outputs: usize,
        metadata: CallableMetadata,
    ) -> Self {
        let identity = CallableIdentity::SemanticFunction(FunctionId(function));
        Self::resolved_inner(
            identity,
            name,
            fallback_policy,
            args,
            requested_outputs,
            metadata,
        )
    }

    fn resolved_inner(
        identity: CallableIdentity,
        display_name: Option<String>,
        fallback_policy: CallableFallbackPolicy,
        args: Vec<Value>,
        requested_outputs: usize,
        mut metadata: CallableMetadata,
    ) -> Self {
        metadata.display_name = metadata.display_name.or(display_name);
        Self {
            target: CallableTarget::Resolved {
                identity,
                fallback_policy,
            },
            args,
            requested_outputs,
            metadata,
        }
    }

    fn feval_semantic(
        function: usize,
        name: String,
        fallback_policy: CallableFallbackPolicy,
        args: Vec<Value>,
        requested_outputs: usize,
    ) -> Self {
        Self::semantic_inner(
            function,
            Some(name.clone()),
            fallback_policy,
            args,
            requested_outputs,
            CallableMetadata::feval(Some(name)),
        )
    }

    fn feval_forward(func_value: Value, args: Vec<Value>, requested_outputs: usize) -> Self {
        Self {
            target: CallableTarget::FevalForward(func_value),
            args,
            requested_outputs,
            metadata: CallableMetadata::feval(None),
        }
    }

    pub(crate) fn dynamic_named(
        name: String,
        args: Vec<Value>,
        requested_outputs: usize,
        fallback_policy: CallableFallbackPolicy,
        call_kind: CallableCallKind,
    ) -> Self {
        Self::resolved_inner(
            CallableIdentity::DynamicName(SymbolName(name.clone())),
            Some(name.clone()),
            fallback_policy,
            args,
            requested_outputs,
            CallableMetadata {
                call_kind,
                display_name: Some(name),
                ..CallableMetadata::default()
            },
        )
    }

    pub(crate) fn resolved(
        identity: CallableIdentity,
        display_name: Option<String>,
        args: Vec<Value>,
        requested_outputs: usize,
        fallback_policy: CallableFallbackPolicy,
        call_kind: CallableCallKind,
    ) -> Self {
        Self::resolved_inner(
            identity,
            display_name.clone(),
            fallback_policy,
            args,
            requested_outputs,
            CallableMetadata {
                call_kind,
                display_name,
                ..CallableMetadata::default()
            },
        )
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
            Value::String(text) => {
                if let Some(name) = Self::parse_at_handle_name(&text) {
                    if let Some(function) = semantic_registry.resolve_name(&name) {
                        return Self::feval_semantic(
                            function.0,
                            name,
                            CallableFallbackPolicy::None,
                            args,
                            requested_outputs,
                        );
                    }
                    return Self::dynamic_named(
                        name,
                        args,
                        requested_outputs,
                        CallableFallbackPolicy::RuntimeNameResolution,
                        CallableCallKind::Feval,
                    );
                }
                Self::feval_forward(Value::String(text), args, requested_outputs)
            }
            Value::CharArray(ca) if ca.rows == 1 => {
                let text: String = ca.data.iter().collect();
                if let Some(name) = Self::parse_at_handle_name(&text) {
                    if let Some(function) = semantic_registry.resolve_name(&name) {
                        return Self::feval_semantic(
                            function.0,
                            name,
                            CallableFallbackPolicy::None,
                            args,
                            requested_outputs,
                        );
                    }
                    return Self::dynamic_named(
                        name,
                        args,
                        requested_outputs,
                        CallableFallbackPolicy::RuntimeNameResolution,
                        CallableCallKind::Feval,
                    );
                }
                Self::feval_forward(Value::CharArray(ca), args, requested_outputs)
            }
            Value::Closure(closure) => {
                Self::from_closure(closure, args, requested_outputs, semantic_registry)
            }
            Value::FunctionHandle(name) => {
                if let Some(function) = semantic_registry.resolve_name(&name) {
                    return Self::feval_semantic(
                        function.0,
                        name,
                        CallableFallbackPolicy::None,
                        args,
                        requested_outputs,
                    );
                }
                Self::dynamic_named(
                    name,
                    args,
                    requested_outputs,
                    CallableFallbackPolicy::RuntimeNameResolution,
                    CallableCallKind::Feval,
                )
            }
            Value::SemanticFunctionHandle { name, function } => Self::feval_semantic(
                function,
                name,
                CallableFallbackPolicy::None,
                args,
                requested_outputs,
            ),
            other => Self::feval_forward(other, args, requested_outputs),
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
            return Self::feval_semantic(
                function,
                name,
                CallableFallbackPolicy::None,
                call_args,
                requested_outputs,
            );
        }
        if let Some(function) = semantic_registry.resolve_name(&name) {
            return Self::feval_semantic(
                function.0,
                name,
                CallableFallbackPolicy::None,
                call_args,
                requested_outputs,
            );
        }
        Self::dynamic_named(
            name,
            call_args,
            requested_outputs,
            CallableFallbackPolicy::RuntimeNameResolution,
            CallableCallKind::Feval,
        )
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

async fn call_builtin_with_requested_outputs(
    name: &str,
    args: &[Value],
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
    if requested_outputs != 1 {
        runmat_runtime::call_builtin_async_with_outputs(name, args, requested_outputs).await
    } else {
        runmat_runtime::call_builtin_async(name, args).await
    }
}

async fn forward_named_fallback(
    name: String,
    args: Vec<Value>,
    requested_outputs: usize,
    fallback_policy: CallableFallbackPolicy,
) -> Result<Value, RuntimeError> {
    match fallback_policy {
        CallableFallbackPolicy::RuntimeNameResolution
        | CallableFallbackPolicy::ObjectDispatchThenRuntimeNameResolution => {
            forward_builtin_feval(Value::FunctionHandle(name), args, requested_outputs).await
        }
        CallableFallbackPolicy::None
        | CallableFallbackPolicy::ObjectDispatch
        | CallableFallbackPolicy::ExternalBoundary => unreachable!(),
    }
}

async fn execute_resolved_callable(
    identity: CallableIdentity,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
    fallback_policy: CallableFallbackPolicy,
) -> Result<Value, RuntimeError> {
    match identity {
        CallableIdentity::Builtin(id) => {
            call_builtin_with_requested_outputs(&id.0, &args, requested_outputs).await
        }
        CallableIdentity::SemanticFunction(function) => {
            if let Some(result) = runmat_runtime::user_functions::try_call_semantic_function(
                function.0,
                &args,
                requested_outputs,
            )
            .await
            {
                return result;
            }
            Err(semantic_unavailable_error(function.0, &metadata))
        }
        other => match fallback_policy {
            CallableFallbackPolicy::RuntimeNameResolution
            | CallableFallbackPolicy::ObjectDispatchThenRuntimeNameResolution => {
                let request = runmat_runtime::user_functions::SemanticCallableRequest::resolved(
                    other.clone(),
                    fallback_policy,
                    args.clone(),
                    requested_outputs,
                    runmat_runtime::user_functions::SemanticCallableKind::Other,
                );
                if let Some(result) =
                    runmat_runtime::user_functions::try_call_semantic_descriptor(request).await
                {
                    return result;
                }
                let Some(name) = other.display_name() else {
                    return Err(undefined_name_error("<unnamed callable>", &metadata));
                };
                forward_named_fallback(name, args, requested_outputs, fallback_policy).await
            }
            CallableFallbackPolicy::None
            | CallableFallbackPolicy::ObjectDispatch
            | CallableFallbackPolicy::ExternalBoundary => {
                let name = other
                    .display_name()
                    .unwrap_or_else(|| "<unnamed callable>".into());
                Err(undefined_name_error(&name, &metadata))
            }
        },
    }
}

async fn try_execute_resolved_callable(
    identity: CallableIdentity,
    args: Vec<Value>,
    requested_outputs: usize,
    _metadata: CallableMetadata,
    fallback_policy: CallableFallbackPolicy,
) -> Result<Option<Value>, RuntimeError> {
    match identity {
        CallableIdentity::Builtin(id) => {
            call_builtin_with_requested_outputs(&id.0, &args, requested_outputs)
                .await
                .map(Some)
        }
        CallableIdentity::SemanticFunction(function) => {
            if let Some(result) = runmat_runtime::user_functions::try_call_semantic_function(
                function.0,
                &args,
                requested_outputs,
            )
            .await
            {
                return result.map(Some);
            }
            Ok(None)
        }
        other => match fallback_policy {
            CallableFallbackPolicy::RuntimeNameResolution
            | CallableFallbackPolicy::ObjectDispatchThenRuntimeNameResolution => {
                let request = runmat_runtime::user_functions::SemanticCallableRequest::resolved(
                    other.clone(),
                    fallback_policy,
                    args.clone(),
                    requested_outputs,
                    runmat_runtime::user_functions::SemanticCallableKind::Other,
                );
                if let Some(result) =
                    runmat_runtime::user_functions::try_call_semantic_descriptor(request).await
                {
                    return result.map(Some);
                }
                let Some(name) = other.display_name() else {
                    return Ok(None);
                };
                forward_named_fallback(name, args, requested_outputs, fallback_policy)
                    .await
                    .map(Some)
            }
            CallableFallbackPolicy::None
            | CallableFallbackPolicy::ObjectDispatch
            | CallableFallbackPolicy::ExternalBoundary => Ok(None),
        },
    }
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
        CallableTarget::Resolved {
            identity,
            fallback_policy,
        } => {
            execute_resolved_callable(identity, args, requested_outputs, metadata, fallback_policy)
                .await
        }
        CallableTarget::FevalForward(func_value) => {
            forward_builtin_feval(func_value, args, requested_outputs).await
        }
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
        CallableTarget::Resolved {
            identity,
            fallback_policy,
        } => {
            try_execute_resolved_callable(
                identity,
                args,
                requested_outputs,
                metadata,
                fallback_policy,
            )
            .await
        }
        CallableTarget::FevalForward(func_value) => {
            forward_builtin_feval(func_value, args, requested_outputs)
                .await
                .map(Some)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{execute_callable_descriptor, CallableCallKind, CallableDescriptor};
    use futures::executor::block_on;
    use runmat_builtins::{Tensor, Value};
    use runmat_hir::{BuiltinId, CallableFallbackPolicy, CallableIdentity};

    #[test]
    fn builtin_descriptor_uses_requested_outputs_for_multi_result_calls() {
        let input = Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0], vec![1, 3]).expect("tensor"));
        let descriptor = CallableDescriptor::resolved(
            CallableIdentity::Builtin(BuiltinId("max".to_string())),
            Some("max".to_string()),
            vec![input],
            2,
            CallableFallbackPolicy::None,
            CallableCallKind::Direct,
        );
        let value = block_on(execute_callable_descriptor(descriptor)).expect("execute descriptor");
        match value {
            Value::OutputList(values) => assert_eq!(values.len(), 2),
            other => panic!("expected two-output list from builtin descriptor, got {other:?}"),
        }
    }

    #[test]
    fn builtin_descriptor_uses_requested_outputs_for_zero_result_calls() {
        let args = vec![Value::Num(9.0)];
        let expected = block_on(runmat_runtime::call_builtin_async_with_outputs(
            "sqrt", &args, 0,
        ))
        .expect("runtime builtin with explicit zero outputs");
        let descriptor = CallableDescriptor::resolved(
            CallableIdentity::Builtin(BuiltinId("sqrt".to_string())),
            Some("sqrt".to_string()),
            args,
            0,
            CallableFallbackPolicy::None,
            CallableCallKind::Direct,
        );
        let value = block_on(execute_callable_descriptor(descriptor)).expect("execute descriptor");
        assert_eq!(value, expected);
    }
}
