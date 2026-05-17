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
    pub(crate) fn semantic(
        function: FunctionId,
        args: Vec<Value>,
        requested_outputs: usize,
    ) -> Self {
        Self::semantic_inner(
            function.0,
            None,
            CallableFallbackPolicy::None,
            args,
            requested_outputs,
            CallableMetadata::default(),
        )
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

    pub(crate) fn semantic_named(
        function: FunctionId,
        name: String,
        args: Vec<Value>,
        requested_outputs: usize,
    ) -> Self {
        Self::semantic_inner(
            function.0,
            Some(name.clone()),
            CallableFallbackPolicy::None,
            args,
            requested_outputs,
            CallableMetadata {
                display_name: Some(name),
                ..CallableMetadata::default()
            },
        )
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
                    return Self::feval_semantic(
                        function.0,
                        name,
                        CallableFallbackPolicy::RuntimeNameResolution,
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
                CallableFallbackPolicy::RuntimeNameResolution,
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

async fn execute_resolved_callable(
    identity: CallableIdentity,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
    fallback_policy: CallableFallbackPolicy,
) -> Result<Value, RuntimeError> {
    match identity {
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
            if matches!(
                fallback_policy,
                CallableFallbackPolicy::BuiltinByName
                    | CallableFallbackPolicy::RuntimeNameResolution
            ) {
                if let Some(name) = metadata.display_name.clone() {
                    return forward_builtin_feval(Value::FunctionHandle(name), args).await;
                }
            }
            Err(semantic_unavailable_error(function.0, &metadata))
        }
        other => match fallback_policy {
            CallableFallbackPolicy::BuiltinByName
            | CallableFallbackPolicy::RuntimeNameResolution => {
                let Some(name) = other.display_name() else {
                    return Err(undefined_name_error("<unnamed callable>", &metadata));
                };
                forward_builtin_feval(Value::FunctionHandle(name), args).await
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
    metadata: CallableMetadata,
    fallback_policy: CallableFallbackPolicy,
) -> Result<Option<Value>, RuntimeError> {
    match identity {
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
            if matches!(
                fallback_policy,
                CallableFallbackPolicy::BuiltinByName
                    | CallableFallbackPolicy::RuntimeNameResolution
            ) {
                if let Some(name) = metadata.display_name {
                    return forward_builtin_feval(Value::FunctionHandle(name), args)
                        .await
                        .map(Some);
                }
            }
            Ok(None)
        }
        other => {
            if !matches!(
                fallback_policy,
                CallableFallbackPolicy::BuiltinByName
                    | CallableFallbackPolicy::RuntimeNameResolution
            ) {
                return Ok(None);
            }
            let Some(name) = other.display_name() else {
                return Ok(None);
            };
            forward_builtin_feval(Value::FunctionHandle(name), args)
                .await
                .map(Some)
        }
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
            forward_builtin_feval(func_value, args).await.map(Some)
        }
    }
}
