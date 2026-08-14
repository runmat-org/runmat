use crate::call::identity::strict_callable_display_name;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_types::{
    BuiltinId, CallableFallbackPolicy, CallableIdentity, FunctionId, QualifiedName, SymbolName,
};
use runmat_value::{Closure, Value};

/// Executor adapter used only to map stable source names to semantic functions.
pub trait FunctionNameResolver {
    fn resolve_function(&self, name: &str) -> Option<FunctionId>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallableCallKind {
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
pub struct CallableMetadata {
    pub call_kind: CallableCallKind,
    pub display_name: Option<String>,
    pub source_id: Option<runmat_types::SourceId>,
    pub span: Option<runmat_types::Span>,
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
pub enum CallableTarget {
    Resolved {
        identity: CallableIdentity,
        fallback_policy: CallableFallbackPolicy,
    },
    FevalForward(Value),
}

#[derive(Debug, Clone)]
pub struct CallableDescriptor {
    pub target: CallableTarget,
    pub args: Vec<Value>,
    pub requested_outputs: usize,
    pub metadata: CallableMetadata,
}

impl CallableDescriptor {
    fn parse_handle_name(text: &str) -> Option<String> {
        let handle = text.trim().strip_prefix('@').unwrap_or(text.trim()).trim();
        if handle.is_empty() {
            None
        } else {
            Some(handle.to_string())
        }
    }

    fn is_at_prefixed_text(text: &str) -> bool {
        text.trim().starts_with('@')
    }

    fn qualified_identity_from_name(name: &str) -> CallableIdentity {
        if Self::is_well_formed_qualified_name(name) {
            let segments = name
                .split('.')
                .map(|segment| SymbolName(segment.to_string()))
                .collect::<Vec<_>>();
            CallableIdentity::ExternalName(QualifiedName(segments))
        } else {
            // Preserve malformed dotted names as a single segment instead of silently normalizing.
            CallableIdentity::ExternalName(QualifiedName(vec![SymbolName(name.to_string())]))
        }
    }

    fn is_well_formed_qualified_name(name: &str) -> bool {
        let segments = name.split('.').collect::<Vec<_>>();
        segments.len() > 1 && segments.iter().all(|segment| !segment.is_empty())
    }

    fn function_inner(
        function: usize,
        name: Option<String>,
        fallback_policy: CallableFallbackPolicy,
        args: Vec<Value>,
        requested_outputs: usize,
        metadata: CallableMetadata,
    ) -> Self {
        let identity = CallableIdentity::BoundFunction(FunctionId(function));
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
        metadata.display_name = metadata
            .display_name
            .or(display_name)
            .or_else(|| strict_callable_display_name(&identity));
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
        Self::function_inner(
            function,
            Some(name.clone()),
            fallback_policy,
            args,
            requested_outputs,
            CallableMetadata::feval(Some(name)),
        )
    }

    fn feval_resolved_name(
        identity: CallableIdentity,
        name: String,
        fallback_policy: CallableFallbackPolicy,
        args: Vec<Value>,
        requested_outputs: usize,
    ) -> Self {
        Self::resolved_inner(
            identity,
            Some(name.clone()),
            fallback_policy,
            args,
            requested_outputs,
            CallableMetadata::feval(Some(name)),
        )
    }

    fn resolve_named_target(
        name: &str,
        function_resolver: &impl FunctionNameResolver,
    ) -> (CallableIdentity, CallableFallbackPolicy) {
        if let Some(function) = function_resolver.resolve_function(name) {
            return (
                CallableIdentity::BoundFunction(function),
                CallableFallbackPolicy::None,
            );
        }
        if runmat_builtins::builtin_name_is_known(name) {
            return (
                CallableIdentity::Builtin(BuiltinId(name.to_string())),
                CallableFallbackPolicy::None,
            );
        }
        if Self::is_well_formed_qualified_name(name) {
            return (
                Self::qualified_identity_from_name(name),
                CallableFallbackPolicy::ExternalBoundary,
            );
        }
        (
            CallableIdentity::DynamicName(SymbolName(name.to_string())),
            CallableFallbackPolicy::RuntimeNameResolution,
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

    pub fn resolved(
        identity: CallableIdentity,
        args: Vec<Value>,
        requested_outputs: usize,
        fallback_policy: CallableFallbackPolicy,
        call_kind: CallableCallKind,
    ) -> Self {
        Self::resolved_inner(
            identity,
            None,
            fallback_policy,
            args,
            requested_outputs,
            CallableMetadata {
                call_kind,
                ..CallableMetadata::default()
            },
        )
    }

    pub fn from_feval_value(
        func_val: Value,
        args: Vec<Value>,
        requested_outputs: usize,
        function_resolver: &impl FunctionNameResolver,
    ) -> Self {
        match func_val {
            Value::String(text) => {
                if Self::is_at_prefixed_text(&text) {
                    return Self::feval_forward(Value::String(text), args, requested_outputs);
                }
                if let Some(name) = Self::parse_handle_name(&text) {
                    let (identity, fallback_policy) =
                        Self::resolve_named_target(&name, function_resolver);
                    return Self::feval_resolved_name(
                        identity,
                        name,
                        fallback_policy,
                        args,
                        requested_outputs,
                    );
                }
                Self::feval_forward(Value::String(text), args, requested_outputs)
            }
            Value::CharArray(ca) if ca.rows == 1 => {
                let text: String = ca.data.iter().collect();
                if Self::is_at_prefixed_text(&text) {
                    return Self::feval_forward(Value::CharArray(ca), args, requested_outputs);
                }
                if let Some(name) = Self::parse_handle_name(&text) {
                    let (identity, fallback_policy) =
                        Self::resolve_named_target(&name, function_resolver);
                    return Self::feval_resolved_name(
                        identity,
                        name,
                        fallback_policy,
                        args,
                        requested_outputs,
                    );
                }
                Self::feval_forward(Value::CharArray(ca), args, requested_outputs)
            }
            Value::StringArray(sa) if sa.data.len() == 1 => {
                if Self::is_at_prefixed_text(&sa.data[0]) {
                    return Self::feval_forward(Value::StringArray(sa), args, requested_outputs);
                }
                if let Some(name) = Self::parse_handle_name(&sa.data[0]) {
                    let (identity, fallback_policy) =
                        Self::resolve_named_target(&name, function_resolver);
                    return Self::feval_resolved_name(
                        identity,
                        name,
                        fallback_policy,
                        args,
                        requested_outputs,
                    );
                }
                Self::feval_forward(Value::StringArray(sa), args, requested_outputs)
            }
            Value::Closure(closure) => {
                Self::from_closure(closure, args, requested_outputs, function_resolver)
            }
            Value::FunctionHandle(name) => {
                let (identity, fallback_policy) =
                    Self::resolve_named_target(&name, function_resolver);
                Self::feval_resolved_name(identity, name, fallback_policy, args, requested_outputs)
            }
            Value::ExternalFunctionHandle(name) => {
                let (identity, fallback_policy) =
                    Self::resolve_named_target(&name, function_resolver);
                Self::feval_resolved_name(identity, name, fallback_policy, args, requested_outputs)
            }
            Value::MethodFunctionHandle(name) => Self::feval_resolved_name(
                CallableIdentity::Method(runmat_types::MethodId(name.clone())),
                name,
                CallableFallbackPolicy::RuntimeNameResolution,
                args,
                requested_outputs,
            ),
            Value::BoundFunctionHandle { name, function } => Self::feval_semantic(
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
        function_resolver: &impl FunctionNameResolver,
    ) -> Self {
        let name = closure.function_name;
        let mut call_args = closure.captures;
        call_args.extend(args);
        if let Some(function) = closure.bound_function {
            return Self::feval_semantic(
                function,
                name,
                CallableFallbackPolicy::None,
                call_args,
                requested_outputs,
            );
        }
        if let Some(function) = function_resolver.resolve_function(&name) {
            return Self::feval_semantic(
                function.0,
                name,
                CallableFallbackPolicy::None,
                call_args,
                requested_outputs,
            );
        }
        let (identity, fallback_policy) = Self::resolve_named_target(&name, function_resolver);
        Self::feval_resolved_name(
            identity,
            name,
            fallback_policy,
            call_args,
            requested_outputs,
        )
    }
}

fn function_unavailable_error(function: usize, metadata: &CallableMetadata) -> RuntimeError {
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
    semantic_error(
        "UndefinedSemanticFunction",
        format!(
            "{}{} could not invoke semantic function {function}{location}",
            metadata.call_kind.label(),
            display,
        ),
    )
}

fn undefined_identity_error(
    identity: &CallableIdentity,
    metadata: &CallableMetadata,
) -> RuntimeError {
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
    semantic_error(
        "UndefinedFunction",
        format!(
            "Undefined function in {}: {identity:?}{location}",
            metadata.call_kind.label()
        ),
    )
}

async fn call_builtin_with_requested_outputs(
    name: &str,
    args: &[Value],
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
    crate::call_builtin_async_with_outputs(name, args, requested_outputs).await
}

async fn forward_named_fallback(
    name: String,
    args: Vec<Value>,
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
    match crate::call_builtin_async_with_outputs(&name, &args, requested_outputs).await {
        Ok(value) => Ok(value),
        Err(err) if err.identifier() == Some("RunMat:UndefinedFunction") => {
            crate::call_feval_async_with_outputs(
                Value::FunctionHandle(name),
                &args,
                requested_outputs,
            )
            .await
        }
        Err(err) => Err(err),
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
        CallableIdentity::BoundFunction(function) => {
            if let Some(result) = crate::user_functions::try_call_semantic_function(
                function.0,
                &args,
                requested_outputs,
            )
            .await
            {
                return result;
            }
            Err(function_unavailable_error(function.0, &metadata))
        }
        CallableIdentity::ExternalFunction {
            function,
            display_name,
        } => {
            if let Some(result) = crate::user_functions::try_call_external_function(
                crate::user_functions::ExternalFunctionCall {
                    function: function.0,
                    display_name,
                    arguments: args.clone(),
                    requested_outputs,
                },
            )
            .await
            {
                return result;
            }
            if let Some(result) = crate::user_functions::try_call_semantic_function(
                function.0,
                &args,
                requested_outputs,
            )
            .await
            {
                return result;
            }
            Err(function_unavailable_error(function.0, &metadata))
        }
        other => {
            let request = crate::user_functions::CallableRequest::resolved(
                other.clone(),
                fallback_policy,
                args.clone(),
                requested_outputs,
            );
            if let Some(result) = crate::user_functions::try_call_semantic_descriptor(request).await
            {
                return result;
            }
            let Some(name) = fallback_policy.vm_fallback_name_for(&other) else {
                return Err(undefined_identity_error(&other, &metadata));
            };
            forward_named_fallback(name, args, requested_outputs).await
        }
    }
}

async fn try_execute_resolved_callable(
    identity: CallableIdentity,
    args: Vec<Value>,
    requested_outputs: usize,
    fallback_policy: CallableFallbackPolicy,
) -> Result<Option<Value>, RuntimeError> {
    match identity {
        CallableIdentity::Builtin(id) => {
            call_builtin_with_requested_outputs(&id.0, &args, requested_outputs)
                .await
                .map(Some)
        }
        CallableIdentity::BoundFunction(function) => {
            if let Some(result) = crate::user_functions::try_call_semantic_function(
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
        CallableIdentity::ExternalFunction {
            function,
            display_name,
        } => {
            if let Some(result) = crate::user_functions::try_call_external_function(
                crate::user_functions::ExternalFunctionCall {
                    function: function.0,
                    display_name,
                    arguments: args.clone(),
                    requested_outputs,
                },
            )
            .await
            {
                return result.map(Some);
            }
            if let Some(result) = crate::user_functions::try_call_semantic_function(
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
        other => {
            let request = crate::user_functions::CallableRequest::resolved(
                other.clone(),
                fallback_policy,
                args.clone(),
                requested_outputs,
            );
            if let Some(result) = crate::user_functions::try_call_semantic_descriptor(request).await
            {
                return result.map(Some);
            }
            let Some(name) = fallback_policy.vm_fallback_name_for(&other) else {
                return Ok(None);
            };
            match forward_named_fallback(name, args, requested_outputs).await {
                Ok(value) => Ok(Some(value)),
                Err(err) if err.identifier() == Some("RunMat:UndefinedFunction") => Ok(None),
                Err(err) => Err(err),
            }
        }
    }
}

pub async fn execute_callable_descriptor(
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
            crate::call_feval_async_with_outputs(func_value, &args, requested_outputs).await
        }
    }
}

pub async fn try_execute_callable_descriptor(
    descriptor: CallableDescriptor,
) -> Result<Option<Value>, RuntimeError> {
    let CallableDescriptor {
        target,
        args,
        requested_outputs,
        metadata: _,
    } = descriptor;
    match target {
        CallableTarget::Resolved {
            identity,
            fallback_policy,
        } => {
            try_execute_resolved_callable(identity, args, requested_outputs, fallback_policy).await
        }
        CallableTarget::FevalForward(func_value) => {
            crate::call_feval_async_with_outputs(func_value, &args, requested_outputs)
                .await
                .map(Some)
        }
    }
}
