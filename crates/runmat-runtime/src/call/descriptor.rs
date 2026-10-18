use crate::call::identity::strict_callable_display_name;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_types::{
    BuiltinId, CallableFallbackPolicy, CallableIdentity, FunctionId, QualifiedName, SymbolName,
};
use runmat_value::{Closure, Value};

/// The resolver exhausted callable selection at this source callsite. A
/// selected callee's error never carries this cause, even if it uses the same
/// public diagnostic identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MissingCallableTarget {
    source: crate::source_context::RuntimeSourceRef,
    invocation_arity: crate::context::RuntimeInvocationArity,
}

impl MissingCallableTarget {
    pub fn in_source_script(&self, source: crate::source_context::RuntimeSourceRef) -> bool {
        self.source == source
            && self.invocation_arity == crate::context::RuntimeInvocationArity::Script
    }
}

impl std::fmt::Display for MissingCallableTarget {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "no callable target selected at source {:?}",
            self.source.source_id()
        )
    }
}

impl std::error::Error for MissingCallableTarget {}

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

#[derive(Debug, Clone, PartialEq)]
pub struct CallableMetadata {
    pub call_kind: CallableCallKind,
    pub display_name: Option<String>,
    pub source_id: Option<runmat_types::SourceId>,
    pub source_ref: Option<crate::source_context::RuntimeSourceRef>,
    pub source_invocation: Option<crate::context::RuntimeInvocationArity>,
    pub span: Option<runmat_types::Span>,
    pub function_input_callsite: Option<crate::callsite::CallsiteInfo>,
}

impl Default for CallableMetadata {
    fn default() -> Self {
        Self {
            call_kind: CallableCallKind::Direct,
            display_name: None,
            source_id: None,
            source_ref: None,
            source_invocation: None,
            span: None,
            function_input_callsite: None,
        }
    }
}

impl CallableMetadata {
    fn feval(display_name: Option<String>) -> Self {
        Self {
            call_kind: CallableCallKind::Feval,
            display_name,
            source_id: None,
            source_ref: None,
            source_invocation: None,
            span: None,
            function_input_callsite: None,
        }
    }
}

/// One closure stored in its canonical Value representation so root scans can
/// borrow the complete capture and lexical-activation graph without cloning it.
#[derive(Debug, Clone)]
pub struct RetainedClosureValue(runmat_value::SharedValueGeneration);

impl RetainedClosureValue {
    pub fn new(closure: Closure) -> Self {
        Self(runmat_value::SharedValueGeneration::unbounded(
            Value::Closure(closure),
        ))
    }

    pub fn as_value(&self) -> &Value {
        self.0.value()
    }

    pub fn as_closure(&self) -> &Closure {
        match self.0.value() {
            Value::Closure(closure) => closure,
            _ => unreachable!("retained closure is constructed only from Closure"),
        }
    }

    /// Reuse an already-rooted callable generation without cloning its
    /// capture payload. The caller retains the supplied read lease until this
    /// root-free descriptor owner is installed in another rooted boundary.
    pub(crate) fn from_read_lease(
        lease: &crate::execution::RetainedValueReadLease,
    ) -> Result<Self, RuntimeError> {
        if !matches!(lease.value(), Value::Closure(_)) {
            return Err(crate::runtime_error::semantic_error(
                "CallableArguments",
                "retained callable owner is not a closure",
            ));
        }
        Ok(Self(lease.shared_generation()))
    }

    pub(crate) fn generation(&self) -> runmat_value::SharedValueGeneration {
        self.0.clone()
    }

    pub(crate) fn retain_for_call(
        &self,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> Result<crate::execution::RetainedValueReadLease, RuntimeError> {
        crate::execution::RetainedValueReadLease::retain_shared_generation(
            self.generation(),
            runtime,
        )
    }
}

impl PartialEq for RetainedClosureValue {
    fn eq(&self, other: &Self) -> bool {
        self.as_value() == other.as_value()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CallableTarget {
    Bound(runmat_value::BoundFunctionBinding),
    /// Preserve the traceable closure and its ancestor environment through
    /// queued callback admission; flattened captures alone are insufficient.
    BoundLexical {
        binding: runmat_value::BoundFunctionBinding,
        closure: RetainedClosureValue,
    },
    CapturedResolved {
        identity: CallableIdentity,
        fallback_policy: CallableFallbackPolicy,
        closure: RetainedClosureValue,
    },
    Resolved {
        identity: CallableIdentity,
        fallback_policy: CallableFallbackPolicy,
    },
    FevalForward(Value),
}

impl CallableTarget {
    /// Expose every value graph retained by a deferred callable target.
    pub(crate) fn visit_retained_values(
        &self,
        visitor: &mut dyn FnMut(&Value) -> Option<()>,
    ) -> Option<()> {
        match self {
            Self::BoundLexical { closure, .. } | Self::CapturedResolved { closure, .. } => {
                visitor(closure.as_value())
            }
            Self::FevalForward(value) => visitor(value),
            Self::Bound(_) | Self::Resolved { .. } => Some(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallableDescriptor {
    pub target: CallableTarget,
    pub args: Vec<Value>,
    pub requested_outputs: usize,
    pub metadata: CallableMetadata,
}

impl CallableDescriptor {
    /// Visit arguments and the callable target as one retained graph.
    pub(crate) fn visit_retained_values(
        &self,
        visitor: &mut dyn FnMut(&Value) -> Option<()>,
    ) -> Option<()> {
        for value in &self.args {
            visitor(value)?;
        }
        self.target.visit_retained_values(visitor)
    }

    pub fn bound(
        binding: runmat_value::BoundFunctionBinding,
        args: Vec<Value>,
        requested_outputs: usize,
        call_kind: CallableCallKind,
    ) -> Self {
        Self {
            target: CallableTarget::Bound(binding),
            args,
            requested_outputs,
            metadata: CallableMetadata {
                call_kind,
                ..CallableMetadata::default()
            },
        }
    }

    pub fn with_function_input_callsite(mut self, callsite: crate::callsite::CallsiteInfo) -> Self {
        self.metadata.function_input_callsite = Some(callsite);
        self
    }
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

    fn feval_bound(
        binding: runmat_value::BoundFunctionBinding,
        name: String,
        args: Vec<Value>,
        requested_outputs: usize,
    ) -> Self {
        Self {
            target: CallableTarget::Bound(binding),
            args,
            requested_outputs,
            metadata: CallableMetadata::feval(Some(name)),
        }
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
                Self::feval_resolved_name(
                    identity,
                    name.into_name(),
                    fallback_policy,
                    args,
                    requested_outputs,
                )
            }
            Value::ExternalFunctionHandle(name) => {
                let (identity, fallback_policy) =
                    Self::resolve_named_target(&name, function_resolver);
                Self::feval_resolved_name(
                    identity,
                    name.into_name(),
                    fallback_policy,
                    args,
                    requested_outputs,
                )
            }
            Value::MethodFunctionHandle(name) => Self::feval_resolved_name(
                CallableIdentity::Method(runmat_types::MethodId(name.name().to_owned())),
                name.into_name(),
                CallableFallbackPolicy::RuntimeNameResolution,
                args,
                requested_outputs,
            ),
            Value::BoundFunctionHandle(handle) => {
                let data = handle.into_data();
                Self::feval_bound(
                    runmat_value::BoundFunctionBinding {
                        function: data.function,
                        authority: data.authority,
                    },
                    data.name,
                    args,
                    requested_outputs,
                )
            }
            other => Self::feval_forward(other, args, requested_outputs),
        }
    }

    fn from_closure(
        closure: Closure,
        args: Vec<Value>,
        requested_outputs: usize,
        function_resolver: &impl FunctionNameResolver,
    ) -> Self {
        let name = closure.function_name.clone();
        if let Some(binding) = closure.binding.clone() {
            return Self {
                target: CallableTarget::BoundLexical {
                    binding,
                    closure: RetainedClosureValue::new(closure),
                },
                args,
                requested_outputs,
                metadata: CallableMetadata::feval(Some(name)),
            };
        }
        let (identity, fallback_policy) = Self::resolve_named_target(&name, function_resolver);
        Self {
            target: CallableTarget::CapturedResolved {
                identity,
                fallback_policy,
                closure: RetainedClosureValue::new(closure),
            },
            args,
            requested_outputs,
            metadata: CallableMetadata::feval(Some(name)),
        }
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
    let message = format!(
        "Undefined function in {}: {identity:?}{location}",
        metadata.call_kind.label()
    );
    match (metadata.source_ref, metadata.source_invocation) {
        (Some(source), Some(invocation_arity)) => crate::runtime_error::semantic_error_with_source(
            "UndefinedFunction",
            message,
            MissingCallableTarget {
                source,
                invocation_arity,
            },
        ),
        _ => semantic_error("UndefinedFunction", message),
    }
}

async fn call_builtin_with_requested_outputs(
    runtime: Option<&crate::context::RuntimeContext>,
    name: &str,
    args: &[Value],
    requested_outputs: usize,
    callsite: Option<crate::callsite::CallsiteInfo>,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    match runtime {
        Some(runtime) => {
            runtime
                .scope(crate::callsite::with_callsite(
                    callsite,
                    crate::call_builtin_sequence_async_with_outputs_in_context(
                        runtime,
                        name,
                        args,
                        requested_outputs,
                    ),
                ))
                .await?
        }
        None => {
            crate::callsite::with_callsite(
                callsite,
                crate::call_builtin_sequence_async_with_outputs(name, args, requested_outputs),
            )
            .await?
        }
    }
}

async fn forward_named_fallback(
    runtime: Option<&crate::context::RuntimeContext>,
    name: String,
    args: Vec<Value>,
    requested_outputs: usize,
    callsite: Option<crate::callsite::CallsiteInfo>,
    source_ref: Option<crate::source_context::RuntimeSourceRef>,
    source_invocation: Option<crate::context::RuntimeInvocationArity>,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    let invoke = crate::callsite::with_callsite(
        callsite,
        crate::dispatcher::try_call_builtin_sequence_async_with_outputs(
            &name,
            &args,
            requested_outputs,
            runtime,
        ),
    );
    match runtime {
        Some(runtime) => runtime.scope(invoke).await??,
        None => invoke.await??,
    }
    .ok_or_else(|| {
        let builder = crate::build_runtime_error(format!("Undefined function: {name}"))
            .with_identifier("RunMat:UndefinedFunction")
            .with_failure_origin(crate::runtime_error::RuntimeFailureOrigin::LanguageEvaluation);
        match (source_ref, source_invocation) {
            (Some(source), Some(invocation_arity)) => builder
                .with_source(MissingCallableTarget {
                    source,
                    invocation_arity,
                })
                .build(),
            _ => builder.build(),
        }
    })
}

#[path = "descriptor/execution.rs"]
mod execution;
pub(crate) use execution::try_execute_retained_bound_closure_in_context;

#[path = "descriptor/resolved_execution.rs"]
mod resolved_execution;

pub fn execute_callable_descriptor(
    descriptor: CallableDescriptor,
) -> impl std::future::Future<Output = Result<crate::execution::RootedValueSequence, RuntimeError>>
{
    // Construct the selected execution future now: a bound lexical target
    // roots its closure before an unpolled caller can enqueue or discard it.
    execution::execute(None, descriptor)
}

pub fn execute_callable_descriptor_in_context<'a>(
    runtime: &'a crate::context::RuntimeContext,
    descriptor: CallableDescriptor,
) -> impl std::future::Future<Output = Result<crate::execution::RootedValueSequence, RuntimeError>> + 'a
{
    runtime.scope(execution::execute(Some(runtime), descriptor))
}

#[cfg(test)]
#[path = "descriptor/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "descriptor/lexical_roots_tests.rs"]
mod lexical_roots_tests;

pub fn try_execute_callable_descriptor(
    descriptor: CallableDescriptor,
) -> impl std::future::Future<Output = Result<Option<crate::execution::RootedValueSequence>, RuntimeError>>
{
    execution::try_execute(None, descriptor)
}

pub fn try_execute_callable_descriptor_in_context<'a>(
    runtime: &'a crate::context::RuntimeContext,
    descriptor: CallableDescriptor,
) -> impl std::future::Future<
    Output = Result<Option<crate::execution::RootedValueSequence>, RuntimeError>,
> + 'a {
    runtime.scope(execution::try_execute(Some(runtime), descriptor))
}
