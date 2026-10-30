use crate::call::identity::strict_callable_display_name;
use crate::call::identity::CallableIdentityOwner;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_types::{
    CallableFallbackPolicy, CallableIdentity, FunctionId, QualifiedName, SymbolName,
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

/// Whether an unresolved retained name can be frozen to a builtin target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NameResolutionCompleteness {
    /// A selected registry can prove that a semantic name is absent.
    Complete,
    /// Lookup still belongs to the original invocation's runtime name owner.
    Deferred,
}

/// Executor adapter used only to map stable source names to semantic functions.
pub trait FunctionNameResolver {
    fn resolve_function(&self, name: &str) -> Option<FunctionId>;

    /// Absence is conservative by default. An exact registry adapter may opt
    /// in only when `None` proves absence from its selected name environment.
    fn name_resolution_completeness(&self) -> NameResolutionCompleteness {
        NameResolutionCompleteness::Deferred
    }

    /// Retain the exact selected definition when the resolver owns one.
    /// Numeric identity remains a lookup hint, never retained execution
    /// authority. Resolvers without a definition owner keep the legacy
    /// current-publication behavior through the default.
    fn resolve_bound_function(&self, _name: &str) -> Option<runmat_value::BoundFunctionBinding> {
        None
    }
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
    pub display_name: Option<runmat_value::Utf8Text>,
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
    fn feval(display_name: Option<runmat_value::Utf8Text>) -> Self {
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
pub struct RetainedClosureValue(RetainedClosureOwner);

#[derive(Debug, Clone)]
enum RetainedClosureOwner {
    Generation(runmat_value::SharedValueGeneration),
    Captured(super::CapturedValueLease),
}

impl RetainedClosureValue {
    pub fn new(closure: Closure) -> Self {
        Self(RetainedClosureOwner::Generation(
            runmat_value::SharedValueGeneration::unbounded(Value::Closure(closure)),
        ))
    }

    pub fn as_value(&self) -> &Value {
        match &self.0 {
            RetainedClosureOwner::Generation(owner) => owner.value(),
            RetainedClosureOwner::Captured(owner) => owner.value(),
        }
    }

    pub fn as_closure(&self) -> &Closure {
        match self.as_value() {
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
        Ok(Self(RetainedClosureOwner::Generation(
            lease.shared_generation(),
        )))
    }

    pub(crate) fn from_callable_owner(
        owner: &super::RetainedCallableTarget,
    ) -> Result<Self, RuntimeError> {
        if !matches!(owner.value(), Value::Closure(_)) {
            return Err(crate::runtime_error::semantic_error(
                "CallableArguments",
                "retained callable owner is not a closure",
            ));
        }
        Ok(Self(match owner {
            super::RetainedCallableTarget::Direct(owner) => {
                RetainedClosureOwner::Generation(owner.shared_generation())
            }
            super::RetainedCallableTarget::Captured(owner) => {
                RetainedClosureOwner::Captured(owner.clone())
            }
        }))
    }

    #[cfg(test)]
    pub(crate) fn generation(&self) -> Option<runmat_value::SharedValueGeneration> {
        match &self.0 {
            RetainedClosureOwner::Generation(owner) => Some(owner.clone()),
            RetainedClosureOwner::Captured(_) => None,
        }
    }

    pub(crate) fn matches_callable_owner(&self, owner: &super::RetainedCallableTarget) -> bool {
        match (&self.0, owner) {
            (
                RetainedClosureOwner::Generation(left),
                super::RetainedCallableTarget::Direct(right),
            ) => left.ptr_eq(&right.shared_generation()),
            (
                RetainedClosureOwner::Captured(left),
                super::RetainedCallableTarget::Captured(right),
            ) => left.same_capture(right),
            _ => false,
        }
    }

    pub(crate) fn retain_for_call(
        &self,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> Result<super::RetainedCallableTarget, RuntimeError> {
        match &self.0 {
            RetainedClosureOwner::Generation(owner) => Ok(super::RetainedCallableTarget::Direct(
                crate::execution::RetainedValueReadLease::retain_shared_generation(
                    owner.clone(),
                    runtime,
                )?,
            )),
            RetainedClosureOwner::Captured(owner) => {
                Ok(super::RetainedCallableTarget::Captured(owner.clone()))
            }
        }
    }
}

impl PartialEq for RetainedClosureValue {
    fn eq(&self, other: &Self) -> bool {
        self.as_value() == other.as_value()
    }
}

/// Immutable callable target generation used by a repeatable forwarded
/// request. Its external `RetainedValueReadLease` remains with the repeatable
/// descriptor owner, so this root-free alias never creates a root cycle.
#[derive(Debug, Clone)]
pub struct RetainedCallableValue(super::RetainedCallableTarget);

impl RetainedCallableValue {
    pub(crate) fn from_read_lease(lease: &crate::execution::RetainedValueReadLease) -> Self {
        Self(super::RetainedCallableTarget::Direct(lease.clone()))
    }

    pub fn value(&self) -> &Value {
        self.0.value()
    }

    pub(crate) fn matches_callable_owner(&self, owner: &super::RetainedCallableTarget) -> bool {
        self.0.same_owner(owner)
    }
}

impl PartialEq for RetainedCallableValue {
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CallableTarget {
    Scoped(runmat_value::ScopedFunctionHandle),
    Bound(runmat_value::BoundFunctionBinding),
    /// Preserve the traceable closure and its ancestor environment through
    /// queued callback admission; flattened captures alone are insufficient.
    BoundLexical {
        binding: runmat_value::BoundFunctionBinding,
        closure: RetainedClosureValue,
    },
    CapturedResolved {
        identity: CallableIdentityOwner,
        fallback_policy: CallableFallbackPolicy,
        closure: RetainedClosureValue,
    },
    Resolved {
        identity: CallableIdentityOwner,
        fallback_policy: CallableFallbackPolicy,
    },
    FevalForward(Value),
    RetainedFevalForward(RetainedCallableValue),
    /// A selected builtin whose name borrows the original admitted generation.
    RetainedBuiltin(RetainedCallableValue),
}

impl CallableTarget {
    /// Borrow only a builtin identity already selected by the canonical name
    /// owner. This does not resolve arbitrary text or copy its admitted bytes.
    pub fn selected_builtin_name(&self) -> Option<&str> {
        match self {
            Self::RetainedBuiltin(value) => named_selection::retained_name(value.value()),
            Self::Resolved {
                identity,
                fallback_policy: CallableFallbackPolicy::None,
            } => match identity.as_identity() {
                CallableIdentity::Builtin(name) => Some(&name.0),
                _ => None,
            },
            _ => None,
        }
    }

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
            Self::RetainedFevalForward(value) | Self::RetainedBuiltin(value) => {
                visitor(value.value())
            }
            Self::Scoped(_) | Self::Bound(_) | Self::Resolved { .. } => Some(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallableDescriptor<A = Vec<Value>> {
    pub target: CallableTarget,
    pub args: A,
    pub requested_outputs: usize,
    pub metadata: CallableMetadata,
}

impl CallableDescriptor {
    /// Attach the existing rooted owner to a target-only selection. Never
    /// discard an ordinary descriptor's supplied arguments.
    pub fn with_retained_arguments(
        self,
        args: crate::call::BuiltinArguments,
    ) -> Result<CallableDescriptor<crate::call::BuiltinArguments>, RuntimeError> {
        if !self.args.is_empty() {
            return Err(semantic_error(
                "CallableArguments",
                "retained arguments require a target-only descriptor",
            ));
        }
        Ok(CallableDescriptor {
            target: self.target,
            args,
            requested_outputs: self.requested_outputs,
            metadata: self.metadata,
        })
    }
}

impl CallableDescriptor {
    /// Select a repeatable target from one already-admitted immutable callable
    /// generation. This is the retained-owner arm of the same descriptor
    /// selector used by `from_feval_value`: it may only return a descriptor
    /// whose selected authority and metadata borrow or share that generation.
    /// In particular it never copies a name into an unclaimed `String`.
    pub(crate) fn from_retained_feval_value(
        owner: &crate::execution::RetainedValueReadLease,
        requested_outputs: usize,
        function_resolver: &impl FunctionNameResolver,
    ) -> Result<Self, RuntimeError> {
        let descriptor = match owner.value() {
            Value::ScopedFunctionHandle(handle) => Self {
                target: CallableTarget::Scoped(handle.clone()),
                args: Vec::new(),
                requested_outputs,
                metadata: CallableMetadata::feval(None),
            },
            Value::BoundFunctionHandle(handle) => Self {
                target: CallableTarget::Bound(runmat_value::BoundFunctionBinding {
                    function: handle.function,
                    authority: handle.authority.clone(),
                }),
                args: Vec::new(),
                requested_outputs,
                metadata: CallableMetadata::feval(Some(handle.name.clone())),
            },
            Value::Closure(closure) => {
                let retained = RetainedClosureValue::from_read_lease(owner)?;
                let binding = closure
                    .binding
                    .clone()
                    .or_else(|| function_resolver.resolve_bound_function(&closure.function_name));
                if let Some(binding) = binding {
                    Self {
                        target: CallableTarget::BoundLexical {
                            binding,
                            closure: retained,
                        },
                        args: Vec::new(),
                        requested_outputs,
                        // Share the immutable generation's admitted text owner;
                        // diagnostics retain its name without copying bytes.
                        metadata: CallableMetadata::feval(Some(closure.function_name.clone())),
                    }
                } else if let Some(function) =
                    function_resolver.resolve_function(&closure.function_name)
                {
                    Self {
                        target: CallableTarget::CapturedResolved {
                            identity: CallableIdentity::BoundFunction(function).into(),
                            fallback_policy: CallableFallbackPolicy::None,
                            closure: retained,
                        },
                        args: Vec::new(),
                        requested_outputs,
                        metadata: CallableMetadata::feval(Some(closure.function_name.clone())),
                    }
                } else {
                    Self {
                        target: CallableTarget::RetainedFevalForward(
                            RetainedCallableValue::from_read_lease(owner),
                        ),
                        args: Vec::new(),
                        requested_outputs,
                        metadata: CallableMetadata::feval(None),
                    }
                }
            }
            Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::String(_)
            | Value::StringArray(_) => {
                let retained = RetainedCallableValue::from_read_lease(owner);
                let target = match named_selection::retained_name(owner.value()) {
                    Some(name) => match named_selection::select(name, function_resolver) {
                        named_selection::Selected::Bound(binding) => CallableTarget::Bound(binding),
                        named_selection::Selected::Resolved(
                            named_selection::Resolved::Function(function),
                        ) => CallableTarget::Resolved {
                            identity: CallableIdentity::BoundFunction(function).into(),
                            fallback_policy: CallableFallbackPolicy::None,
                        },
                        named_selection::Selected::Resolved(named_selection::Resolved::Builtin)
                            if function_resolver.name_resolution_completeness()
                                == NameResolutionCompleteness::Complete =>
                        {
                            CallableTarget::RetainedBuiltin(retained)
                        }
                        named_selection::Selected::Resolved(
                            named_selection::Resolved::Builtin
                            | named_selection::Resolved::Qualified
                            | named_selection::Resolved::Dynamic,
                        ) => CallableTarget::RetainedFevalForward(retained),
                    },
                    None => CallableTarget::RetainedFevalForward(retained),
                };
                Self {
                    target,
                    args: Vec::new(),
                    requested_outputs,
                    metadata: CallableMetadata::feval(None),
                }
            }
            _ => Self {
                target: CallableTarget::RetainedFevalForward(
                    RetainedCallableValue::from_read_lease(owner),
                ),
                args: Vec::new(),
                requested_outputs,
                metadata: CallableMetadata::feval(None),
            },
        };
        Ok(descriptor)
    }

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
        identity: CallableIdentityOwner,
        display_name: Option<runmat_value::Utf8Text>,
        fallback_policy: CallableFallbackPolicy,
        args: Vec<Value>,
        requested_outputs: usize,
        mut metadata: CallableMetadata,
    ) -> Self {
        metadata.display_name =
            metadata
                .display_name
                .or(display_name)
                .or_else(|| match &identity {
                    CallableIdentityOwner::Owned(identity) => {
                        strict_callable_display_name(identity).map(Into::into)
                    }
                    // A retained caller already has its exact immutable source
                    // owner; presentation text is supplied independently through
                    // metadata, never rebuilt by copying identity/name segments.
                    CallableIdentityOwner::Retained(_) => None,
                });
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
        name: runmat_value::Utf8Text,
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
        let name: runmat_value::Utf8Text = name.into();
        Self::resolved_inner(
            identity.into(),
            Some(name.clone()),
            fallback_policy,
            args,
            requested_outputs,
            CallableMetadata::feval(Some(name)),
        )
    }

    fn feval_named(
        name: String,
        args: Vec<Value>,
        requested_outputs: usize,
        function_resolver: &impl FunctionNameResolver,
    ) -> Self {
        match named_selection::select(&name, function_resolver) {
            named_selection::Selected::Bound(binding) => {
                Self::feval_bound(binding, name.into(), args, requested_outputs)
            }
            named_selection::Selected::Resolved(selected) => {
                let (identity, fallback_policy) = selected.owned(&name);
                Self::feval_resolved_name(identity, name, fallback_policy, args, requested_outputs)
            }
        }
    }

    fn resolve_named_target(
        name: &str,
        function_resolver: &impl FunctionNameResolver,
    ) -> (CallableIdentity, CallableFallbackPolicy) {
        named_selection::resolve(name, function_resolver).owned(name)
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
        Self::resolved_with_identity_owner(
            identity.into(),
            args,
            requested_outputs,
            fallback_policy,
            call_kind,
        )
    }

    pub fn resolved_retained(
        identity: std::sync::Arc<CallableIdentity>,
        args: Vec<Value>,
        requested_outputs: usize,
        fallback_policy: CallableFallbackPolicy,
        call_kind: CallableCallKind,
    ) -> Self {
        Self::resolved_with_identity_owner(
            identity.into(),
            args,
            requested_outputs,
            fallback_policy,
            call_kind,
        )
    }

    pub fn resolved_with_identity_owner(
        identity: CallableIdentityOwner,
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
                if text.as_str().is_some_and(Self::is_at_prefixed_text) {
                    return Self::feval_forward(Value::String(text), args, requested_outputs);
                }
                if let Some(name) = text.as_str().and_then(Self::parse_handle_name) {
                    return Self::feval_named(name, args, requested_outputs, function_resolver);
                }
                Self::feval_forward(Value::String(text), args, requested_outputs)
            }
            Value::CharArray(ca) if ca.rows == 1 => {
                let Ok(Some(text)) = ca.try_row_string() else {
                    return Self::feval_forward(Value::CharArray(ca), args, requested_outputs);
                };
                if Self::is_at_prefixed_text(&text) {
                    return Self::feval_forward(Value::CharArray(ca), args, requested_outputs);
                }
                if let Some(name) = Self::parse_handle_name(&text) {
                    return Self::feval_named(name, args, requested_outputs, function_resolver);
                }
                Self::feval_forward(Value::CharArray(ca), args, requested_outputs)
            }
            Value::StringArray(sa) if sa.data().len() == 1 => {
                if sa.data()[0].as_str().is_some_and(Self::is_at_prefixed_text) {
                    return Self::feval_forward(Value::StringArray(sa), args, requested_outputs);
                }
                if let Some(name) = sa.data()[0].as_str().and_then(Self::parse_handle_name) {
                    return Self::feval_named(name, args, requested_outputs, function_resolver);
                }
                Self::feval_forward(Value::StringArray(sa), args, requested_outputs)
            }
            Value::ScopedFunctionHandle(handle) => Self {
                target: CallableTarget::Scoped(handle),
                args,
                requested_outputs,
                metadata: CallableMetadata::feval(None),
            },
            Value::Closure(closure) => {
                Self::from_closure(closure, args, requested_outputs, function_resolver)
            }
            Value::FunctionHandle(name) => Self::feval_named(
                name.into_name_unbounded(),
                args,
                requested_outputs,
                function_resolver,
            ),
            Value::ExternalFunctionHandle(name) => Self::feval_named(
                name.into_name_unbounded(),
                args,
                requested_outputs,
                function_resolver,
            ),
            Value::MethodFunctionHandle(name) => Self::feval_resolved_name(
                CallableIdentity::Method(runmat_types::MethodId(name.name().to_owned())),
                name.into_name_unbounded(),
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
                identity: identity.into(),
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
        "{} in {}{location}",
        crate::call::identity::undefined_callable_message(identity),
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
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    callsite: Option<crate::callsite::CallsiteInfo>,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    crate::context::check_active_host()?;
    let invoke = crate::callsite::with_callsite(
        callsite,
        crate::dispatcher::try_call_retained_builtin_sequence_with_outputs(
            name,
            args,
            requested_outputs,
            runtime,
        ),
    );
    let result = match runtime {
        Some(runtime) => runtime.scope(invoke).await??,
        None => invoke.await??,
    }
    .ok_or_else(|| crate::dispatcher::undefined_builtin_error(name));
    crate::context::check_active_host()?;
    result
}

async fn forward_named_fallback(
    runtime: Option<&crate::context::RuntimeContext>,
    name: String,
    args: impl Into<crate::call::BuiltinArguments>,
    requested_outputs: usize,
    callsite: Option<crate::callsite::CallsiteInfo>,
    source_ref: Option<crate::source_context::RuntimeSourceRef>,
    source_invocation: Option<crate::context::RuntimeInvocationArity>,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    let invoke = crate::callsite::with_callsite(
        callsite,
        crate::dispatcher::try_call_retained_builtin_sequence_with_outputs(
            &name,
            args.into(),
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
pub(crate) use execution::invoke_selected_retained_closure;

#[path = "descriptor/resolved_execution.rs"]
mod resolved_execution;

#[path = "descriptor/repeatable.rs"]
mod repeatable;
mod named_selection;
pub use repeatable::RepeatableCallable;

/// Execute an exact selected method binding with a new callee's supplied
/// inputs, retaining the original argument forest through normalization.
pub(crate) async fn execute_retained_bound_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    binding: runmat_value::BoundFunctionBinding,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    call_kind: CallableCallKind,
) -> Result<crate::execution::RootedValueSequence, crate::RuntimeError> {
    execution::invoke_bound(
        runtime,
        binding,
        args.into_supplied_function_arguments(runtime)?,
        requested_outputs,
        CallableMetadata {
            call_kind,
            ..CallableMetadata::default()
        },
        None,
    )
    .await
}

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

/// Execute the same selected target with its canonical retained argument owner.
pub fn execute_retained_callable_descriptor(
    runtime: Option<&crate::context::RuntimeContext>,
    descriptor: CallableDescriptor<crate::call::BuiltinArguments>,
) -> impl std::future::Future<Output = Result<crate::execution::RootedValueSequence, RuntimeError>> + '_
{
    let work = execution::execute(runtime, descriptor);
    async move {
        match runtime {
            Some(runtime) => runtime.scope(work).await,
            None => work.await,
        }
    }
}
/// Optional selection differs only in resolver decline, never in argument ownership.
pub fn try_execute_retained_callable_descriptor(
    runtime: Option<&crate::context::RuntimeContext>,
    descriptor: CallableDescriptor<crate::call::BuiltinArguments>,
) -> impl std::future::Future<
    Output = Result<Option<crate::execution::RootedValueSequence>, RuntimeError>,
> + '_ {
    let work = execution::try_execute(runtime, descriptor);
    async move {
        match runtime {
            Some(runtime) => runtime.scope(work).await,
            None => work.await,
        }
    }
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
