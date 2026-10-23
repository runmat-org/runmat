use super::*;
use crate::call::identity::CallableIdentityOwner;

#[derive(Debug)]
pub struct CallableRequest {
    identity: CallableIdentityOwner,
    bound: Option<runmat_value::BoundFunctionBinding>,
    fallback_policy: CallableFallbackPolicy,
    args: CallableRequestArguments,
    requested_outputs: usize,
    function_input_callsite: Option<crate::callsite::CallsiteInfo>,
    selected_runtime: Option<crate::context::RuntimeContext>,
}

#[cfg(test)]
mod identity_owner_tests {
    use super::*;

    #[test]
    fn rooted_request_retains_the_same_qualified_identity_owner() {
        let original = std::sync::Arc::new(crate::call::identity::external_qualified_identity(
            "pkg.Point",
            "origin",
        ));
        let args = crate::call::RootedFunctionArguments::ordinary(
            crate::execution::RootedValues::new(Vec::new()).unwrap(),
        );
        let request = CallableRequest::rooted_resolved_with_identity_owner(
            std::sync::Arc::clone(&original).into(),
            CallableFallbackPolicy::ExternalBoundary,
            args,
            1,
        );
        let CallableIdentityOwner::Retained(retained) = &request.identity else {
            panic!("request deep-copied source identity")
        };
        assert!(std::sync::Arc::ptr_eq(&original, retained));
        assert_eq!(std::sync::Arc::strong_count(&original), 2);
        drop(request);
        assert_eq!(std::sync::Arc::strong_count(&original), 1);
    }
}

#[derive(Debug)]
enum CallableRequestArguments {
    Owned(Vec<Value>),
    Rooted(crate::call::RootedFunctionArguments),
}

impl CallableRequestArguments {
    fn into_rooted(
        self,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> Result<crate::call::RootedFunctionArguments, RuntimeError> {
        match self {
            Self::Rooted(arguments) => Ok(arguments),
            Self::Owned(arguments) => match runtime {
                Some(runtime) => crate::execution::RootedValues::in_context(arguments, runtime)
                    .map(crate::call::RootedFunctionArguments::ordinary),
                None => crate::execution::RootedValues::new(arguments)
                    .map(crate::call::RootedFunctionArguments::ordinary)
                    .map_err(|error| error.into_runtime_error()),
            },
        }
    }
}

impl CallableRequest {
    pub fn semantic(function: usize, args: Vec<Value>, requested_outputs: usize) -> Self {
        Self {
            identity: CallableIdentity::BoundFunction(runmat_types::FunctionId(function)).into(),
            bound: None,
            fallback_policy: CallableFallbackPolicy::None,
            args: CallableRequestArguments::Owned(args),
            requested_outputs,
            function_input_callsite: None,
            selected_runtime: None,
        }
    }

    pub fn bound(
        binding: runmat_value::BoundFunctionBinding,
        args: Vec<Value>,
        requested_outputs: usize,
    ) -> Self {
        Self {
            identity: CallableIdentity::BoundFunction(runmat_types::FunctionId(binding.function))
                .into(),
            bound: Some(binding),
            fallback_policy: CallableFallbackPolicy::None,
            args: CallableRequestArguments::Owned(args),
            requested_outputs,
            function_input_callsite: None,
            selected_runtime: None,
        }
    }

    pub fn resolved(
        identity: CallableIdentity,
        fallback_policy: CallableFallbackPolicy,
        args: Vec<Value>,
        requested_outputs: usize,
    ) -> Self {
        Self::resolved_with_identity_owner(
            identity.into(),
            fallback_policy,
            args,
            requested_outputs,
        )
    }

    pub fn resolved_retained(
        identity: std::sync::Arc<CallableIdentity>,
        fallback_policy: CallableFallbackPolicy,
        args: Vec<Value>,
        requested_outputs: usize,
    ) -> Self {
        Self::resolved_with_identity_owner(
            identity.into(),
            fallback_policy,
            args,
            requested_outputs,
        )
    }

    pub fn resolved_with_identity_owner(
        identity: CallableIdentityOwner,
        fallback_policy: CallableFallbackPolicy,
        args: Vec<Value>,
        requested_outputs: usize,
    ) -> Self {
        Self {
            identity,
            bound: None,
            fallback_policy,
            args: CallableRequestArguments::Owned(args),
            requested_outputs,
            function_input_callsite: None,
            selected_runtime: None,
        }
    }

    pub(crate) fn rooted_resolved(
        identity: CallableIdentity,
        fallback_policy: CallableFallbackPolicy,
        args: crate::call::RootedFunctionArguments,
        requested_outputs: usize,
    ) -> Self {
        Self::rooted_resolved_with_identity_owner(
            identity.into(),
            fallback_policy,
            args,
            requested_outputs,
        )
    }

    pub(crate) fn rooted_resolved_with_identity_owner(
        identity: CallableIdentityOwner,
        fallback_policy: CallableFallbackPolicy,
        args: crate::call::RootedFunctionArguments,
        requested_outputs: usize,
    ) -> Self {
        Self {
            identity,
            bound: None,
            fallback_policy,
            args: CallableRequestArguments::Rooted(args),
            requested_outputs,
            function_input_callsite: None,
            selected_runtime: None,
        }
    }

    pub fn with_function_input_callsite(
        mut self,
        callsite: Option<crate::callsite::CallsiteInfo>,
    ) -> Self {
        self.function_input_callsite = callsite;
        self
    }

    pub(crate) fn with_selected_runtime(
        mut self,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> Self {
        self.selected_runtime = runtime.cloned();
        self
    }
}

pub async fn try_call_semantic_function(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
) -> Option<Result<crate::execution::RootedValueSequence, RuntimeError>> {
    try_call_semantic_function_in_context(None, function, args, requested_outputs).await
}

/// The borrowed ABI needs one owned handle collection for queued callbacks.
/// Reuse the canonical final-slot constructor: original roots survive its
/// allocator callbacks, and backing is admitted before any Value is copied.
fn materialize_borrowed_arguments(
    runtime: Option<&crate::context::RuntimeContext>,
    args: &[Value],
) -> Result<crate::execution::RootedValues, RuntimeError> {
    if runtime.is_none() {
        // None remains a context-free embedding call, not an ambient managed
        // grant. Reject an active finite/cancelled scope before allocating;
        // never pass that guard's context to the selected callback.
        crate::execution::rooted_continuation::unbounded_callable_cancellation(None)?;
    }
    crate::execution::RootedValues::from_borrowed_arguments(args, runtime)
}

/// Carry an explicitly selected invocation owner into a semantic body. A
/// callback-selected context must not be replaced by an invoker's installation
/// context, or Java/Python reentry loses its exact active-lane authority.
pub async fn try_call_semantic_function_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    function: usize,
    args: &[Value],
    requested_outputs: usize,
) -> Option<Result<crate::execution::RootedValueSequence, RuntimeError>> {
    let arguments = materialize_borrowed_arguments(runtime, args);
    let arguments = match arguments {
        Ok(arguments) => crate::call::RootedFunctionArguments::ordinary(arguments),
        Err(error) => return Some(Err(error)),
    };
    try_call_rooted_semantic_function_in_context(runtime, function, arguments, requested_outputs)
        .await
}

/// Invoke the canonical numeric semantic port with its original segmented
/// argument owners. Captured closure Values remain borrowed from the retained
/// callable generation; no compatibility Vec is manufactured.
pub async fn try_call_rooted_semantic_function_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    function: usize,
    arguments: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
) -> Option<Result<crate::execution::RootedValueSequence, RuntimeError>> {
    let invoker = current_semantic_function_invoker();
    let invoker = invoker?;
    Some(
        match invoker(runtime.cloned(), function, arguments, requested_outputs) {
            Ok(future) => future.await,
            Err(error) => Err(error),
        },
    )
}

pub async fn try_call_bound_function(
    binding: runmat_value::BoundFunctionBinding,
    args: &[Value],
    requested_outputs: usize,
) -> Option<Result<crate::execution::RootedValueSequence, RuntimeError>> {
    let runtime = crate::context::legacy::active()?;
    if matches!(
        &binding.authority,
        runmat_value::BoundFunctionAuthority::SelectedNative { .. }
    ) {
        return try_call_bound_function_with_environment_in_context(
            &runtime,
            binding,
            args,
            requested_outputs,
            None,
        )
        .await;
    }
    if matches!(
        &binding.authority,
        runmat_value::BoundFunctionAuthority::Local { .. }
    ) && !binding.authority.admits_local(&runtime.session_lease())
    {
        return Some(Err(crate::runtime_error::semantic_error(
            "UnavailableBoundFunction",
            "bound function belongs to another or ended session",
        )));
    }
    let invoker = current_bound_function_invoker()?;
    let args = match materialize_borrowed_arguments(Some(&runtime), args) {
        Ok(args) => args,
        Err(error) => return Some(Err(error)),
    };
    Some(
        match invoker(
            runtime,
            binding,
            crate::call::RootedFunctionArguments::ordinary(args),
            requested_outputs,
            None,
        ) {
            Ok(future) => future.await,
            Err(error) => Err(error),
        },
    )
}

pub async fn try_call_bound_function_in_context(
    runtime: &crate::context::RuntimeContext,
    binding: runmat_value::BoundFunctionBinding,
    args: &[Value],
    requested_outputs: usize,
) -> Option<Result<crate::execution::RootedValueSequence, RuntimeError>> {
    try_call_bound_function_with_environment_in_context(
        runtime,
        binding,
        args,
        requested_outputs,
        None,
    )
    .await
}

pub fn try_call_bound_function_with_environment_in_context<'a>(
    runtime: &'a crate::context::RuntimeContext,
    binding: runmat_value::BoundFunctionBinding,
    args: &[Value],
    requested_outputs: usize,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
) -> impl std::future::Future<
    Output = Option<Result<crate::execution::RootedValueSequence, RuntimeError>>,
> + 'a {
    // The public future itself can be queued without being polled. Acquire
    // both leases at construction; invoker selection and user code stay lazy.
    let admission = lexical_environment
        .as_ref()
        .map(crate::call::lexical::root_environment)
        .transpose()
        .and_then(|roots| {
            materialize_borrowed_arguments(Some(runtime), args).map(|args| {
                (
                    roots.unwrap_or_default(),
                    crate::call::RootedFunctionArguments::ordinary(args),
                )
            })
        });
    invoke_rooted_bound_function_with_environment_in_context(
        runtime,
        binding,
        admission,
        requested_outputs,
        lexical_environment,
    )
}

/// Execute one already-rooted, segmented function entry without flattening a
/// closure's immutable capture generation into its supplied argument vector.
pub fn try_call_rooted_bound_function_with_environment_in_context<'a>(
    runtime: &'a crate::context::RuntimeContext,
    binding: runmat_value::BoundFunctionBinding,
    arguments: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
) -> impl std::future::Future<
    Output = Option<Result<crate::execution::RootedValueSequence, RuntimeError>>,
> + 'a {
    let admission = if let Some(callable) = arguments.callable() {
        // The callable's external root lease already traces its exact lexical
        // ancestor chain. Re-rooting each activation here would manufacture
        // another unadmitted root Vec for every repeated selected call.
        let captured_environment = match callable.value() {
            runmat_value::Value::Closure(closure) => closure.lexical_environment.as_ref(),
            _ => None,
        };
        if captured_environment != lexical_environment.as_ref() {
            Err(crate::runtime_error::semantic_error(
                "UnavailableBoundFunction",
                "selected callable and lexical environment disagree",
            ))
        } else {
            Ok((Vec::new(), arguments))
        }
    } else {
        lexical_environment
            .as_ref()
            .map(crate::call::lexical::root_environment)
            .transpose()
            .map(|roots| (roots.unwrap_or_default(), arguments))
    };
    invoke_rooted_bound_function_with_environment_in_context(
        runtime,
        binding,
        admission,
        requested_outputs,
        lexical_environment,
    )
}

/// Legacy-context adapter for an already-rooted selected call. It never
/// flattens a captured prefix; absence of an active context remains an
/// unavailable invoker, not an implicit ambient fallback.
pub async fn try_call_rooted_bound_function(
    binding: runmat_value::BoundFunctionBinding,
    arguments: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
) -> Option<Result<crate::execution::RootedValueSequence, RuntimeError>> {
    let runtime = crate::context::legacy::active()?;
    try_call_rooted_bound_function_with_environment_in_context(
        &runtime,
        binding,
        arguments,
        requested_outputs,
        None,
    )
    .await
}

async fn invoke_rooted_bound_function_with_environment_in_context(
    runtime: &crate::context::RuntimeContext,
    binding: runmat_value::BoundFunctionBinding,
    admission: Result<
        (
            Vec<runmat_gc::ExplicitRoot>,
            crate::call::RootedFunctionArguments,
        ),
        RuntimeError,
    >,
    requested_outputs: usize,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
) -> Option<Result<crate::execution::RootedValueSequence, RuntimeError>> {
    let (_environment_roots, args) = match admission {
        Ok(admission) => admission,
        Err(error) => return Some(Err(error)),
    };
    if let runmat_value::BoundFunctionAuthority::SelectedNative { origin, lease } =
        &binding.authority
    {
        return Some(
            super::source_facts::invoke_selected_native_lease(
                runtime.clone(),
                binding.function,
                *origin,
                Rc::clone(lease),
                args,
                requested_outputs,
                lexical_environment,
            )
            .await,
        );
    }
    if matches!(
        &binding.authority,
        runmat_value::BoundFunctionAuthority::Local { .. }
    ) && !binding.authority.admits_local(&runtime.session_lease())
    {
        return Some(Err(crate::runtime_error::semantic_error(
            "UnavailableBoundFunction",
            "bound function belongs to another or ended session",
        )));
    }
    let invoker = runtime
        .selected_call_projection()
        .borrow()
        .bound_invoker()
        .unwrap_or_else(|| runtime.state().call.borrow().bound_invoker.clone())?;
    Some(
        match invoker(
            runtime.clone(),
            binding,
            args,
            requested_outputs,
            lexical_environment,
        ) {
            Ok(future) => future.await,
            Err(error) => Err(error),
        },
    )
}

pub async fn try_call_external_function(
    call: ExternalFunctionCall,
) -> Option<Result<crate::execution::RootedValueSequence, RuntimeError>> {
    try_call_external_function_in_context(None, call).await
}

pub async fn try_call_external_function_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    call: ExternalFunctionCall,
) -> Option<Result<crate::execution::RootedValueSequence, RuntimeError>> {
    let invoker = current_external_function_invoker()?;
    Some(match invoker(runtime.cloned(), call) {
        Ok(future) => future.await,
        Err(error) => Err(error),
    })
}

pub(crate) async fn try_call_external_function_owned_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    call: ExternalFunctionCall,
) -> std::ops::ControlFlow<
    Result<crate::execution::RootedValueSequence, RuntimeError>,
    ExternalFunctionCall,
> {
    let Some(invoker) = current_external_function_invoker() else {
        return std::ops::ControlFlow::Continue(call);
    };
    std::ops::ControlFlow::Break(match invoker(runtime.cloned(), call) {
        Ok(future) => future.await,
        Err(error) => Err(error),
    })
}

pub async fn try_call_lexical_function(
    call: crate::call::lexical::LexicalCall,
) -> Option<Result<crate::call::lexical::RootedLexicalCallResult, RuntimeError>> {
    try_call_lexical_function_in_context(None, call).await
}

pub async fn try_call_lexical_function_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    call: crate::call::lexical::LexicalCall,
) -> Option<Result<crate::call::lexical::RootedLexicalCallResult, RuntimeError>> {
    let invoker = current_lexical_function_invoker()?;
    let _input_roots = match crate::execution::RootedValueLease::for_values(
        call.arguments
            .iter()
            .chain(call.captures.iter().map(|capture| &capture.value)),
    ) {
        Ok(roots) => roots,
        Err(error) => return Some(Err(error.into_runtime_error())),
    };
    let _environment_roots = match call.environment.as_ref() {
        Some(environment) => match crate::call::lexical::root_environment(environment) {
            Ok(roots) => roots,
            Err(error) => return Some(Err(error)),
        },
        None => Vec::new(),
    };
    Some(match invoker(runtime.cloned(), call) {
        Ok(future) => future.await,
        Err(error) => Err(error),
    })
}

pub async fn try_call_semantic_function_by_name(
    name: &str,
    args: &[Value],
    requested_outputs: usize,
) -> Option<Result<crate::execution::RootedValueSequence, RuntimeError>> {
    try_call_semantic_function_by_name_in_context(None, name, args, requested_outputs).await
}

pub async fn try_call_semantic_function_by_name_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    name: &str,
    args: &[Value],
    requested_outputs: usize,
) -> Option<Result<crate::execution::RootedValueSequence, RuntimeError>> {
    let function = resolve_semantic_function_by_name(name)?;
    try_call_selected_semantic_name_in_context(runtime, name, function, args, requested_outputs)
        .await
}

async fn try_call_selected_semantic_name_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    name: &str,
    function: usize,
    args: &[Value],
    requested_outputs: usize,
) -> Option<Result<crate::execution::RootedValueSequence, RuntimeError>> {
    match selected_scoped_name_target(name, function) {
        Some(ScopedSemanticNameTarget::Retained(binding)) => match runtime {
            Some(runtime) => {
                try_call_bound_function_in_context(runtime, binding, args, requested_outputs).await
            }
            None => try_call_bound_function(binding, args, requested_outputs).await,
        },
        Some(ScopedSemanticNameTarget::Published(_)) | None => {
            try_call_semantic_function_in_context(runtime, function, args, requested_outputs).await
        }
    }
}

fn selected_scoped_name_target(name: &str, function: usize) -> Option<ScopedSemanticNameTarget> {
    current_scoped_name_target_resolver()
        .and_then(|resolver| resolver(name))
        .filter(|target| target.function().0 == function)
}

pub(crate) type RootedSemanticAttempt = std::ops::ControlFlow<
    Result<crate::execution::RootedValueSequence, RuntimeError>,
    crate::call::RootedFunctionArguments,
>;

/// Attempt one already-rooted segmented semantic selection. Absence of an
/// invoker returns the exact argument owner; a selected body is terminal.
pub(crate) async fn try_call_rooted_semantic_name_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    name: &str,
    arguments: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
) -> RootedSemanticAttempt {
    let mut input_callsite = None;
    try_call_rooted_semantic_name_with_callsite_in_context(
        runtime,
        name,
        arguments,
        requested_outputs,
        &mut input_callsite,
    )
    .await
}

async fn try_call_rooted_semantic_name_with_callsite_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    name: &str,
    arguments: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
    input_callsite: &mut Option<crate::callsite::CallsiteInfo>,
) -> RootedSemanticAttempt {
    let Some(function) = resolve_semantic_function_by_name(name) else {
        return std::ops::ControlFlow::Continue(arguments);
    };
    match selected_scoped_name_target(name, function) {
        Some(ScopedSemanticNameTarget::Retained(binding)) => {
            let selected_native = matches!(
                &binding.authority,
                runmat_value::BoundFunctionAuthority::SelectedNative { .. }
            );
            if !selected_native && current_bound_function_invoker().is_none() {
                return std::ops::ControlFlow::Continue(arguments);
            }
            let selected = async {
                match runtime {
                    Some(runtime) => {
                        try_call_rooted_bound_function_with_environment_in_context(
                            runtime,
                            binding,
                            arguments,
                            requested_outputs,
                            None,
                        )
                        .await
                    }
                    None => {
                        try_call_rooted_bound_function(binding, arguments, requested_outputs).await
                    }
                }
            };
            // Only the selected body receives caller-expression facts. A
            // declined resolution keeps both original owners untouched.
            let result = if let Some(callsite) = input_callsite.take() {
                flatten_callsite_result(
                    crate::callsite::with_function_input_callsite(Some(callsite), selected).await,
                )
            } else {
                selected.await
            };
            std::ops::ControlFlow::Break(result.unwrap_or_else(|| {
                Err(crate::runtime_error::semantic_error(
                    "UnavailableBoundFunction",
                    "selected semantic function has no rooted invoker",
                ))
            }))
        }
        Some(ScopedSemanticNameTarget::Published(_)) | None => {
            if current_semantic_function_invoker().is_none() {
                return std::ops::ControlFlow::Continue(arguments);
            }
            let selected = try_call_rooted_semantic_function_in_context(
                runtime,
                function,
                arguments,
                requested_outputs,
            );
            let result = if let Some(callsite) = input_callsite.take() {
                flatten_callsite_result(
                    crate::callsite::with_function_input_callsite(Some(callsite), selected).await,
                )
            } else {
                selected.await
            };
            std::ops::ControlFlow::Break(
                result.expect("semantic invoker was checked before rooted dispatch"),
            )
        }
    }
}

pub fn resolve_semantic_function_by_name(name: &str) -> Option<usize> {
    let resolver = current_semantic_function_resolver()?;
    resolver(name)
}

pub async fn try_load_and_call_dynamic_function(
    name: String,
    args: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
    phase: DynamicFunctionLoadPhase,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> super::DynamicFunctionLoadAttempt {
    let Some(context) = crate::context::legacy::active() else {
        return std::ops::ControlFlow::Continue(args);
    };
    let Some(loader) = context.state().call.borrow().dynamic_loader.clone() else {
        return std::ops::ControlFlow::Continue(args);
    };
    super::await_dynamic_function_load(loader(
        context,
        name,
        args,
        requested_outputs,
        phase,
        input_callsite,
    ))
    .await
}

/// Select an observed source definition without invoking its body. The
/// registered session loader performs the same source/cache/admission path as
/// `@name`; only its exact counted binding leaves this rooted result.
pub async fn select_observed_source_binding_in_context(
    runtime: crate::context::RuntimeContext,
    name: String,
    source: crate::builtins::common::path_search::FileCandidate,
) -> Result<Option<runmat_value::BoundFunctionBinding>, RuntimeError> {
    let loader = runtime.state().call.borrow().dynamic_loader.clone();
    let Some(loader) = loader else {
        return Ok(None);
    };
    let arguments = crate::call::RootedFunctionArguments::ordinary(
        crate::execution::RootedValues::in_context(Vec::new(), &runtime)?,
    );
    let result = super::await_dynamic_function_load(loader(
        runtime,
        name,
        arguments,
        1,
        DynamicFunctionLoadPhase::Handle {
            class: None,
            selected_source: Some(source),
        },
        None,
    ))
    .await;
    let sequence = match result {
        std::ops::ControlFlow::Break(result) => result?,
        std::ops::ControlFlow::Continue(_) => return Ok(None),
    };
    let [Value::BoundFunctionHandle(handle)] = sequence.value().as_slice() else {
        return Err(crate::build_runtime_error(
            "selected source handle did not return one bound callable",
        )
        .with_identifier("RunMat:DynamicEval:SelectedSource")
        .build());
    };
    Ok(Some(runmat_value::BoundFunctionBinding {
        function: handle.function,
        authority: handle.authority.clone(),
    }))
}

pub async fn try_clear_dynamic_functions(
    request: DynamicFunctionClearRequest,
) -> Option<Result<(), RuntimeError>> {
    let context = crate::context::legacy::active()?;
    let clearer = context.state().call.borrow().dynamic_clearer.clone()?;
    Some(match clearer(context, request) {
        Ok(future) => future.await,
        Err(error) => Err(error),
    })
}

pub async fn try_refresh_dynamic_functions(
    request: DynamicFunctionRefreshRequest,
) -> Option<Result<(), RuntimeError>> {
    let context = crate::context::legacy::active()?;
    let refresher = context.state().call.borrow().dynamic_refresher.clone()?;
    let mut cooperation = context.cpu_cooperation();
    if let Err(source) = cooperation.checkpoint(0).await {
        return Some(Err(refresh_execution_error(&context, source)));
    }
    let future = match refresher(context.clone(), request) {
        Ok(future) => future,
        Err(error) => return Some(Err(error)),
    };
    Some(
        cooperation
            .await_cancellable(future)
            .await
            .map_err(|source| refresh_execution_error(&context, source))
            .and_then(std::convert::identity),
    )
}

fn refresh_execution_error(
    context: &crate::context::RuntimeContext,
    source: crate::execution::ExecutionServiceError,
) -> RuntimeError {
    if matches!(&source, crate::execution::ExecutionServiceError::Cancelled) {
        return crate::build_runtime_error("source-cache refresh was cancelled")
            .with_identifier(format!("{}:Cancelled", context.error_namespace()))
            .with_source(source)
            .build();
    }
    source.into_runtime_error()
}

#[path = "call_dispatch/route_phases.rs"]
mod route_phases;
use route_phases::{
    try_bound_identity, try_external_foreign_adapters, try_named_fallback,
    try_registered_constructor_source, try_registered_external_static,
};

pub(crate) async fn try_call_semantic_descriptor_owned(
    request: CallableRequest,
) -> super::CallableAttempt {
    let CallableRequest {
        identity,
        bound,
        fallback_policy,
        args,
        requested_outputs,
        mut function_input_callsite,
        selected_runtime,
    } = request;
    let mut args = match args.into_rooted(selected_runtime.as_ref()) {
        Ok(arguments) => arguments,
        Err(error) => return std::ops::ControlFlow::Break(Err(error)),
    };
    if let CallableIdentity::BoundFunction(function) = identity.as_identity() {
        return try_bound_identity(
            *function,
            bound,
            args,
            requested_outputs,
            function_input_callsite,
            selected_runtime.as_ref(),
        )
        .await;
    }
    if !fallback_policy.allows_semantic_name_resolution_for(identity.as_identity()) {
        return std::ops::ControlFlow::Continue(args);
    }
    let Some(name) = fallback_policy.resolution_name_for(identity.as_identity()) else {
        return std::ops::ControlFlow::Continue(args);
    };
    // An external-qualified name may denote a registered public static method.
    // Select its typed body through the same authority as direct dotted calls;
    // this does not grant arbitrary external names VM/builtin name fallback.
    if let CallableIdentity::ExternalName(qualified) = identity.as_identity() {
        args = match try_registered_external_static(
            qualified,
            &name,
            args,
            requested_outputs,
            &mut function_input_callsite,
            selected_runtime.as_ref(),
        )
        .await
        {
            std::ops::ControlFlow::Break(result) => return std::ops::ControlFlow::Break(result),
            std::ops::ControlFlow::Continue(args) => args,
        };
    }
    if matches!(identity.as_identity(), CallableIdentity::ExternalName(_)) {
        args = match try_external_foreign_adapters(&name, args, requested_outputs).await {
            std::ops::ControlFlow::Break(result) => return std::ops::ControlFlow::Break(result),
            std::ops::ControlFlow::Continue(args) => args,
        };
    }
    let registered_constructor = matches!(
        identity.as_identity(),
        CallableIdentity::DynamicName(_) | CallableIdentity::ExternalName(_)
    )
    .then(|| crate::class_registry::registered_constructor_identity(&name))
    .flatten();
    if let Some(class) = registered_constructor {
        return try_registered_constructor_source(
            class,
            name,
            args,
            requested_outputs,
            function_input_callsite,
        )
        .await;
    }
    try_named_fallback(
        identity.as_identity(),
        name,
        args,
        requested_outputs,
        function_input_callsite,
        selected_runtime.as_ref(),
    )
    .await
}

pub async fn try_call_semantic_descriptor(
    request: CallableRequest,
) -> Option<Result<crate::execution::RootedValueSequence, RuntimeError>> {
    match try_call_semantic_descriptor_owned(request).await {
        std::ops::ControlFlow::Break(result) => Some(result),
        std::ops::ControlFlow::Continue(_) => None,
    }
}

fn flatten_callsite_result<T>(
    result: Result<Option<Result<T, RuntimeError>>, RuntimeError>,
) -> Option<Result<T, RuntimeError>> {
    match result {
        Ok(result) => result,
        Err(error) => Some(Err(error)),
    }
}
