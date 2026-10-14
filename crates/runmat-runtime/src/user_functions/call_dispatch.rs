use super::*;

#[derive(Debug, Clone)]
pub struct CallableRequest {
    identity: CallableIdentity,
    bound: Option<runmat_value::BoundFunctionBinding>,
    fallback_policy: CallableFallbackPolicy,
    args: Vec<Value>,
    requested_outputs: usize,
    function_input_callsite: Option<crate::callsite::CallsiteInfo>,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
    selected_runtime: Option<crate::context::RuntimeContext>,
}

impl CallableRequest {
    pub fn semantic(function: usize, args: Vec<Value>, requested_outputs: usize) -> Self {
        Self {
            identity: CallableIdentity::BoundFunction(runmat_types::FunctionId(function)),
            bound: None,
            fallback_policy: CallableFallbackPolicy::None,
            args,
            requested_outputs,
            function_input_callsite: None,
            lexical_environment: None,
            selected_runtime: None,
        }
    }

    pub fn bound(
        binding: runmat_value::BoundFunctionBinding,
        args: Vec<Value>,
        requested_outputs: usize,
    ) -> Self {
        Self {
            identity: CallableIdentity::BoundFunction(runmat_types::FunctionId(binding.function)),
            bound: Some(binding),
            fallback_policy: CallableFallbackPolicy::None,
            args,
            requested_outputs,
            function_input_callsite: None,
            lexical_environment: None,
            selected_runtime: None,
        }
    }

    pub fn resolved(
        identity: CallableIdentity,
        fallback_policy: CallableFallbackPolicy,
        args: Vec<Value>,
        requested_outputs: usize,
    ) -> Self {
        Self {
            identity,
            bound: None,
            fallback_policy,
            args,
            requested_outputs,
            function_input_callsite: None,
            lexical_environment: None,
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

    pub(crate) fn with_lexical_environment(
        mut self,
        environment: Option<runmat_value::LexicalEnvironment>,
    ) -> Self {
        self.lexical_environment = environment;
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

/// Carry an explicitly selected invocation owner into a semantic body. A
/// callback-selected context must not be replaced by an invoker's installation
/// context, or Java/Python reentry loses its exact active-lane authority.
pub async fn try_call_semantic_function_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    function: usize,
    args: &[Value],
    requested_outputs: usize,
) -> Option<Result<crate::execution::RootedValueSequence, RuntimeError>> {
    let invoker = current_semantic_function_invoker();
    let invoker = invoker?;
    Some(invoker(runtime.cloned(), function, args, requested_outputs).await)
}

pub async fn try_call_bound_function(
    binding: runmat_value::BoundFunctionBinding,
    args: &[Value],
    requested_outputs: usize,
) -> Option<Result<crate::execution::RootedValueSequence, RuntimeError>> {
    let runtime = crate::context::legacy::active()?;
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
    let args = match crate::execution::RootedValues::new(args.to_vec()) {
        Ok(args) => args,
        Err(error) => return Some(Err(error.into_runtime_error())),
    };
    Some(invoker(runtime, binding, args, requested_outputs, None).await)
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
            crate::execution::RootedValues::new(args.to_vec())
                .map(|args| (roots.unwrap_or_default(), args))
                .map_err(|error| error.into_runtime_error())
        });
    async move {
        let (_environment_roots, args) = match admission {
            Ok(admission) => admission,
            Err(error) => return Some(Err(error)),
        };
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
            invoker(
                runtime.clone(),
                binding,
                args,
                requested_outputs,
                lexical_environment,
            )
            .await,
        )
    }
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
    Some(invoker(runtime.cloned(), call).await)
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
    Some(invoker(runtime.cloned(), call).await)
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

pub fn resolve_semantic_function_by_name(name: &str) -> Option<usize> {
    let resolver = current_semantic_function_resolver()?;
    resolver(name)
}

pub async fn try_load_and_call_dynamic_function(
    name: String,
    args: Vec<Value>,
    requested_outputs: usize,
    phase: DynamicFunctionLoadPhase,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> Option<Result<crate::execution::RootedValueSequence, RuntimeError>> {
    let context = crate::context::legacy::active()?;
    let loader = context.state().call.borrow().dynamic_loader.clone()?;
    loader(
        context,
        name,
        args,
        requested_outputs,
        phase,
        input_callsite,
    )
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
    let result = loader(
        runtime,
        name,
        Vec::new(),
        1,
        DynamicFunctionLoadPhase::Handle {
            class: None,
            selected_source: Some(source),
        },
        None,
    )
    .await;
    let Some(result) = result else {
        return Ok(None);
    };
    let sequence = result?;
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
    Some(clearer(context, request).await)
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
    Some(
        cooperation
            .await_cancellable(refresher(context.clone(), request))
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

pub async fn try_call_semantic_descriptor(
    request: CallableRequest,
) -> Option<Result<crate::execution::RootedValueSequence, RuntimeError>> {
    let CallableRequest {
        identity,
        bound,
        fallback_policy,
        args,
        requested_outputs,
        function_input_callsite,
        lexical_environment,
        selected_runtime,
    } = request;
    if let CallableIdentity::BoundFunction(function) = identity {
        if let Some(binding) = bound {
            if let Some(context) = selected_runtime.clone().or_else(|| {
                lexical_environment
                    .as_ref()
                    .and_then(|_| crate::context::legacy::active())
            }) {
                return crate::callsite::with_function_input_callsite(
                    function_input_callsite,
                    try_call_bound_function_with_environment_in_context(
                        &context,
                        binding,
                        &args,
                        requested_outputs,
                        lexical_environment,
                    ),
                )
                .await;
            }
            return crate::callsite::with_function_input_callsite(
                function_input_callsite,
                try_call_bound_function(binding, &args, requested_outputs),
            )
            .await;
        }
        return crate::callsite::with_function_input_callsite(
            function_input_callsite,
            try_call_semantic_function_in_context(
                selected_runtime.as_ref(),
                function.0,
                &args,
                requested_outputs,
            ),
        )
        .await;
    }
    if !fallback_policy.allows_semantic_name_resolution_for(&identity) {
        return None;
    }
    let name = fallback_policy.resolution_name_for(&identity)?;
    if matches!(identity, CallableIdentity::ExternalName(_)) {
        if let Some(context) = crate::context::legacy::active() {
            if let Some(result) = crate::foreign::try_invoke_clibgen(
                context.clone(),
                &name,
                args.clone(),
                requested_outputs,
            )
            .await
            {
                return Some(result);
            }
            if let Some(result) = crate::foreign::try_invoke_clib(
                context.clone(),
                &name,
                args.clone(),
                requested_outputs,
            )
            .await
            {
                return Some(result);
            }
            if let Some(result) =
                crate::foreign::try_invoke_python(context, &name, args.clone(), requested_outputs)
                    .await
            {
                return Some(result);
            }
        }
    }
    let registered_constructor = matches!(
        identity,
        CallableIdentity::DynamicName(_) | CallableIdentity::ExternalName(_)
    )
    .then(|| crate::class_registry::registered_constructor_identity(&name))
    .flatten();
    if let Some(class) = registered_constructor {
        // A registered constructor with an already bound semantic/builtin body,
        // or no constructor declaration at all, keeps its existing runtime path.
        // Only an unresolved external declaration may select class source. The
        // loader must prove that source owns this exact typed constructor.
        let constructor = class
            .qualified_name()
            .0
            .last()
            .map(|name| runmat_types::MethodName::from(name.0.as_str()))
            .and_then(|method| crate::class_registry::lookup_bound_method(&class, &method))
            .or_else(|| {
                crate::class_registry::lookup_bound_method(
                    &class,
                    &runmat_types::MethodName::from(class.display_name()),
                )
            });
        if let Some(constructor) = constructor.filter(|method| {
            method.declaring_class == class
                && !method.declaration.is_static
                && matches!(&method.callable, CallableIdentity::ExternalName(_))
                && method.fallback == CallableFallbackPolicy::ExternalBoundary
        }) {
            // Registration can precede installation of the semantic invoker.
            // Preserve that existing constructor priority before source lookup.
            let declared = &constructor.declaration.function_name;
            let owner_qualified = format!(
                "{}.{}",
                constructor.declaring_class,
                class
                    .qualified_name()
                    .0
                    .last()
                    .expect("registered class identity has a final segment")
                    .0
            );
            if resolve_semantic_function_by_name(declared).is_none()
                && resolve_semantic_function_by_name(&owner_qualified).is_none()
            {
                if let Some(result) = try_load_and_call_dynamic_function(
                    name,
                    args,
                    requested_outputs,
                    DynamicFunctionLoadPhase::RegisteredConstructorSource(class),
                    function_input_callsite,
                )
                .await
                {
                    return Some(result);
                }
            }
        }
        // Constructor fallback never enters generic semantic-name or MEX
        // dispatch, which could select an unrelated callable by spelling.
        return None;
    }
    if matches!(
        identity,
        CallableIdentity::DynamicName(_)
            | CallableIdentity::Imported(_)
            | CallableIdentity::ExternalName(_)
    ) {
        // A private source selected by the active caller is already an exact
        // semantic target. Public names retain the loader's MEX-first order.
        if let Some(target) = resolve_semantic_function_by_name(&name)
            .and_then(|function| selected_scoped_name_target(&name, function))
        {
            let selected = async {
                match target {
                    ScopedSemanticNameTarget::Published(function) => {
                        try_call_semantic_function_in_context(
                            selected_runtime.as_ref(),
                            function.0,
                            &args,
                            requested_outputs,
                        )
                        .await
                    }
                    ScopedSemanticNameTarget::Retained(binding) => {
                        match selected_runtime.as_ref() {
                            Some(runtime) => {
                                try_call_bound_function_in_context(
                                    runtime,
                                    binding,
                                    &args,
                                    requested_outputs,
                                )
                                .await
                            }
                            None => {
                                try_call_bound_function(binding, &args, requested_outputs).await
                            }
                        }
                    }
                }
            };
            if let Some(result) = crate::callsite::with_function_input_callsite(
                function_input_callsite.clone(),
                selected,
            )
            .await
            {
                return Some(result);
            }
        }
        if let Some(result) = try_load_and_call_dynamic_function(
            name.clone(),
            args.clone(),
            requested_outputs,
            DynamicFunctionLoadPhase::BeforeSemantic,
            function_input_callsite.clone(),
        )
        .await
        {
            return Some(result);
        }
    }
    if let Some(function) = resolve_semantic_function_by_name(&name) {
        if let Some(result) = crate::callsite::with_function_input_callsite(
            function_input_callsite.clone(),
            try_call_selected_semantic_name_in_context(
                selected_runtime.as_ref(),
                &name,
                function,
                &args,
                requested_outputs,
            ),
        )
        .await
        {
            return Some(result);
        }
    }
    if matches!(
        identity,
        CallableIdentity::DynamicName(_)
            | CallableIdentity::Imported(_)
            | CallableIdentity::ExternalName(_)
    ) {
        if let Some(result) = try_load_and_call_dynamic_function(
            name.clone(),
            args.clone(),
            requested_outputs,
            DynamicFunctionLoadPhase::AfterSemantic,
            function_input_callsite,
        )
        .await
        {
            return Some(result);
        }
    }
    if matches!(
        identity,
        CallableIdentity::ExternalName(_) | CallableIdentity::Imported(_)
    ) {
        if let Some(context) = crate::context::legacy::active() {
            if let Some(result) =
                crate::foreign::try_invoke_java(context, &name, args, requested_outputs).await
            {
                return Some(result);
            }
        }
    }
    None
}
