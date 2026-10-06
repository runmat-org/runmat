use super::*;

#[derive(Debug, Clone)]
pub struct CallableRequest {
    identity: CallableIdentity,
    bound: Option<runmat_value::BoundFunctionBinding>,
    fallback_policy: CallableFallbackPolicy,
    args: Vec<Value>,
    requested_outputs: usize,
    function_input_callsite: Option<crate::callsite::CallsiteInfo>,
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
        }
    }

    pub fn with_function_input_callsite(
        mut self,
        callsite: Option<crate::callsite::CallsiteInfo>,
    ) -> Self {
        self.function_input_callsite = callsite;
        self
    }
}

pub async fn try_call_semantic_function(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
) -> Option<Result<ValueSequence, RuntimeError>> {
    let invoker = current_semantic_function_invoker();
    let invoker = invoker?;
    Some(invoker(function, args, requested_outputs).await)
}

pub async fn try_call_bound_function(
    binding: runmat_value::BoundFunctionBinding,
    args: &[Value],
    requested_outputs: usize,
) -> Option<Result<ValueSequence, RuntimeError>> {
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
    Some(invoker(runtime, binding, args, requested_outputs).await)
}

pub async fn try_call_bound_function_in_context(
    runtime: &crate::context::RuntimeContext,
    binding: runmat_value::BoundFunctionBinding,
    args: &[Value],
    requested_outputs: usize,
) -> Option<Result<ValueSequence, RuntimeError>> {
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
    let invoker = runtime.state().call.borrow().bound_invoker.clone()?;
    let args = match crate::execution::RootedValues::new(args.to_vec()) {
        Ok(args) => args,
        Err(error) => return Some(Err(error.into_runtime_error())),
    };
    Some(invoker(runtime.clone(), binding, args, requested_outputs).await)
}

pub async fn try_call_external_function(
    call: ExternalFunctionCall,
) -> Option<Result<ValueSequence, RuntimeError>> {
    let invoker = current_external_function_invoker()?;
    Some(invoker(call).await)
}

pub async fn try_call_lexical_function(
    call: crate::call::lexical::LexicalCall,
) -> Option<Result<crate::call::lexical::LexicalCallResult, RuntimeError>> {
    let invoker = current_lexical_function_invoker()?;
    Some(invoker(call).await)
}

pub async fn try_call_semantic_function_by_name(
    name: &str,
    args: &[Value],
    requested_outputs: usize,
) -> Option<Result<ValueSequence, RuntimeError>> {
    let function = resolve_semantic_function_by_name(name)?;
    try_call_semantic_function(function, args, requested_outputs).await
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
) -> Option<Result<ValueSequence, RuntimeError>> {
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
) -> Option<Result<ValueSequence, RuntimeError>> {
    let CallableRequest {
        identity,
        bound,
        fallback_policy,
        args,
        requested_outputs,
        function_input_callsite,
    } = request;
    if let CallableIdentity::BoundFunction(function) = identity {
        if let Some(binding) = bound {
            return crate::callsite::with_function_input_callsite(
                function_input_callsite,
                try_call_bound_function(binding, &args, requested_outputs),
            )
            .await;
        }
        return crate::callsite::with_function_input_callsite(
            function_input_callsite,
            try_call_semantic_function(function.0, &args, requested_outputs),
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
    if matches!(identity, CallableIdentity::DynamicName(_))
        && runmat_types::ClassIdentity::new(name.clone())
            .ok()
            .and_then(|identity| crate::class_registry::get_class(&identity))
            .is_some()
    {
        // Constructor calls for class names must flow through runtime constructor dispatch,
        // not generic semantic name resolution.
        return None;
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
            try_call_semantic_function(function, &args, requested_outputs),
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
