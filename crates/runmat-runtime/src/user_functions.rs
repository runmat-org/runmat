use crate::RuntimeError;
use runmat_thread_local::runmat_thread_local;
use runmat_types::{CallableFallbackPolicy, CallableIdentity, SourceId};
use runmat_value::{Value, ValueSequence};
use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;

pub type UserFunctionFuture = Pin<Box<dyn Future<Output = Result<ValueSequence, RuntimeError>>>>;
pub type DynamicFunctionLoadFuture =
    Pin<Box<dyn Future<Output = Option<Result<ValueSequence, RuntimeError>>>>>;
pub type DynamicFunctionClearFuture = Pin<Box<dyn Future<Output = Result<(), RuntimeError>>>>;
pub type FunctionInvoker = dyn Fn(usize, &[Value], usize) -> UserFunctionFuture;
#[derive(Debug, Clone)]
pub struct ExternalFunctionCall {
    pub function: usize,
    pub display_name: String,
    pub arguments: Vec<Value>,
    pub requested_outputs: usize,
}
pub type ExternalFunctionInvoker = dyn Fn(ExternalFunctionCall) -> UserFunctionFuture;
pub type LexicalFunctionFuture =
    Pin<Box<dyn Future<Output = Result<crate::call::lexical::LexicalCallResult, RuntimeError>>>>;
pub type LexicalFunctionInvoker =
    dyn Fn(crate::call::lexical::LexicalCall) -> LexicalFunctionFuture;
pub type FunctionResolver = dyn Fn(&str) -> Option<usize> + Send + Sync;

/// The point in callable resolution at which a session-backed loader runs.
///
/// Native callable artifacts such as MEX modules have language-defined
/// precedence over source functions. Ordinary dynamically discovered source
/// files are consulted only after the analyzed semantic registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicFunctionLoadPhase {
    BeforeSemantic,
    AfterSemantic,
}

pub type DynamicFunctionLoader = dyn Fn(
    crate::context::RuntimeContext,
    String,
    Vec<Value>,
    usize,
    DynamicFunctionLoadPhase,
    Option<crate::callsite::CallsiteInfo>,
) -> DynamicFunctionLoadFuture;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicFunctionClearRequest {
    All,
    NativeExtensions,
    Named(String),
}

/// Refresh the source lookup inventory without clearing live native modules or workspace values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicFunctionRefreshRequest {
    Default,
    Path,
    Toolbox,
    ToolboxCache,
}

pub type DynamicFunctionRefresher = dyn Fn(
    crate::context::RuntimeContext,
    DynamicFunctionRefreshRequest,
) -> DynamicFunctionClearFuture;

pub type DynamicFunctionClearer = dyn Fn(
    crate::context::RuntimeContext,
    DynamicFunctionClearRequest,
) -> DynamicFunctionClearFuture;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFunctionInfo {
    pub source_id: SourceId,
    pub name: String,
    pub function: usize,
    pub role: runmat_types::FunctionSourceRole,
    /// Immutable source-program binding, not the session-local numeric ID.
    pub binding_origin: Option<runmat_execution::FunctionBindingOrigin>,
}

pub fn source_function_for(function: usize) -> Option<SourceFunctionInfo> {
    if let Some(state) = active_state() {
        return state
            .call
            .borrow()
            .source_functions
            .as_ref()?
            .iter()
            .find(|info| info.function == function)
            .cloned();
    }
    SOURCE_FUNCTION_CATALOG.with(|slot| {
        slot.borrow()
            .as_ref()?
            .iter()
            .find(|info| info.function == function)
            .cloned()
    })
}

/// Resolve only an exact immutable program/function binding. A reused numeric
/// session ID or matching display name is never sufficient authority.
pub fn resolve_binding_origin(origin: runmat_execution::FunctionBindingOrigin) -> Option<usize> {
    fn unique(
        catalog: &[SourceFunctionInfo],
        origin: runmat_execution::FunctionBindingOrigin,
    ) -> Option<usize> {
        let mut matches = catalog
            .iter()
            .filter(|info| info.binding_origin == Some(origin))
            .map(|info| info.function);
        let function = matches.next()?;
        matches.next().is_none().then_some(function)
    }
    if let Some(state) = active_state() {
        return unique(state.call.borrow().source_functions.as_deref()?, origin);
    }
    SOURCE_FUNCTION_CATALOG.with(|slot| unique(slot.borrow().as_deref()?, origin))
}

/// Bind a semantic function while preserving its source role at the value
/// owner. A missing catalog is explicitly Unknown, never inferred from name.
pub fn bound_function_handle(name: String, function: usize) -> Value {
    let source_role = source_function_for(function)
        .map_or(runmat_types::FunctionSourceRole::Unknown, |info| info.role);
    Value::BoundFunctionHandle {
        name,
        function,
        source_role,
        source_file: source_file_for(function),
    }
}

/// Capture the registered display path at value creation, never by a later
/// numeric function-ID lookup after the value crosses a session boundary.
pub fn source_file_for(function: usize) -> Option<String> {
    let info = source_function_for(function)?;
    crate::source_context::source_info(info.source_id)
        .and_then(|source| source.fullpath_name)
        .map(|path| path.to_string())
}

#[derive(Debug, Clone)]
pub struct CallableRequest {
    identity: CallableIdentity,
    fallback_policy: CallableFallbackPolicy,
    args: Vec<Value>,
    requested_outputs: usize,
    function_input_callsite: Option<crate::callsite::CallsiteInfo>,
}

impl CallableRequest {
    pub fn semantic(function: usize, args: Vec<Value>, requested_outputs: usize) -> Self {
        Self {
            identity: CallableIdentity::BoundFunction(runmat_types::FunctionId(function)),
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

runmat_thread_local! {
    static SEMANTIC_FUNCTION_INVOKER: RefCell<Option<Rc<FunctionInvoker>>> =
        const { RefCell::new(None) };
    static EXTERNAL_FUNCTION_INVOKER: RefCell<Option<Rc<ExternalFunctionInvoker>>> =
        const { RefCell::new(None) };
    static LEXICAL_FUNCTION_INVOKER: RefCell<Option<Rc<LexicalFunctionInvoker>>> =
        const { RefCell::new(None) };
    static SEMANTIC_FUNCTION_RESOLVER: RefCell<Option<Arc<FunctionResolver>>> =
        const { RefCell::new(None) };
    static SOURCE_FUNCTION_CATALOG: RefCell<Option<Arc<Vec<SourceFunctionInfo>>>> =
        const { RefCell::new(None) };
    static ACTIVE_SEMANTIC_FUNCTION_STACK: RefCell<Vec<usize>> =
        const { RefCell::new(Vec::new()) };
}

pub struct FunctionInvokerGuard {
    previous: Option<Rc<FunctionInvoker>>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

pub struct ExternalFunctionInvokerGuard {
    previous: Option<Rc<ExternalFunctionInvoker>>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

pub struct LexicalFunctionInvokerGuard {
    previous: Option<Rc<LexicalFunctionInvoker>>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

pub struct FunctionResolverGuard {
    previous: Option<Arc<FunctionResolver>>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

pub struct SourceFunctionCatalogGuard {
    previous: Option<Arc<Vec<SourceFunctionInfo>>>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

pub struct ActiveSemanticFunctionGuard {
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

impl Drop for FunctionInvokerGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        if let Some(state) = &self.state {
            state.call.borrow_mut().semantic_invoker = previous;
        } else {
            SEMANTIC_FUNCTION_INVOKER.with(|slot| {
                *slot.borrow_mut() = previous;
            });
        }
    }
}

impl Drop for ExternalFunctionInvokerGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        if let Some(state) = &self.state {
            state.call.borrow_mut().external_invoker = previous;
        } else {
            EXTERNAL_FUNCTION_INVOKER.with(|slot| {
                *slot.borrow_mut() = previous;
            });
        }
    }
}

impl Drop for LexicalFunctionInvokerGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        if let Some(state) = &self.state {
            state.call.borrow_mut().lexical_invoker = previous;
        } else {
            LEXICAL_FUNCTION_INVOKER.with(|slot| {
                *slot.borrow_mut() = previous;
            });
        }
    }
}

impl Drop for FunctionResolverGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        if let Some(state) = &self.state {
            state.call.borrow_mut().semantic_resolver = previous;
        } else {
            SEMANTIC_FUNCTION_RESOLVER.with(|slot| {
                *slot.borrow_mut() = previous;
            });
        }
    }
}

impl Drop for SourceFunctionCatalogGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        if let Some(state) = &self.state {
            state.call.borrow_mut().source_functions = previous;
        } else {
            SOURCE_FUNCTION_CATALOG.with(|slot| {
                *slot.borrow_mut() = previous;
            });
        }
    }
}

impl Drop for ActiveSemanticFunctionGuard {
    fn drop(&mut self) {
        if let Some(state) = &self.state {
            state.call.borrow_mut().active_functions.pop();
        } else {
            ACTIVE_SEMANTIC_FUNCTION_STACK.with(|slot| {
                slot.borrow_mut().pop();
            });
        }
    }
}

pub fn install_semantic_function_invoker(
    invoker: Option<Arc<FunctionInvoker>>,
) -> FunctionInvokerGuard {
    replace_semantic_function_invoker(invoker.map(|invoker| {
        Rc::new(move |function, arguments: &[Value], requested_outputs| {
            invoker(function, arguments, requested_outputs)
        }) as Rc<FunctionInvoker>
    }))
}

pub fn install_local_semantic_function_invoker(
    invoker: Rc<FunctionInvoker>,
) -> FunctionInvokerGuard {
    replace_semantic_function_invoker(Some(invoker))
}

pub fn clear_semantic_function_invoker() -> FunctionInvokerGuard {
    replace_semantic_function_invoker(None)
}

fn replace_semantic_function_invoker(invoker: Option<Rc<FunctionInvoker>>) -> FunctionInvokerGuard {
    if let Some(state) = active_state() {
        let previous = std::mem::replace(&mut state.call.borrow_mut().semantic_invoker, invoker);
        return FunctionInvokerGuard {
            previous,
            state: Some(state),
        };
    }
    let previous =
        SEMANTIC_FUNCTION_INVOKER.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), invoker));
    FunctionInvokerGuard {
        previous,
        state: None,
    }
}

pub fn install_external_function_invoker(
    invoker: Option<Arc<ExternalFunctionInvoker>>,
) -> ExternalFunctionInvokerGuard {
    replace_external_function_invoker(
        invoker.map(|invoker| Rc::new(move |call| invoker(call)) as Rc<ExternalFunctionInvoker>),
    )
}

pub fn install_local_external_function_invoker(
    invoker: Rc<ExternalFunctionInvoker>,
) -> ExternalFunctionInvokerGuard {
    replace_external_function_invoker(Some(invoker))
}

fn replace_external_function_invoker(
    invoker: Option<Rc<ExternalFunctionInvoker>>,
) -> ExternalFunctionInvokerGuard {
    if let Some(state) = active_state() {
        let previous = std::mem::replace(&mut state.call.borrow_mut().external_invoker, invoker);
        return ExternalFunctionInvokerGuard {
            previous,
            state: Some(state),
        };
    }
    let previous =
        EXTERNAL_FUNCTION_INVOKER.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), invoker));
    ExternalFunctionInvokerGuard {
        previous,
        state: None,
    }
}

pub fn install_lexical_function_invoker(
    invoker: Option<Arc<LexicalFunctionInvoker>>,
) -> LexicalFunctionInvokerGuard {
    replace_lexical_function_invoker(
        invoker.map(|invoker| Rc::new(move |call| invoker(call)) as Rc<LexicalFunctionInvoker>),
    )
}

pub fn install_local_lexical_function_invoker(
    invoker: Rc<LexicalFunctionInvoker>,
) -> LexicalFunctionInvokerGuard {
    replace_lexical_function_invoker(Some(invoker))
}

fn replace_lexical_function_invoker(
    invoker: Option<Rc<LexicalFunctionInvoker>>,
) -> LexicalFunctionInvokerGuard {
    if let Some(state) = active_state() {
        let previous = std::mem::replace(&mut state.call.borrow_mut().lexical_invoker, invoker);
        return LexicalFunctionInvokerGuard {
            previous,
            state: Some(state),
        };
    }
    let previous =
        LEXICAL_FUNCTION_INVOKER.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), invoker));
    LexicalFunctionInvokerGuard {
        previous,
        state: None,
    }
}

pub fn install_semantic_function_resolver(
    resolver: Option<Arc<FunctionResolver>>,
) -> FunctionResolverGuard {
    if let Some(state) = active_state() {
        let previous = std::mem::replace(&mut state.call.borrow_mut().semantic_resolver, resolver);
        return FunctionResolverGuard {
            previous,
            state: Some(state),
        };
    }
    let previous = SEMANTIC_FUNCTION_RESOLVER
        .with(|slot| std::mem::replace(&mut *slot.borrow_mut(), resolver));
    FunctionResolverGuard {
        previous,
        state: None,
    }
}

pub fn install_source_function_catalog(
    catalog: Option<Arc<Vec<SourceFunctionInfo>>>,
) -> SourceFunctionCatalogGuard {
    if let Some(state) = active_state() {
        let previous = std::mem::replace(&mut state.call.borrow_mut().source_functions, catalog);
        return SourceFunctionCatalogGuard {
            previous,
            state: Some(state),
        };
    }
    let previous =
        SOURCE_FUNCTION_CATALOG.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), catalog));
    SourceFunctionCatalogGuard {
        previous,
        state: None,
    }
}

pub fn push_active_semantic_function(function: usize) -> ActiveSemanticFunctionGuard {
    if let Some(state) = active_state() {
        state.call.borrow_mut().active_functions.push(function);
        return ActiveSemanticFunctionGuard { state: Some(state) };
    }
    ACTIVE_SEMANTIC_FUNCTION_STACK.with(|slot| {
        slot.borrow_mut().push(function);
    });
    ActiveSemanticFunctionGuard { state: None }
}

pub fn current_semantic_function_invoker() -> Option<Rc<FunctionInvoker>> {
    if let Some(state) = active_state() {
        return state.call.borrow().semantic_invoker.clone();
    }
    SEMANTIC_FUNCTION_INVOKER.with(|slot| slot.borrow().clone())
}

pub fn current_external_function_invoker() -> Option<Rc<ExternalFunctionInvoker>> {
    if let Some(state) = active_state() {
        return state.call.borrow().external_invoker.clone();
    }
    EXTERNAL_FUNCTION_INVOKER.with(|slot| slot.borrow().clone())
}

pub fn current_lexical_function_invoker() -> Option<Rc<LexicalFunctionInvoker>> {
    if let Some(state) = active_state() {
        return state.call.borrow().lexical_invoker.clone();
    }
    LEXICAL_FUNCTION_INVOKER.with(|slot| slot.borrow().clone())
}

pub fn current_semantic_function_resolver() -> Option<Arc<FunctionResolver>> {
    if let Some(state) = active_state() {
        return state.call.borrow().semantic_resolver.clone();
    }
    SEMANTIC_FUNCTION_RESOLVER.with(|slot| slot.borrow().clone())
}

pub fn current_active_semantic_function() -> Option<usize> {
    if let Some(state) = active_state() {
        return state.call.borrow().active_functions.last().copied();
    }
    ACTIVE_SEMANTIC_FUNCTION_STACK.with(|slot| slot.borrow().last().copied())
}

/// Seed a newly-created standalone runtime context from the legacy
/// thread-local call environment before entering its scope.
///
/// Embedders that already provide an explicit `RuntimeContext` must configure
/// that context directly. This bridge exists for the standalone VM entrypoints
/// that historically accepted resolver/invoker guards without a context.
pub fn inherit_legacy_call_environment(context: &crate::context::RuntimeContext) {
    let mut call = context.state().call.borrow_mut();
    call.semantic_invoker = SEMANTIC_FUNCTION_INVOKER.with(|slot| slot.borrow().clone());
    call.external_invoker = EXTERNAL_FUNCTION_INVOKER.with(|slot| slot.borrow().clone());
    call.lexical_invoker = LEXICAL_FUNCTION_INVOKER.with(|slot| slot.borrow().clone());
    call.semantic_resolver = SEMANTIC_FUNCTION_RESOLVER.with(|slot| slot.borrow().clone());
    call.source_functions = SOURCE_FUNCTION_CATALOG.with(|slot| slot.borrow().clone());
    call.active_functions = ACTIVE_SEMANTIC_FUNCTION_STACK.with(|slot| slot.borrow().clone());
}

pub fn source_functions_for(source_id: SourceId) -> Vec<SourceFunctionInfo> {
    if let Some(state) = active_state() {
        return source_functions_in_catalog(
            state.call.borrow().source_functions.as_deref(),
            source_id,
        );
    }
    SOURCE_FUNCTION_CATALOG
        .with(|slot| source_functions_in_catalog(slot.borrow().as_deref(), source_id))
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

fn source_functions_in_catalog(
    catalog: Option<&Vec<SourceFunctionInfo>>,
    source_id: SourceId,
) -> Vec<SourceFunctionInfo> {
    catalog
        .map(|catalog| {
            catalog
                .iter()
                .filter(|info| info.source_id == source_id)
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

fn active_state() -> Option<std::rc::Rc<crate::context::RuntimeContextState>> {
    crate::context::legacy::active().map(|context| std::rc::Rc::clone(context.state()))
}

pub async fn try_call_semantic_descriptor(
    request: CallableRequest,
) -> Option<Result<ValueSequence, RuntimeError>> {
    let CallableRequest {
        identity,
        fallback_policy,
        args,
        requested_outputs,
        function_input_callsite,
    } = request;
    if let CallableIdentity::BoundFunction(function) = identity {
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

#[cfg(test)]
#[path = "user_functions/input_callsite_tests.rs"]
mod input_callsite_tests;

#[cfg(test)]
#[path = "user_functions/binding_origin_tests.rs"]
mod binding_origin_tests;

#[cfg(test)]
mod lexical_tests {
    use super::*;
    use crate::call::lexical::{LexicalCall, LexicalCallResult, LexicalCapture};

    #[test]
    fn lexical_invoker_preserves_binding_identity_and_restores_scope() {
        assert!(current_lexical_function_invoker().is_none());
        let guard = install_lexical_function_invoker(Some(Arc::new(|mut call| {
            Box::pin(async move {
                assert_eq!(call.function, 7);
                assert_eq!(call.arguments, vec![Value::Num(2.0)]);
                call.captures[0].value = Value::Num(5.0);
                Ok(LexicalCallResult {
                    outputs: runmat_value::ValueSequence::single(Value::Num(7.0)).unwrap(),
                    captures: call.captures,
                })
            })
        })));
        let result = futures::executor::block_on(try_call_lexical_function(LexicalCall {
            function: 7,
            captures: vec![LexicalCapture {
                binding: runmat_types::BindingId(3),
                value: Value::Num(1.0),
            }],
            arguments: vec![Value::Num(2.0)],
            requested_outputs: 1,
        }))
        .expect("lexical invoker is installed")
        .expect("lexical call succeeds");
        assert_eq!(result.outputs.into_values(), vec![Value::Num(7.0)]);
        assert_eq!(result.captures[0].value, Value::Num(5.0));
        drop(guard);
        assert!(current_lexical_function_invoker().is_none());
    }

    #[test]
    fn external_invoker_preserves_identity_kind_and_restores_scope() {
        assert!(current_external_function_invoker().is_none());
        let guard = install_external_function_invoker(Some(Arc::new(|call| {
            crate::sequence::single_value_future(async move {
                assert_eq!(call.function, 0);
                assert_eq!(call.display_name, "published");
                assert_eq!(call.arguments, vec![Value::Num(2.0)]);
                assert_eq!(call.requested_outputs, 1);
                Ok(Value::Num(12.0))
            })
        })));
        let result =
            futures::executor::block_on(try_call_external_function(ExternalFunctionCall {
                function: 0,
                display_name: "published".into(),
                arguments: vec![Value::Num(2.0)],
                requested_outputs: 1,
            }))
            .expect("external invoker is installed")
            .expect("external call succeeds");
        assert_eq!(result.into_values(), vec![Value::Num(12.0)]);
        drop(guard);
        assert!(current_external_function_invoker().is_none());
    }
}
