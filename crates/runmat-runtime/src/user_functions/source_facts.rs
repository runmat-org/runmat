use super::*;

#[cfg(test)]
#[path = "source_facts/explicit_context_tests.rs"]
mod explicit_context_tests;

type StaticSiteResolver = dyn Fn(
    usize,
    Option<&SourceFunctionInfo>,
    &Arc<runmat_value::FunctionDefinitionOwner>,
) -> Option<Vec<runmat_value::StaticSiteDefinitionLease>>;

type SelectedClassResolver = dyn Fn(
    &runmat_value::BoundFunctionBinding,
) -> Option<Rc<crate::class_registry::SelectedClassProjection>>;

type ExternalSourceResolver =
    dyn Fn(runmat_execution::FunctionBindingOrigin) -> Option<SelectedExternalSourceTarget>;

pub type SelectedExternalInvocation = dyn Fn(
    crate::context::RuntimeContext,
    crate::call::RootedFunctionArguments,
    usize,
    Option<runmat_value::LexicalEnvironment>,
) -> crate::context::RuntimeServiceFuture<
    Result<crate::execution::RootedValueSequence, crate::RuntimeError>,
>;

/// Identity and the existing invocation owner projected together from one
/// verified linked Native IR body. The local coordinate never names its
/// portable origin; source issuance independently authenticates that origin.
pub struct LinkedNativeInvocation {
    pub compiled_function: runmat_types::FunctionId,
    pub origin: runmat_execution::FunctionBindingOrigin,
    pub invoke: Rc<SelectedExternalInvocation>,
}

#[derive(Clone)]
struct SelectedNativeInvocation {
    origin: runmat_execution::FunctionBindingOrigin,
    function: runmat_types::FunctionId,
    source: crate::source_context::RuntimeSourceRef,
    owner: crate::source_context::SelectedProgramSourceOwner,
    invoke: Rc<SelectedExternalInvocation>,
}

impl std::fmt::Debug for SelectedNativeInvocation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SelectedNativeInvocation")
            .field("origin", &self.origin)
            .field("function", &self.function)
            .finish_non_exhaustive()
    }
}

impl runmat_value::SelectedNativeFunctionLease for SelectedNativeInvocation {
    fn compiled_function(&self) -> runmat_types::FunctionId {
        self.function
    }
    fn origin(&self) -> runmat_execution::FunctionBindingOrigin {
        self.origin
    }

    fn is_live(&self) -> bool {
        self.owner.is_live() && self.owner.owns_source(self.source)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl SelectedNativeInvocation {
    fn admits(
        &self,
        runtime: &crate::context::RuntimeContext,
        function: usize,
        origin: runmat_execution::FunctionBindingOrigin,
    ) -> bool {
        self.origin == origin
            && self.function.0 == function
            && runmat_value::SelectedNativeFunctionLease::is_live(self)
            && self.owner.session_lease().ptr_eq(&runtime.session_lease())
    }
}

/// Validate the actual installed native owner, not only its portable label.
pub(crate) fn selected_native_binding_admitted(
    runtime: &crate::context::RuntimeContext,
    binding: &runmat_value::BoundFunctionBinding,
    origin: runmat_execution::FunctionBindingOrigin,
) -> bool {
    let runmat_value::BoundFunctionAuthority::SelectedNative {
        origin: issued,
        lease,
    } = &binding.authority
    else {
        return false;
    };
    *issued == origin
        && lease
            .as_any()
            .downcast_ref::<SelectedNativeInvocation>()
            .is_some_and(|target| target.admits(runtime, binding.function, origin))
}

pub(super) async fn invoke_selected_native_lease(
    runtime: crate::context::RuntimeContext,
    function: usize,
    origin: runmat_execution::FunctionBindingOrigin,
    lease: Rc<dyn runmat_value::SelectedNativeFunctionLease>,
    arguments: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
) -> Result<crate::execution::RootedValueSequence, crate::RuntimeError> {
    let Some(target) = lease.as_any().downcast_ref::<SelectedNativeInvocation>() else {
        return Err(crate::runtime_error::semantic_error(
            "UnavailableBoundFunction",
            "selected native binding has no installed runtime implementation",
        ));
    };
    let target = target.clone();
    if !target.admits(&runtime, function, origin) {
        return Err(crate::runtime_error::semantic_error(
            "UnavailableBoundFunction",
            "selected native binding belongs to another or ended source session",
        ));
    }
    (target.invoke)(runtime, arguments, requested_outputs, lexical_environment).await
}

/// An exact linked function and issued source, retained for one selected
/// invocation. This is not VM bytecode and cannot be built from a name or
/// session-local function number alone.
#[derive(Clone)]
pub struct SelectedExternalSourceTarget {
    origin: runmat_execution::FunctionBindingOrigin,
    source: crate::source_context::RuntimeSourceRef,
    owner: crate::source_context::SelectedProgramSourceOwner,
    function: runmat_types::FunctionId,
    fixed_outputs: usize,
    has_varargout: bool,
    selected_product: Option<runmat_execution::Digest>,
    native_ir: runmat_execution::Digest,
    native_function: runmat_execution::Digest,
    source_ordinal: runmat_types::ProgramSourceId,
    invocation: Rc<SelectedNativeInvocation>,
}

impl std::fmt::Debug for SelectedExternalSourceTarget {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SelectedExternalSourceTarget")
            .field("origin", &self.origin)
            .field("source", &self.source)
            .field("function", &self.function)
            .finish_non_exhaustive()
    }
}

impl SelectedExternalSourceTarget {
    pub fn from_linked_function(
        runtime: &crate::context::RuntimeContext,
        owner: crate::source_context::SelectedProgramSourceOwner,
        info: &SourceFunctionInfo,
        source_ordinal: runmat_types::ProgramSourceId,
        fixed_outputs: usize,
        has_varargout: bool,
        selected_product: Option<runmat_execution::Digest>,
        native_ir: runmat_execution::Digest,
        native_function: runmat_execution::Digest,
        invocation: LinkedNativeInvocation,
    ) -> Option<Self> {
        let origin = info.binding_origin?;
        let source = runtime.program_source_ref(source_ordinal)?;
        if origin.program != runtime.program_revision()?.identity_digest().ok()?
            || origin != invocation.origin
            || info.function != invocation.compiled_function.0
            || owner.source_ordinal_for(source) != Some(source_ordinal)
            || !owner.matches_issued_source_id(source, info.source_id)
            || !owner.is_live()
        {
            return None;
        }
        Some(Self {
            origin,
            source,
            owner: owner.clone(),
            function: invocation.compiled_function,
            fixed_outputs,
            has_varargout,
            selected_product,
            native_ir,
            native_function,
            source_ordinal,
            invocation: Rc::new(SelectedNativeInvocation {
                origin,
                function: invocation.compiled_function,
                source,
                owner: owner.clone(),
                invoke: invocation.invoke,
            }),
        })
    }

    pub fn origin(&self) -> runmat_execution::FunctionBindingOrigin {
        self.origin
    }
    pub fn function(&self) -> usize {
        self.function.0
    }
    pub fn binding(&self) -> Option<runmat_value::BoundFunctionBinding> {
        runmat_value::BoundFunctionBinding::selected_native(
            self.function,
            Rc::clone(&self.invocation) as Rc<dyn runmat_value::SelectedNativeFunctionLease>,
        )
    }
    pub fn output_arity(&self) -> (usize, bool) {
        (self.fixed_outputs, self.has_varargout)
    }
    pub fn selected_product(&self) -> Option<runmat_execution::Digest> {
        self.selected_product
    }
    pub fn native_ir(&self) -> runmat_execution::Digest {
        self.native_ir
    }
    pub fn native_function(&self) -> runmat_execution::Digest {
        self.native_function
    }
    pub fn source_ordinal(&self) -> runmat_types::ProgramSourceId {
        self.source_ordinal
    }
    pub fn linked_import(
        &self,
        caller: runmat_execution::executable::ExecutableStaticCaller,
        site: runmat_execution::value::GraphStaticReferenceSite,
    ) -> Option<runmat_execution::executable::LinkedNativeStaticImport> {
        // An ordinary linked program owns executable native leases, but only
        // a selected deployment product can authorize a portable static link.
        let selected_product = self.selected_product?;
        self.is_live()
            .then_some(runmat_execution::executable::LinkedNativeStaticImport {
                caller,
                site,
                target: self.origin,
                source: self.source_ordinal,
                selected_product,
                native_ir: self.native_ir,
                native_function: self.native_function,
            })
    }
    pub fn source(&self) -> crate::source_context::RuntimeSourceRef {
        self.source
    }
    pub fn is_live(&self) -> bool {
        self.owner.is_live() && self.owner.owns_source(self.source)
    }

    pub fn same_issuing_session(
        &self,
        owner: &crate::source_context::SelectedProgramSourceOwner,
    ) -> bool {
        self.is_live() && self.owner.session_lease().ptr_eq(&owner.session_lease())
    }

    pub fn same_loaded_target(&self, other: &Self) -> bool {
        self.origin == other.origin
            && self.source == other.source
            && self.owner.same_issuance(&other.owner)
            && self.selected_product == other.selected_product
            && self.native_ir == other.native_ir
            && self.native_function == other.native_function
            && Rc::ptr_eq(&self.invocation, &other.invocation)
    }
}

/// The selected executable owner supplies both exact lookup and source-group
/// enumeration. It is separate from the published portable-origin catalog:
/// retired bodies can create local handles without becoming name/transfer
/// authority again.
pub struct ExecutionSourceFacts {
    by_function: Rc<dyn Fn(usize) -> Option<SourceFunctionInfo>>,
    by_source: Rc<dyn Fn(SourceId) -> Vec<SourceFunctionInfo>>,
    by_owners: Rc<dyn Fn(usize) -> Option<DefinitionOwnerGraph>>,
    by_static_sites: Rc<StaticSiteResolver>,
    by_origin: Rc<dyn Fn(runmat_execution::FunctionBindingOrigin) -> Option<SourceFunctionInfo>>,
    selected_class: Option<Rc<SelectedClassResolver>>,
    external_source: Option<Rc<ExternalSourceResolver>>,
}

pub type DefinitionOwnerGraph = Arc<Vec<(usize, Arc<runmat_value::FunctionDefinitionOwner>)>>;

impl ExecutionSourceFacts {
    pub fn new(
        by_function: Rc<dyn Fn(usize) -> Option<SourceFunctionInfo>>,
        by_source: Rc<dyn Fn(SourceId) -> Vec<SourceFunctionInfo>>,
        by_owners: Rc<dyn Fn(usize) -> Option<DefinitionOwnerGraph>>,
        by_origin: Rc<
            dyn Fn(runmat_execution::FunctionBindingOrigin) -> Option<SourceFunctionInfo>,
        >,
    ) -> Self {
        Self {
            by_function,
            by_source,
            by_owners,
            by_static_sites: Rc::new(|_, _, _| Some(Vec::new())),
            by_origin,
            selected_class: None,
            external_source: None,
        }
    }

    /// Read an exact selected constructor's cached class projection through
    /// this invocation's source/definition issuer; no class-name lookup.
    pub fn with_selected_class(mut self, selected: Rc<SelectedClassResolver>) -> Self {
        self.selected_class = Some(selected);
        self
    }

    pub fn with_external_source(mut self, selected: Rc<ExternalSourceResolver>) -> Self {
        self.external_source = Some(selected);
        self
    }

    pub fn external_source(
        &self,
        origin: runmat_execution::FunctionBindingOrigin,
    ) -> Option<SelectedExternalSourceTarget> {
        let target = self.external_source.as_ref()?(origin)?;
        (target.origin() == origin && target.is_live()).then_some(target)
    }

    pub fn selected_class(
        &self,
        binding: &runmat_value::BoundFunctionBinding,
    ) -> Option<Rc<crate::class_registry::SelectedClassProjection>> {
        self.selected_class.as_ref()?(binding)
    }

    /// Selected VM execution may issue counted, site-qualified imports. A
    /// catalog-only resolver cannot manufacture them from names or origins.
    pub fn with_static_sites(mut self, by_static_sites: Rc<StaticSiteResolver>) -> Self {
        self.by_static_sites = by_static_sites;
        self
    }

    fn function(&self, function: usize) -> Option<SourceFunctionInfo> {
        (self.by_function)(function)
    }

    fn source_group(&self, source: SourceId) -> Vec<SourceFunctionInfo> {
        (self.by_source)(source)
    }

    fn owners(&self, function: usize) -> Option<DefinitionOwnerGraph> {
        (self.by_owners)(function)
    }

    fn static_sites(
        &self,
        function: usize,
        info: Option<&SourceFunctionInfo>,
        owner: &std::sync::Arc<runmat_value::FunctionDefinitionOwner>,
    ) -> Option<Vec<runmat_value::StaticSiteDefinitionLease>> {
        (self.by_static_sites)(function, info, owner)
    }

    fn origin(
        &self,
        origin: runmat_execution::FunctionBindingOrigin,
    ) -> Option<SourceFunctionInfo> {
        (self.by_origin)(origin)
    }
}

pub type ExecutionSourceResolver = ExecutionSourceFacts;

pub fn selected_class_projection_for_binding(
    binding: &runmat_value::BoundFunctionBinding,
) -> Option<Rc<crate::class_registry::SelectedClassProjection>> {
    selected_execution_facts()?.selected_class(binding)
}

/// Observe a linked selected function only through the active invocation's
/// execution-source resolver, never a later name or portable-origin catalog.
pub fn selected_external_source_target(
    origin: runmat_execution::FunctionBindingOrigin,
) -> Option<SelectedExternalSourceTarget> {
    selected_execution_facts()?.external_source(origin)
}

#[derive(Debug, Clone)]
pub struct SourceFunctionInfo {
    pub source_id: SourceId,
    pub name: String,
    pub function: usize,
    pub role: runmat_types::FunctionSourceRole,
    /// Immutable source-program binding, not the session-local numeric ID.
    pub binding_origin: Option<runmat_execution::FunctionBindingOrigin>,
    /// Display path captured by the publishing definition; never recovered
    /// from a later numeric SourceId after the definition is retired.
    pub source_file: Option<String>,
    /// Opaque identity of the actual executable definition, not just its
    /// name or session-local number.
    pub definition_owners: Option<DefinitionOwnerGraph>,
}

pub fn source_function_for(function: usize) -> Option<SourceFunctionInfo> {
    if let Some(resolver) = selected_execution_facts() {
        return resolver.function(function);
    }
    selected_source_catalog()?
        .iter()
        .find(|info| info.function == function)
        .cloned()
}

/// Resolve only an exact immutable program/function binding. A reused numeric
/// session ID or matching display name is never sufficient authority.
pub fn resolve_binding_origin(origin: runmat_execution::FunctionBindingOrigin) -> Option<usize> {
    fn function(info: &SourceFunctionInfo) -> usize {
        info.function
    }
    if let Some(resolver) = selected_execution_facts() {
        return resolver.origin(origin).as_ref().map(function);
    }
    unique_origin(selected_source_catalog()?.as_ref(), origin).map(function)
}

/// Resolve an immutable portable origin into the exact selected definition
/// graph. When the catalog owns definitions, the returned value carries local
/// definition leases; copying the value therefore retains its callable graph.
pub fn resolve_binding_origin_binding(
    origin: runmat_execution::FunctionBindingOrigin,
) -> Option<runmat_value::BoundFunctionBinding> {
    let info = if let Some(resolver) = selected_execution_facts() {
        resolver.origin(origin)?
    } else {
        unique_origin(selected_source_catalog()?.as_ref(), origin)?.clone()
    };
    Some(runmat_value::BoundFunctionBinding {
        function: info.function,
        authority: authority_for_selected_info(info.function, Some(&info)),
    })
}

/// Resolve a portable origin against one invocation-owned exact source graph.
/// This path never consults or mutates the ambient resolver slot.
pub(crate) fn resolve_binding_origin_binding_in(
    resolver: &ExecutionSourceResolver,
    origin: runmat_execution::FunctionBindingOrigin,
) -> Option<runmat_value::BoundFunctionBinding> {
    let info = resolver.origin(origin)?;
    if info.binding_origin != Some(origin) {
        return None;
    }
    Some(runmat_value::BoundFunctionBinding {
        function: info.function,
        authority: authority_for_info(
            info.function,
            Some(&info),
            Some(resolver),
            current_session_lease(),
        ),
    })
}

/// Bind an exact portable origin to an explicitly selected invocation.
///
/// The caller must derive `resolver` from the same admitted executable that
/// supplied `origin`; neither payload bytes nor the ambient resolver certify
/// that pairing. The issued lease belongs to `runtime`, not a later poll's
/// ambient session. Missing or mismatched origins fail closed.
pub fn resolve_binding_origin_binding_in_context(
    resolver: &ExecutionSourceResolver,
    origin: runmat_execution::FunctionBindingOrigin,
    runtime: &crate::context::RuntimeContext,
) -> Option<runmat_value::BoundFunctionBinding> {
    let info = resolver.origin(origin)?;
    if info.binding_origin != Some(origin) {
        return None;
    }
    Some(runmat_value::BoundFunctionBinding {
        function: info.function,
        authority: authority_for_info(
            info.function,
            Some(&info),
            Some(resolver),
            runtime.session_lease(),
        ),
    })
}

/// Resolve an exact origin using only this runtime's selected source facts.
/// An explicitly absent invocation projection masks the session state; neither
/// another active runtime nor the compatibility catalog may supply authority.
pub fn resolve_binding_origin_binding_from_runtime(
    runtime: &crate::context::RuntimeContext,
    origin: runmat_execution::FunctionBindingOrigin,
) -> Option<runmat_value::BoundFunctionBinding> {
    let resolver = execution_facts_from_runtime(runtime)?;
    resolve_binding_origin_binding_in_context(&resolver, origin, runtime)
}

fn unique_origin(
    catalog: &[SourceFunctionInfo],
    origin: runmat_execution::FunctionBindingOrigin,
) -> Option<&SourceFunctionInfo> {
    let mut matches = catalog
        .iter()
        .filter(|info| info.binding_origin == Some(origin));
    let info = matches.next()?;
    matches.next().is_none().then_some(info)
}

/// Bind a semantic function while preserving its source role at the value
/// owner. A missing catalog is explicitly Unknown, never inferred from name.
pub fn bound_function_handle(name: String, function: usize) -> Value {
    let info = source_function_for(function);
    let source_role = info
        .as_ref()
        .map_or(runmat_types::FunctionSourceRole::Unknown, |info| info.role);
    let source_file = info.as_ref().and_then(|info| info.source_file.clone());
    Value::bound_function_handle(
        name,
        function,
        authority_for_selected_info(function, info.as_ref()),
        source_role,
        source_file,
    )
}

pub fn bound_function_handle_from_source_info(info: SourceFunctionInfo) -> Value {
    let authority = authority_for_selected_info(info.function, Some(&info));
    Value::bound_function_handle(
        info.name,
        info.function,
        authority,
        info.role,
        info.source_file,
    )
}

/// Issue local execution authority from the exact published definition. A
/// portable-only catalog can issue an immutable origin, but a missing catalog
/// never fabricates authority from a display name or numeric function ID.
pub fn bound_function_authority_for(function: usize) -> runmat_value::BoundFunctionAuthority {
    let info = source_function_for(function);
    authority_for_selected_info(function, info.as_ref())
}

fn authority_for_selected_info(
    function: usize,
    info: Option<&SourceFunctionInfo>,
) -> runmat_value::BoundFunctionAuthority {
    let selected = selected_execution_facts();
    authority_for_info(function, info, selected.as_deref(), current_session_lease())
}

fn authority_for_info(
    function: usize,
    info: Option<&SourceFunctionInfo>,
    selected: Option<&ExecutionSourceResolver>,
    issuer: std::sync::Weak<()>,
) -> runmat_value::BoundFunctionAuthority {
    use runmat_value::{BoundFunctionAuthority, FunctionDefinitionOwner, LocalExport};

    let owners = info
        .and_then(|info| info.definition_owners.clone())
        .or_else(|| selected.and_then(|facts| facts.owners(function)));
    if let Some(owners) = owners.as_ref() {
        let Some((_, owner)) = owners.iter().find(|(id, _)| *id == function) else {
            return dead_bound_authority(function);
        };
        let static_sites = match selected {
            Some(facts) => match facts.static_sites(function, info, owner) {
                Some(sites) => sites,
                None => return dead_bound_authority(function),
            },
            None => Vec::new(),
        };
        let export = info
            .as_ref()
            .and_then(|info| info.binding_origin)
            .map_or(LocalExport::Ephemeral, LocalExport::Exact);
        return BoundFunctionAuthority::local_with_static_sites(
            issuer,
            FunctionDefinitionOwner::issue(owner, function),
            owners
                .iter()
                .filter(|(id, _)| *id != function)
                .map(|(id, owner)| FunctionDefinitionOwner::issue(owner, *id))
                .collect(),
            static_sites,
            export,
        );
    }
    if let (Some(info), Some(selected)) = (info, selected) {
        if let Some(origin) = info.binding_origin {
            if let Some(target) = selected.external_source(origin) {
                return target
                    .binding()
                    .filter(|binding| binding.function == function)
                    .map_or_else(
                        || dead_bound_authority(function),
                        |binding| binding.authority,
                    );
            }
        }
    }
    if selected.is_some() {
        return dead_bound_authority(function);
    }
    if let Some(origin) = info.and_then(|info| info.binding_origin) {
        return BoundFunctionAuthority::portable(origin);
    }
    dead_bound_authority(function)
}

fn dead_bound_authority(function: usize) -> runmat_value::BoundFunctionAuthority {
    runmat_value::BoundFunctionAuthority::unavailable(function)
}

pub fn bound_function_binding_for(function: usize) -> runmat_value::BoundFunctionBinding {
    runmat_value::BoundFunctionBinding {
        function,
        authority: bound_function_authority_for(function),
    }
}

/// Capture the registered display path at value creation, never by a later
/// numeric function-ID lookup after the value crosses a session boundary.
pub fn source_file_for(function: usize) -> Option<String> {
    source_function_for(function)?.source_file
}

pub fn source_functions_for(source_id: SourceId) -> Vec<SourceFunctionInfo> {
    if let Some(facts) = selected_execution_facts() {
        return facts.source_group(source_id);
    }
    let catalog = selected_source_catalog();
    source_functions_in_catalog(catalog.as_deref(), source_id)
}

fn selected_execution_facts() -> Option<Rc<ExecutionSourceFacts>> {
    if let Some(runtime) = crate::context::legacy::active() {
        return execution_facts_from_runtime(&runtime);
    }
    EXECUTION_SOURCE_RESOLVER.with(|slot| slot.borrow().clone())
}

fn execution_facts_from_runtime(
    runtime: &crate::context::RuntimeContext,
) -> Option<Rc<ExecutionSourceFacts>> {
    runtime
        .selected_call_projection()
        .borrow()
        .execution_source_resolver()
        .unwrap_or_else(|| {
            runtime
                .state()
                .call
                .borrow()
                .execution_source_resolver
                .clone()
        })
}

fn selected_source_catalog() -> Option<Arc<Vec<SourceFunctionInfo>>> {
    if let Some(runtime) = crate::context::legacy::active() {
        return runtime
            .call_projection()
            .borrow()
            .source_functions()
            .unwrap_or_else(|| runtime.state().call.borrow().source_functions.clone());
    }
    SOURCE_FUNCTION_CATALOG.with(|slot| slot.borrow().clone())
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
