use super::*;

#[cfg(test)]
#[path = "source_facts/explicit_context_tests.rs"]
mod explicit_context_tests;

type StaticSiteResolver = dyn Fn(
    &SourceFunctionInfo,
    &Arc<runmat_value::FunctionDefinitionOwner>,
) -> Option<Vec<runmat_value::StaticSiteDefinitionLease>>;

type SelectedClassResolver = dyn Fn(
    &runmat_value::BoundFunctionBinding,
) -> Option<Rc<crate::class_registry::SelectedClassProjection>>;

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
            by_static_sites: Rc::new(|_, _| Some(Vec::new())),
            by_origin,
            selected_class: None,
        }
    }

    /// Read an exact selected constructor's cached class projection through
    /// this invocation's source/definition issuer; no class-name lookup.
    pub fn with_selected_class(mut self, selected: Rc<SelectedClassResolver>) -> Self {
        self.selected_class = Some(selected);
        self
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
        info: &SourceFunctionInfo,
        owner: &std::sync::Arc<runmat_value::FunctionDefinitionOwner>,
    ) -> Option<Vec<runmat_value::StaticSiteDefinitionLease>> {
        (self.by_static_sites)(info, owner)
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
            Some(facts) => match info.and_then(|info| facts.static_sites(info, owner)) {
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
        return runtime
            .call_projection()
            .borrow()
            .execution_source_resolver()
            .unwrap_or_else(|| {
                runtime
                    .state()
                    .call
                    .borrow()
                    .execution_source_resolver
                    .clone()
            });
    }
    EXECUTION_SOURCE_RESOLVER.with(|slot| slot.borrow().clone())
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
