use super::*;

/// The selected executable owner supplies both exact lookup and source-group
/// enumeration. It is separate from the published portable-origin catalog:
/// retired bodies can create local handles without becoming name/transfer
/// authority again.
pub struct ExecutionSourceFacts {
    by_function: Rc<dyn Fn(usize) -> Option<SourceFunctionInfo>>,
    by_source: Rc<dyn Fn(SourceId) -> Vec<SourceFunctionInfo>>,
    by_owners: Rc<dyn Fn(usize) -> Option<DefinitionOwnerGraph>>,
}

pub type DefinitionOwnerGraph = Arc<Vec<(usize, Arc<runmat_value::FunctionDefinitionOwner>)>>;

impl ExecutionSourceFacts {
    pub fn new(
        by_function: Rc<dyn Fn(usize) -> Option<SourceFunctionInfo>>,
        by_source: Rc<dyn Fn(SourceId) -> Vec<SourceFunctionInfo>>,
        by_owners: Rc<dyn Fn(usize) -> Option<DefinitionOwnerGraph>>,
    ) -> Self {
        Self {
            by_function,
            by_source,
            by_owners,
        }
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
}

pub type ExecutionSourceResolver = ExecutionSourceFacts;

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
    let info = source_function_for(function);
    let source_role = info
        .as_ref()
        .map_or(runmat_types::FunctionSourceRole::Unknown, |info| info.role);
    let source_file = info.as_ref().and_then(|info| info.source_file.clone());
    Value::BoundFunctionHandle {
        name,
        function,
        authority: authority_for_selected_info(function, info.as_ref()),
        source_role,
        source_file,
    }
}

pub fn bound_function_handle_from_source_info(info: SourceFunctionInfo) -> Value {
    let authority = authority_for_selected_info(info.function, Some(&info));
    Value::BoundFunctionHandle {
        name: info.name,
        function: info.function,
        authority,
        source_role: info.role,
        source_file: info.source_file,
    }
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
    use runmat_value::{BoundFunctionAuthority, FunctionDefinitionOwner, LocalExport};

    let selected = selected_execution_facts();
    let owners = info
        .and_then(|info| info.definition_owners.clone())
        .or_else(|| selected.as_ref().and_then(|facts| facts.owners(function)));
    if let Some(owners) = owners.as_ref() {
        let Some((_, owner)) = owners.iter().find(|(id, _)| *id == function) else {
            return dead_bound_authority(function);
        };
        let export = info
            .as_ref()
            .and_then(|info| info.binding_origin)
            .map_or(LocalExport::Ephemeral, LocalExport::Exact);
        let issuer = active_state().map_or_else(std::sync::Weak::new, |state| {
            Arc::downgrade(&state.session_lease)
        });
        return BoundFunctionAuthority::local(
            issuer,
            FunctionDefinitionOwner::issue(owner, function),
            owners
                .iter()
                .filter(|(id, _)| *id != function)
                .map(|(id, owner)| FunctionDefinitionOwner::issue(owner, *id))
                .collect(),
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
    if let Some(state) = active_state() {
        return source_functions_in_catalog(
            state.call.borrow().source_functions.as_deref(),
            source_id,
        );
    }
    SOURCE_FUNCTION_CATALOG
        .with(|slot| source_functions_in_catalog(slot.borrow().as_deref(), source_id))
}

fn selected_execution_facts() -> Option<Rc<ExecutionSourceFacts>> {
    if let Some(state) = active_state() {
        return state.call.borrow().execution_source_resolver.clone();
    }
    EXECUTION_SOURCE_RESOLVER.with(|slot| slot.borrow().clone())
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
