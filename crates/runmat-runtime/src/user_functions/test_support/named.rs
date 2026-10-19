use crate::context::RuntimeContext;
use runmat_value::{BoundFunctionBinding, FunctionDefinitionOwner};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub struct ExactNamedFixtures {
    _bound: super::super::BoundFunctionInvokerGuard,
    _resolver: super::super::FunctionResolverGuard,
    _catalog: super::super::SourceFunctionCatalogGuard,
    bindings: HashMap<String, BoundFunctionBinding>,
    owners: super::super::DefinitionOwnerGraph,
}

impl ExactNamedFixtures {
    pub fn binding(&self, name: &str) -> BoundFunctionBinding {
        self.bindings
            .get(name)
            .unwrap_or_else(|| panic!("unknown exact named fixture {name}"))
            .clone()
    }

    pub fn owner(&self, function: usize) -> Arc<FunctionDefinitionOwner> {
        self.owners
            .iter()
            .find_map(|(known, owner)| (*known == function).then(|| Arc::clone(owner)))
            .unwrap_or_else(|| panic!("unknown exact fixture function {function}"))
    }
}

/// Install name lookup and definition authority from one exact owner graph.
/// The returned bindings must not be replaced with independently issued
/// leases: name resolution and invocation intentionally select these owners.
pub fn install_exact_named_forwarders(
    runtime: &RuntimeContext,
    fixtures: &[(&str, usize)],
) -> Result<ExactNamedFixtures, &'static str> {
    if fixtures.is_empty() {
        return Err("exact named fixtures cannot be empty");
    }
    if fixtures.iter().any(|(name, _)| name.is_empty()) {
        return Err("exact named fixture names cannot be empty");
    }
    let mut names = HashSet::with_capacity(fixtures.len());
    let mut functions = HashSet::with_capacity(fixtures.len());
    if fixtures
        .iter()
        .any(|(name, function)| !names.insert(*name) || !functions.insert(*function))
    {
        return Err("exact named fixtures require unique names and functions");
    }

    let owners = Arc::new(
        fixtures
            .iter()
            .map(|(_, function)| (*function, Arc::new(FunctionDefinitionOwner::default())))
            .collect::<Vec<_>>(),
    );
    let catalog = Arc::new(
        fixtures
            .iter()
            .map(|(name, function)| super::super::SourceFunctionInfo {
                source_id: runmat_types::SourceId(*function),
                name: (*name).to_string(),
                function: *function,
                role: runmat_types::FunctionSourceRole::Unknown,
                binding_origin: None,
                source_file: None,
                definition_owners: Some(Arc::clone(&owners)),
            })
            .collect::<Vec<_>>(),
    );
    let resolved = Arc::new(
        fixtures
            .iter()
            .map(|(name, function)| ((*name).to_string(), *function))
            .collect::<HashMap<_, _>>(),
    );

    let entry = runtime.enter();
    let catalog_guard = super::super::install_source_function_catalog(Some(catalog))
        .expect("exact named fixture catalog projection");
    let resolver_guard =
        super::super::install_semantic_function_resolver(Some(Arc::new(move |name| {
            resolved.get(name).copied()
        })))
        .expect("exact named fixture resolver projection");
    let bindings = fixtures
        .iter()
        .map(|(name, function)| {
            (
                (*name).to_string(),
                super::super::bound_function_binding_for(*function),
            )
        })
        .collect::<HashMap<_, _>>();
    let session = runtime.session_lease();
    if bindings
        .values()
        .any(|binding| !binding.authority.admits_local(&session))
    {
        return Err("exact named fixtures could not issue selected-session authority");
    }
    let bound_guard = super::install_bound_forwarders(bindings.values().cloned().collect());
    drop(entry);

    Ok(ExactNamedFixtures {
        _bound: bound_guard,
        _resolver: resolver_guard,
        _catalog: catalog_guard,
        bindings,
        owners,
    })
}

#[cfg(test)]
#[path = "named/tests.rs"]
mod tests;
