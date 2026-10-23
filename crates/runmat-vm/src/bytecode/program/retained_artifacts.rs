use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use runmat_execution::executable::ExecutableStaticImport;
use runmat_execution::FunctionBindingOrigin;
use runmat_hir::FunctionId;
use runmat_runtime::user_functions::{DefinitionOwnerGraph, SourceFunctionInfo};
use runmat_value::FunctionDefinitionOwner;

use super::{function_remap, FunctionBytecode, FunctionRegistry};
use crate::program_execution::RuntimeBoundExecutableUnit;

#[path = "retained_artifacts/ids.rs"]
mod ids;
use ids::DestinationFunctionIds;
#[path = "retained_artifacts/fingerprints.rs"]
mod fingerprints;
#[path = "retained_artifacts/types.rs"]
mod types;
pub use types::{
    PreparedRetainedCallableUnits, RetainedCallableStageError, RetainedCallableStageLimits,
    SelectedFingerprintInvariant,
};
#[path = "retained_artifacts/class_declarations.rs"]
mod class_declarations;
#[path = "retained_artifacts/private_companions.rs"]
mod private_companions;
#[path = "retained_artifacts/work_limits.rs"]
mod work_limits;
use private_companions::PrivateCompanionGraphs;

pub(super) fn selected_runtime_origin_fingerprints(
    registry: &FunctionRegistry,
) -> Result<HashMap<FunctionBindingOrigin, String>, RetainedCallableStageError> {
    fingerprints::normalized_selected(registry)
}

pub(super) fn selected_runtime_body_fingerprint(
    registry: &FunctionRegistry,
    function: FunctionId,
) -> Result<(FunctionBindingOrigin, String), RetainedCallableStageError> {
    fingerprints::normalized_selected_body(registry, function)
}

pub(super) fn selected_runtime_body_fingerprints(
    registry: &FunctionRegistry,
    functions: &[FunctionId],
) -> Result<HashMap<FunctionId, (FunctionBindingOrigin, String)>, RetainedCallableStageError> {
    fingerprints::normalized_selected_bodies(registry, functions)
}

pub(super) fn selected_runtime_body_fingerprints_with_imports(
    registry: &FunctionRegistry,
    functions: &[FunctionId],
    imports: &[ExecutableStaticImport],
) -> Result<HashMap<FunctionId, (FunctionBindingOrigin, String)>, RetainedCallableStageError> {
    fingerprints::normalized_selected_bodies_with_imports(registry, functions, imports)
}

pub(super) fn selected_runtime_body_fingerprints_by_product(
    registry: &FunctionRegistry,
    inputs: &[(FunctionId, &[ExecutableStaticImport])],
) -> Result<HashMap<FunctionId, (FunctionBindingOrigin, String)>, RetainedCallableStageError> {
    fingerprints::normalized_selected_bodies_by_product(registry, inputs)
}

/// Prepare imported executable definitions as retained-only bodies. The
/// destination registry is cloned and changed only after every unit, origin,
/// source owner, graph, and remap has validated. Every unit must already be
/// bound into `destination_runtime`; its retained runtime then owns those
/// target-session leases. Numeric IDs from the artifact are never reused as
/// destination authority.
pub fn prepare_retained_callable_units(
    destination_runtime: &runmat_runtime::context::RuntimeContext,
    destination: &FunctionRegistry,
    units: Vec<RuntimeBoundExecutableUnit>,
    limits: RetainedCallableStageLimits,
) -> Result<PreparedRetainedCallableUnits, RetainedCallableStageError> {
    work_limits::validate(&units, limits)?;
    for unit in &units {
        unit.validate_runtime_owner(destination_runtime)
            .map_err(|_| RetainedCallableStageError::ForeignSourceOwner)?;
        let mut source_roots = HashMap::new();
        for (function, body) in &unit.registry().functions {
            let source = body
                .source_id
                .ok_or(RetainedCallableStageError::MissingRuntimeSource)?;
            source_roots.entry(source).or_insert(*function);
        }
        for function in source_roots.into_values() {
            if unit
                .registry()
                .definition_dependency_owners(function)
                .is_none()
            {
                return Err(RetainedCallableStageError::IncompleteDefinitionGraph);
            }
        }
    }
    let linked_constructorless_classes = class_declarations::constructorless_from_units(&units)?;

    let mut allocator = DestinationFunctionIds::new(destination);
    let mut origin_destinations = HashMap::<FunctionBindingOrigin, (FunctionId, String)>::new();
    let mut imported = HashMap::<FunctionId, FunctionBytecode>::new();
    let mut imported_unit = HashMap::<FunctionId, usize>::new();

    for (unit_index, unit) in units.iter().enumerate() {
        let fingerprints = fingerprints::normalized(unit.registry())?;
        let mut source_ids = unit
            .registry()
            .functions
            .keys()
            .copied()
            .collect::<Vec<_>>();
        source_ids.sort_by_key(|function| function.0);
        let mut local_origins = HashSet::new();
        let mut remap = HashMap::with_capacity(source_ids.len());
        for source_id in &source_ids {
            let body = &unit.registry().functions[source_id];
            let origin = body
                .binding_origin
                .ok_or(RetainedCallableStageError::MissingOrigin {
                    function: *source_id,
                })?;
            if !local_origins.insert(origin) {
                return Err(RetainedCallableStageError::DuplicateOrigin);
            }
            let destination_id = match origin_destinations.get(&origin) {
                Some((destination_id, fingerprint)) => {
                    if fingerprint != &fingerprints[&origin] {
                        return Err(RetainedCallableStageError::ConflictingOrigin);
                    }
                    *destination_id
                }
                None => {
                    let destination_id = allocator.allocate()?;
                    origin_destinations
                        .insert(origin, (destination_id, fingerprints[&origin].clone()));
                    destination_id
                }
            };
            remap.insert(*source_id, destination_id);
        }
        for source_id in source_ids {
            let body = &unit.registry().functions[&source_id];
            let origin = body
                .binding_origin
                .ok_or(RetainedCallableStageError::MissingOrigin {
                    function: source_id,
                })?;
            let destination_id = remap[&source_id];
            if imported.contains_key(&destination_id) {
                debug_assert_eq!(
                    origin_destinations
                        .get(&origin)
                        .map(|(function, _)| *function),
                    Some(destination_id)
                );
                continue;
            }
            let mut body = body.clone();
            if body.selected_class_product.is_some() {
                return Err(RetainedCallableStageError::IncompleteDefinitionGraph);
            }
            // Binding a cloned admitted unit may preserve the deserialized
            // owner Arc across two selections with different dependencies.
            // A new retained publication issues a new definition graph; its
            // class seal and escaped handles must not mutate the prior one.
            body.definition_owner = Arc::new(FunctionDefinitionOwner::default());
            function_remap::remap_closed_function(&mut body, &remap)
                .map_err(|_| RetainedCallableStageError::IncompleteDefinitionGraph)?;
            imported.insert(destination_id, body);
            imported_unit.insert(destination_id, unit_index);
        }
    }

    // The owner graph contains only the exact selected imported products.
    // Current named definitions in the destination are not dependency
    // authority for a retained callable.
    let graph_registry = FunctionRegistry::new(imported.clone());
    let mut companion_graphs = PrivateCompanionGraphs::new(
        &graph_registry,
        &imported_unit,
        &linked_constructorless_classes,
    )?;
    let mut destination_ids = imported.keys().copied().collect::<Vec<_>>();
    destination_ids.sort_by_key(|function| function.0);
    let mut origin_facts = Vec::with_capacity(destination_ids.len());
    let mut owner_graphs = HashMap::new();
    for destination_id in &destination_ids {
        let body = &imported[destination_id];
        let source = body
            .source_id
            .ok_or(RetainedCallableStageError::MissingRuntimeSource)?;
        let definition_owners = match owner_graphs.get(&source) {
            Some(owners) => Arc::clone(owners),
            None => {
                let owners = companion_graphs.graph_for(*destination_id)?;
                owner_graphs.insert(source, Arc::clone(&owners));
                owners
            }
        };
        let source_id = source
            .resolve_in(destination_runtime)
            .ok_or(RetainedCallableStageError::MissingRuntimeSource)?;
        origin_facts.push(SourceFunctionInfo {
            source_id,
            name: body.display_name.clone(),
            function: destination_id.0,
            role: body.source_role,
            binding_origin: body.binding_origin,
            source_file: runmat_runtime::source_context::source_info_in_context(
                destination_runtime,
                source_id,
            )
            .and_then(|source| source.fullpath_name.map(|path| path.to_string()))
            .or_else(|| body.source_file.clone()),
            definition_owners: Some(definition_owners),
        });
    }
    // One temporary lease per staged body is sufficient. Per-root owner
    // graphs above share these owners and must not turn a wide source group
    // into a quadratic number of live lease records.
    let staging_leases = destination_ids
        .iter()
        .map(|function| {
            let body = &imported[function];
            FunctionDefinitionOwner::issue(&body.definition_owner, function.0)
        })
        .collect();

    let mut registry = destination.clone();
    registry.insert_retained_definitions(imported.into_values());
    Ok(PreparedRetainedCallableUnits::new(
        registry,
        Arc::new(origin_facts),
        staging_leases,
        units,
        linked_constructorless_classes
            .into_iter()
            .map(|linked| linked.declaration)
            .collect(),
    ))
}

/// Canonical per-origin fingerprints with the admitted static-import sites.
pub(crate) fn admitted_callable_origin_fingerprints_with_imports(
    registry: &FunctionRegistry,
    source_map: &runmat_execution::ExecutableSourceMapV2,
    imports: &[ExecutableStaticImport],
) -> Result<HashMap<FunctionBindingOrigin, String>, RetainedCallableStageError> {
    fingerprints::normalized_admitted_with_imports(registry, source_map, imports)
}

pub(crate) fn admitted_semantic_body_digests(
    registry: &FunctionRegistry,
    source_map: &runmat_execution::ExecutableSourceMapV2,
    imports: &[ExecutableStaticImport],
) -> Result<
    HashMap<FunctionBindingOrigin, runmat_execution::executable::SelectedBodySemanticDigest>,
    RetainedCallableStageError,
> {
    fingerprints::admitted_semantic_body_digests(registry, source_map, imports)
}

fn owners_graph(owners: &[(FunctionId, Arc<FunctionDefinitionOwner>)]) -> DefinitionOwnerGraph {
    Arc::new(
        owners
            .iter()
            .map(|(function, owner)| (function.0, Arc::clone(owner)))
            .collect(),
    )
}

impl FunctionRegistry {
    /// Exact origins whose executable envelopes remain exportable. Published
    /// named definitions remain eligible; retired-only definitions require a
    /// live exact owner graph.
    pub fn retained_callable_origins(&self) -> HashSet<FunctionBindingOrigin> {
        self.functions
            .values()
            .filter_map(|body| body.binding_origin)
            .chain(
                self.retired
                    .values()
                    .flatten()
                    .filter(|body| body.definition_owner.has_handles())
                    .filter_map(|body| body.binding_origin),
            )
            .collect()
    }

    fn insert_retained_definitions(
        &mut self,
        definitions: impl IntoIterator<Item = FunctionBytecode>,
    ) {
        self.execution_stacks.take();
        self.prune_retired();
        for definition in definitions {
            self.retired
                .entry(definition.function)
                .or_default()
                .push(definition);
        }
    }
}

#[cfg(test)]
#[path = "retained_artifacts_tests.rs"]
mod tests;
