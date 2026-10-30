//! Frozen original ownership for foreign semantic functions, shared without recompilation.
use super::ExecutableUnit;
use runmat_types::FunctionId;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    rc::Rc,
};

mod source_projection;

impl ExecutableUnit {
    pub(crate) fn owns_callable(&self, function: FunctionId) -> bool {
        owns_callable_parts(self.mir(), self.analysis(), self.functions(), function)
    }

    pub(crate) fn original_dependencies(&self) -> &BTreeMap<FunctionId, Rc<Self>> {
        &self.original_dependencies
    }

    pub(crate) fn foreign_functions(&self) -> Result<Vec<FunctionId>, String> {
        Ok(self
            .functions()
            .functions
            .keys()
            .copied()
            .filter(|id| !self.owns_callable(*id))
            .collect())
    }

    pub(crate) fn bind_original_dependencies(
        mut self,
        owners: &HashMap<FunctionId, Rc<Self>>,
    ) -> Result<Self, String> {
        self.reset_direct_code_cache();
        for function in self.foreign_functions()? {
            let owner = owners.get(&function).ok_or_else(|| {
                format!(
                    "semantic function {} has no original compiler owner",
                    function.0
                )
            })?;
            let projection = self
                .functions()
                .get(function)
                .ok_or("foreign function projection is absent")?;
            original_projection(self.source_map(), owner, projection)?;
            self.original_dependencies
                .insert(function, Rc::clone(owner));
        }
        Ok(self)
    }

    pub(super) fn portable_dependency_products(
        &self,
    ) -> Result<Vec<runmat_execution::ExecutableDependencyProduct>, String> {
        let mut visited = HashSet::new();
        let mut pending = self
            .original_dependencies
            .values()
            .cloned()
            .collect::<Vec<_>>();
        let mut products = BTreeMap::new();
        let mut encoded_bytes = 0_u64;
        while let Some(owner) = pending.pop() {
            if !visited.insert(Rc::as_ptr(&owner)) {
                continue;
            }
            if visited.len() > runmat_execution::EXECUTABLE_DEPENDENCY_MAX_PRODUCTS {
                return Err("original dependency inventory exceeds product limit".into());
            }
            let selected = owner
                .analysis()
                .functions
                .iter()
                .filter_map(|f| match &f.callable.identity {
                    Some(runmat_types::CallableIdentity::BoundFunction(id)) => Some(*id),
                    _ => None,
                })
                .find(|id| owner.owns_callable(*id))
                .map(|id| {
                    (
                        owner
                            .functions()
                            .get(id)
                            .expect("owned function exists")
                            .display_name
                            .clone(),
                        id,
                    )
                });
            let local = owner.portable_local_envelope_for_selection(selected)?;
            let product = runmat_execution::ExecutableDependencyProduct {
                manifest: local.manifest,
                payloads: local.payloads,
            };
            let descriptor = product.descriptor().map_err(|e| e.to_string())?;
            if let std::collections::btree_map::Entry::Vacant(entry) =
                products.entry(descriptor.identity)
            {
                encoded_bytes = encoded_bytes
                    .checked_add(descriptor.encoded_length)
                    .filter(|bytes| {
                        *bytes <= runmat_execution::EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES as u64
                    })
                    .ok_or("original dependencies exceed envelope budget")?;
                entry.insert(product);
                pending.extend(owner.original_dependencies.values().cloned());
            }
        }
        Ok(products.into_values().collect())
    }
}

/// Validate a compiler projection against its exact original admitted body.
/// The returned body carries the original nonwire compiler evidence, not any
/// authority reconstructed from the projection's serialized fields.
pub(super) fn original_projection(
    selected_map: &crate::ExecutableSourceMap,
    owner: &ExecutableUnit,
    projection: &runmat_vm::FunctionBytecode,
) -> Result<runmat_vm::FunctionBytecode, String> {
    let origin = projection
        .binding_origin
        .ok_or("foreign function projection has no original binding origin")?;
    let function = owner
        .local_function_for_origin(origin)?
        .ok_or("dependency is not the original function owner")?;
    if function != projection.function {
        return Err("foreign function differs from its original compiler coordinate".into());
    }
    let mut original = owner
        .functions()
        .get(function)
        .ok_or("original function is absent")?
        .clone();
    source_projection::normalize_original(
        selected_map,
        owner.source_map(),
        projection,
        &mut original,
    )?;
    // Coverage sites are scoped to the compiling caller. All executable
    // instructions, source identities, spans and ABI remain exact.
    original.coverage_sites = projection.coverage_sites.clone();
    let original_wire = serde_json::to_value(&original).map_err(|e| e.to_string())?;
    let projected_wire = serde_json::to_value(projection).map_err(|e| e.to_string())?;
    if projected_wire != original_wire {
        let changed = projected_wire
            .as_object()
            .ok_or("invalid function projection")?
            .iter()
            .filter_map(|(field, value)| {
                (original_wire.get(field) != Some(value)).then_some(field.as_str())
            })
            .collect::<Vec<_>>();
        return Err(format!(
            "foreign function {} differs from its original owner in {changed:?}",
            function.0
        ));
    }
    Ok(original)
}

pub(super) fn owns_callable_parts(
    mir: &runmat_mir::MirAssembly,
    analysis: &runmat_mir::analysis::AnalysisStore,
    functions: &runmat_vm::FunctionRegistry,
    function: FunctionId,
) -> bool {
    let id = function;
    let Some(metadata) = mir.functions.get(&id) else {
        return false;
    };
    let Some(bytecode) = functions.get(id) else {
        return false;
    };
    !matches!(metadata.kind, runmat_hir::FunctionKind::SyntheticEntrypoint)
        && bytecode.display_name == metadata.name.0
        && bytecode
            .source_id
            .and_then(|source| source.program())
            .is_some_and(|source| source.0 == metadata.source.0)
        && analysis
            .function_for_local(function)
            .is_some_and(|facts| facts.function == bytecode.portable_function)
}
