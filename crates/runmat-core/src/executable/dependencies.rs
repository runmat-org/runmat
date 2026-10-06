//! Frozen original ownership for foreign semantic functions, shared without recompilation.
use super::ExecutableUnit;
use runmat_types::ProgramFunctionId;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    rc::Rc,
};

impl ExecutableUnit {
    pub(crate) fn owns_callable(&self, function: ProgramFunctionId) -> bool {
        owns_callable_parts(self.mir(), self.analysis(), self.functions(), function)
    }

    pub(crate) fn original_dependencies(&self) -> &BTreeMap<ProgramFunctionId, Rc<Self>> {
        &self.original_dependencies
    }

    pub(crate) fn foreign_functions(&self) -> Result<Vec<ProgramFunctionId>, String> {
        self.functions()
            .functions
            .keys()
            .map(|id| {
                u32::try_from(id.0)
                    .map(ProgramFunctionId)
                    .map_err(|_| "function identity exceeds portable schema".to_string())
            })
            .filter(|id| id.as_ref().map_or(true, |id| !self.owns_callable(*id)))
            .collect()
    }

    pub(crate) fn bind_original_dependencies(
        mut self,
        owners: &HashMap<ProgramFunctionId, Rc<Self>>,
    ) -> Result<Self, String> {
        for function in self.foreign_functions()? {
            let owner = owners.get(&function).ok_or_else(|| {
                format!(
                    "semantic function {} has no original compiler owner",
                    function.0
                )
            })?;
            if !owner.owns_callable(function) {
                return Err("dependency is not the original function owner".into());
            }
            let id = runmat_hir::FunctionId(function.0 as usize);
            let mut projection = self
                .functions()
                .get(id)
                .ok_or("foreign function projection is absent")?
                .clone();
            let mut original = owner
                .functions()
                .get(id)
                .ok_or("original function is absent")?
                .clone();
            // Coverage sites are scoped to the compiling caller. All executable
            // instructions, source identities, spans and ABI remain exact.
            projection.coverage_sites.clear();
            original.coverage_sites.clear();
            let projection = serde_json::to_value(&projection).map_err(|e| e.to_string())?;
            let original = serde_json::to_value(&original).map_err(|e| e.to_string())?;
            if projection != original {
                let changed = projection
                    .as_object()
                    .ok_or("invalid function projection")?
                    .iter()
                    .filter_map(|(field, value)| {
                        (original.get(field) != Some(value)).then_some(field.as_str())
                    })
                    .collect::<Vec<_>>();
                return Err(format!(
                    "foreign function {} differs from its original owner in {changed:?}",
                    function.0
                ));
            }
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
                .find(|f| owner.owns_callable(f.function))
                .map(|f| {
                    let id = runmat_hir::FunctionId(f.function.0 as usize);
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

pub(super) fn owns_callable_parts(
    mir: &runmat_mir::MirAssembly,
    analysis: &runmat_mir::analysis::AnalysisStore,
    functions: &runmat_vm::FunctionRegistry,
    function: ProgramFunctionId,
) -> bool {
    let id = runmat_hir::FunctionId(function.0 as usize);
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
            .is_some_and(|source| source.0 == metadata.source.0 as usize)
        && analysis
            .functions
            .iter()
            .any(|facts| facts.function == function)
}
