//! Bounded retained owner graphs for an admitted private-source partition.
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use runmat_hir::FunctionId;
use runmat_runtime::user_functions::DefinitionOwnerGraph;

use super::super::{function_remap, BytecodeSourceRef, FunctionRegistry, PrivateSourceAlias};
use super::{owners_graph, RetainedCallableStageError};

type Partition = (usize, PrivateSourceAlias);

pub(super) struct PrivateCompanionGraphs<'a> {
    registry: &'a FunctionRegistry,
    imported_unit: &'a HashMap<FunctionId, usize>,
    by_partition: HashMap<Partition, Vec<FunctionId>>,
    by_source: HashMap<(usize, BytecodeSourceRef), Vec<FunctionId>>,
    cached_partition: HashMap<Partition, DefinitionOwnerGraph>,
}

impl<'a> PrivateCompanionGraphs<'a> {
    pub(super) fn new(
        registry: &'a FunctionRegistry,
        imported_unit: &'a HashMap<FunctionId, usize>,
    ) -> Self {
        let mut by_partition = HashMap::<Partition, Vec<FunctionId>>::new();
        let mut by_source = HashMap::<(usize, BytecodeSourceRef), Vec<FunctionId>>::new();
        for (id, body) in &registry.functions {
            let unit = imported_unit[id];
            if let Some(alias) = body
                .private_source_owner
                .as_ref()
                .and_then(|owner| owner.aliases().first())
            {
                // Admission validated every alias partition as equal/disjoint.
                // Its first sorted alias is a canonical key, not a filename.
                // Retain all admitted bodies in that owner partition so its
                // callers share one flat owner graph instead of N copies of
                // N private siblings. This can extend a same-owner primary
                // body's lifetime, but does not publish its ordinary name;
                // admission work limits still bound the graph.
                by_partition
                    .entry((unit, alias.clone()))
                    .or_default()
                    .push(*id);
            }
            if let Some(source) = body.source_id {
                by_source.entry((unit, source)).or_default().push(*id);
            }
        }
        for ids in by_partition.values_mut().chain(by_source.values_mut()) {
            ids.sort_by_key(|id| id.0);
        }
        Self {
            registry,
            imported_unit,
            by_partition,
            by_source,
            cached_partition: HashMap::new(),
        }
    }

    fn partition_for(&self, id: FunctionId) -> Option<Partition> {
        let unit = *self.imported_unit.get(&id)?;
        let alias = self
            .registry
            .get_current(id)?
            .private_source_owner
            .as_ref()?
            .aliases()
            .first()?
            .clone();
        Some((unit, alias))
    }

    pub(super) fn graph_for(
        &mut self,
        root: FunctionId,
    ) -> Result<DefinitionOwnerGraph, RetainedCallableStageError> {
        let partition = self.partition_for(root);
        if let Some(cached) = partition
            .as_ref()
            .and_then(|key| self.cached_partition.get(key))
        {
            return Ok(Arc::clone(cached));
        }
        let mut pending = partition
            .as_ref()
            .and_then(|key| self.by_partition.get(key))
            .cloned()
            .unwrap_or_else(|| vec![root]);
        let mut visited = HashSet::new();
        let mut expanded_partitions = HashSet::new();
        if let Some(key) = partition.as_ref() {
            expanded_partitions.insert(key.clone());
        }
        let mut expanded_sources = HashSet::new();
        let mut owners = Vec::new();
        while let Some(id) = pending.pop() {
            if !visited.insert(id) {
                continue;
            }
            let body = self
                .registry
                .get_current(id)
                .ok_or(RetainedCallableStageError::IncompleteDefinitionGraph)?;
            let unit = *self
                .imported_unit
                .get(&id)
                .ok_or(RetainedCallableStageError::IncompleteDefinitionGraph)?;
            owners.push((id, Arc::clone(&body.definition_owner)));
            if let Some(source) = body.source_id {
                if expanded_sources.insert((unit, source)) {
                    pending.extend(
                        self.by_source
                            .get(&(unit, source))
                            .into_iter()
                            .flatten()
                            .copied(),
                    );
                }
            }
            if let Some(key) = self.partition_for(id) {
                if expanded_partitions.insert(key.clone()) {
                    pending.extend(self.by_partition.get(&key).into_iter().flatten().copied());
                }
            }
            for instruction in &body.instructions {
                function_remap::visit_instruction(instruction, |dependency| {
                    pending.push(dependency)
                });
            }
        }
        owners.sort_by_key(|(id, _)| id.0);
        let graph = owners_graph(&owners);
        if let Some(key) = partition {
            self.cached_partition.insert(key, Arc::clone(&graph));
        }
        Ok(graph)
    }
}
