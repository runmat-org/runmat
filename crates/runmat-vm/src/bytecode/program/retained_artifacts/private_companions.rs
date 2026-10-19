//! Bounded retained owner graphs for an admitted private-source partition.
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use runmat_hir::FunctionId;
use runmat_runtime::user_functions::DefinitionOwnerGraph;
use runmat_types::{ClassIdentity, ClassMethodRole};

use super::super::{
    function_remap, super_constructor_dependencies, BytecodeSourceRef, FunctionRegistry,
    PrivateSourceAlias,
};
use super::{
    class_declarations::LinkedConstructorlessClass, owners_graph, RetainedCallableStageError,
};

type Partition = (usize, PrivateSourceAlias);

pub(super) struct PrivateCompanionGraphs<'a> {
    registry: &'a FunctionRegistry,
    imported_unit: &'a HashMap<FunctionId, usize>,
    by_partition: HashMap<Partition, Vec<FunctionId>>,
    by_source: HashMap<(usize, BytecodeSourceRef), Vec<FunctionId>>,
    linked_methods: HashMap<ClassIdentity, Vec<FunctionId>>,
    cached_partition: HashMap<Partition, DefinitionOwnerGraph>,
    super_constructors: super_constructor_dependencies::SuperConstructorIndex<'a>,
    linked_constructorless_classes: HashSet<ClassIdentity>,
}

impl<'a> PrivateCompanionGraphs<'a> {
    pub(super) fn new(
        registry: &'a FunctionRegistry,
        imported_unit: &'a HashMap<FunctionId, usize>,
        linked_constructorless_classes: &[LinkedConstructorlessClass],
    ) -> Result<Self, RetainedCallableStageError> {
        let mut by_partition = HashMap::<Partition, Vec<FunctionId>>::new();
        let mut by_source = HashMap::<(usize, BytecodeSourceRef), Vec<FunctionId>>::new();
        let selected_classes = linked_constructorless_classes
            .iter()
            .map(|selected| (selected.declaration.class.clone(), selected))
            .collect::<HashMap<_, _>>();
        if selected_classes.len() != linked_constructorless_classes.len() {
            return Err(RetainedCallableStageError::IncompleteDefinitionGraph);
        }
        let mut linked_methods = HashMap::<ClassIdentity, Vec<FunctionId>>::new();
        for (id, body) in &registry.functions {
            let unit = imported_unit[id];
            if let Some(owner) = body.class_method_owner.as_ref() {
                if let Some(selected) = selected_classes.get(&owner.declaring_class) {
                    if unit != selected.unit_index
                        || owner.role != ClassMethodRole::Ordinary
                        || !selected.methods.contains(&owner.method)
                    {
                        return Err(RetainedCallableStageError::IncompleteDefinitionGraph);
                    }
                    linked_methods
                        .entry(owner.declaring_class.clone())
                        .or_default()
                        .push(*id);
                }
            }
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
        for ids in linked_methods.values_mut() {
            ids.sort_by_key(|id| id.0);
        }
        Ok(Self {
            registry,
            imported_unit,
            by_partition,
            by_source,
            linked_methods,
            cached_partition: HashMap::new(),
            super_constructors: super_constructor_dependencies::SuperConstructorIndex::new(
                registry,
            ),
            linked_constructorless_classes: selected_classes.into_keys().collect(),
        })
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
                if let Some((requesting_class, super_class)) =
                    super_constructor_dependencies::instruction_classes(instruction)
                {
                    if body.class_method_owner.as_ref().is_none_or(|owner| {
                        owner.is_static || &owner.declaring_class != requesting_class
                    }) {
                        return Err(RetainedCallableStageError::IncompleteDefinitionGraph);
                    }
                    if let Some(target) = self
                        .super_constructors
                        .target(super_class, body.binding_origin)
                        .map_err(|_| RetainedCallableStageError::IncompleteDefinitionGraph)?
                    {
                        pending.push(target);
                    } else if let Some(declaration) = self
                        .super_constructors
                        .declaration(super_class, body.binding_origin)
                        .map_err(|_| RetainedCallableStageError::IncompleteDefinitionGraph)?
                    {
                        pending.push(declaration);
                    } else if !self.linked_constructorless_classes.contains(super_class)
                        && !runmat_runtime::class_registry::has_intrinsic_implicit_constructor(
                            super_class,
                        )
                    {
                        return Err(RetainedCallableStageError::IncompleteDefinitionGraph);
                    }
                    // A selected declaration-only parent can have a separate
                    // exact declaration target *and* ordinary method bodies.
                    // Retain those checked bodies even when the implicit
                    // constructor resolver chose its declaration above.
                    if self.linked_constructorless_classes.contains(super_class) {
                        pending.extend(
                            self.linked_methods
                                .get(super_class)
                                .into_iter()
                                .flatten()
                                .copied(),
                        );
                    }
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::{FunctionBytecode, Instr};
    use runmat_types::{ClassIdentity, ClassMethodOwner, ClassMethodRole, MethodName};

    fn constructor(class: &str, function: usize) -> FunctionBytecode {
        FunctionBytecode {
            function: FunctionId(function),
            class_method_owner: Some(ClassMethodOwner {
                declaring_class: ClassIdentity::new(class).expect("class"),
                method: MethodName::from(class),
                is_static: false,
                role: ClassMethodRole::Constructor,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn staged_derived_graph_retains_companion_base_constructor() {
        let derived = FunctionBytecode {
            instructions: vec![Instr::CallSuperConstructorMulti {
                current_class: ClassIdentity::new("Derived").expect("derived"),
                super_class: ClassIdentity::new("Base").expect("base"),
                arg_count: 0,
                out_count: 1,
            }],
            ..constructor("Derived", 1)
        };
        let registry = FunctionRegistry::new(HashMap::from([
            (FunctionId(1), derived),
            (FunctionId(2), constructor("Base", 2)),
        ]));
        let units = HashMap::from([(FunctionId(1), 0), (FunctionId(2), 0)]);
        let graph = PrivateCompanionGraphs::new(&registry, &units, &[])
            .expect("selected class index")
            .graph_for(FunctionId(1))
            .expect("complete typed graph");
        assert_eq!(graph.len(), 2);
        assert!(graph.iter().any(|(id, _)| *id == 2));
    }

    #[test]
    fn staged_methodless_script_parent_requires_selected_declaration() {
        let base = ClassIdentity::new("MethodlessBase").expect("base");
        let derived = FunctionBytecode {
            instructions: vec![Instr::CallSuperConstructorMulti {
                current_class: ClassIdentity::new("Derived").expect("derived"),
                super_class: base.clone(),
                arg_count: 0,
                out_count: 1,
            }],
            ..constructor("Derived", 1)
        };
        let registry = FunctionRegistry::new(HashMap::from([(FunctionId(1), derived)]));
        let units = HashMap::from([(FunctionId(1), 0)]);
        assert!(PrivateCompanionGraphs::new(&registry, &units, &[])
            .expect("empty selected class index")
            .graph_for(FunctionId(1))
            .is_err());
        let selected = [LinkedConstructorlessClass {
            declaration: runmat_runtime::context::RetainedClassDeclaration {
                class: base,
                parent: None,
                constructor: None,
            },
            unit_index: 1,
            methods: HashSet::new(),
        }];
        let graph = PrivateCompanionGraphs::new(&registry, &units, &selected)
            .expect("selected class index")
            .graph_for(FunctionId(1))
            .expect("methodless source declaration is selected");
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn selected_constructorless_parent_methods_join_exact_handle_graph() {
        let base = ClassIdentity::new("Base").expect("base");
        let derived = FunctionBytecode {
            instructions: vec![Instr::CallSuperConstructorMulti {
                current_class: ClassIdentity::new("Derived").expect("derived"),
                super_class: base.clone(),
                arg_count: 0,
                out_count: 1,
            }],
            ..constructor("Derived", 1)
        };
        let method = |id, name: &str| FunctionBytecode {
            function: FunctionId(id),
            class_method_owner: Some(ClassMethodOwner {
                declaring_class: base.clone(),
                method: MethodName::from(name),
                is_static: false,
                role: ClassMethodRole::Ordinary,
            }),
            ..Default::default()
        };
        let declaration = FunctionBytecode {
            function: FunctionId(4),
            instructions: vec![Instr::RegisterClass {
                name: base.clone(),
                super_class: None,
                is_sealed: false,
                is_abstract: false,
                properties: Vec::new(),
                methods: Vec::new(),
                events: Vec::new(),
                enumerations: Vec::new(),
            }],
            ..Default::default()
        };
        let registry = FunctionRegistry::new(HashMap::from([
            (FunctionId(1), derived),
            (FunctionId(2), method(2, "marker")),
            (FunctionId(3), method(3, "secondMarker")),
            (FunctionId(4), declaration),
        ]));
        let units = HashMap::from([
            (FunctionId(1), 0),
            (FunctionId(2), 1),
            (FunctionId(3), 1),
            (FunctionId(4), 1),
        ]);
        let selected = [LinkedConstructorlessClass {
            declaration: runmat_runtime::context::RetainedClassDeclaration {
                class: base.clone(),
                parent: None,
                constructor: None,
            },
            unit_index: 1,
            methods: HashSet::from([MethodName::from("marker"), MethodName::from("secondMarker")]),
        }];
        let mut graphs = PrivateCompanionGraphs::new(&registry, &units, &selected)
            .expect("exact selected class methods");
        let graph = graphs.graph_for(FunctionId(1)).expect("complete graph");
        assert_eq!(graph.len(), 4);
        assert!(graph.iter().any(|(id, _)| *id == 2));
        assert!(graph.iter().any(|(id, _)| *id == 3));
        assert!(graph.iter().any(|(id, _)| *id == 4));

        let wrong_unit = HashMap::from([
            (FunctionId(1), 0),
            (FunctionId(2), 2),
            (FunctionId(3), 1),
            (FunctionId(4), 1),
        ]);
        assert!(PrivateCompanionGraphs::new(&registry, &wrong_unit, &selected).is_err());
    }
}
