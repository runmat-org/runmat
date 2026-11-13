//! Linked classes whose admitted source has no constructor body.

use std::collections::{BTreeMap, HashSet};

use runmat_runtime::context::RetainedClassDeclaration;
use runmat_types::{ClassIdentity, ClassMethodRole, MethodName};

use super::super::super_constructor_dependencies::same_class_registration;
use super::{RetainedCallableStageError, RuntimeBoundExecutableUnit};
use crate::bytecode::Instr;

/// Exact selected declaration unit and its declared method identities. The
/// retained owner graph uses this to keep method bodies from that unit, never
/// same-named bodies from another selected source.
pub(super) struct LinkedConstructorlessClass {
    pub declaration: RetainedClassDeclaration,
    pub unit_index: usize,
    pub methods: HashSet<MethodName>,
}

/// Project once from validated runtime-bound units. The full instruction
/// comparison rejects conflicting same-class declarations before any staged
/// registry or caller binding is published.
pub(super) fn constructorless_from_units(
    units: &[RuntimeBoundExecutableUnit],
) -> Result<Vec<LinkedConstructorlessClass>, RetainedCallableStageError> {
    let mut declarations = BTreeMap::<
        ClassIdentity,
        (
            usize,
            &Instr,
            RetainedClassDeclaration,
            Vec<runmat_types::MethodName>,
        ),
    >::new();
    for (unit_index, unit) in units.iter().enumerate() {
        let instructions = unit.bytecode().instructions.iter().chain(
            unit.registry()
                .functions
                .values()
                .flat_map(|body| body.instructions.iter()),
        );
        for instruction in instructions {
            let Instr::RegisterClass {
                name,
                super_class,
                methods,
                ..
            } = instruction
            else {
                continue;
            };
            let candidate = RetainedClassDeclaration {
                class: name.clone(),
                parent: super_class.clone(),
                constructor: None,
            };
            match declarations.entry(name.clone()) {
                std::collections::btree_map::Entry::Occupied(existing) => {
                    let (selected_unit, selected_instruction, ..) = existing.get();
                    if *selected_unit != unit_index
                        || !same_class_registration(selected_instruction, instruction)
                    {
                        return Err(RetainedCallableStageError::IncompleteDefinitionGraph);
                    }
                }
                std::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert((
                        unit_index,
                        instruction,
                        candidate,
                        methods.iter().map(|method| method.name.clone()).collect(),
                    ));
                }
            }
        }
    }
    let mut constructorless = Vec::new();
    for (unit_index, _, declaration, methods) in declarations.into_values() {
        let owners = units
            .iter()
            .flat_map(|unit| unit.registry().functions.values())
            .filter_map(|body| {
                body.class_method_owner.as_ref().filter(|owner| {
                    owner.declaring_class == declaration.class
                        && owner.role == ClassMethodRole::Constructor
                })
            })
            .collect::<Vec<_>>();
        if owners.is_empty() {
            constructorless.push(LinkedConstructorlessClass {
                declaration,
                unit_index,
                methods: methods.into_iter().collect(),
            });
        } else if owners.iter().any(|owner| !methods.contains(&owner.method)) {
            return Err(RetainedCallableStageError::IncompleteDefinitionGraph);
        }
    }
    Ok(constructorless)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, rc::Rc};

    use runmat_hir::FunctionId;

    use super::*;
    use crate::bytecode::{FunctionBytecode, FunctionRegistry};

    fn unit(
        runtime: &runmat_runtime::context::RuntimeContext,
        ids: &[usize],
    ) -> RuntimeBoundExecutableUnit {
        let class = ClassIdentity::new("ConstructorlessBase").expect("class");
        let bodies = ids
            .iter()
            .map(|id| {
                (
                    FunctionId(*id),
                    FunctionBytecode {
                        function: FunctionId(*id),
                        instructions: vec![Instr::RegisterClass {
                            name: class.clone(),
                            super_class: None,
                            is_sealed: false,
                            is_abstract: false,
                            properties: Vec::new(),
                            methods: Vec::new(),
                            events: Vec::new(),
                            enumerations: Vec::new(),
                        }],
                        ..Default::default()
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        RuntimeBoundExecutableUnit::for_test(FunctionRegistry::new(bodies), runtime.clone())
    }

    #[test]
    fn identical_repeated_declarations_coalesce_only_within_one_admitted_unit() {
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let first = unit(&runtime, &[1, 2]);
        assert_eq!(
            constructorless_from_units(&[first])
                .expect("one admitted product")
                .len(),
            1
        );
        let first = unit(&runtime, &[1]);
        let second = unit(&runtime, &[2]);
        assert!(constructorless_from_units(&[first, second]).is_err());
    }
}
