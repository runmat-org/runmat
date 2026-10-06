use std::collections::HashSet;
use std::sync::Arc;

use runmat_hir::FunctionId;
use runmat_value::FunctionDefinitionOwner;

use super::{function_remap, FunctionRegistry};

pub(super) fn owners(
    registry: &FunctionRegistry,
    function: FunctionId,
) -> Option<Vec<(FunctionId, Arc<FunctionDefinitionOwner>)>> {
    let mut visited = HashSet::new();
    let mut owners = Vec::new();
    let mut pending = Vec::new();
    // A handle to a function file also owns its local definitions. Calls to
    // localfunctions() can issue those handles after the file is retired,
    // even when no static call edge references a particular local.
    if let Some(source) = registry.get(function)?.source_id {
        pending.extend(registry.functions_for_source_ref(source).iter().copied());
    }
    pending.push(function);
    while let Some(function) = pending.pop() {
        if !visited.insert(function) {
            continue;
        }
        let body = registry.get(function)?;
        owners.push((function, Arc::clone(&body.definition_owner)));
        for instruction in &body.instructions {
            function_remap::visit_instruction(instruction, |dependency| pending.push(dependency));
        }
    }
    Some(owners)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::{FunctionBytecode, Instr};
    use std::collections::HashMap;

    #[test]
    fn static_helper_and_bound_closure_are_retained_transitively() {
        let first = FunctionBytecode {
            function: FunctionId(1),
            instructions: vec![Instr::CallSemanticFunctionMulti(FunctionId(2), 0, 1)],
            ..Default::default()
        };
        let second = FunctionBytecode {
            function: FunctionId(2),
            instructions: vec![Instr::CreateSemanticClosure(
                FunctionId(3),
                "nested".into(),
                0,
                runmat_types::FunctionSourceRole::Anonymous,
            )],
            ..Default::default()
        };
        let third = FunctionBytecode {
            function: FunctionId(3),
            ..Default::default()
        };
        let registry = FunctionRegistry::new(HashMap::from([
            (FunctionId(1), first),
            (FunctionId(2), second),
            (FunctionId(3), third),
        ]));
        assert_eq!(
            registry
                .definition_dependency_owners(FunctionId(1))
                .unwrap()
                .len(),
            3
        );
    }

    #[test]
    fn unresolved_static_callee_cannot_issue_incomplete_bundle() {
        let caller = FunctionBytecode {
            function: FunctionId(1),
            instructions: vec![Instr::CallSemanticFunctionMulti(FunctionId(2), 0, 1)],
            ..Default::default()
        };
        let registry = FunctionRegistry::new(HashMap::from([(FunctionId(1), caller)]));
        assert!(registry
            .definition_dependency_owners(FunctionId(1))
            .is_none());
    }

    #[test]
    fn long_dependency_chain_and_cycle_are_bounded_by_the_worklist() {
        const COUNT: usize = 4096;
        let functions = (0..COUNT)
            .map(|index| {
                let dependency = (index + 1) % COUNT;
                (
                    FunctionId(index),
                    FunctionBytecode {
                        function: FunctionId(index),
                        instructions: vec![Instr::CallSemanticFunctionMulti(
                            FunctionId(dependency),
                            0,
                            1,
                        )],
                        ..Default::default()
                    },
                )
            })
            .collect();
        let registry = FunctionRegistry::new(functions);
        assert_eq!(
            registry
                .definition_dependency_owners(FunctionId(0))
                .unwrap()
                .len(),
            COUNT
        );
    }
}
