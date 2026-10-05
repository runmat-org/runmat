use std::collections::HashSet;
use std::sync::Arc;

use runmat_hir::FunctionId;
use runmat_types::CallableIdentity;
use runmat_value::FunctionDefinitionOwner;

use crate::bytecode::{BytecodeSubscriptStep, Instr};

use super::FunctionRegistry;

pub(super) fn owners(
    registry: &FunctionRegistry,
    function: FunctionId,
) -> Option<Vec<(FunctionId, Arc<FunctionDefinitionOwner>)>> {
    let mut visited = HashSet::new();
    let mut owners = Vec::new();
    let mut pending = vec![function];
    while let Some(function) = pending.pop() {
        if !visited.insert(function) {
            continue;
        }
        let body = registry.get(function)?;
        owners.push((function, Arc::clone(&body.definition_owner)));
        for instruction in &body.instructions {
            visit_static_dependencies(instruction, |dependency| pending.push(dependency));
        }
    }
    Some(owners)
}

fn identity_dependency(identity: &CallableIdentity) -> Option<FunctionId> {
    match identity {
        CallableIdentity::BoundFunction(function)
        | CallableIdentity::AnonymousFunction(function)
        | CallableIdentity::ExternalFunction { function, .. } => Some(*function),
        CallableIdentity::Builtin(_)
        | CallableIdentity::Imported(_)
        | CallableIdentity::Method(_)
        | CallableIdentity::DynamicName(_)
        | CallableIdentity::ExternalName(_) => None,
    }
}

fn visit_subscript_dependencies(
    steps: &[BytecodeSubscriptStep],
    mut visit: impl FnMut(FunctionId),
) {
    for step in steps {
        if let BytecodeSubscriptStep::DottedInvoke {
            semantic_fallback: Some(identity),
            ..
        } = step
        {
            if let Some(dependency) = identity_dependency(identity) {
                visit(dependency);
            }
        }
    }
}

fn visit_static_dependencies(instruction: &Instr, mut visit: impl FnMut(FunctionId)) {
    match instruction {
        Instr::CreateSemanticClosure(function, ..)
        | Instr::CreateBoundFunctionHandle(function, ..)
        | Instr::CreateExternalBoundFunctionHandle(function, ..)
        | Instr::CreateSemanticFuture(function, ..)
        | Instr::CreateSemanticFutureExpandMultiOutput(function, ..)
        | Instr::CallSemanticFunctionMulti(function, ..)
        | Instr::CallSemanticFunctionMultiUsingOutputSlot(function, ..)
        | Instr::CallSemanticFunctionExpandMultiOutput(function, ..)
        | Instr::CallSemanticNestedFunctionMulti { function, .. }
        | Instr::CallSemanticNestedFunctionMultiUsingOutputSlot { function, .. }
        | Instr::CallSemanticNestedFunctionExpandMultiOutput { function, .. } => {
            visit(*function);
        }
        Instr::CallFunctionMulti { identity, .. }
        | Instr::CallFunctionMultiUsingOutputSlot { identity, .. }
        | Instr::CallFunctionExpandMultiOutput { identity, .. }
        | Instr::CallWorkspaceFirstMulti { identity, .. }
        | Instr::CallWorkspaceFirstMultiUsingOutputSlot { identity, .. }
        | Instr::CallWorkspaceFirstExpandMultiOutput { identity, .. }
        | Instr::CallWorkspaceFirstExpandMultiOutputUsingOutputSlot { identity, .. }
        | Instr::CallMethodOrMemberIndexMulti { identity, .. }
        | Instr::CallMethodOrMemberIndexExpandMultiOutput { identity, .. } => {
            if let Some(dependency) = identity_dependency(identity) {
                visit(dependency);
            }
        }
        Instr::ReadSubscriptPath { steps, .. } | Instr::CaptureSubscriptPath { steps, .. } => {
            visit_subscript_dependencies(steps, &mut visit);
        }
        Instr::BeginSubscriptEndReceiver { prefix } => {
            visit_subscript_dependencies(prefix, &mut visit);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::FunctionBytecode;
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
