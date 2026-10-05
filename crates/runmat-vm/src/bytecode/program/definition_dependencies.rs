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
    collect(registry, function, &mut visited, &mut owners)?;
    Some(owners)
}

fn collect(
    registry: &FunctionRegistry,
    function: FunctionId,
    visited: &mut HashSet<FunctionId>,
    owners: &mut Vec<(FunctionId, Arc<FunctionDefinitionOwner>)>,
) -> Option<()> {
    if !visited.insert(function) {
        return Some(());
    }
    let body = registry.get(function)?;
    owners.push((function, Arc::clone(&body.definition_owner)));
    for instruction in &body.instructions {
        for dependency in static_dependencies(instruction) {
            collect(registry, dependency, visited, owners)?;
        }
    }
    Some(())
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

fn subscript_dependencies(steps: &[BytecodeSubscriptStep], output: &mut Vec<FunctionId>) {
    for step in steps {
        if let BytecodeSubscriptStep::DottedInvoke {
            semantic_fallback: Some(identity),
            ..
        } = step
        {
            output.extend(identity_dependency(identity));
        }
    }
}

fn static_dependencies(instruction: &Instr) -> Vec<FunctionId> {
    let mut output = Vec::new();
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
            output.push(*function);
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
            output.extend(identity_dependency(identity));
        }
        Instr::ReadSubscriptPath { steps, .. } | Instr::CaptureSubscriptPath { steps, .. } => {
            subscript_dependencies(steps, &mut output);
        }
        Instr::BeginSubscriptEndReceiver { prefix } => {
            subscript_dependencies(prefix, &mut output);
        }
        _ => {}
    }
    output
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
}
