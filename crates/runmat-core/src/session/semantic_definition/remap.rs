use runmat_hir::FunctionId;
use runmat_types::CallableIdentity;
use runmat_vm::{BytecodeSubscriptStep, Instr};
use std::collections::HashMap;

fn remap_callable(identity: &mut CallableIdentity, remap: &HashMap<FunctionId, FunctionId>) {
    let function = match identity {
        CallableIdentity::BoundFunction(function)
        | CallableIdentity::AnonymousFunction(function)
        | CallableIdentity::ExternalFunction { function, .. } => function,
        CallableIdentity::Builtin(_)
        | CallableIdentity::Imported(_)
        | CallableIdentity::Method(_)
        | CallableIdentity::DynamicName(_)
        | CallableIdentity::ExternalName(_) => return,
    };
    if let Some(mapped) = remap.get(function) {
        *function = *mapped;
    }
}

fn remap_subscript_steps(
    steps: &mut [BytecodeSubscriptStep],
    remap: &HashMap<FunctionId, FunctionId>,
) {
    for step in steps {
        if let BytecodeSubscriptStep::DottedInvoke {
            semantic_fallback: Some(identity),
            ..
        } = step
        {
            remap_callable(identity, remap);
        }
    }
}

/// Rebind every function identity carried by executable bytecode, including
/// callable identities nested in subscript-path instructions.
pub(in crate::session) fn remap_semantic_function_instr(
    instr: &mut Instr,
    remap: &HashMap<FunctionId, FunctionId>,
) {
    match instr {
        Instr::CreateSemanticClosure(function, _, _, _)
        | Instr::CreateBoundFunctionHandle(function, _, _)
        | Instr::CreateSemanticFuture(function, _, _)
        | Instr::CreateSemanticFutureExpandMultiOutput(function, _, _)
        | Instr::CallSemanticFunctionMulti(function, _, _)
        | Instr::CallSemanticFunctionMultiUsingOutputSlot(function, _, _)
        | Instr::CallSemanticFunctionExpandMultiOutput(function, _, _)
        | Instr::CallSemanticNestedFunctionMulti { function, .. }
        | Instr::CallSemanticNestedFunctionMultiUsingOutputSlot { function, .. }
        | Instr::CallSemanticNestedFunctionExpandMultiOutput { function, .. } => {
            if let Some(mapped) = remap.get(function) {
                *function = *mapped;
            }
        }
        Instr::ReadSubscriptPath { steps, .. } | Instr::CaptureSubscriptPath { steps, .. } => {
            remap_subscript_steps(steps, remap)
        }
        Instr::BeginSubscriptEndReceiver { prefix } => remap_subscript_steps(prefix, remap),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{MemberName, ObjectIndexingContext, SequenceUse};

    fn dotted(identity: CallableIdentity) -> BytecodeSubscriptStep {
        BytecodeSubscriptStep::DottedInvoke {
            member: MemberName::from("local_method"),
            arguments: Vec::new(),
            semantic_fallback: Some(identity),
        }
    }

    #[test]
    fn session_rebinds_function_id_in_every_subscript_path_carrier() {
        let remap = HashMap::from([(FunctionId(0), FunctionId(17))]);
        let mut instructions = [
            Instr::ReadSubscriptPath {
                steps: vec![dotted(CallableIdentity::BoundFunction(FunctionId(0)))],
                selection: SequenceUse::RequireSingle,
                context: ObjectIndexingContext::Expression,
                to_sequence_register: false,
            },
            Instr::CaptureSubscriptPath {
                steps: vec![dotted(CallableIdentity::AnonymousFunction(FunctionId(0)))],
                sequence_slot: 0,
                context: ObjectIndexingContext::Expression,
            },
            Instr::BeginSubscriptEndReceiver {
                prefix: vec![dotted(CallableIdentity::ExternalFunction {
                    function: FunctionId(0),
                    display_name: "local_method".into(),
                })],
            },
        ];
        for instruction in &mut instructions {
            remap_semantic_function_instr(instruction, &remap);
        }
        let steps = instructions.iter().map(|instruction| match instruction {
            Instr::ReadSubscriptPath { steps, .. } | Instr::CaptureSubscriptPath { steps, .. } => {
                steps
            }
            Instr::BeginSubscriptEndReceiver { prefix } => prefix,
            _ => unreachable!(),
        });
        for steps in steps {
            assert!(matches!(
                &steps[0],
                BytecodeSubscriptStep::DottedInvoke {
                    semantic_fallback: Some(
                        CallableIdentity::BoundFunction(FunctionId(17))
                            | CallableIdentity::AnonymousFunction(FunctionId(17))
                            | CallableIdentity::ExternalFunction {
                                function: FunctionId(17),
                                ..
                            }
                    ),
                    ..
                }
            ));
        }
    }
}
