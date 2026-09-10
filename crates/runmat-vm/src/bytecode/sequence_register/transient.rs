use super::super::Instr;

const REGISTER_NAME: &str = "comma-separated sequence register";

pub(super) fn validate(instructions: &[Instr]) -> Result<(), String> {
    let mut live_producer = None;
    for (pc, instruction) in instructions.iter().enumerate() {
        if is_producer(instruction) {
            if let Some(producer) = live_producer {
                return Err(format!(
                    "{REGISTER_NAME} is overwritten at instruction {pc} before the value produced at instruction {producer} is consumed"
                ));
            }
            if matches!(instruction, Instr::CaptureCallOutputSequence)
                && !pc
                    .checked_sub(1)
                    .and_then(|previous| instructions.get(previous))
                    .is_some_and(is_output_slot_call)
            {
                return Err(format!(
                    "{REGISTER_NAME} capture at instruction {pc} must immediately follow a dynamically sized call"
                ));
            }
            live_producer = Some(pc);
            continue;
        }
        if is_consumer(instruction) {
            let Some(producer) = live_producer.take() else {
                return Err(format!(
                    "{REGISTER_NAME} is consumed at instruction {pc} without a producer"
                ));
            };
            if producer + 1 != pc {
                return Err(format!(
                    "{REGISTER_NAME} produced at instruction {producer} must be consumed immediately, before instruction {pc}"
                ));
            }
            continue;
        }
        if let Some(producer) = live_producer {
            return Err(format!(
                "{REGISTER_NAME} produced at instruction {producer} crosses instruction {pc} instead of being consumed immediately"
            ));
        }
    }
    if let Some(producer) = live_producer {
        return Err(format!(
            "{REGISTER_NAME} produced at instruction {producer} remains live at the function boundary"
        ));
    }
    Ok(())
}

fn is_producer(instruction: &Instr) -> bool {
    matches!(
        instruction,
        Instr::LoadMemberSequenceUsingOutputSlot { .. }
            | Instr::LoadMemberDynamicSequenceUsingOutputSlot { .. }
            | Instr::ReadSubscriptPath {
                to_sequence_register: true,
                ..
            }
            | Instr::CaptureCallOutputSequence
            | Instr::CaptureScalarSequence
    )
}

fn is_consumer(instruction: &Instr) -> bool {
    matches!(
        instruction,
        Instr::StoreMemberSequence(_)
            | Instr::StoreMemberDynamicSequence
            | Instr::CommitPreparedOutputTargets { .. }
    )
}

fn is_output_slot_call(instruction: &Instr) -> bool {
    matches!(
        instruction,
        Instr::CallFevalMultiUsingOutputSlot(..)
            | Instr::CallFevalExpandMultiOutputUsingOutputSlot(..)
            | Instr::CallBuiltinMultiUsingOutputSlot(..)
            | Instr::CallFunctionMultiUsingOutputSlot { .. }
            | Instr::CallWorkspaceFirstMultiUsingOutputSlot { .. }
            | Instr::CallSemanticFunctionMultiUsingOutputSlot(..)
            | Instr::CallSemanticNestedFunctionMultiUsingOutputSlot { .. }
            | Instr::CallWorkspaceFirstExpandMultiOutputUsingOutputSlot { .. }
    )
}
