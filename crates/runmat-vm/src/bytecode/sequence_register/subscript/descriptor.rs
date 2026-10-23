use super::super::Instr;

pub(super) fn validate_subscript_descriptors(instructions: &[Instr]) -> Result<(), String> {
    for (pc, instruction) in instructions.iter().enumerate() {
        let (steps, selection, to_register) = match instruction {
            Instr::ReadSubscriptPath {
                steps,
                selection,
                to_sequence_register,
                ..
            } => (steps, Some(*selection), *to_sequence_register),
            Instr::CaptureSubscriptPath { steps, .. } => (steps, None, false),
            Instr::BeginSubscriptEndReceiver { prefix } => (prefix, None, false),
            _ => continue,
        };
        if !matches!(instruction, Instr::BeginSubscriptEndReceiver { .. }) && steps.is_empty() {
            return Err(format!("subscript path at instruction {pc} is empty"));
        }
        let dotted = steps.iter().any(|step| {
            matches!(
                step,
                super::super::super::BytecodeSubscriptStep::DottedInvoke { .. }
            )
        });
        if dotted
            && (steps.len() != 1 || matches!(instruction, Instr::BeginSubscriptEndReceiver { .. }))
        {
            return Err(format!(
                "dotted invocation at instruction {pc} must be its own execution segment"
            ));
        }
        for step in steps {
            match step {
                super::super::super::BytecodeSubscriptStep::Braces { selectors }
                    if selectors.iter().any(|selector| {
                        matches!(
                            selector,
                            super::super::super::BytecodeSubscriptSelector::Expansion(_)
                        )
                    }) =>
                {
                    return Err(format!(
                        "brace step at instruction {pc} contains an application expansion"
                    ));
                }
                super::super::super::BytecodeSubscriptStep::Parentheses { selectors }
                    if selectors.iter().any(|selector| {
                        matches!(
                            selector,
                            super::super::super::BytecodeSubscriptSelector::Expansion(_)
                        )
                    }) && matches!(
                        pc.checked_sub(1)
                            .and_then(|previous| instructions.get(previous)),
                        Some(Instr::FinishSubscriptEndReceiver { .. })
                    ) =>
                {
                    return Err(format!("expanded parentheses at instruction {pc} cannot use fixed-position end preparation"));
                }
                _ => {}
            }
        }
        if dotted
            && !matches!(
                pc.checked_sub(1)
                    .and_then(|previous| instructions.get(previous)),
                Some(Instr::FinishDottedInvokeReceiver { .. })
            )
        {
            return Err(format!(
                "dotted invocation at instruction {pc} has no prepared execution segment"
            ));
        }
        steps.iter().try_fold(0usize, |total, step| {
            total.checked_add(step.operand_count()).ok_or_else(|| {
                format!("subscript path operand count overflows at instruction {pc}")
            })
        })?;
        if to_register && selection != Some(runmat_types::SequenceUse::SelectDestinationCardinality)
        {
            return Err(format!(
                "subscript path at instruction {pc} writes the sequence register without destination-cardinality selection"
            ));
        }
    }
    for (pc, instruction) in instructions.iter().enumerate() {
        let Instr::FinishDottedInvokeReceiver {
            argument_operand_count,
            ..
        } = instruction
        else {
            continue;
        };
        let Some(next) = instructions.get(pc + 1) else {
            return Err(format!(
                "dotted invocation at instruction {pc} has no execution segment"
            ));
        };
        let steps = match next {
            Instr::ReadSubscriptPath { steps, .. } | Instr::CaptureSubscriptPath { steps, .. } => {
                steps.as_slice()
            }
            _ => {
                return Err(format!(
                    "dotted invocation at instruction {pc} is not followed by its execution segment"
                ))
            }
        };
        let [super::super::super::BytecodeSubscriptStep::DottedInvoke { arguments, .. }] = steps
        else {
            return Err(format!(
                "dotted invocation at instruction {pc} has an invalid execution segment"
            ));
        };
        let count = arguments.iter().try_fold(0usize, |total, argument| {
            let width = match argument {
                super::super::super::BytecodeDottedInvokeArgument::Colon => 0,
                super::super::super::BytecodeDottedInvokeArgument::Call(spec) => {
                    spec.stack_operand_count()
                }
            };
            total.checked_add(width)
        });
        if count != Some(*argument_operand_count) {
            return Err(format!(
                "dotted invocation at instruction {pc} declares the wrong argument operand count"
            ));
        }
    }
    Ok(())
}
