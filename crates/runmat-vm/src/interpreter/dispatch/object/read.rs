use super::{write_sequence_register, ObjectDispatchContext};
use runmat_runtime::object::resolve as obj_resolve;
use runmat_runtime::sequence::ResolveValueSequence;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

pub(super) async fn dispatch(
    instr: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
    sequence_register: &mut super::super::SequenceState,
    context: ObjectDispatchContext<'_>,
) -> Result<bool, RuntimeError> {
    let runtime = context.runtime;
    let vars = context.vars;
    let caller_function_name =
        (!context.current_function_name.is_empty()).then_some(context.current_function_name);
    match instr {
        crate::bytecode::Instr::LoadMember(field) => {
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let value = obj_resolve::load_member_with_context(
                Some(runtime),
                base,
                field.0.clone(),
                false,
                caller_function_name,
            )
            .await?;
            value.with_owned(|value| stack.push(value));
            Ok(true)
        }
        crate::bytecode::Instr::LoadMemberOrInit(field) => {
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let value = obj_resolve::load_member_with_context(
                Some(runtime),
                base,
                field.0.clone(),
                true,
                caller_function_name,
            )
            .await?;
            value.with_owned(|value| stack.push(value));
            Ok(true)
        }
        crate::bytecode::Instr::LoadMemberDynamic => {
            let name_val = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let name: String = (&name_val).try_into()?;
            let value = obj_resolve::load_member_dynamic_with_context(
                Some(runtime),
                base,
                name,
                false,
                caller_function_name,
            )
            .await?;
            value.with_owned(|value| stack.push(value));
            Ok(true)
        }
        crate::bytecode::Instr::LoadMemberDynamicOrInit => {
            let name_val = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let name: String = (&name_val).try_into()?;
            let value = obj_resolve::load_member_dynamic_with_context(
                Some(runtime),
                base,
                name,
                true,
                caller_function_name,
            )
            .await?;
            value.with_owned(|value| stack.push(value));
            Ok(true)
        }
        crate::bytecode::Instr::LoadMemberSequence { member, selection } => {
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let values = obj_resolve::read_member_sequence_with_context(
                Some(runtime),
                base,
                member.0.clone(),
                false,
                caller_function_name,
            )
            .await?;
            values.with_owned_sequence(|values| {
                stack.extend(values.resolve(*selection, Default::default())?);
                Ok::<_, RuntimeError>(())
            })?;
            Ok(true)
        }
        crate::bytecode::Instr::LoadMemberDynamicSequence { selection } => {
            let name: String = (&stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?)
                .try_into()?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let values = obj_resolve::read_member_sequence_with_context(
                Some(runtime),
                base,
                name,
                false,
                caller_function_name,
            )
            .await?;
            values.with_owned_sequence(|values| {
                stack.extend(values.resolve(*selection, Default::default())?);
                Ok::<_, RuntimeError>(())
            })?;
            Ok(true)
        }
        crate::bytecode::Instr::MemberSequenceCardinality => {
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let count = obj_resolve::member_sequence_cardinality(&base)?;
            stack.push(Value::Num(count as f64));
            Ok(true)
        }
        crate::bytecode::Instr::LoadMemberSequenceUsingOutputSlot {
            member,
            output_count_slot,
        } => {
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let count = super::super::requested_outputs_from_slot(vars, *output_count_slot)?;
            let sequence = obj_resolve::read_member_sequence_with_context(
                Some(runtime),
                base,
                member.0.clone(),
                false,
                caller_function_name,
            )
            .await?;
            sequence.with_owned_sequence(|sequence| {
                let values = sequence.resolve(
                    runmat_types::SequenceUse::SelectPrefix { count },
                    Default::default(),
                )?;
                write_sequence_register(stack, sequence_register, values)
            })?;
            Ok(true)
        }
        crate::bytecode::Instr::LoadMemberDynamicSequenceUsingOutputSlot { output_count_slot } => {
            let name: String = (&stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?)
                .try_into()?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let count = super::super::requested_outputs_from_slot(vars, *output_count_slot)?;
            let sequence = obj_resolve::read_member_sequence_with_context(
                Some(runtime),
                base,
                name,
                false,
                caller_function_name,
            )
            .await?;
            sequence.with_owned_sequence(|sequence| {
                let values = sequence.resolve(
                    runmat_types::SequenceUse::SelectPrefix { count },
                    Default::default(),
                )?;
                write_sequence_register(stack, sequence_register, values)
            })?;
            Ok(true)
        }
        _ => Ok(false),
    }
}
