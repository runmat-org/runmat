//! Exact ordinary-load occurrences from the same compiler graph walk.
use super::*;

pub(super) fn project(
    selected: &runmat_accelerate::FusionGroupPlan,
    loads: &[AccelLoad],
) -> Option<runmat_accelerate::fusion_exec::VmHostInputUses> {
    let layout = selected.group.stack_layout.as_ref()?;
    let mut offsets = HashSet::new();
    let entry = layout
        .compiler_variable_bindings
        .iter()
        .map(|binding| {
            if binding.stack_offset >= layout.required_stack_operands
                || !offsets.insert(binding.stack_offset)
                || !layout.bindings.contains(binding)
                || !selected.inputs.contains(&binding.value_id)
            {
                return None;
            }
            Some((binding.value_id, binding.stack_offset))
        })
        .collect::<Option<Vec<_>>>()?;
    let loads = loads
        .iter()
        .filter(|load| {
            selected.group.span.start <= load.pc
                && load.pc <= selected.group.span.end
                && selected.inputs.contains(&load.value_id)
        })
        .map(|load| runmat_accelerate::fusion_exec::VmHostLoadUse {
            pc: load.pc,
            binding: load.binding.clone(),
            value_id: load.value_id,
        })
        .collect();
    Some(runmat_accelerate::fusion_exec::VmHostInputUses { entry, loads })
}
