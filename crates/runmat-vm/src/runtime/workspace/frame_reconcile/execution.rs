use super::*;
use crate::interpreter::vars::InterpreterVarsRetirement;

struct FrameReconcileRetirement {
    _replacements: Replacements,
    _value_backing: Option<InterpreterVarsRetirement>,
    _names: FrameVec<WorkspaceName>,
    _values: FrameVec<InterpreterSlot>,
}

fn retain_names(
    retired: &mut FrameVec<WorkspaceName>,
    names: impl IntoIterator<Item = Option<WorkspaceName>>,
) {
    retired.extend(names.into_iter().flatten());
}

impl PreparedFrameReconcile {
    pub(super) fn checkpoint(&self) -> Result<(), RuntimeError> {
        if !self.token.is_prepared() {
            return Err(unavailable());
        }
        if let (Some(context), Some(receipt)) = (&self.context, &self.receipt) {
            receipt.checkpoint(context).map_err(allocation)?;
        }
        if !self.token.begin_publication() {
            return Err(unavailable());
        }
        Ok(())
    }

    pub(super) fn commit(mut self) {
        assert!(self.token.is_publishing());
        let mut retired_names = self
            .retired_names
            .take()
            .expect("prepared reconciliation retirement names");
        let mut retired_values = self
            .retired_values
            .take()
            .expect("prepared reconciliation retirement values");
        let retired = with_frame_mut_unlocked(self.frame, |frame| {
            let ws = &mut frame.state;
            let mut replacements = Replacements::default();
            macro_rules! replace {
                ($prepared:ident, $field:ident) => {
                    if let Some(backing) = self.replacements.$prepared.take() {
                        replacements.$prepared = Some(std::mem::replace(&mut ws.$field, backing));
                    }
                };
            }
            replace!(names, names);
            replace!(assigned, assigned);
            replace!(assigned_names, assigned_names_this_execution);
            replace!(assigned_ids, assigned_ids_this_execution);
            replace!(removed_slots, removed_slots_this_execution);
            replace!(removed_ids, removed_ids_this_execution);
            replace!(removed_names, removed_names_this_execution);
            replace!(lifecycle, slot_lifecycle);
            replace!(reconciliation, pending_var_reconciliation);
            let value_backing = self
                .replacements
                .values
                .take()
                .map(|backing| ws.values.install_replacement(backing));

            for (name, slot) in &self.removed {
                if let Some((canonical, _)) = ws.names.remove_entry(name.as_str()) {
                    retired_names.push(canonical);
                }
                if let Some(canonical) = ws.assigned.take(name.as_str()) {
                    retired_names.push(canonical);
                }
                if let Some(value) = ws.values.replace_owned(*slot, Value::Num(0.0)) {
                    retired_values.push(value);
                }
                retain_names(
                    &mut retired_names,
                    super::super::mark_slot_unassigned_retaining(ws, *slot, name.clone()),
                );
                if !ws.pending_var_reconciliation.contains(slot) {
                    ws.pending_var_reconciliation.insert(*slot);
                }
            }
            for index in 0..self.incoming.len() {
                let Some((name, slot)) = self.incoming[index].as_ref() else {
                    continue;
                };
                let value = std::mem::replace(&mut self.values[index].1, Value::Num(0.0));
                if let Some(existing) = ws.names.get_mut(name.as_str()) {
                    *existing = *slot;
                } else {
                    ws.names.insert(name.clone(), *slot);
                }
                if ws.values.len() <= *slot {
                    ws.values.resize_owned(*slot + 1, Value::Num(0.0));
                }
                if let Some(value) = ws.values.replace_owned(*slot, value) {
                    retired_values.push(value);
                }
                if !ws.assigned.contains(name.as_str()) {
                    ws.assigned.insert(name.clone());
                }
                if !ws.assigned_names_this_execution.contains(name.as_str()) {
                    ws.assigned_names_this_execution.insert(name.clone());
                }
                if !ws.assigned_ids_this_execution.contains(slot) {
                    ws.assigned_ids_this_execution.insert(*slot);
                }
                if !ws.pending_var_reconciliation.contains(slot) {
                    ws.pending_var_reconciliation.insert(*slot);
                }
                retain_names(
                    &mut retired_names,
                    super::super::mark_slot_assigned_retaining(ws, *slot, name),
                );
            }
            FrameReconcileRetirement {
                _replacements: replacements,
                _value_backing: value_backing,
                _names: retired_names,
                _values: retired_values,
            }
        })
        .expect("prepared reconciliation holds its exact frame through commit");
        // The phase becomes available before GC, foreign, or admitted backing
        // owners retire, and the workspace stack is no longer borrowed.
        drop(self);
        drop(retired);
    }
}
