use std::collections::{HashMap, HashSet};

use runmat_runtime::RuntimeError;
use runmat_value::Value;

use super::{lexical, persistent};

pub(super) enum AssignmentRootPresence {
    Present,
    Absent(String),
}

impl AssignmentRootPresence {
    pub fn is_absent(&self) -> bool {
        matches!(self, Self::Absent(_))
    }
}

/// The same canonical binding refresh used by an indexed-assignment load and
/// by a prepared destination before it observes its root.
pub(super) struct AssignmentSlotAuthority<'a> {
    pub context: &'a crate::bytecode::program::ExecutionContext,
    pub global_aliases: &'a runmat_runtime::workspace::bindings::FrameGlobalBindings,
    pub persistent_aliases: &'a crate::interpreter::persistent::PersistentAliases,
    pub missing_input_slots: &'a HashSet<usize>,
    pub var_names: &'a HashMap<usize, String>,
    pub current_function_name: &'a str,
}

impl AssignmentSlotAuthority<'_> {
    pub fn refresh(
        &self,
        vars: &mut Vec<Value>,
        slot: usize,
    ) -> Result<AssignmentRootPresence, RuntimeError> {
        let global_binding = self.global_aliases.binding_for_slot(slot);
        let global_value = crate::runtime::globals::current_bound_value(self.global_aliases, slot);
        if let Some(global_value) = global_value.as_ref() {
            if slot >= vars.len() {
                vars.resize(slot + 1, Value::Num(0.0));
            }
            vars[slot] = global_value.clone();
            crate::runtime::workspace::refresh_workspace_slot(slot, &vars[slot]);
        }
        let authority = self.context.runtime.session_variable_authority();
        persistent::refresh(
            self.persistent_aliases,
            vars,
            &authority,
            self.current_function_name,
            slot,
        )?;
        let persistent_present =
            persistent::selected_presence(self.persistent_aliases, &authority, slot)?;
        let lexical_assigned = lexical::refresh(self.context, vars, slot)?;
        if self.missing_input_slots.contains(&slot) {
            return Err(crate::interpreter::errors::mex(
                "NotEnoughInputs",
                "Not enough input arguments.",
            ));
        }
        let absent = if global_binding.is_some() {
            global_value.is_none()
        } else if let Some(present) = persistent_present {
            !present
        } else if !lexical_assigned {
            true
        } else {
            matches!(
                (
                crate::runtime::workspace::workspace_slot_assigned(slot),
                crate::runtime::workspace::workspace_slot_name(slot),
                self.var_names.get(&slot),
                ),
                (Some(false), Some(slot_name), Some(var_name)) if slot_name == *var_name
            )
        };
        if absent {
            if let Some(value) = vars.get_mut(slot) {
                *value = Value::Num(0.0);
                // A missing canonical alias is not a workspace `clear`: do
                // not record a removal in this frame's completion report.
                crate::runtime::workspace::refresh_workspace_slot(slot, value);
            }
            let name = self
                .var_names
                .get(&slot)
                .cloned()
                .or_else(|| crate::runtime::workspace::workspace_slot_name(slot))
                .unwrap_or_else(|| format!("var_{slot}"));
            Ok(AssignmentRootPresence::Absent(name))
        } else {
            Ok(AssignmentRootPresence::Present)
        }
    }

    pub fn publish(&self, vars: &[Value], slot: usize) -> Result<(), RuntimeError> {
        if let Some(value) = vars.get(slot) {
            crate::runtime::globals::update_global_store(slot, value, self.global_aliases);
        }
        persistent::publish_store(
            self.persistent_aliases,
            vars,
            &self.context.runtime.session_variable_authority(),
            self.current_function_name,
            slot,
        )?;
        lexical::publish(self.context, vars, slot)
    }
}
