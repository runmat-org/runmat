use std::collections::{HashMap, HashSet};

use crate::interpreter::vars::InterpreterVars;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

use super::{lexical, persistent};

pub(super) fn empty_cell_root() -> Value {
    Value::Cell(
        runmat_value::CellArray::new(Vec::new(), 0, 0)
            .expect("the canonical empty cell shape is valid"),
    )
}

fn empty_declared_root() -> Value {
    Value::Tensor(
        runmat_value::Tensor::new(Vec::new(), vec![0, 0])
            .expect("the canonical declared-empty numeric shape is valid"),
    )
}

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
        vars: &mut InterpreterVars,
        slot: usize,
    ) -> Result<AssignmentRootPresence, RuntimeError> {
        let global_binding = self.global_aliases.binding_handle_for_slot(slot);
        let global_value = global_binding
            .as_ref()
            .and_then(crate::runtime::globals::current_bound_value_for_handle);
        if let Some(global_value) =
            global_value.or_else(|| global_binding.as_ref().map(|_| empty_declared_root()))
        {
            if slot >= vars.len() {
                vars.resize_owned(slot + 1, Value::Num(0.0));
            }
            let _retired = vars.replace_owned(slot, global_value);
            crate::runtime::workspace::refresh_workspace_slot_from_vars(slot, vars);
        }
        let authority = self.context.runtime.session_variable_authority();
        persistent::refresh(
            self.persistent_aliases,
            vars,
            &authority,
            self.current_function_name,
            slot,
        )?;
        let lexical_assigned = lexical::refresh(self.context, vars, slot)?;
        if self.missing_input_slots.contains(&slot) {
            return Err(crate::interpreter::errors::mex(
                "NotEnoughInputs",
                "Not enough input arguments.",
            ));
        }
        let absent = if global_binding.is_some() || self.persistent_aliases.contains_key(&slot) {
            // A declaration itself owns a live, initially empty numeric
            // binding even if the session has not stored its payload yet.
            false
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
            if vars.get(slot).is_some() {
                let _retired = vars.replace_owned(slot, Value::Num(0.0));
                // A missing canonical alias is not a workspace `clear`: do
                // not record a removal in this frame's completion report.
                crate::runtime::workspace::refresh_workspace_slot_from_vars(slot, vars);
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

    pub fn publish(&self, vars: &InterpreterVars, slot: usize) -> Result<(), RuntimeError> {
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
