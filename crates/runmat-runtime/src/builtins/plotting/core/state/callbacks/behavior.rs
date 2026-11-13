//! Retain the original invocation selection without activating a GUI role.

use super::*;

pub(crate) fn retain_behavior(
    value: Value,
    builtin: &'static str,
) -> BuiltinResult<PlotCallableToken> {
    retain_behavior_in_context(crate::context::legacy::active().as_ref(), value, builtin)
}

pub(crate) fn retain_behavior_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    value: Value,
    builtin: &'static str,
) -> BuiltinResult<PlotCallableToken> {
    let empty = match &value {
        Value::String(runmat_value::StringElement::Text(text)) => text.trim().is_empty(),
        Value::CharArray(chars) => chars.data().is_empty(),
        Value::Tensor(tensor) => tensor.is_empty(),
        Value::Cell(cell) => cell.data().is_empty(),
        _ => false,
    };
    let authority = if empty {
        None
    } else {
        runtime
            .map(|runtime| runtime.capture_background_callback(false))
            .transpose()?
    };
    retain_root_in_context(runtime, value, authority, builtin)
}

impl PlotCallableToken {
    pub(crate) fn shares_lease(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub(crate) fn source_text(&self, runtime: &crate::context::RuntimeContext) -> Option<String> {
        self.belongs_to(runtime)
            .then(|| runtime.plot_behavior_source(self.0.slot))
            .flatten()
    }
    pub(crate) fn belongs_to(&self, runtime: &crate::context::RuntimeContext) -> bool {
        self.0.owner == PlotCallableOwner::Session(runtime.local_identity())
            && self.0.live.load(Ordering::Acquire)
    }

    pub(crate) fn prepare_behavior(
        &self,
        runtime: &crate::context::RuntimeContext,
    ) -> BuiltinResult<Option<crate::context::RuntimeContext>> {
        if !self.belongs_to(runtime) {
            return Err(plotting_error(
                "graphics",
                "callback belongs to another or ended session",
            ));
        }
        runtime.prepare_plot_behavior(self.0.slot)
    }

    pub(crate) fn resolve_in_context(
        &self,
        runtime: &crate::context::RuntimeContext,
    ) -> BuiltinResult<Value> {
        if !self.belongs_to(runtime) {
            return Err(plotting_error(
                "graphics",
                "callback belongs to another or ended session",
            ));
        }
        runtime
            .plot_callback(self.0.slot)
            .ok_or_else(|| plotting_error("graphics", "callback root has retired"))
    }
}
