//! Original-definition projection into the single shared named-selector owner.
use super::*;
use crate::builtins::math::optim::common::{model_indexing as shared, model_policy::Policy};
impl shared::Axes for View {
    fn shape(&self) -> &[usize] {
        self.selection().shape()
    }
    fn has_names(&self, axis: usize) -> bool {
        matches!(self.selection().axis_sources().get(axis),Some(OptimizationVariableAxisSource::DefinitionAxis(source)) if self.definition().index_names()[*source].is_some())
    }
    fn name(&self, axis: usize, position: usize) -> Option<&runmat_value::Utf8Text> {
        super::super::properties::axis_name(self, axis, position)
    }
}
fn policy() -> Policy {
    Policy {
        identity: runmat_builtins::OPTIMVAR_CATALOG_ENTRY.identity,
        invalid: &runmat_builtins::OPTIMVAR_ERROR_INVALID_INPUT,
    }
}
pub(super) use shared::is_name_selector;
pub(super) fn vector_axis(view: &View) -> usize {
    shared::vector_axis(view)
}
pub(super) fn axis(view: &View, count: usize, index: usize) -> BuiltinResult<usize> {
    shared::axis(view, count, index, policy())
}
pub(super) async fn indices(
    view: &View,
    axis: usize,
    value: &Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Value> {
    shared::indices(view, axis, value, cooperation, policy()).await
}
