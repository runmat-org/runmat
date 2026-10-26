//! Borrow original native children; never create a provisional object header.
use super::*;
use crate::execution::{RootedValueAdmissionError, RootedValueHandles};

#[derive(Clone)]
pub(super) struct Values<'a> {
    input: &'a dyn OptimizationNativeSource,
    index: usize,
}
impl<'a> Values<'a> {
    pub(super) fn new(input: &'a impl OptimizationNativeSource) -> Self {
        Self { input, index: 0 }
    }
}
impl<'a> Iterator for Values<'a> {
    type Item = &'a Value;
    fn next(&mut self) -> Option<Self::Item> {
        let value = self.input.value_at(self.index)?;
        self.index = self.index.saturating_add(1);
        Some(value)
    }
}
#[derive(Clone)]
pub(super) struct Handles<'a> {
    input: &'a dyn OptimizationNativeSource,
    index: usize,
}
impl<'a> Handles<'a> {
    pub(super) fn new(input: &'a impl OptimizationNativeSource) -> Self {
        Self { input, index: 0 }
    }
}
impl Iterator for Handles<'_> {
    type Item = runmat_gc::GcHandle;
    fn next(&mut self) -> Option<Self::Item> {
        let value = self.input.handle_at(self.index)?;
        self.index = self.index.saturating_add(1);
        Some(value)
    }
}
pub(super) fn retain(
    input: &impl OptimizationNativeSource,
    policy: Policy,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<RootedValueHandles> {
    match context {
        Some(context) => RootedValueHandles::for_values_and_handles_iterator_in_context(
            Values::new(input),
            Handles::new(input),
            context,
        )
        .map_err(|source| root_error(source, policy, Some(context))),
        None => {
            let selected = crate::context::PreparedUnselectedScope::prepare(None)?;
            let _scope = selected.enter();
            RootedValueHandles::for_values_and_handles_iterator_unbounded(
                Values::new(input),
                Handles::new(input),
            )
            .map_err(|source| policy.gc("model child root admission failed", source))
        }
    }
}
pub(super) fn root_error(
    source: RootedValueAdmissionError,
    policy: Policy,
    context: Option<&RuntimeContext>,
) -> RuntimeError {
    match source {
        RootedValueAdmissionError::Host(source) => allocation(context, source),
        RootedValueAdmissionError::Gc(source) => {
            policy.gc("model child root admission failed", source)
        }
    }
}
