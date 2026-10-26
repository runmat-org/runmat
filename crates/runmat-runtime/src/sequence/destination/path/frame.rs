//! Retain the original method proof once, before destination user work.
use super::SequenceDestinationBuilder;
use crate::context::RuntimeContext;
use crate::object::protocol::ObjectAccessContext;
use crate::RuntimeError;

impl SequenceDestinationBuilder {
    /// Only actual execution-frame producers call this for a class method.
    /// Text/capture backing is not covered by the control block's layout quote.
    pub fn with_method_frame_factory_in_context(
        mut self,
        runtime: &RuntimeContext,
        factory: impl FnOnce() -> Result<ObjectAccessContext, RuntimeError>,
    ) -> Result<Self, RuntimeError> {
        self.control.check_owner(Some(runtime))?;
        if self.method_frame.is_some()
            || !self.steps.is_empty()
            || self.object_suffix.is_some()
            || !self.selector_roots.is_empty()
        {
            return Err(crate::runtime_error::semantic_error(
                "PreparedDestinationState",
                "method proof must be captured once before destination traversal",
            ));
        }
        crate::context::host_allocation::require_unbounded_workspace(
            Some(runtime),
            "prepared assignment",
            "retained class-method frame text has no complete finite backing quote",
        )?;
        let frame = {
            let _scope = runtime.enter();
            factory()?
        };
        self.method_frame = Some(self.control.boxed(Some(runtime), frame)?);
        Ok(self)
    }
}
