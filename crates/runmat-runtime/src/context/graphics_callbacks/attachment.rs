//! The fresh owning queue stages all callbacks before Core publishes GUI ports.

use super::*;

impl RuntimeGraphicsCallbackQueue {
    /// Install retained properties into a new unpublished same-session queue.
    /// Failure retires only this fresh queue; original properties stay unchanged.
    #[cfg(feature = "plot-core")]
    pub fn install_retained_graphics_properties(
        &self,
        selected: &RuntimeContext,
    ) -> Result<(), RuntimeError> {
        if self.is_closed()
            || self.0.owner != Some(selected.local_identity())
            || !selected
                .service_ports()
                .graphics_callback_queue()
                .is_some_and(|queue| queue.points_to(self))
        {
            return Err(build_runtime_error(
                "graphics attachment context does not own this live queue",
            )
            .with_identifier("RunMat:graphics:CallbackSessionMismatch")
            .build());
        }
        if !self.0.registrations.borrow().is_empty()
            || !self.0.property_registrations.borrow().is_empty()
        {
            return Err(
                build_runtime_error("graphics attachment requires a fresh callback queue")
                    .with_identifier("RunMat:graphics:CallbackAttachmentNotFresh")
                    .build(),
            );
        }
        let result = (|| {
            if selected.is_cancelled() {
                return Err(crate::execution::ExecutionServiceError::Cancelled.into_runtime_error());
            }
            let prepared = crate::builtins::plotting::state::mode_properties::PreparedPropertyAttachment::prepare(
                selected, self, self.0.maximum_registrations.get(),
            )?;
            if selected.is_cancelled() {
                return Err(crate::execution::ExecutionServiceError::Cancelled.into_runtime_error());
            }
            prepared.commit()
        })();
        if result.is_err() {
            self.close();
        }
        result
    }
}
