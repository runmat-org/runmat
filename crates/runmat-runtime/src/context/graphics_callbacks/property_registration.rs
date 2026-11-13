//! Queue admission and exact Figure-property publication are one local transaction.

use super::*;

impl RuntimeGraphicsCallbackQueue {
    pub(super) fn replace_property_transaction<T>(
        &self,
        selected: RuntimeContext,
        receiver: GraphicsReceiverRef,
        kind: GraphicsCallbackKind,
        callback: RegisteredCallback,
        publish: impl FnOnce(GraphicsCallbackIngress) -> Result<T, RuntimeError>,
    ) -> Result<T, RuntimeError> {
        let representation = callback.representation();
        let figure = self.validate_graphics_source(&selected, &receiver)?;
        self.prune_retired_graphics_properties();
        let mut properties = self.0.property_registrations.borrow_mut();
        if let Some(property) = properties
            .iter_mut()
            .find(|property| property.kind == kind && property.receiver.shares_lease(&receiver))
        {
            let id = property.owner.id;
            if !self.0.registrations.borrow().contains_key(&id) {
                return Err(
                    build_runtime_error("graphics callback property owner is retired")
                        .with_identifier("RunMat:graphics:CallbackSourceRetired")
                        .build(),
                );
            }
            let new_live = Arc::new(CallbackClaim::default());
            let ingress = GraphicsCallbackIngress {
                representation,
                registration: id,
                figure,
                kind,
                live: Arc::clone(&new_live),
                closed: Arc::clone(&self.0.closed),
                sender: self.0.sender.clone(),
                admission_wake: Arc::clone(&self.0.admission_wake),
            };
            // The bounded publisher validates before changing its slot. Until
            // it succeeds, the old queue record and claim are untouched.
            let output = publish(ingress)?;
            let old_live = std::mem::replace(&mut property.owner.live, Arc::clone(&new_live));
            let old = self.0.registrations.borrow_mut().insert(
                id,
                Registration {
                    selected: selected
                        .with_cleanup_command_origin(None)
                        .with_graphics_callback_mutation(None),
                    callback,
                    figure,
                    kind,
                    receiver: Some(receiver),
                    live: new_live,
                },
            );
            debug_assert!(old.is_some(), "property slot has one registration owner");
            drop(properties);
            old_live.retire();
            drop(old);
            return Ok(output);
        }
        drop(properties);
        let (owner, ingress) =
            self.register_inner(selected, figure, kind, Some(receiver.clone()), callback)?;
        // Failure drops this new, unpublished owner; no old property is retired.
        let output = publish(ingress)?;
        self.0
            .property_registrations
            .borrow_mut()
            .push(PropertyRegistration {
                receiver,
                kind,
                owner,
            });
        Ok(output)
    }
}
