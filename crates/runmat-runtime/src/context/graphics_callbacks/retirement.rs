//! Exact session-owner retirement for callbacks and host acknowledgements.

use super::*;

impl RuntimeGraphicsCallbackQueue {
    /// Revoke this session's callback authority even if an embedding host
    /// retained a strong queue clone. Closing is idempotent. A callback that
    /// already began is reported Indeterminate, never silently replayable.
    pub fn close(&self) {
        if self.0.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        // Wake the one session-lane idle pump even when an embedding host
        // retains its strong queue clone after session retirement.
        self.0.admission_wake.notify_one();

        let in_flight = self.0.in_flight.borrow_mut().take();
        if let Some(state) = &in_flight {
            let envelope = state.envelope();
            // A property owner may already have been removed while its
            // cooperation turn is pending. Fence that exact envelope too.
            envelope.claim.retire();
            if let Some(reply) = &envelope.reply {
                match state {
                    InFlight::Ready(_) => {
                        // Ready can also describe the narrow interval after
                        // the reply's DISPATCHED CAS but before process()
                        // publishes Invoking. The queued rejection wins only
                        // before that CAS; otherwise close the started reply.
                        reply.reject_queued(GraphicsCallbackHostFailure::Retired);
                        reply.close_started();
                    }
                    InFlight::Invoking(_) | InFlight::Indeterminate(_) => reply.close_started(),
                }
            }
        }

        // Detach all Rc-local owners under short borrows. Registration and
        // callable destructors may reenter the queue, so none run while a
        // RefCell guard is held. Retire claims before releasing the roots.
        let properties = std::mem::take(&mut *self.0.property_registrations.borrow_mut());
        let registrations = std::mem::take(&mut *self.0.registrations.borrow_mut());
        for registration in registrations.values() {
            registration.live.retire();
        }
        for property in &properties {
            property.owner.live.retire();
        }

        // Host submission racing close may land just after this drain. Its
        // post-send closed check rejects its own ticket; no closed queue may
        // process an envelope or retain a live callable registration.
        loop {
            let received = { self.0.receiver.borrow().try_recv() };
            let Ok(envelope) = received else {
                break;
            };
            envelope.claim.retire();
            if let Some(reply) = envelope.reply {
                reply.reject_queued(GraphicsCallbackHostFailure::Retired);
            }
        }
        drop(in_flight);
        drop(properties);
        drop(registrations);
    }

    pub fn is_closed(&self) -> bool {
        self.0.closed.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::RuntimeExecutionService;
    use std::future::Future;
    use std::num::NonZeroUsize;
    use std::task::Poll;

    fn selected() -> RuntimeContext {
        RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
    }

    fn queue() -> RuntimeGraphicsCallbackQueue {
        RuntimeGraphicsCallbackQueue::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(2).unwrap(),
        )
    }

    fn event() -> GraphicsCallbackHostEvent {
        GraphicsCallbackHostEvent {
            figure: 7,
            kind: GraphicsCallbackKind::PanActionPre,
            position: None,
        }
    }

    #[test]
    fn close_retires_queued_tickets_and_callable_roots_with_external_queue_clone() {
        struct Dropped {
            count: Rc<Cell<usize>>,
            owner: WeakRuntimeGraphicsCallbackQueue,
        }
        impl Drop for Dropped {
            fn drop(&mut self) {
                let owner = self.owner.upgrade().expect("external queue stays alive");
                assert!(owner.0.registrations.borrow_mut().is_empty());
                assert!(owner.0.property_registrations.borrow_mut().is_empty());
                self.count.set(self.count.get() + 1);
            }
        }
        let owner = queue();
        let external = owner.clone();
        let drops = Rc::new(Cell::new(0));
        let witness = Dropped {
            count: drops.clone(),
            owner: owner.downgrade(),
        };
        let callback: Rc<GraphicsCallbackResponder> = Rc::new(move |_| {
            let _keep_witness = &witness;
            Box::pin(async { Ok(GraphicsCallbackResult::Completed) })
        });
        let (_registration, ingress) = owner
            .register_response(
                selected(),
                7,
                GraphicsCallbackKind::PanActionPre,
                callback.clone(),
            )
            .unwrap();
        drop(callback);
        let first = ingress.request(event()).unwrap();
        let second = ingress.request(event()).unwrap();
        owner.close();
        owner.close();
        assert!(external.is_closed());
        assert!(!ingress.is_live());
        assert_eq!(drops.get(), 1, "closed owner releases the callable root");
        assert_eq!(
            futures::executor::block_on(first),
            Err(GraphicsCallbackHostFailure::Retired)
        );
        assert_eq!(
            futures::executor::block_on(second),
            Err(GraphicsCallbackHostFailure::Retired)
        );
        assert_eq!(
            ingress.request(event()).err().unwrap().failure,
            GraphicsCallbackIngressFailure::Retired
        );
        assert_eq!(
            futures::executor::block_on(external.process())
                .unwrap_err()
                .identifier(),
            Some("RunMat:graphics:CallbackSessionRetired")
        );
    }

    #[test]
    fn closing_started_callback_reports_indeterminate_without_replay() {
        let owner = queue();
        let calls = Rc::new(Cell::new(0));
        let callback: Rc<GraphicsCallbackResponder> = {
            let calls = calls.clone();
            Rc::new(move |_| {
                calls.set(calls.get() + 1);
                Box::pin(futures::future::pending())
            })
        };
        let (_registration, ingress) = owner
            .register_response(selected(), 7, GraphicsCallbackKind::PanActionPre, callback)
            .unwrap();
        let ticket = ingress.request(event()).unwrap();
        let mut processing = Box::pin(owner.process());
        futures::executor::block_on(futures::future::poll_fn(|context| {
            assert!(processing.as_mut().poll(context).is_pending());
            Poll::Ready(())
        }));
        assert_eq!(calls.get(), 1);
        owner.close();
        assert_eq!(
            ticket.disposition(),
            GraphicsCallbackDisposition::Indeterminate
        );
        assert_eq!(
            futures::executor::block_on(ticket),
            Err(GraphicsCallbackHostFailure::Indeterminate)
        );
        drop(processing);
        assert_eq!(calls.get(), 1);
        assert_eq!(
            futures::executor::block_on(owner.process())
                .unwrap_err()
                .identifier(),
            Some("RunMat:graphics:CallbackSessionRetired")
        );
    }

    #[test]
    fn close_between_reply_begin_and_invoking_publication_is_indeterminate() {
        let owner = queue();
        let calls = Rc::new(Cell::new(0));
        let callback: Rc<GraphicsCallbackResponder> = {
            let calls = calls.clone();
            Rc::new(move |_| {
                calls.set(calls.get() + 1);
                Box::pin(async { Ok(GraphicsCallbackResult::Completed) })
            })
        };
        let (_registration, ingress) = owner
            .register_response(selected(), 7, GraphicsCallbackKind::PanActionPre, callback)
            .unwrap();
        let ticket = ingress.request(event()).unwrap();
        let envelope = Arc::new(owner.0.receiver.borrow().try_recv().unwrap());
        *owner.0.in_flight.borrow_mut() = Some(InFlight::Ready(Arc::clone(&envelope)));
        let mut invocation = envelope.claim.begin().expect("live claim");
        let reply = envelope.reply.as_ref().expect("host request");
        assert!(reply.begin(), "dispatch CAS wins before close");

        // No callback has been called yet, but the host can no longer safely
        // replay the request: close observes Ready with a dispatched reply.
        owner.close();
        assert_eq!(calls.get(), 0);
        assert_eq!(
            ticket.disposition(),
            GraphicsCallbackDisposition::Indeterminate
        );
        invocation.terminal();
        reply.complete(Ok(GraphicsCallbackResult::Completed));
        assert_eq!(
            futures::executor::block_on(ticket),
            Err(GraphicsCallbackHostFailure::Indeterminate)
        );
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn dropped_queued_ticket_after_claim_acquisition_cannot_publish_callback() {
        let owner = queue();
        let calls = Rc::new(Cell::new(0));
        let callback: Rc<GraphicsCallbackResponder> = {
            let calls = calls.clone();
            Rc::new(move |_| {
                calls.set(calls.get() + 1);
                Box::pin(async { Ok(GraphicsCallbackResult::Completed) })
            })
        };
        let (_registration, ingress) = owner
            .register_response(selected(), 7, GraphicsCallbackKind::PanActionPre, callback)
            .unwrap();
        let mut ticket = ingress.request(event()).unwrap();
        let envelope = owner.0.receiver.borrow().try_recv().unwrap();
        let mut invocation = envelope.claim.begin().expect("claim acquired");
        assert_eq!(ticket.abandon(), GraphicsCallbackDisposition::NotPublished);
        assert!(!envelope.reply.as_ref().expect("host request").begin());
        invocation.abort_before_invocation();
        drop(invocation);
        assert_eq!(calls.get(), 0);
        assert_eq!(
            envelope.claim.retire(),
            GraphicsCallbackDisposition::NotPublished
        );
        drop(ticket);
    }
}
