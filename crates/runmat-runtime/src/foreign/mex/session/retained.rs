//! Visibility supplied by the actual MEX session owner, not its Core consumer.
use super::*;

impl MexRuntimeSession {
    pub(super) fn visit_session_retained_values(
        &self,
        visitor: &mut dyn FnMut(&Value) -> Option<()>,
    ) -> Option<()> {
        // Native lanes can own persistent MxArrays, callback RuntimeContexts and
        // in-flight command values on another thread. Isolated clients retain
        // origin RuntimeContexts. Their complete visibility is not yet exposed;
        // a module path or an idle-looking queue does not prove their absence.
        if self.shutdown_in_progress.get()
            || self.native_lane.try_borrow().ok()?.is_some()
            || !self.native_modules.try_borrow().ok()?.is_empty()
            || !self.isolated.try_borrow().ok()?.is_empty()
            || !self.active_isolated.try_borrow().ok()?.is_empty()
        {
            return None;
        }
        for context in self.native_values.try_borrow().ok()?.values() {
            context.visit_retained_handles(&mut |handle| {
                // Only the lightweight handle identity is wrapped. Managed
                // payloads remain borrowed through the existing Value visitor.
                visitor(&Value::HandleObject(handle.clone()))
            })?;
        }
        // Artifact paths/identities and isolation policy contain no Values.
        // In particular, an uninitialized default adapter is provably empty.
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foreign::ForeignAdapter;

    #[test]
    fn mex_uninitialized_and_installed_metadata_have_complete_empty_visibility() {
        let session = MexRuntimeSession::new();
        let mut count = 0;
        assert_eq!(
            session.visit_retained_values(&mut |_| {
                count += 1;
                Some(())
            }),
            Some(())
        );
        session.installed_modules.borrow_mut().insert(
            "declared".into(),
            InstalledMexArtifact {
                path: PathBuf::from("declared.mex"),
                identity: "metadata-only".into(),
            },
        );
        assert_eq!(
            session.visit_retained_values(&mut |_| {
                count += 1;
                Some(())
            }),
            Some(())
        );
        assert_eq!(count, 0);
    }

    #[test]
    fn mex_conversion_context_visits_managed_targets_without_payload_copy() {
        runmat_gc::gc_test_context(|| {
            let tensor = runmat_value::Tensor::new(vec![2.0; 8192], vec![1, 8192]).unwrap();
            let root = runmat_gc::gc_allocate_rooted(Value::Tensor(tensor)).unwrap();
            let handle = runmat_value::HandleRef {
                class_name: "RetainedPayload".into(),
                target: root.handle(),
                valid: true,
            };
            let original =
                runmat_gc::gc_with_value(&handle.target, |value| value as *const Value).unwrap();
            let context = Rc::new(MxValueContext::new());
            let encoded = context
                .encode(
                    &Value::HandleObject(handle.clone()),
                    runmat_mex::MxApiMode::InterleavedComplex,
                    runmat_mex::MxBoundaryInterface::CMatrix,
                )
                .unwrap();
            drop(encoded);
            let session = MexRuntimeSession::new();
            session
                .native_values
                .borrow_mut()
                .insert(PathBuf::from("owner"), context);
            let mut seen = 0;
            assert_eq!(
                session.visit_retained_values(&mut |value| {
                    let Value::HandleObject(retained) = value else {
                        panic!("retained handle")
                    };
                    assert_eq!(
                        runmat_gc::gc_handle_addr(&retained.target),
                        runmat_gc::gc_handle_addr(&handle.target)
                    );
                    runmat_gc::gc_with_value(&retained.target, |target| {
                        assert!(
                            std::ptr::eq(target, original),
                            "managed payload is borrowed"
                        );
                    })
                    .unwrap();
                    seen += 1;
                    Some(())
                }),
                Some(())
            );
            assert_eq!(seen, 1);
            assert_eq!(session.visit_retained_values(&mut |_| None), None);
        });
    }

    #[test]
    fn mex_unobservable_or_borrowed_state_never_claims_empty_visibility() {
        let session = MexRuntimeSession::new();
        {
            let _borrow = session.native_values.borrow_mut();
            assert_eq!(session.visit_retained_values(&mut |_| Some(())), None);
        }
        session
            .active_isolated
            .borrow_mut()
            .insert(PathBuf::from("active"));
        assert_eq!(session.visit_retained_values(&mut |_| Some(())), None);
        session.active_isolated.borrow_mut().clear();
        session
            .native_modules
            .borrow_mut()
            .insert(PathBuf::from("persistent"));
        assert_eq!(session.visit_retained_values(&mut |_| Some(())), None);
        session.native_modules.borrow_mut().clear();
        session.shutdown_in_progress.set(true);
        assert_eq!(session.visit_retained_values(&mut |_| Some(())), None);
        session.shutdown_in_progress.set(false);
        assert_eq!(session.visit_retained_values(&mut |_| Some(())), Some(()));
    }
}
