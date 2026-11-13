//! Visibility supplied by the actual MEX session owner, not its Core consumer.
use super::*;
#[cfg(test)]
use runmat_mex::DirectMexBoundaryHostServices;

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
    use std::sync::atomic::{AtomicBool, Ordering};

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
                identity: runmat_types::ForeignArtifactIdentity::new("metadata-only")
                    .expect("fixture artifact identity"),
                interop: runmat_types::InteropManifest::empty(),
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
            session.native_values.borrow_mut().insert(
                PathBuf::from("owner"),
                Rc::new(RegisteredMxValueContext::new(context).unwrap()),
            );
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

    #[test]
    fn persistent_token_is_rooted_before_input_root_retires_and_survives_collection() {
        runmat_gc::gc_test_context(|| {
            let session = MexRuntimeSession::new();
            let path = PathBuf::from("persistent.mex");
            let input = runmat_gc::gc_allocate_rooted(Value::Num(73.0)).unwrap();
            let target = input.handle();
            let handle = runmat_value::HandleRef {
                class_name: "PersistentProbe".into(),
                target,
                valid: true,
            };
            let context = session.native_value_context(&path).unwrap();
            let token = context
                .encode(
                    &Value::HandleObject(handle),
                    runmat_mex::MxApiMode::InterleavedComplex,
                    runmat_mex::MxBoundaryInterface::CMatrix,
                )
                .unwrap();
            session.native_modules.borrow_mut().insert(path.clone());

            let target_identity = (target.addr(), target.epoch());
            let observed = Arc::new(AtomicBool::new(false));
            let saw_retirement = Arc::clone(&observed);
            let _observer = runmat_gc::RootRetirementObserver::register(Arc::new(move |event| {
                let runmat_gc::RootRetirement::Explicit(retired) = event else {
                    return;
                };
                if (retired.addr(), retired.epoch()) == target_identity {
                    assert_eq!(
                        runmat_gc::gc_is_reachable_excluding(retired, &[]),
                        Some(true)
                    );
                    saw_retirement.store(true, Ordering::SeqCst);
                }
            }))
            .unwrap();
            input.unroot().unwrap();
            assert!(observed.load(Ordering::SeqCst));
            runmat_gc::gc_collect_major().unwrap();
            assert_eq!(
                runmat_gc::gc_with_value(&target, |value| value.clone()).unwrap(),
                Value::Num(73.0)
            );

            // A later native callback can still resolve the original token.
            let restored = context.decode(&token).unwrap();
            assert!(matches!(restored, Value::HandleObject(ref value) if value.target == target));
            drop(restored);

            // The successful clear path retires its registration only after
            // native clear/mexAtExit completes.
            session.native_modules.borrow_mut().remove(&path);
            session.release_native_value_context(&path);
            drop(context);
            runmat_gc::gc_collect_major().unwrap();
            assert!(runmat_gc::gc_with_value(&target, |_| ()).is_err());
        });
    }

    #[test]
    fn shutdown_releases_every_persistent_token_registration() {
        runmat_gc::gc_test_context(|| {
            let session = MexRuntimeSession::new();
            let mut targets = Vec::new();
            for path in ["first.mex", "second.mex"] {
                let input = runmat_gc::gc_allocate_rooted(Value::Num(1.0)).unwrap();
                let target = input.handle();
                let context = session.native_value_context(Path::new(path)).unwrap();
                context
                    .encode(
                        &Value::HandleObject(runmat_value::HandleRef {
                            class_name: "ShutdownProbe".into(),
                            target,
                            valid: true,
                        }),
                        runmat_mex::MxApiMode::InterleavedComplex,
                        runmat_mex::MxBoundaryInterface::CMatrix,
                    )
                    .unwrap();
                input.unroot().unwrap();
                targets.push(target);
            }
            runmat_gc::gc_collect_major().unwrap();
            for target in &targets {
                assert!(runmat_gc::gc_with_value(target, |_| ()).is_ok());
            }
            // Exercise the public shutdown path; no native lane is needed for
            // these synthetic contexts, and shutdown still retires every root.
            let runtime =
                RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
            futures::executor::block_on(session.shutdown(runtime)).unwrap();
            runmat_gc::gc_collect_major().unwrap();
            for target in &targets {
                assert!(runmat_gc::gc_with_value(target, |_| ()).is_err());
            }
        });
    }

    #[test]
    fn forced_session_disposal_abandons_pending_native_service_without_stale_gc_root() {
        runmat_gc::gc_test_context(|| {
            let runtime =
                RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
            let session = MexRuntimeSession::new();
            let path = PathBuf::from("abandoned.mex");
            let input = runmat_gc::gc_allocate_rooted(Value::Num(29.0)).unwrap();
            let target = input.handle();
            let registration = session.native_value_registration(&path).unwrap();
            let token = registration
                .context()
                .encode(
                    &Value::HandleObject(runmat_value::HandleRef {
                        class_name: "AbandonedProbe".into(),
                        target,
                        valid: true,
                    }),
                    runmat_mex::MxApiMode::InterleavedComplex,
                    runmat_mex::MxBoundaryInterface::CMatrix,
                )
                .unwrap();
            let lane = Rc::new(NativeMexLane::spawn().unwrap());
            lane.hold_test_services(Rc::new(
                DirectMexBoundaryHostServices::new_with_origin_lease(
                    Rc::new(RuntimeMexHostServices::new(runtime)),
                    registration.context(),
                    runmat_mex::MxApiMode::InterleavedComplex,
                    runmat_mex::MxBoundaryInterface::CMatrix,
                    registration.clone(),
                ),
            ));
            *session.native_lane.borrow_mut() = Some(lane);
            input.unroot().unwrap();
            runmat_gc::gc_collect_major().unwrap();
            assert!(runmat_gc::gc_with_value(&target, |_| ()).is_ok());
            assert!(matches!(
                registration.context().decode(&token),
                Ok(Value::HandleObject(_))
            ));

            drop(registration);
            drop(session);
            runmat_gc::gc_collect_major().unwrap();
            assert!(runmat_gc::gc_with_value(&target, |_| ()).is_err());
        });
    }
}
