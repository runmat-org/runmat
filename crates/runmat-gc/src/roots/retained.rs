//! Value projection on the same registration that owns GC root liveness.
use super::*;

/// A GC root whose retained Values are visible for non-GC resource ownership.
/// The same lifetime and exclusive-access invariants as `GcRoot::scan` apply.
/// Like scanning, projection must not register/unregister roots or reenter the
/// root scanner: the registration read lock protects all projected lifetimes.
pub trait ValueRoot: GcRoot {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()>;
}

pub(crate) enum RegisteredRoot {
    Opaque(Box<dyn GcRoot>),
    Values(Box<dyn ValueRoot>),
}

impl RegisteredRoot {
    pub(super) fn visit_retained_values(
        &self,
        visitor: &mut dyn FnMut(&Value) -> Option<()>,
    ) -> Option<()> {
        match self {
            Self::Opaque(_) => None,
            Self::Values(root) => root.visit_retained_values(visitor),
        }
    }
}

impl GcRoot for RegisteredRoot {
    fn scan(&self) -> Vec<GcHandle> {
        match self {
            Self::Opaque(root) => root.scan(),
            Self::Values(root) => root.scan(),
        }
    }
    fn description(&self) -> String {
        match self {
            Self::Opaque(root) => root.description(),
            Self::Values(root) => root.description(),
        }
    }
    fn estimated_size(&self) -> usize {
        match self {
            Self::Opaque(root) => root.estimated_size(),
            Self::Values(root) => root.estimated_size(),
        }
    }
    fn is_active(&self) -> bool {
        match self {
            Self::Opaque(root) => root.is_active(),
            Self::Values(root) => root.is_active(),
        }
    }
}

impl ValueRoot for StackRoot {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        // SAFETY: precisely the same registered-vector lifetime and exclusive
        // scan requirement as StackRoot::scan; registration owns both views.
        for value in unsafe { self.stack.as_ref() } {
            visitor(value)?;
        }
        Some(())
    }
}
impl ValueRoot for VariableArrayRoot {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        // SAFETY: same registered-vector invariant as VariableArrayRoot::scan.
        for value in unsafe { self.vars.as_ref() } {
            visitor(value)?;
        }
        Some(())
    }
}
impl ValueRoot for GlobalRoot {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        for value in &self.values {
            visitor(value)?;
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrowed_projection_does_not_copy_large_host_values_and_stops_on_failure() {
        let values = vec![
            Value::Tensor(runmat_value::Tensor::new(vec![3.0; 65_536], vec![256, 256]).unwrap()),
            Value::Num(7.0),
        ];
        // SAFETY: values remain pinned and immutable throughout both visits.
        let root = unsafe { StackRoot::new(NonNull::from(&values), "borrowed".into()) };
        let mut index = 0;
        root.visit_retained_values(&mut |value| {
            assert!(
                std::ptr::eq(value, &values[index]),
                "visitor must receive the actual stored Value"
            );
            index += 1;
            Some(())
        })
        .unwrap();
        assert_eq!(index, 2);
        let mut seen = 0;
        assert!(root
            .visit_retained_values(&mut |_| {
                seen += 1;
                None
            })
            .is_none());
        assert_eq!(seen, 1);
    }

    #[test]
    fn nested_stack_aliases_follow_the_existing_registration_lifetime() {
        let scanner = RootScanner::new();
        let outer = vec![Value::String("caller-owned".into())];
        let inner = vec![outer[0].clone()];
        // SAFETY: both vectors remain fixed and live until unregister below.
        let outer_id = scanner
            .register_value_root(Box::new(unsafe {
                StackRoot::new(NonNull::from(&outer), "outer".into())
            }))
            .unwrap();
        let inner_id = scanner
            .register_value_root(Box::new(unsafe {
                StackRoot::new(NonNull::from(&inner), "inner".into())
            }))
            .unwrap();
        assert_eq!(scanner.retained_values().unwrap().len(), 2);
        assert_eq!(
            scanner.retained_values_excluding(&[inner_id]),
            Some(outer.clone())
        );
        assert_eq!(
            scanner.retained_values().unwrap().len(),
            2,
            "projection exclusion never unregisters roots"
        );
        scanner.unregister_root(inner_id).unwrap();
        assert_eq!(scanner.retained_values(), Some(outer.clone()));
        scanner.unregister_root(outer_id).unwrap();
        assert_eq!(scanner.retained_values(), Some(vec![]));
    }

    #[test]
    fn explicit_root_values_remain_visible_until_unregistered() {
        crate::gc_test_context(|| {
            let root = crate::gc_allocate_rooted(Value::String("explicit-owner".into())).unwrap();
            assert!(crate::gc_retained_values()
                .unwrap()
                .contains(&Value::String("explicit-owner".into())));
            let handle = root.handle();
            crate::gc_with_value_mut(&handle, |_| {
                assert!(
                    crate::gc_retained_values().is_none(),
                    "a borrowed target is not proof of no aliases"
                );
            })
            .unwrap();
            drop(root);
            assert!(!crate::gc_retained_values()
                .unwrap()
                .contains(&Value::String("explicit-owner".into())));
        });
    }

    #[test]
    fn exclusion_is_registration_scoped_not_target_scoped() {
        crate::gc_test_context(|| {
            let created = crate::gc_allocate_rooted(Value::String("target".into())).unwrap();
            let target = created.handle();
            let value = Value::HandleObject(runmat_value::HandleRef {
                class_name: "Probe".into(),
                target,
                valid: true,
            });
            let excluded = crate::gc_register_value_root(Box::new(GlobalRoot::new(
                vec![value],
                "excluded temporary owner".into(),
            )))
            .unwrap();
            created.unroot().unwrap();
            assert_eq!(
                crate::gc_is_reachable_excluding(target, &[excluded]),
                Some(false)
            );
            let alias = crate::gc_root(target).unwrap();
            assert_eq!(
                crate::gc_is_reachable_excluding(target, &[excluded]),
                Some(true)
            );
            drop(alias);
            let mut holder = runmat_value::UnboundedObjectBuilder::new("Holder");
            holder.insert(
                "nested".into(),
                Value::HandleObject(runmat_value::HandleRef {
                    class_name: "Probe".into(),
                    target,
                    valid: true,
                }),
            );
            let nested = crate::gc_allocate_rooted(Value::Object(holder.finish())).unwrap();
            assert_eq!(
                crate::gc_is_reachable_excluding(target, &[excluded]),
                Some(true)
            );
            drop(nested);
            assert_eq!(
                crate::gc_is_reachable_excluding(target, &[excluded]),
                Some(false)
            );
            crate::gc_unregister_root(excluded).unwrap();
        });
    }

    #[test]
    fn batched_registration_graph_preserves_callback_dependency_and_independent_alias() {
        crate::gc_test_context(|| {
            let first = crate::gc_allocate_rooted(Value::Num(1.0)).unwrap();
            let second = crate::gc_allocate_rooted(Value::Num(2.0)).unwrap();
            let a = first.handle();
            let b = second.handle();
            let handle_value = |target| {
                Value::HandleObject(runmat_value::HandleRef {
                    class_name: "Probe".into(),
                    target,
                    valid: true,
                })
            };
            let temporary = crate::ValueRootRegistration::new(Box::new(GlobalRoot::new(
                vec![handle_value(a), handle_value(b)],
                "detached values".into(),
            )))
            .unwrap();
            let callback = crate::ValueRootRegistration::new(Box::new(GlobalRoot::new(
                vec![handle_value(b)],
                "active A callback".into(),
            )))
            .unwrap();
            first.unroot().unwrap();
            second.unroot().unwrap();
            let inspect = || {
                crate::gc_with_registered_root_reachability(
                    &[a, b],
                    &[callback.id()],
                    &[temporary.id()],
                    |snapshot| snapshot.unwrap(),
                )
            };
            let graph = inspect();
            assert!(!graph.outside_reaches(0));
            assert!(!graph.outside_reaches(1));
            assert!(!graph.source_reaches(0, 0));
            assert!(graph.source_reaches(0, 1));
            let alias = crate::gc_root(b).unwrap();
            assert!(inspect().outside_reaches(1));
            drop(alias);
        });
    }

    #[test]
    fn opaque_root_is_unavailable_even_when_gc_handle_scan_is_empty() {
        let scanner = RootScanner::new();
        let id = scanner
            .register_root(Box::new(GlobalRoot::new(vec![], "opaque".into())))
            .unwrap();
        assert!(scanner.scan_roots().unwrap().is_empty());
        assert!(scanner.retained_values().is_none());
        scanner.unregister_root(id).unwrap();
        assert_eq!(scanner.retained_values(), Some(vec![]));
    }
}
