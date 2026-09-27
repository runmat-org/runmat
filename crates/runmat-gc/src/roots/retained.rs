//! Value projection on the same registration that owns GC root liveness.
use super::*;

/// A GC root whose retained Values are visible for non-GC resource ownership.
/// The same lifetime and exclusive-access invariants as `GcRoot::scan` apply.
pub trait ValueRoot: GcRoot {
    fn retained_values(&self) -> Option<Vec<Value>>;
}

pub(super) enum RegisteredRoot {
    Opaque(Box<dyn GcRoot>),
    Values(Box<dyn ValueRoot>),
}

impl RegisteredRoot {
    pub(super) fn retained_values(&self) -> Option<Vec<Value>> {
        match self {
            Self::Opaque(_) => None,
            Self::Values(root) => root.retained_values(),
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
    fn retained_values(&self) -> Option<Vec<Value>> {
        // SAFETY: precisely the same registered-vector lifetime and exclusive
        // scan requirement as StackRoot::scan; registration owns both views.
        Some(unsafe { self.stack.as_ref() }.clone())
    }
}
impl ValueRoot for VariableArrayRoot {
    fn retained_values(&self) -> Option<Vec<Value>> {
        // SAFETY: same registered-vector invariant as VariableArrayRoot::scan.
        Some(unsafe { self.vars.as_ref() }.clone())
    }
}
impl ValueRoot for GlobalRoot {
    fn retained_values(&self) -> Option<Vec<Value>> {
        Some(self.values.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
