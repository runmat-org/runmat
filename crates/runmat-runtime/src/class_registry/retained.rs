//! Retained values owned by the existing class registry, for GC and residency.
use super::*;

/// Complete current-thread registry roots unless another thread or an active
/// borrow makes alias visibility unavailable. `None` never means no roots.
pub fn visit_static_property_retained_values(
    visitor: &mut dyn FnMut(&Value) -> Option<()>,
) -> Option<()> {
    let current = std::thread::current().id();
    if STATIC_VALUE_THREADS
        .lock()
        .ok()?
        .iter()
        .any(|thread| *thread != current)
    {
        return None;
    }
    let mut result = Some(());
    visit_values(&mut |value| {
        if result.is_some() {
            result = visitor(value);
        }
    })?;
    result
}

pub(super) fn visit_values(visitor: &mut impl FnMut(&Value)) -> Option<()> {
    FALLBACK_STATE.with(|state| {
        let state = state.try_borrow().ok()?;
        visit_state(&state, visitor);
        Some(())
    })?;
    CONTEXT_STATES.with(|states| {
        let mut states = states.try_borrow_mut().ok()?;
        states.retain(|state| state.strong_count() > 0);
        for state in states.iter().filter_map(std::rc::Weak::upgrade) {
            let classes = state.classes.try_borrow().ok()?;
            visit_state(&classes, visitor);
        }
        Some(())
    })
}

fn visit_state(state: &RuntimeClassState, visitor: &mut impl FnMut(&Value)) {
    for value in state.static_values.values() {
        visitor(value);
    }
    for class in state.classes.values() {
        for value in class
            .properties
            .values()
            .filter_map(|property| property.default_value.as_ref())
        {
            visitor(value);
        }
    }
}

pub(super) fn note_default_roots(class: &RuntimeClass) {
    struct GcPresence(bool);
    impl Tracer for GcPresence {
        fn mark(&mut self, _: GcHandle) {
            self.0 = true;
        }
    }
    for value in class
        .properties
        .values()
        .filter_map(|property| property.default_value.as_ref())
    {
        let mut gc = GcPresence(false);
        value.trace(&mut gc);
        // A GC-owned default may acquire resident children later. Reuse the
        // registry's existing thread-presence guard for those identities too.
        if gc.0 || crate::dispatcher::value_contains_gpu(value) {
            ensure_gc_root_provider();
            mark_static_values_thread_active();
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_values_include_class_defaults_and_published_statics() {
        let mut state = RuntimeClassState::default();
        state.classes.clear();
        let property = RuntimeProperty {
            name: "payload".into(),
            is_static: false,
            is_constant: false,
            is_dependent: false,
            get_access: MemberAccess::Public,
            set_access: MemberAccess::Public,
            default_value: Some(Value::String("default".into())),
        };
        let class = RuntimeClass {
            events: Default::default(),
            name: "RetainedRoots".into(),
            parent: None,
            properties: [(property.name.clone(), property)].into(),
            methods: HashMap::new(),
        };
        state.classes.insert(class.name.clone(), class);
        state.static_values.insert(
            ("RetainedRoots".into(), "static".into()),
            Value::String("static".into()),
        );
        let mut values = Vec::new();
        visit_state(&state, &mut |value| values.push(value.clone()));
        assert_eq!(
            values,
            vec![
                Value::String("static".into()),
                Value::String("default".into())
            ]
        );
    }

    #[test]
    fn registry_borrow_conflict_is_unavailable_not_empty() {
        FALLBACK_STATE.with(|state| {
            let _borrow = state.borrow_mut();
            assert!(visit_values(&mut |_| panic!("borrowed registry is unavailable")).is_none());
        });
    }
}
