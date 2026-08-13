//! Session-scoped class registrations, executable method bindings, and static values.
//!
//! Immutable class/member declarations belong to `runmat-types`. These records
//! are the live runtime projection: they may contain executable names and live
//! default/static values and therefore intentionally remain outside the static
//! declaration schema.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::Once;
use std::thread::ThreadId;

use runmat_gc_api::{GcHandle, Trace, Tracer};
use runmat_types::MemberAccess;
use runmat_value::Value;

#[derive(Debug, Clone)]
pub struct RuntimeProperty {
    pub name: String,
    pub is_static: bool,
    pub is_constant: bool,
    pub is_dependent: bool,
    pub get_access: MemberAccess,
    pub set_access: MemberAccess,
    pub default_value: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct RuntimeMethod {
    pub name: String,
    pub is_static: bool,
    pub is_abstract: bool,
    pub is_sealed: bool,
    pub access: MemberAccess,
    pub function_name: String,
    pub implicit_class_argument: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RuntimeClass {
    pub name: String,
    pub parent: Option<String>,
    pub properties: HashMap<String, RuntimeProperty>,
    pub methods: HashMap<String, RuntimeMethod>,
}

/// Session-aware registration check for lazily installed builtin classes.
///
/// Unlike a thread-local boolean, this consults the active runtime context's
/// class registry, so two interleaved sessions on one native or WASM thread
/// cannot cause one another to skip registration.
pub struct ClassRegistration {
    class_name: &'static str,
}

impl ClassRegistration {
    pub const fn new(class_name: &'static str) -> Self {
        Self { class_name }
    }

    fn get(&self) -> bool {
        with_state(|state| state.registrations.contains(self.class_name))
    }

    pub fn ensure(&self, register: impl FnOnce()) {
        if self.get() {
            return;
        }
        register();
        with_state_mut(|state| {
            state.registrations.insert(self.class_name);
        });
    }
}

thread_local! {
    static FALLBACK_STATE: RefCell<RuntimeClassState> = RefCell::new(RuntimeClassState::default());
    static CONTEXT_STATES: RefCell<Vec<std::rc::Weak<crate::context::RuntimeContextState>>> =
        const { RefCell::new(Vec::new()) };
    static STATIC_VALUE_THREAD_REGISTRATION: StaticValueThreadRegistration =
        const { StaticValueThreadRegistration };
}

#[derive(Debug)]
pub(crate) struct RuntimeClassState {
    classes: HashMap<String, RuntimeClass>,
    sealed: HashSet<String>,
    abstract_classes: HashSet<String>,
    static_values: HashMap<(String, String), Value>,
    enumerations: HashMap<String, HashSet<String>>,
    registrations: HashSet<&'static str>,
}

impl Default for RuntimeClassState {
    fn default() -> Self {
        Self {
            classes: primitive_class_registry(),
            sealed: HashSet::new(),
            abstract_classes: HashSet::new(),
            static_values: HashMap::new(),
            enumerations: HashMap::new(),
            registrations: HashSet::new(),
        }
    }
}

pub(crate) fn register_context_state(state: &std::rc::Rc<crate::context::RuntimeContextState>) {
    CONTEXT_STATES.with(|states| {
        let mut states = states.borrow_mut();
        states.retain(|state| state.strong_count() > 0);
        if !states
            .iter()
            .filter_map(std::rc::Weak::upgrade)
            .any(|existing| std::rc::Rc::ptr_eq(&existing, state))
        {
            states.push(std::rc::Rc::downgrade(state));
        }
    });
}

static STATIC_VALUE_THREADS: once_cell::sync::Lazy<Mutex<HashSet<ThreadId>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashSet::new()));

struct StaticValueThreadRegistration;

fn ensure_gc_root_provider() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        runmat_gc::register_external_root_provider(
            "runmat-runtime-class-statics",
            static_property_gc_roots,
            static_property_values_exist_on_other_threads,
        );
    });
}

impl Drop for StaticValueThreadRegistration {
    fn drop(&mut self) {
        if let Ok(mut threads) = STATIC_VALUE_THREADS.lock() {
            threads.remove(&std::thread::current().id());
        }
    }
}

fn mark_static_values_thread_active() {
    STATIC_VALUE_THREAD_REGISTRATION.with(|_| {});
    if let Ok(mut threads) = STATIC_VALUE_THREADS.lock() {
        threads.insert(std::thread::current().id());
    }
}

pub fn static_property_values_exist_on_other_threads() -> bool {
    let current = std::thread::current().id();
    STATIC_VALUE_THREADS
        .lock()
        .map(|threads| threads.iter().any(|thread_id| *thread_id != current))
        .unwrap_or(false)
}

pub fn static_property_gc_roots() -> Vec<GcHandle> {
    struct RootCollector(Vec<GcHandle>);
    impl Tracer for RootCollector {
        fn mark(&mut self, handle: GcHandle) {
            self.0.push(handle);
        }
    }
    let mut collector = RootCollector(Vec::new());
    FALLBACK_STATE.with(|state| {
        for value in state.borrow().static_values.values() {
            value.trace(&mut collector);
        }
    });
    CONTEXT_STATES.with(|states| {
        let mut states = states.borrow_mut();
        states.retain(|state| state.strong_count() > 0);
        for state in states.iter().filter_map(std::rc::Weak::upgrade) {
            for value in state.classes.borrow().static_values.values() {
                value.trace(&mut collector);
            }
        }
    });
    collector.0
}

fn primitive_class_registry() -> HashMap<String, RuntimeClass> {
    let mut registry = HashMap::new();
    for class_name in [
        "double", "single", "logical", "int8", "int16", "int32", "int64", "uint8", "uint16",
        "uint32", "uint64",
    ] {
        let method = RuntimeMethod {
            name: "zeros".into(),
            is_static: true,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: "zeros".into(),
            implicit_class_argument: Some(class_name.into()),
        };
        registry.insert(
            class_name.into(),
            RuntimeClass {
                name: class_name.into(),
                parent: None,
                properties: HashMap::new(),
                methods: HashMap::from([("zeros".into(), method)]),
            },
        );
    }
    for (name, parent) in [
        ("handle", None),
        ("dynamicprops", Some("handle")),
        ("matlab.metadata.Property", None),
        ("matlab.metadata.DynamicProperty", Some("handle")),
    ] {
        registry.insert(
            name.into(),
            RuntimeClass {
                name: name.into(),
                parent: parent.map(str::to_owned),
                properties: HashMap::new(),
                methods: HashMap::new(),
            },
        );
    }
    if let Some(class) = registry.get_mut("matlab.metadata.DynamicProperty") {
        class.methods.insert(
            "delete".into(),
            RuntimeMethod {
                name: "delete".into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: "matlab.metadata.DynamicProperty.delete".into(),
                implicit_class_argument: None,
            },
        );
    }
    registry
}

pub fn register_class(def: RuntimeClass) {
    register_class_with_modifiers(def, false, false);
}

pub fn register_class_with_sealed(def: RuntimeClass, is_sealed: bool) {
    register_class_with_modifiers(def, is_sealed, false);
}

pub fn register_class_with_modifiers(def: RuntimeClass, is_sealed: bool, is_abstract: bool) {
    let class_name = def.name.clone();
    with_state_mut(|state| {
        state.classes.insert(class_name.clone(), def);
        set_membership(&mut state.sealed, &class_name, is_sealed);
        set_membership(&mut state.abstract_classes, &class_name, is_abstract);
        state.enumerations.entry(class_name).or_default();
    });
}

fn set_membership(registry: &mut HashSet<String>, name: &str, present: bool) {
    if present {
        registry.insert(name.to_owned());
    } else {
        registry.remove(name);
    }
}

pub fn register_class_enumerations(class_name: &str, members: impl IntoIterator<Item = String>) {
    with_state_mut(|state| {
        let entry = state.enumerations.entry(class_name.to_owned()).or_default();
        entry.clear();
        entry.extend(members);
    });
}

pub fn class_has_enumeration_member(class_name: &str, member: &str) -> bool {
    with_state(|state| {
        state
            .enumerations
            .get(class_name)
            .is_some_and(|members| members.contains(member))
    })
}

pub fn get_class(name: &str) -> Option<RuntimeClass> {
    with_state(|state| state.classes.get(name).cloned())
}

pub fn class_names() -> Vec<String> {
    with_state(|state| state.classes.keys().cloned().collect())
}

pub fn is_class_sealed(name: &str) -> bool {
    with_state(|state| state.sealed.contains(name))
}

pub fn is_class_abstract(name: &str) -> bool {
    with_state(|state| state.abstract_classes.contains(name))
}

pub fn is_class_or_subclass(class_name: &str, ancestor_name: &str) -> bool {
    if class_name == ancestor_name {
        return true;
    }
    with_state(|state| {
        let registry = &state.classes;
        let mut current = Some(class_name.to_owned());
        let mut visited = HashSet::new();
        while let Some(name) = current {
            if !visited.insert(name.clone()) {
                return false;
            }
            if name == ancestor_name {
                return true;
            }
            current = registry.get(&name).and_then(|class| class.parent.clone());
        }
        false
    })
}

pub fn superclass_chain(class_name: &str) -> Option<Vec<String>> {
    with_state(|state| {
        let registry = &state.classes;
        if !registry.contains_key(class_name) {
            return None;
        }
        let mut current = registry
            .get(class_name)
            .and_then(|class| class.parent.clone());
        let mut visited = HashSet::from([class_name.to_owned()]);
        let mut result = Vec::new();
        while let Some(name) = current {
            if !visited.insert(name.clone()) {
                break;
            }
            result.push(name.clone());
            current = registry.get(&name).and_then(|class| class.parent.clone());
        }
        Some(result)
    })
}

pub fn lookup_property(class_name: &str, property: &str) -> Option<(RuntimeProperty, String)> {
    lookup_member(class_name, |class| class.properties.get(property).cloned())
}

pub fn lookup_method(class_name: &str, method: &str) -> Option<(RuntimeMethod, String)> {
    lookup_member(class_name, |class| class.methods.get(method).cloned())
}

fn lookup_member<T>(
    class_name: &str,
    mut member: impl FnMut(&RuntimeClass) -> Option<T>,
) -> Option<(T, String)> {
    with_state(|state| {
        let registry = &state.classes;
        let mut current = Some(class_name.to_owned());
        let mut visited = HashSet::new();
        while let Some(name) = current {
            if !visited.insert(name.clone()) {
                break;
            }
            let class = registry.get(&name)?;
            if let Some(value) = member(class) {
                return Some((value, name));
            }
            current = class.parent.clone();
        }
        None
    })
}

pub fn get_static_property_value(class_name: &str, property: &str) -> Option<Value> {
    with_state(|state| {
        state
            .static_values
            .get(&(class_name.to_owned(), property.to_owned()))
            .cloned()
    })
}

pub fn set_static_property_value(class_name: &str, property: &str, value: Value) {
    ensure_gc_root_provider();
    mark_static_values_thread_active();
    with_state_mut(|state| {
        state
            .static_values
            .insert((class_name.to_owned(), property.to_owned()), value);
    });
}

fn with_state<R>(callback: impl FnOnce(&RuntimeClassState) -> R) -> R {
    if let Some(context) = crate::context::legacy::active() {
        callback(&context.state().classes.borrow())
    } else {
        FALLBACK_STATE.with(|state| callback(&state.borrow()))
    }
}

fn with_state_mut<R>(callback: impl FnOnce(&mut RuntimeClassState) -> R) -> R {
    if let Some(context) = crate::context::legacy::active() {
        callback(&mut context.state().classes.borrow_mut())
    } else {
        FALLBACK_STATE.with(|state| callback(&mut state.borrow_mut()))
    }
}

pub fn set_static_property_value_in_owner(
    class_name: &str,
    property: &str,
    value: Value,
) -> Result<(), String> {
    let Some((_, owner)) = lookup_property(class_name, property) else {
        return Err(format!("Unknown static property '{class_name}.{property}'"));
    };
    set_static_property_value(&owner, property, value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_CLASS_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_class_name(prefix: &str) -> String {
        format!(
            "{prefix}_{}",
            TEST_CLASS_COUNTER.fetch_add(1, Ordering::Relaxed)
        )
    }

    fn empty_class(name: String, parent: Option<String>) -> RuntimeClass {
        RuntimeClass {
            name,
            parent,
            properties: HashMap::new(),
            methods: HashMap::new(),
        }
    }

    #[test]
    fn primitives_and_inheritance_are_session_local_and_resolvable() {
        assert_eq!(
            lookup_method("uint64", "zeros").unwrap().0.function_name,
            "zeros"
        );
        assert!(is_class_or_subclass("dynamicprops", "handle"));
        assert_eq!(
            superclass_chain("dynamicprops"),
            Some(vec!["handle".into()])
        );
    }

    #[test]
    fn every_primitive_exposes_class_preserving_static_zeros_binding() {
        for name in [
            "double", "single", "logical", "int8", "int16", "int32", "int64", "uint8", "uint16",
            "uint32", "uint64",
        ] {
            let (method, owner) = lookup_method(name, "zeros").unwrap();
            assert_eq!(owner, name);
            assert!(method.is_static);
            assert_eq!(method.function_name, "zeros");
            assert_eq!(method.implicit_class_argument.as_deref(), Some(name));
        }
    }

    #[test]
    fn superclass_chain_is_nearest_first_and_handles_missing_parents_and_cycles() {
        let grand = unique_class_name("grand");
        let parent = unique_class_name("parent");
        let child = unique_class_name("child");
        register_class(empty_class(grand.clone(), None));
        register_class(empty_class(parent.clone(), Some(grand.clone())));
        register_class(empty_class(child.clone(), Some(parent.clone())));
        assert_eq!(
            superclass_chain(&child),
            Some(vec![parent.clone(), grand.clone()])
        );
        assert_eq!(superclass_chain(&grand), Some(Vec::new()));
        assert_eq!(superclass_chain("missing-class"), None);

        let orphan = unique_class_name("orphan");
        let missing_parent = unique_class_name("missing_parent");
        register_class(empty_class(orphan.clone(), Some(missing_parent.clone())));
        assert_eq!(superclass_chain(&orphan), Some(vec![missing_parent]));

        let first = unique_class_name("cycle_first");
        let second = unique_class_name("cycle_second");
        register_class(empty_class(first.clone(), Some(second.clone())));
        register_class(empty_class(second.clone(), Some(first.clone())));
        assert_eq!(superclass_chain(&first), Some(vec![second]));
        assert!(!is_class_or_subclass(&first, "missing-ancestor"));
    }

    #[test]
    fn inherited_method_and_property_lookup_terminate_on_cycles() {
        let parent = unique_class_name("lookup_parent");
        let child = unique_class_name("lookup_child");
        let mut parent_class = empty_class(parent.clone(), None);
        parent_class.methods.insert(
            "parentOnly".into(),
            RuntimeMethod {
                name: "parentOnly".into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: "parentOnly_impl".into(),
                implicit_class_argument: None,
            },
        );
        parent_class.properties.insert(
            "parentFlag".into(),
            RuntimeProperty {
                name: "parentFlag".into(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: None,
            },
        );
        register_class(parent_class);
        register_class(empty_class(child.clone(), Some(parent.clone())));
        assert_eq!(lookup_method(&child, "parentOnly").unwrap().1, parent);
        assert_eq!(
            lookup_property(&child, "parentFlag").unwrap().0.name,
            "parentFlag"
        );

        let first = unique_class_name("lookup_cycle_first");
        let second = unique_class_name("lookup_cycle_second");
        register_class(empty_class(first.clone(), Some(second.clone())));
        register_class(empty_class(second, Some(first.clone())));
        assert!(lookup_method(&first, "missing").is_none());
        assert!(lookup_property(&first, "missing").is_none());
    }

    #[test]
    fn modifiers_enumerations_and_inherited_static_values_share_one_registry_owner() {
        let parent = unique_class_name("state_parent");
        let child = unique_class_name("state_child");
        let mut parent_class = empty_class(parent.clone(), None);
        parent_class.properties.insert(
            "Count".into(),
            RuntimeProperty {
                name: "Count".into(),
                is_static: true,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Private,
                default_value: Some(Value::Num(0.0)),
            },
        );
        register_class_with_modifiers(parent_class, true, true);
        register_class(empty_class(child.clone(), Some(parent.clone())));
        register_class_enumerations(&parent, ["Ready".to_owned(), "Done".to_owned()]);

        assert!(is_class_sealed(&parent));
        assert!(is_class_abstract(&parent));
        assert!(class_has_enumeration_member(&parent, "Ready"));
        assert!(!class_has_enumeration_member(&parent, "Missing"));
        set_static_property_value_in_owner(&child, "Count", Value::Num(7.0)).unwrap();
        assert_eq!(
            get_static_property_value(&parent, "Count"),
            Some(Value::Num(7.0))
        );
        assert_eq!(get_static_property_value(&child, "Count"), None);
        assert!(set_static_property_value_in_owner(&child, "Missing", Value::Num(1.0)).is_err());
    }
}
