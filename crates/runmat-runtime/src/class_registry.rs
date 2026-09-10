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
use runmat_types::{
    standard, CallableFallbackPolicy, CallableIdentity, ClassIdentity, MemberAccess, MemberName,
    MethodName, QualifiedName, StaticClassIdentity, SymbolName,
};
use runmat_value::Value;

#[derive(Debug, Clone)]
pub struct RuntimeProperty {
    pub name: MemberName,
    pub is_static: bool,
    pub is_constant: bool,
    pub is_dependent: bool,
    pub get_access: MemberAccess,
    pub set_access: MemberAccess,
    pub default_value: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct RuntimeMethod {
    pub name: MethodName,
    pub is_static: bool,
    pub is_abstract: bool,
    pub is_sealed: bool,
    pub access: MemberAccess,
    /// Declaration-time spelling resolved into `BoundRuntimeMethod::callable`
    /// during class registration. Execution never reinterprets this string.
    pub function_name: String,
    pub implicit_class_argument: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMethodCallingConvention {
    Direct,
    StandardSubstruct,
}

#[derive(Debug, Clone)]
pub struct BoundRuntimeMethod {
    pub registry_generation: u64,
    pub declaration: RuntimeMethod,
    pub declaring_class: ClassIdentity,
    pub callable: CallableIdentity,
    pub fallback: CallableFallbackPolicy,
    pub convention: RuntimeMethodCallingConvention,
}

#[derive(Debug, Clone)]
pub struct RuntimeClass {
    pub name: ClassIdentity,
    pub parent: Option<ClassIdentity>,
    pub properties: HashMap<MemberName, RuntimeProperty>,
    pub methods: HashMap<MethodName, RuntimeMethod>,
}

/// Session-aware registration check for lazily installed builtin classes.
///
/// Unlike a thread-local boolean, this consults the active runtime context's
/// class registry, so two interleaved sessions on one native or WASM thread
/// cannot cause one another to skip registration.
pub struct ClassRegistration {
    class_name: StaticClassIdentity,
}

impl ClassRegistration {
    pub const fn new(class_name: StaticClassIdentity) -> Self {
        Self { class_name }
    }

    fn get(&self) -> bool {
        with_state(|state| state.registrations.contains(&self.class_name))
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

#[derive(Debug, Clone)]
pub(crate) struct RuntimeClassState {
    generation: u64,
    classes: HashMap<ClassIdentity, RuntimeClass>,
    method_bindings: HashMap<(ClassIdentity, MethodName), BoundRuntimeMethod>,
    sealed: HashSet<ClassIdentity>,
    abstract_classes: HashSet<ClassIdentity>,
    static_values: HashMap<(ClassIdentity, String), Value>,
    enumerations: HashMap<ClassIdentity, HashSet<String>>,
    registrations: HashSet<StaticClassIdentity>,
}

impl Default for RuntimeClassState {
    fn default() -> Self {
        let classes = primitive_class_registry();
        let method_bindings = bind_class_methods(&classes, 1);
        Self {
            generation: 1,
            classes,
            method_bindings,
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

fn primitive_class_registry() -> HashMap<ClassIdentity, RuntimeClass> {
    let mut registry = HashMap::new();
    for class_name in [
        standard::DOUBLE,
        standard::SINGLE,
        standard::LOGICAL,
        standard::INT8,
        standard::INT16,
        standard::INT32,
        standard::INT64,
        standard::UINT8,
        standard::UINT16,
        standard::UINT32,
        standard::UINT64,
    ] {
        let method = RuntimeMethod {
            name: "zeros".into(),
            is_static: true,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: "zeros".into(),
            implicit_class_argument: Some(class_name.display_name().to_owned()),
        };
        registry.insert(
            class_name.owned(),
            RuntimeClass {
                name: class_name.owned(),
                parent: None,
                properties: HashMap::new(),
                methods: HashMap::from([("zeros".into(), method)]),
            },
        );
    }
    for (name, parent) in [
        (standard::HANDLE, None),
        (standard::DYNAMIC_PROPERTIES, Some(standard::HANDLE)),
        (standard::METADATA_PROPERTY, None),
        (standard::METADATA_DYNAMIC_PROPERTY, Some(standard::HANDLE)),
    ] {
        registry.insert(
            name.owned(),
            RuntimeClass {
                name: name.owned(),
                parent: parent.map(StaticClassIdentity::owned),
                properties: HashMap::new(),
                methods: HashMap::new(),
            },
        );
    }
    let gpu_array = runmat_builtins::standard_class_declaration(&standard::GPU_ARRAY.owned())
        .expect("gpuArray standard class declaration");
    let methods = gpu_array
        .methods
        .into_iter()
        .map(|method| {
            let function_name = match method.callable {
                runmat_types::CallableIdentity::Builtin(id) => id.0,
                runmat_types::CallableIdentity::ExternalName(name) => name
                    .display_name()
                    .expect("standard class method has a qualified callable identity"),
                identity => panic!(
                    "standard gpuArray method '{}' has unsupported callable identity {identity:?}",
                    method.name
                ),
            };
            let name = method.name;
            (
                name.clone(),
                RuntimeMethod {
                    name,
                    is_static: method.is_static,
                    is_abstract: method.attributes.is_abstract,
                    is_sealed: method.attributes.is_sealed,
                    access: method.attributes.access,
                    function_name,
                    implicit_class_argument: method.implicit_class_argument,
                },
            )
        })
        .collect();
    registry.insert(
        standard::GPU_ARRAY.owned(),
        RuntimeClass {
            name: standard::GPU_ARRAY.owned(),
            parent: None,
            properties: HashMap::new(),
            methods,
        },
    );
    let dynamic_property_class = ClassIdentity::from("matlab.metadata.DynamicProperty");
    if let Some(class) = registry.get_mut(&dynamic_property_class) {
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
        state.generation = state.generation.wrapping_add(1).max(1);
        let bindings = bind_methods_for_class(&def, state.generation).collect::<Vec<_>>();
        state
            .method_bindings
            .retain(|(owner, _), _| owner != &class_name);
        state.method_bindings.extend(bindings);
        state.classes.insert(class_name.clone(), def);
        set_membership(&mut state.sealed, &class_name, is_sealed);
        set_membership(&mut state.abstract_classes, &class_name, is_abstract);
        state.enumerations.entry(class_name).or_default();
    });
}

pub fn class_registry_generation() -> u64 {
    with_state(|state| state.generation)
}

/// Resolve a legacy execution-frame spelling to its registered typed method.
/// This is the only boundary where declaration text is interpreted as caller
/// identity; protocol execution uses the returned identities exclusively.
pub fn caller_method_for_function(function_name: &str) -> Option<(ClassIdentity, MethodName)> {
    with_state(|state| {
        let mut matches = state
            .method_bindings
            .iter()
            .filter(|(_, bound)| bound.declaration.function_name == function_name)
            .map(|((class, method), _)| (class.clone(), method.clone()));
        let unique = matches.next()?;
        matches.next().is_none().then_some(unique)
    })
}

fn set_membership(registry: &mut HashSet<ClassIdentity>, name: &ClassIdentity, present: bool) {
    if present {
        registry.insert(name.clone());
    } else {
        registry.remove(name);
    }
}

pub fn register_class_enumerations(
    class_name: &ClassIdentity,
    members: impl IntoIterator<Item = String>,
) {
    with_state_mut(|state| {
        let entry = state.enumerations.entry(class_name.clone()).or_default();
        entry.clear();
        entry.extend(members);
    });
}

pub fn class_has_enumeration_member(class_name: &ClassIdentity, member: &str) -> bool {
    with_state(|state| {
        state
            .enumerations
            .get(class_name)
            .is_some_and(|members| members.contains(member))
    })
}

pub fn get_class(name: &ClassIdentity) -> Option<RuntimeClass> {
    with_state(|state| state.classes.get(name).cloned())
}

pub fn class_names() -> Vec<ClassIdentity> {
    with_state(|state| state.classes.keys().cloned().collect())
}

/// Resolve the declaring class context for a runtime function name. Executors
/// use this shared lookup to apply identical private/protected member access.
pub fn class_context_for_function(function_name: &str) -> Option<ClassIdentity> {
    caller_method_for_function(function_name).map(|(class, _)| class)
}

pub fn is_class_sealed(name: &ClassIdentity) -> bool {
    with_state(|state| state.sealed.contains(name))
}

pub fn is_class_abstract(name: &ClassIdentity) -> bool {
    with_state(|state| state.abstract_classes.contains(name))
}

pub fn is_class_or_subclass(class_name: &ClassIdentity, ancestor_name: &ClassIdentity) -> bool {
    if class_name == ancestor_name {
        return true;
    }
    with_state(|state| {
        let registry = &state.classes;
        let mut current = Some(class_name.clone());
        let mut visited = HashSet::new();
        while let Some(name) = current {
            if !visited.insert(name.clone()) {
                return false;
            }
            if &name == ancestor_name {
                return true;
            }
            current = registry.get(&name).and_then(|class| class.parent.clone());
        }
        false
    })
}

pub fn superclass_chain(class_name: &ClassIdentity) -> Option<Vec<ClassIdentity>> {
    with_state(|state| {
        let registry = &state.classes;
        if !registry.contains_key(class_name) {
            return None;
        }
        let mut current = registry
            .get(class_name)
            .and_then(|class| class.parent.clone());
        let mut visited = HashSet::from([class_name.clone()]);
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

pub fn lookup_property(
    class_name: &ClassIdentity,
    property: &MemberName,
) -> Option<(RuntimeProperty, ClassIdentity)> {
    lookup_member(class_name, |class| class.properties.get(property).cloned())
}

pub fn lookup_method(
    class_name: &ClassIdentity,
    method: &runmat_types::MethodName,
) -> Option<(RuntimeMethod, ClassIdentity)> {
    lookup_member(class_name, |class| class.methods.get(method).cloned())
}

pub fn lookup_bound_method(
    class_name: &ClassIdentity,
    method: &MethodName,
) -> Option<BoundRuntimeMethod> {
    with_state(|state| {
        let mut current = Some(class_name.clone());
        let mut visited = HashSet::new();
        while let Some(name) = current {
            if !visited.insert(name.clone()) {
                break;
            }
            let class = state.classes.get(&name)?;
            if let Some(binding) = state.method_bindings.get(&(name.clone(), method.clone())) {
                return Some(binding.clone());
            }
            current = class.parent.clone();
        }
        None
    })
}

fn bind_class_methods(
    classes: &HashMap<ClassIdentity, RuntimeClass>,
    generation: u64,
) -> HashMap<(ClassIdentity, MethodName), BoundRuntimeMethod> {
    classes
        .values()
        .flat_map(|class| bind_methods_for_class(class, generation))
        .collect()
}

fn bind_methods_for_class(
    class: &RuntimeClass,
    generation: u64,
) -> impl Iterator<Item = ((ClassIdentity, MethodName), BoundRuntimeMethod)> + '_ {
    class.methods.values().cloned().map(move |declaration| {
        let method = declaration.name.clone();
        let (callable, fallback) = resolve_method_callable(&class.name, &declaration);
        let convention = if crate::object::protocol::ObjectProtocol::from_method(&method)
            .is_some_and(crate::object::protocol::ObjectProtocol::uses_standard_substruct)
        {
            RuntimeMethodCallingConvention::StandardSubstruct
        } else {
            RuntimeMethodCallingConvention::Direct
        };
        (
            (class.name.clone(), method),
            BoundRuntimeMethod {
                registry_generation: generation,
                declaration,
                declaring_class: class.name.clone(),
                callable,
                fallback,
                convention,
            },
        )
    })
}

fn resolve_method_callable(
    owner: &ClassIdentity,
    method: &RuntimeMethod,
) -> (CallableIdentity, CallableFallbackPolicy) {
    let function_name = method.function_name.trim();
    if let Some(function) = crate::call::closures::resolve_method_semantic_function_id(
        owner,
        &method.name.0,
        function_name,
    ) {
        return (
            CallableIdentity::BoundFunction(runmat_types::FunctionId(function)),
            CallableFallbackPolicy::None,
        );
    }
    if let Some(callable) = runmat_builtins::builtin_callable_identity_by_name(function_name) {
        return (callable, CallableFallbackPolicy::None);
    }
    if function_name.is_empty() {
        return (
            crate::call::identity::external_qualified_identity(
                owner.display_name(),
                &method.name.0,
            ),
            CallableFallbackPolicy::ExternalBoundary,
        );
    }
    if function_name.contains('.') {
        return runtime_named_callable(function_name);
    }
    (
        crate::call::identity::external_qualified_identity(owner.display_name(), function_name),
        CallableFallbackPolicy::ExternalBoundary,
    )
}

fn runtime_named_callable(name: &str) -> (CallableIdentity, CallableFallbackPolicy) {
    if let Some(function) = crate::user_functions::resolve_semantic_function_by_name(name.trim()) {
        return (
            CallableIdentity::BoundFunction(runmat_types::FunctionId(function)),
            CallableFallbackPolicy::None,
        );
    }
    let segments: Vec<&str> = name.split('.').collect();
    if segments.len() > 1 && segments.iter().all(|segment| !segment.trim().is_empty()) {
        return (
            CallableIdentity::ExternalName(QualifiedName(
                segments
                    .into_iter()
                    .map(|segment| SymbolName(segment.trim().to_owned()))
                    .collect(),
            )),
            CallableFallbackPolicy::ExternalBoundary,
        );
    }
    (
        CallableIdentity::DynamicName(SymbolName(name.trim().to_owned())),
        CallableFallbackPolicy::RuntimeNameResolution,
    )
}

fn lookup_member<T>(
    class_name: &ClassIdentity,
    mut member: impl FnMut(&RuntimeClass) -> Option<T>,
) -> Option<(T, ClassIdentity)> {
    with_state(|state| {
        let registry = &state.classes;
        let mut current = Some(class_name.clone());
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

pub fn get_static_property_value(class_name: &ClassIdentity, property: &str) -> Option<Value> {
    with_state(|state| {
        state
            .static_values
            .get(&(class_name.clone(), property.to_owned()))
            .cloned()
    })
}

pub fn set_static_property_value(class_name: &ClassIdentity, property: &str, value: Value) {
    ensure_gc_root_provider();
    mark_static_values_thread_active();
    with_state_mut(|state| {
        state
            .static_values
            .insert((class_name.clone(), property.to_owned()), value);
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
    class_name: &ClassIdentity,
    property: &str,
    value: Value,
) -> Result<(), String> {
    let Some((_, owner)) = lookup_property(class_name, &MemberName::from(property)) else {
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

    fn unique_class_name(prefix: &str) -> ClassIdentity {
        ClassIdentity::from(format!(
            "{prefix}_{}",
            TEST_CLASS_COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn empty_class(name: ClassIdentity, parent: Option<ClassIdentity>) -> RuntimeClass {
        RuntimeClass {
            name,
            parent,
            properties: HashMap::new(),
            methods: HashMap::new(),
        }
    }

    fn class_with_method(name: ClassIdentity, method: &str, function: &str) -> RuntimeClass {
        let mut class = empty_class(name, None);
        let declaration = RuntimeMethod {
            name: method.into(),
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: function.into(),
            implicit_class_argument: None,
        };
        class.methods.insert(declaration.name.clone(), declaration);
        class
    }

    #[test]
    fn legacy_function_spelling_only_grants_an_unambiguous_typed_caller() {
        let function = format!(
            "shared_protocol_impl_{}",
            TEST_CLASS_COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let first = unique_class_name("caller_first");
        register_class(class_with_method(first.clone(), "subsref", &function));
        assert_eq!(
            caller_method_for_function(&function),
            Some((first, MethodName::from("subsref")))
        );

        let second = unique_class_name("caller_second");
        register_class(class_with_method(second, "subsref", &function));
        assert_eq!(caller_method_for_function(&function), None);
        assert_eq!(class_context_for_function(&function), None);
    }

    #[test]
    fn primitives_and_inheritance_are_session_local_and_resolvable() {
        let uint64 = runmat_types::standard::UINT64.owned();
        let dynamicprops = ClassIdentity::from("dynamicprops");
        let handle = runmat_types::standard::HANDLE.owned();
        assert_eq!(
            lookup_method(&uint64, &runmat_types::MethodName::from("zeros"))
                .unwrap()
                .0
                .function_name,
            "zeros"
        );
        assert!(is_class_or_subclass(&dynamicprops, &handle));
        assert_eq!(superclass_chain(&dynamicprops), Some(vec![handle]));
    }

    #[test]
    fn every_primitive_exposes_class_preserving_static_zeros_binding() {
        for name in [
            standard::DOUBLE,
            standard::SINGLE,
            standard::LOGICAL,
            standard::INT8,
            standard::INT16,
            standard::INT32,
            standard::INT64,
            standard::UINT8,
            standard::UINT16,
            standard::UINT32,
            standard::UINT64,
        ] {
            let identity = name.owned();
            let (method, owner) =
                lookup_method(&identity, &runmat_types::MethodName::from("zeros")).unwrap();
            assert_eq!(owner.display_name(), name.display_name());
            assert!(method.is_static);
            assert_eq!(method.function_name, "zeros");
            assert_eq!(
                method.implicit_class_argument.as_deref(),
                Some(name.display_name())
            );
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
        assert_eq!(
            superclass_chain(&ClassIdentity::from("missing-class")),
            None
        );

        let orphan = unique_class_name("orphan");
        let missing_parent = unique_class_name("missing_parent");
        register_class(empty_class(orphan.clone(), Some(missing_parent.clone())));
        assert_eq!(superclass_chain(&orphan), Some(vec![missing_parent]));

        let first = unique_class_name("cycle_first");
        let second = unique_class_name("cycle_second");
        register_class(empty_class(first.clone(), Some(second.clone())));
        register_class(empty_class(second.clone(), Some(first.clone())));
        assert_eq!(superclass_chain(&first), Some(vec![second]));
        assert!(!is_class_or_subclass(
            &first,
            &ClassIdentity::from("missing-ancestor")
        ));
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
        assert_eq!(
            lookup_method(&child, &runmat_types::MethodName::from("parentOnly"))
                .unwrap()
                .1,
            parent
        );
        assert_eq!(
            lookup_property(&child, &MemberName::from("parentFlag"))
                .unwrap()
                .0
                .name,
            MemberName::from("parentFlag")
        );

        let first = unique_class_name("lookup_cycle_first");
        let second = unique_class_name("lookup_cycle_second");
        register_class(empty_class(first.clone(), Some(second.clone())));
        register_class(empty_class(second, Some(first.clone())));
        assert!(lookup_method(&first, &runmat_types::MethodName::from("missing")).is_none());
        assert!(lookup_property(&first, &MemberName::from("missing")).is_none());
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
