use super::*;
use crate::{OBJECT_SUBSASGN_METHOD, OBJECT_SUBSREF_METHOD};
use runmat_gc::{GcRoot, RootId, Trace, Tracer};
use runmat_types::MemberAccess;
use std::cell::RefCell;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};

pub(super) static NEXT_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    pub(super) static MAP_REGISTRY: RefCell<HashMap<u64, MapStore>> = RefCell::new(HashMap::new());
    static MAP_ROOT_STATE: RefCell<Option<MapRootState>> = const { RefCell::new(None) };
}

static CONTAINERS_MAP_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(CLASS_IDENTITY);

struct MapRootState {
    root_id: RootId,
    active: Arc<AtomicBool>,
}

struct MapRegistryRoot {
    active: Arc<AtomicBool>,
}

impl GcRoot for MapRegistryRoot {
    fn scan(&self) -> Vec<GcHandle> {
        struct RootCollector {
            roots: Vec<GcHandle>,
        }

        impl Tracer for RootCollector {
            fn mark(&mut self, handle: GcHandle) {
                self.roots.push(handle);
            }
        }

        MAP_REGISTRY.with(|registry| {
            let registry = registry.borrow();
            let mut collector = RootCollector { roots: Vec::new() };
            for store in registry.values() {
                if let Some(storage) = store.storage {
                    collector.mark(storage);
                }
                for entry in &store.entries {
                    entry.key_value.trace(&mut collector);
                    entry.value.trace(&mut collector);
                }
            }
            collector.roots
        })
    }

    fn description(&self) -> String {
        "containers.Map registry values".to_string()
    }

    fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
            && MAP_REGISTRY.with(|registry| {
                registry
                    .try_borrow()
                    .map_or(true, |registry| !registry.is_empty())
            })
    }
}

impl runmat_gc::ValueRoot for MapRegistryRoot {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        MAP_REGISTRY.with(|registry| {
            let registry = registry.try_borrow().ok()?;
            for store in registry.values() {
                if let Some(storage) = store.storage {
                    runmat_gc::gc_with_value(&storage, &mut *visitor).ok()??;
                }
                for entry in &store.entries {
                    visitor(&entry.key_value)?;
                    visitor(&entry.value)?;
                }
            }
            Some(())
        })
    }
}

pub(super) fn ensure_map_registry_root_registered(builtin: &'static str) -> BuiltinResult<()> {
    MAP_ROOT_STATE.with(|state| {
        if state
            .borrow()
            .as_ref()
            .is_some_and(|state| state.active.load(Ordering::Acquire))
        {
            return Ok(());
        }

        let active = Arc::new(AtomicBool::new(true));
        let root_id = runmat_gc::gc_register_value_root(Box::new(MapRegistryRoot {
            active: Arc::clone(&active),
        }))
        .map_err(|e| {
            map_internal(
                format!("containers.Map: failed to register GC root: {e}"),
                builtin,
            )
        })?;
        *state.borrow_mut() = Some(MapRootState { root_id, active });
        Ok(())
    })
}

pub(super) fn deactivate_map_registry_root_if_empty() {
    let empty = MAP_REGISTRY.with(|registry| {
        registry
            .try_borrow()
            .map(|registry| registry.is_empty())
            .unwrap_or(false)
    });
    if empty {
        MAP_ROOT_STATE.with(|state| {
            if let Some(state) = state.borrow_mut().take() {
                state.active.store(false, Ordering::Release);
                if let Err(err) = runmat_gc::gc_unregister_root(state.root_id) {
                    log::warn!("containers.Map: failed to unregister empty registry root: {err}");
                }
            }
        });
    }
}

pub(super) fn ensure_containers_map_class_registered() {
    CONTAINERS_MAP_CLASS_REGISTERED.ensure(|| {
        let mut properties = HashMap::new();
        for name in ["Count", "KeyType", "ValueType"] {
            properties.insert(
                name.into(),
                crate::class_registry::RuntimeProperty {
                    name: name.into(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: true,
                    get_access: MemberAccess::Public,
                    set_access: MemberAccess::Private,
                    default_value: None,
                },
            );
        }

        let mut methods = HashMap::new();
        for (name, function_name) in [
            (runmat_types::StaticMethodName::new("keys"), BUILTIN_KEYS),
            (
                runmat_types::StaticMethodName::new("values"),
                BUILTIN_VALUES,
            ),
            (runmat_types::StaticMethodName::new("isKey"), BUILTIN_IS_KEY),
            (
                runmat_types::StaticMethodName::new("remove"),
                BUILTIN_REMOVE,
            ),
            (OBJECT_SUBSREF_METHOD, BUILTIN_SUBSREF),
            (OBJECT_SUBSASGN_METHOD, BUILTIN_SUBSASGN),
        ] {
            methods.insert(
                name.into(),
                crate::class_registry::RuntimeMethod {
                    name: name.into(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: function_name.into(),
                    implicit_class_argument: None,
                },
            );
        }

        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: CLASS_IDENTITY.into(),
            parent: None,
            properties,
            methods,
        });
    });
}
