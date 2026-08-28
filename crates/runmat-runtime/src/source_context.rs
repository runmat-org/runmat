use runmat_thread_local::runmat_thread_local;
use runmat_types::SourceId;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct SourceInfo {
    pub source_id: Option<SourceId>,
    pub name: Arc<str>,
    pub fullpath_name: Option<Arc<str>>,
    pub text: Arc<str>,
}

runmat_thread_local! {
    static CURRENT_SOURCE: RefCell<Option<SourceInfo>> = const { RefCell::new(None) };
    static SOURCE_CATALOG: RefCell<HashMap<SourceId, SourceInfo>> = RefCell::new(HashMap::new());
}

pub struct SourceContextGuard {
    prev: Option<SourceInfo>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

impl Drop for SourceContextGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        if let Some(state) = &self.state {
            state.source.borrow_mut().current = prev;
        } else {
            CURRENT_SOURCE.with(|slot| {
                *slot.borrow_mut() = prev;
            });
        }
    }
}

pub struct SourceCatalogGuard {
    prev: HashMap<SourceId, SourceInfo>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

impl Drop for SourceCatalogGuard {
    fn drop(&mut self) {
        let prev = std::mem::take(&mut self.prev);
        if let Some(state) = &self.state {
            state.source.borrow_mut().catalog = prev;
        } else {
            SOURCE_CATALOG.with(|catalog| {
                *catalog.borrow_mut() = prev;
            });
        }
    }
}

/// Replace the current source text for this thread.
///
/// This is used for UX features like "show the original expression" in legends and for
/// diagnostics that need to slice the source by byte-span.
pub fn replace_current_source(source: Option<&str>) -> SourceContextGuard {
    replace_current_source_context(None, source)
}

pub fn replace_current_source_context(
    name: Option<&str>,
    source: Option<&str>,
) -> SourceContextGuard {
    let next = source.map(|text| SourceInfo {
        source_id: None,
        name: Arc::<str>::from(name.unwrap_or_default()),
        fullpath_name: None,
        text: Arc::<str>::from(text),
    });
    if let Some(state) = active_state() {
        let prev = std::mem::replace(&mut state.source.borrow_mut().current, next);
        SourceContextGuard {
            prev,
            state: Some(state),
        }
    } else {
        let prev = CURRENT_SOURCE.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), next));
        SourceContextGuard { prev, state: None }
    }
}

pub fn replace_current_source_id(source_id: Option<SourceId>) -> SourceContextGuard {
    if let Some(state) = active_state() {
        let next = source_id.and_then(|id| state.source.borrow().catalog.get(&id).cloned());
        let prev = std::mem::replace(&mut state.source.borrow_mut().current, next);
        SourceContextGuard {
            prev,
            state: Some(state),
        }
    } else {
        let next = source_id
            .and_then(|id| SOURCE_CATALOG.with(|catalog| catalog.borrow().get(&id).cloned()));
        let prev = CURRENT_SOURCE.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), next));
        SourceContextGuard { prev, state: None }
    }
}

pub fn replace_source_catalog(entries: Vec<(SourceId, String, String)>) -> SourceCatalogGuard {
    replace_source_catalog_with_fullpaths(
        entries
            .into_iter()
            .map(|(source_id, name, text)| (source_id, name, None, text))
            .collect(),
    )
}

pub fn replace_source_catalog_with_fullpaths(
    entries: Vec<(SourceId, String, Option<String>, String)>,
) -> SourceCatalogGuard {
    let next = entries
        .into_iter()
        .map(|(source_id, name, fullpath_name, text)| {
            (
                source_id,
                SourceInfo {
                    source_id: Some(source_id),
                    name: Arc::<str>::from(name),
                    fullpath_name: fullpath_name.map(Arc::<str>::from),
                    text: Arc::<str>::from(text),
                },
            )
        })
        .collect::<HashMap<_, _>>();
    if let Some(state) = active_state() {
        let prev = std::mem::replace(&mut state.source.borrow_mut().catalog, next);
        SourceCatalogGuard {
            prev,
            state: Some(state),
        }
    } else {
        let prev =
            SOURCE_CATALOG.with(|catalog| std::mem::replace(&mut *catalog.borrow_mut(), next));
        SourceCatalogGuard { prev, state: None }
    }
}

pub fn source_catalog_entries() -> Vec<(SourceId, String, String)> {
    source_catalog_entries_with_fullpaths()
        .into_iter()
        .map(|(source_id, name, _fullpath_name, text)| (source_id, name, text))
        .collect()
}

pub fn source_catalog_entries_with_fullpaths() -> Vec<(SourceId, String, Option<String>, String)> {
    if let Some(state) = active_state() {
        return catalog_entries(&state.source.borrow().catalog);
    }
    SOURCE_CATALOG.with(|catalog| catalog_entries(&catalog.borrow()))
}

pub fn current_source() -> Option<Arc<str>> {
    if let Some(state) = active_state() {
        return state
            .source
            .borrow()
            .current
            .as_ref()
            .map(|source| Arc::clone(&source.text));
    }
    CURRENT_SOURCE.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|source| Arc::clone(&source.text))
    })
}

pub fn current_source_info() -> Option<SourceInfo> {
    if let Some(state) = active_state() {
        return state.source.borrow().current.clone();
    }
    CURRENT_SOURCE.with(|slot| slot.borrow().clone())
}

pub fn source_info(source_id: SourceId) -> Option<SourceInfo> {
    if let Some(state) = active_state() {
        return state.source.borrow().catalog.get(&source_id).cloned();
    }
    SOURCE_CATALOG.with(|catalog| catalog.borrow().get(&source_id).cloned())
}

/// Seed a newly-created standalone runtime context from the ambient source
/// state that preceded it.
///
/// Embedders normally install source state directly on an active context. The
/// standalone VM compatibility boundary creates that context after callers
/// have installed a legacy catalog, so it must transfer the catalog and
/// current source exactly once before entering the context scope.
pub fn inherit_legacy_source_context(context: &crate::context::RuntimeContext) {
    if active_state().is_some() {
        return;
    }
    let current = CURRENT_SOURCE.with(|slot| slot.borrow().clone());
    let catalog = SOURCE_CATALOG.with(|catalog| catalog.borrow().clone());
    let mut state = context.state().source.borrow_mut();
    state.current = current;
    state.catalog = catalog;
}

fn active_state() -> Option<std::rc::Rc<crate::context::RuntimeContextState>> {
    crate::context::legacy::active().map(|context| std::rc::Rc::clone(context.state()))
}

fn catalog_entries(
    catalog: &HashMap<SourceId, SourceInfo>,
) -> Vec<(SourceId, String, Option<String>, String)> {
    catalog
        .iter()
        .map(|(source_id, source)| {
            (
                *source_id,
                source.name.to_string(),
                source.fullpath_name.as_ref().map(ToString::to_string),
                source.text.to_string(),
            )
        })
        .collect()
}
