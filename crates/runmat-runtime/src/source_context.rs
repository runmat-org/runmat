use runmat_hir::SourceId;
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct SourceInfo {
    pub source_id: Option<SourceId>,
    pub name: Arc<str>,
    pub text: Arc<str>,
}

runmat_thread_local! {
    static CURRENT_SOURCE: RefCell<Option<SourceInfo>> = const { RefCell::new(None) };
    static SOURCE_CATALOG: RefCell<HashMap<SourceId, SourceInfo>> = RefCell::new(HashMap::new());
}

pub struct SourceContextGuard {
    prev: Option<SourceInfo>,
}

impl Drop for SourceContextGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        CURRENT_SOURCE.with(|slot| {
            *slot.borrow_mut() = prev;
        });
    }
}

pub struct SourceCatalogGuard {
    prev: HashMap<SourceId, SourceInfo>,
}

impl Drop for SourceCatalogGuard {
    fn drop(&mut self) {
        let prev = std::mem::take(&mut self.prev);
        SOURCE_CATALOG.with(|catalog| {
            *catalog.borrow_mut() = prev;
        });
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
        text: Arc::<str>::from(text),
    });
    let prev = CURRENT_SOURCE.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), next));
    SourceContextGuard { prev }
}

pub fn replace_current_source_id(source_id: Option<SourceId>) -> SourceContextGuard {
    let next =
        source_id.and_then(|id| SOURCE_CATALOG.with(|catalog| catalog.borrow().get(&id).cloned()));
    let prev = CURRENT_SOURCE.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), next));
    SourceContextGuard { prev }
}

pub fn replace_source_catalog(entries: Vec<(SourceId, String, String)>) -> SourceCatalogGuard {
    let next = entries
        .into_iter()
        .map(|(source_id, name, text)| {
            (
                source_id,
                SourceInfo {
                    source_id: Some(source_id),
                    name: Arc::<str>::from(name),
                    text: Arc::<str>::from(text),
                },
            )
        })
        .collect::<HashMap<_, _>>();
    let prev = SOURCE_CATALOG.with(|catalog| std::mem::replace(&mut *catalog.borrow_mut(), next));
    SourceCatalogGuard { prev }
}

pub fn current_source() -> Option<Arc<str>> {
    CURRENT_SOURCE.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|source| Arc::clone(&source.text))
    })
}

pub fn current_source_info() -> Option<SourceInfo> {
    CURRENT_SOURCE.with(|slot| slot.borrow().clone())
}

pub fn source_info(source_id: SourceId) -> Option<SourceInfo> {
    SOURCE_CATALOG.with(|catalog| catalog.borrow().get(&source_id).cloned())
}
