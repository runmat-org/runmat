use runmat_hir::SourceId;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(crate) struct SourceText {
    pub(crate) name: Arc<str>,
    pub(crate) fullpath_name: Option<Arc<str>>,
    pub(crate) text: Arc<str>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct SourceKey {
    pub(crate) name: Arc<str>,
    pub(crate) fullpath_name: Option<Arc<str>>,
    pub(crate) text: Arc<str>,
}

#[derive(Default)]
pub(crate) struct SourcePool {
    sources: Vec<SourceText>,
    index: HashMap<SourceKey, SourceId>,
}

impl SourcePool {
    pub(crate) fn intern_with_fullpath(
        &mut self,
        name: &str,
        fullpath_name: Option<&str>,
        text: &str,
    ) -> SourceId {
        let name: Arc<str> = Arc::from(name);
        let fullpath_name = fullpath_name.map(Arc::<str>::from);
        let text: Arc<str> = Arc::from(text);
        let key = SourceKey {
            name: Arc::clone(&name),
            fullpath_name: fullpath_name.as_ref().map(Arc::clone),
            text: Arc::clone(&text),
        };
        if let Some(id) = self.index.get(&key) {
            return *id;
        }
        let id = SourceId(self.sources.len());
        self.sources.push(SourceText {
            name,
            fullpath_name,
            text,
        });
        self.index.insert(key, id);
        id
    }

    pub(crate) fn get(&self, id: SourceId) -> Option<&SourceText> {
        self.sources.get(id.0)
    }

    pub(crate) fn entries(&self) -> impl Iterator<Item = (SourceId, &SourceText)> {
        self.sources
            .iter()
            .enumerate()
            .map(|(index, source)| (SourceId(index), source))
    }
}

pub(crate) fn line_col_from_offset(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1;
    let mut line_start = 0;
    for (idx, ch) in source.char_indices() {
        if idx >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            line_start = idx + 1;
        }
    }
    let col = offset.saturating_sub(line_start) + 1;
    (line, col)
}
