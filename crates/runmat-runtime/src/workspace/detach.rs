//! Exact synchronous retirement of current-workspace bindings before callbacks.

use std::collections::HashSet;

use runmat_value::Value;

use super::BaseWorkspaceError;

/// Select bindings from the *current* workspace, never a session-wide fallback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkspaceDetachSelection {
    All,
    Names(Vec<String>),
    Except(Vec<String>),
}

impl WorkspaceDetachSelection {
    pub fn select(&self, entries: Vec<(String, Value)>) -> Vec<(String, Value)> {
        match self {
            Self::All => entries,
            Self::Names(names) => {
                let mut available = entries
                    .into_iter()
                    .collect::<std::collections::HashMap<_, _>>();
                names
                    .iter()
                    .filter_map(|name| available.remove(name).map(|value| (name.clone(), value)))
                    .collect()
            }
            Self::Except(excluded) => {
                let excluded: HashSet<&str> = excluded.iter().map(String::as_str).collect();
                entries
                    .into_iter()
                    .filter(|(name, _)| !excluded.contains(name.as_str()))
                    .collect()
            }
        }
    }
}

/// Removed values plus every managed child root. This owner is created before
/// mutation and retained across any awaited cleanup callback.
#[derive(Debug)]
pub struct DetachedWorkspaceValues {
    entries: Vec<(String, Value)>,
    root: runmat_gc::ValueRootRegistration,
}

impl DetachedWorkspaceValues {
    pub fn new(entries: Vec<(String, Value)>) -> Result<Self, BaseWorkspaceError> {
        let values = entries.iter().map(|(_, value)| value.clone()).collect();
        let root = runmat_gc::ValueRootRegistration::new(Box::new(
            runmat_gc::roots::GlobalRoot::new(values, "detached workspace values".into()),
        ))?;
        Ok(Self { entries, root })
    }

    pub fn entries(&self) -> &[(String, Value)] {
        &self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Only a reachability audit that also retains this owner may exclude its
    /// registration; another root for the same handle remains visible.
    pub fn root_registration(&self) -> runmat_gc::RootId {
        self.root.id()
    }
}
