use std::collections::BTreeMap;

use super::{DependencyGeneration, DependencyKey, DependencySnapshot};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependencyChange {
    pub key: DependencyKey,
    pub previous: Option<DependencyGeneration>,
    pub current: DependencyGeneration,
}

#[derive(Clone, Debug)]
struct TrackedDependency {
    revision: String,
    generation: DependencyGeneration,
}

/// Session-scoped revision-to-generation authority used by entry publication.
#[derive(Default)]
pub struct DependencyTracker {
    dependencies: BTreeMap<DependencyKey, TrackedDependency>,
}

impl DependencyTracker {
    /// Observe an authoritative revision. Re-observing the same revision is a
    /// no-op; a different revision advances exactly that dependency.
    pub fn observe(
        &mut self,
        key: DependencyKey,
        revision: impl Into<String>,
    ) -> Result<DependencyChange, &'static str> {
        let revision = revision.into();
        if revision.is_empty() {
            return Err("dependency revision must not be empty");
        }
        match self.dependencies.get_mut(&key) {
            Some(tracked) if tracked.revision == revision => Ok(DependencyChange {
                key,
                previous: Some(tracked.generation),
                current: tracked.generation,
            }),
            Some(tracked) => {
                let previous = tracked.generation;
                tracked.generation = previous.next().ok_or("dependency generation exhausted")?;
                tracked.revision = revision;
                Ok(DependencyChange {
                    key,
                    previous: Some(previous),
                    current: tracked.generation,
                })
            }
            None => {
                self.dependencies.insert(
                    key.clone(),
                    TrackedDependency {
                        revision,
                        generation: DependencyGeneration::INITIAL,
                    },
                );
                Ok(DependencyChange {
                    key,
                    previous: None,
                    current: DependencyGeneration::INITIAL,
                })
            }
        }
    }

    pub fn generation(&self, key: &DependencyKey) -> Option<DependencyGeneration> {
        self.dependencies.get(key).map(|tracked| tracked.generation)
    }

    pub fn snapshot<'a>(
        &self,
        keys: impl IntoIterator<Item = &'a DependencyKey>,
    ) -> DependencySnapshot {
        let mut snapshot = DependencySnapshot::default();
        for key in keys {
            if let Some(generation) = self.generation(key) {
                snapshot.insert(key.clone(), generation);
            }
        }
        snapshot
    }

    pub fn snapshot_all(&self) -> DependencySnapshot {
        self.snapshot(self.dependencies.keys())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advances_only_the_changed_dependency() {
        let mut tracker = DependencyTracker::default();
        let program = DependencyKey::Program("project".into());
        let catalog = DependencyKey::Catalog("builtins".into());
        assert_eq!(
            tracker.observe(program.clone(), "a").unwrap().current.get(),
            1
        );
        tracker.observe(catalog.clone(), "one").unwrap();
        assert_eq!(
            tracker.observe(program.clone(), "a").unwrap().current.get(),
            1
        );
        assert_eq!(
            tracker.observe(program.clone(), "b").unwrap().current.get(),
            2
        );
        assert_eq!(tracker.generation(&catalog).unwrap().get(), 1);
    }
}
