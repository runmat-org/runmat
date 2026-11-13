use std::collections::BTreeMap;
use std::sync::Arc;

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
    authority: Arc<()>,
}

/// A checked single-key observation, not a clone of the tracker.
pub struct PreparedDependencyObservation {
    authority: Arc<()>,
    key: DependencyKey,
    revision: String,
    baseline: Option<DependencyGeneration>,
    baseline_revision: Option<String>,
    next: DependencyGeneration,
}

pub struct ReadyDependencyObservation<'a> {
    tracker: &'a mut DependencyTracker,
    prepared: PreparedDependencyObservation,
}

impl PreparedDependencyObservation {
    pub fn revision(&self) -> &str {
        &self.revision
    }

    pub fn admit_commit(
        self,
        tracker: &mut DependencyTracker,
    ) -> Result<ReadyDependencyObservation<'_>, &'static str> {
        if !Arc::ptr_eq(&self.authority, &tracker.authority)
            || tracker.generation(&self.key) != self.baseline
            || tracker
                .dependencies
                .get(&self.key)
                .map(|tracked| tracked.revision.as_str())
                != self.baseline_revision.as_deref()
        {
            return Err("dependency changed after observation preparation");
        }
        Ok(ReadyDependencyObservation {
            tracker,
            prepared: self,
        })
    }
}

impl ReadyDependencyObservation<'_> {
    pub fn commit(self) -> DependencyChange {
        apply_prepared(self.tracker, self.prepared)
    }

    pub fn commit_with_snapshot(self) -> (DependencyChange, DependencySnapshot) {
        let tracker = self.tracker;
        let change = apply_prepared(&mut *tracker, self.prepared);
        (change, tracker.snapshot_all())
    }
}

fn apply_prepared(
    tracker: &mut DependencyTracker,
    prepared: PreparedDependencyObservation,
) -> DependencyChange {
    let PreparedDependencyObservation {
        authority: _,
        key,
        revision,
        baseline,
        baseline_revision: _,
        next,
    } = prepared;
    match tracker.dependencies.get_mut(&key) {
        Some(tracked) if tracked.revision == revision => {}
        Some(tracked) => {
            tracked.revision = revision;
            tracked.generation = next;
        }
        None => {
            tracker.dependencies.insert(
                key.clone(),
                TrackedDependency {
                    revision,
                    generation: next,
                },
            );
        }
    }
    DependencyChange {
        key,
        previous: baseline,
        current: next,
    }
}

impl DependencyTracker {
    /// Observe an authoritative revision. Re-observing the same revision is a
    /// no-op; a different revision advances exactly that dependency.
    pub fn observe(
        &mut self,
        key: DependencyKey,
        revision: impl Into<String>,
    ) -> Result<DependencyChange, &'static str> {
        self.prepare_observation(key, revision)?
            .admit_commit(self)
            .map(ReadyDependencyObservation::commit)
    }

    pub fn prepare_observation(
        &self,
        key: DependencyKey,
        revision: impl Into<String>,
    ) -> Result<PreparedDependencyObservation, &'static str> {
        let revision = revision.into();
        if revision.is_empty() {
            return Err("dependency revision must not be empty");
        }
        let baseline = self.generation(&key);
        let baseline_revision = self
            .dependencies
            .get(&key)
            .map(|tracked| tracked.revision.clone());
        let next = match self.dependencies.get(&key) {
            Some(tracked) if tracked.revision == revision => tracked.generation,
            Some(tracked) => tracked
                .generation
                .next()
                .ok_or("dependency generation exhausted")?,
            None => DependencyGeneration::INITIAL,
        };
        Ok(PreparedDependencyObservation {
            authority: Arc::clone(&self.authority),
            key,
            revision,
            baseline,
            baseline_revision,
            next,
        })
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

    #[test]
    fn stale_prepared_observation_does_not_replace_newer_revision() {
        let mut tracker = DependencyTracker::default();
        let key = DependencyKey::Program("project".into());
        tracker.observe(key.clone(), "first").unwrap();
        let prepared = tracker.prepare_observation(key.clone(), "stale").unwrap();
        tracker.observe(key.clone(), "newer").unwrap();
        assert!(prepared.admit_commit(&mut tracker).is_err());
        assert_eq!(tracker.dependencies.get(&key).unwrap().revision, "newer");
        assert_eq!(tracker.generation(&key).unwrap().get(), 2);
    }

    #[test]
    fn prepared_observation_is_bound_to_its_tracker_across_moves() {
        let key = DependencyKey::Program("project".into());
        let mut owner = DependencyTracker::default();
        let mut other = DependencyTracker::default();
        owner.observe(key.clone(), "first").unwrap();
        other.observe(key.clone(), "first").unwrap();
        let prepared = owner.prepare_observation(key.clone(), "second").unwrap();
        assert!(prepared.admit_commit(&mut other).is_err());
        assert_eq!(other.dependencies.get(&key).unwrap().revision, "first");

        let prepared = owner.prepare_observation(key.clone(), "second").unwrap();
        let mut moved = owner;
        prepared.admit_commit(&mut moved).unwrap().commit();
        assert_eq!(moved.dependencies.get(&key).unwrap().revision, "second");
    }

    #[test]
    fn exhausted_generation_rejects_preparation_without_mutation() {
        let mut tracker = DependencyTracker::default();
        let key = DependencyKey::Program("project".into());
        tracker.dependencies.insert(
            key.clone(),
            TrackedDependency {
                revision: "current".into(),
                generation: DependencyGeneration::at_limit_for_test(),
            },
        );
        assert!(tracker.prepare_observation(key.clone(), "next").is_err());
        assert_eq!(tracker.dependencies.get(&key).unwrap().revision, "current");
        assert_eq!(tracker.generation(&key).unwrap().get(), u64::MAX);
    }
}
