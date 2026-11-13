//! Stable ranked beam retention. Payloads are constructed only for entries that
//! survive the existing width. A failed compact constructor preserves the beam.
use std::cmp::Ordering;

pub(super) struct RankedFrontier<T> {
    width: usize,
    entries: Vec<T>,
}

impl<T> RankedFrontier<T> {
    pub(super) fn new(width: usize) -> Self {
        Self {
            width,
            entries: Vec::with_capacity(width),
        }
    }

    #[cfg(test)]
    pub(super) fn retain_with<C, E>(
        &mut self,
        candidate: &C,
        compare: impl Fn(&C, &T) -> Ordering,
        create: impl FnOnce() -> Result<T, E>,
    ) -> Result<(), E> {
        // Insert after equal entries, matching stable sort followed by truncate.
        let position = self
            .entries
            .iter()
            .position(|entry| compare(candidate, entry) == Ordering::Less)
            .unwrap_or(self.entries.len());
        if position >= self.width {
            return Ok(());
        }
        let entry = create()?;
        if self.entries.len() == self.width {
            self.entries.pop();
        }
        self.entries.insert(position, entry);
        Ok(())
    }

    pub(super) fn into_entries(self) -> Vec<T> {
        self.entries
    }

    /// Retain the best representative of each equivalence class before ranking.
    /// At most the declared width survives, including replacement of a class.
    pub(super) fn retain_distinct_by(
        &mut self,
        candidate: T,
        compare: impl Fn(&T, &T) -> Ordering,
        same_class: impl Fn(&T, &T) -> bool,
    ) {
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| same_class(&candidate, entry))
        {
            if compare(&candidate, &self.entries[index]) != Ordering::Less {
                return;
            }
            self.entries.remove(index);
        }
        self.retain(candidate, compare);
    }

    /// Move an already constructed compact record without cloning its payload.
    pub(super) fn retain(&mut self, candidate: T, compare: impl Fn(&T, &T) -> Ordering) {
        let position = self
            .entries
            .iter()
            .position(|entry| compare(&candidate, entry) == Ordering::Less)
            .unwrap_or(self.entries.len());
        if position >= self.width {
            return;
        }
        if self.entries.len() == self.width {
            self.entries.pop();
        }
        self.entries.insert(position, candidate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn distinct_frontier_matches_complete_grouped_ranking_and_bounds_live_branches() {
        for width in [0, 1, 2, 8] {
            let candidates = (0..4096)
                .map(|ordinal| (ordinal % 17, ordinal * 13 % 31, ordinal))
                .collect::<Vec<_>>();
            let mut grouped = std::collections::BTreeMap::new();
            for candidate in &candidates {
                let entry = grouped.entry(candidate.0).or_insert(*candidate);
                if candidate.1 < entry.1 {
                    *entry = *candidate;
                }
            }
            let mut expected = grouped.into_values().collect::<Vec<_>>();
            expected.sort_by_key(|candidate| (candidate.1, candidate.2));
            expected.truncate(width);
            let mut frontier = RankedFrontier::new(width);
            for candidate in candidates {
                frontier.retain_distinct_by(
                    candidate,
                    |left, right| left.1.cmp(&right.1),
                    |left, right| left.0 == right.0,
                );
                assert!(frontier.entries.len() <= width);
                let classes = frontier
                    .entries
                    .iter()
                    .map(|entry| entry.0)
                    .collect::<std::collections::BTreeSet<_>>();
                assert_eq!(classes.len(), frontier.entries.len());
            }
            assert_eq!(frontier.into_entries(), expected);
        }
    }

    // witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: For 4096 ranked synthetic payloads, widths zero, one, two and eight exactly match
    //   independent stable full-sort/truncate selection including ties. Retained payload counts never
    //   exceed width; construction adds at most one compact trial. Eviction/destruction release
    //   entries, and constructor failure preserves the previous beam. This carrier witness does not
    //   measure native mesh payload bytes, search cost, whole-task peak or physical quality
    //   convergence.
    #[test]
    fn streamed_retention_matches_stable_full_sort_and_bounds_live_payloads() {
        for width in [0, 1, 2, 8] {
            let live = Cell::new(0usize);
            let peak = Cell::new(0usize);
            struct Payload<'a> {
                rank: u32,
                ordinal: usize,
                live: &'a Cell<usize>,
            }
            impl Drop for Payload<'_> {
                fn drop(&mut self) {
                    self.live.set(self.live.get() - 1);
                }
            }
            let candidates = (0..4096)
                .map(|ordinal| ((ordinal * 17 % 31) as u32, ordinal))
                .collect::<Vec<_>>();
            let mut expected = candidates.clone();
            expected.sort_by_key(|entry| entry.0);
            expected.truncate(width);
            let mut compact_frontier = RankedFrontier::new(width);
            for candidate in &candidates {
                compact_frontier.retain(*candidate, |left, right| left.0.cmp(&right.0));
            }
            assert_eq!(compact_frontier.into_entries(), expected);
            let mut frontier = RankedFrontier::new(width);
            for candidate in &candidates {
                frontier
                    .retain_with::<_, std::convert::Infallible>(
                        candidate,
                        |candidate, entry: &Payload<'_>| candidate.0.cmp(&entry.rank),
                        || {
                            live.set(live.get() + 1);
                            peak.set(peak.get().max(live.get()));
                            Ok(Payload {
                                rank: candidate.0,
                                ordinal: candidate.1,
                                live: &live,
                            })
                        },
                    )
                    .unwrap();
                assert!(live.get() <= width);
            }
            let entries = frontier.into_entries();
            assert_eq!(
                entries
                    .iter()
                    .map(|entry| (entry.rank, entry.ordinal))
                    .collect::<Vec<_>>(),
                expected
            );
            assert!(peak.get() <= width + 1);
            drop(entries);
            assert_eq!(live.get(), 0);
        }
        let mut frontier = RankedFrontier::new(1);
        frontier.retain(3_u32, |left, right| left.cmp(right));
        assert_eq!(
            frontier.retain_with(&1_u32, |left, right| left.cmp(right), || Err("failed")),
            Err("failed")
        );
        assert_eq!(frontier.into_entries(), vec![3]);
    }
}
