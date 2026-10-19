//! Retention selects complete inventories; a study is never split into child runs.
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

#[derive(Clone, Copy, Debug)]
pub(in crate::analysis::storage) struct Limits {
    /// Zero means unlimited, matching the existing artifact retention policy.
    pub total: usize,
    pub per_kind: usize,
}

pub(in crate::analysis::storage) struct Unit<K> {
    pub id: String,
    pub counts: HashMap<K, usize>,
}

/// The incoming unit is required. Existing units must be ordered newest first.
/// Return only complete eviction identities, before the caller mutates storage.
pub(in crate::analysis::storage) fn evictions<K: Clone + Eq + Hash>(
    limits: Limits,
    incoming: &Unit<K>,
    existing: &[Unit<K>],
) -> Result<Vec<String>, String> {
    let mut total = inventory_size(incoming)?;
    let mut counts = incoming.counts.clone();
    if (limits.total > 0 && total > limits.total)
        || (limits.per_kind > 0 && counts.values().any(|n| *n > limits.per_kind))
    {
        return Err("complete publication exceeds artifact retention capacity".into());
    }
    let mut seen = HashSet::from([incoming.id.as_str()]);
    let mut remove = Vec::new();
    for unit in existing {
        if !seen.insert(&unit.id) {
            return Err("retention inventory contains duplicate units".into());
        }
        let next_total = total
            .checked_add(inventory_size(unit)?)
            .ok_or("retention inventory count overflow")?;
        let mut fits = limits.total == 0 || next_total <= limits.total;
        for (kind, size) in &unit.counts {
            let next = counts
                .get(kind)
                .copied()
                .unwrap_or(0)
                .checked_add(*size)
                .ok_or("retention kind count overflow")?;
            fits &= limits.per_kind == 0 || next <= limits.per_kind;
        }
        if fits {
            total = next_total;
            for (kind, size) in &unit.counts {
                *counts.entry(kind.clone()).or_default() += size;
            }
        } else {
            remove.push(unit.id.clone());
        }
    }
    Ok(remove)
}

fn inventory_size<K>(unit: &Unit<K>) -> Result<usize, String> {
    if unit.counts.is_empty() || unit.counts.values().any(|size| *size == 0) {
        return Err("retention unit has an empty inventory".into());
    }
    unit.counts.values().try_fold(0usize, |sum, size| {
        sum.checked_add(*size)
            .ok_or_else(|| "retention inventory count overflow".into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn unit(id: &str, counts: &[(&str, usize)]) -> Unit<String> {
        Unit {
            id: id.into(),
            counts: counts.iter().map(|(k, n)| ((*k).into(), *n)).collect(),
        }
    }
    #[test]
    fn mixed_kind_studies_are_evicted_as_whole_inventories() {
        let incoming = unit("new", &[("thermal", 2)]);
        let older = [
            unit("mixed", &[("thermal", 1), ("structural", 2)]),
            unit("structural", &[("structural", 1)]),
        ];
        assert_eq!(
            evictions(
                Limits {
                    total: 4,
                    per_kind: 2
                },
                &incoming,
                &older
            )
            .unwrap(),
            ["mixed"]
        );
        assert_eq!(
            evictions(
                Limits {
                    total: 5,
                    per_kind: 0
                },
                &incoming,
                &older
            )
            .unwrap(),
            ["structural"]
        );
    }
    #[test]
    fn required_publication_must_fit_without_splitting_or_eviction() {
        let incoming = unit("new", &[("thermal", 3), ("structural", 2)]);
        assert!(evictions(
            Limits {
                total: 4,
                per_kind: 0
            },
            &incoming,
            &[]
        )
        .is_err());
        assert!(evictions(
            Limits {
                total: 0,
                per_kind: 2
            },
            &incoming,
            &[]
        )
        .is_err());
        assert!(evictions(
            Limits {
                total: 0,
                per_kind: 0
            },
            &incoming,
            &[]
        )
        .unwrap()
        .is_empty());
    }
    #[test]
    fn malformed_or_overflowed_inventories_reject_before_selection() {
        let incoming = unit("new", &[("a", 1)]);
        assert!(evictions(
            Limits {
                total: 0,
                per_kind: 0
            },
            &incoming,
            &[unit("new", &[("a", 1)])]
        )
        .is_err());
        assert!(evictions(
            Limits {
                total: 0,
                per_kind: 0
            },
            &incoming,
            &[unit("old", &[])]
        )
        .is_err());
        assert!(evictions(
            Limits {
                total: 0,
                per_kind: 0
            },
            &incoming,
            &[unit("old", &[("a", usize::MAX)])]
        )
        .is_err());
    }
}
