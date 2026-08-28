use std::collections::BTreeMap;

use thiserror::Error;

use super::{CoverageAggregate, CoverageFragment, CoverageSite};

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum CoverageMergeError {
    #[error("coverage fragments have different program revisions: '{expected}' and '{actual}'")]
    ProgramRevisionMismatch { expected: String, actual: String },
    #[error("coverage site '{site_id}' has conflicting definitions")]
    ConflictingSite { site_id: String },
    #[error("coverage counter key {counter_key} identifies both '{first}' and '{second}'")]
    CounterCollision {
        counter_key: u64,
        first: String,
        second: String,
    },
    #[error("coverage fragment references unknown counter key {counter_key}")]
    UnknownCounter { counter_key: u64 },
}

pub fn merge_coverage(
    fragments: impl IntoIterator<Item = CoverageFragment>,
) -> Result<CoverageAggregate, CoverageMergeError> {
    let mut program_revision: Option<String> = None;
    let mut sites = BTreeMap::<String, CoverageSite>::new();
    let mut keys = BTreeMap::<u64, String>::new();
    let mut counts = BTreeMap::<String, u64>::new();

    for fragment in fragments {
        match &program_revision {
            Some(expected) if expected != &fragment.program_revision => {
                return Err(CoverageMergeError::ProgramRevisionMismatch {
                    expected: expected.clone(),
                    actual: fragment.program_revision,
                });
            }
            None => program_revision = Some(fragment.program_revision.clone()),
            _ => {}
        }
        let fragment_keys = fragment
            .sites
            .iter()
            .map(|site| (site.counter_key, site.id.clone()))
            .collect::<BTreeMap<_, _>>();
        for site in fragment.sites {
            if let Some(first) = keys.insert(site.counter_key, site.id.clone()) {
                if first != site.id {
                    return Err(CoverageMergeError::CounterCollision {
                        counter_key: site.counter_key,
                        first,
                        second: site.id,
                    });
                }
            }
            if let Some(existing) = sites.get(&site.id) {
                if existing != &site {
                    return Err(CoverageMergeError::ConflictingSite { site_id: site.id });
                }
            } else {
                sites.insert(site.id.clone(), site);
            }
        }
        for (counter_key, count) in fragment.counts {
            let site_id = fragment_keys
                .get(&counter_key)
                .ok_or(CoverageMergeError::UnknownCounter { counter_key })?;
            let total = counts.entry(site_id.clone()).or_default();
            *total = total.saturating_add(count);
        }
    }

    Ok(CoverageAggregate {
        program_revision,
        sites: sites.into_values().collect(),
        counts,
    })
}

pub fn merge_aggregates(
    aggregates: impl IntoIterator<Item = CoverageAggregate>,
) -> Result<CoverageAggregate, CoverageMergeError> {
    let fragments = aggregates
        .into_iter()
        .filter_map(|aggregate| {
            let program_revision = aggregate.program_revision?;
            let counts = aggregate
                .sites
                .iter()
                .filter_map(|site| {
                    aggregate
                        .counts
                        .get(&site.id)
                        .copied()
                        .map(|count| (site.counter_key, count))
                })
                .collect();
            Some(CoverageFragment {
                program_revision,
                plan_revision: "merged".into(),
                sites: aggregate.sites,
                counts,
            })
        })
        .collect::<Vec<_>>();
    merge_coverage(fragments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage::CoverageMetric;

    fn site(id: &str, counter_key: u64) -> CoverageSite {
        CoverageSite {
            id: id.into(),
            counter_key,
            metric: CoverageMetric::Statement,
            owner_identity: "root".into(),
            relative_path: "src/example.m".into(),
            semantic_path: "example".into(),
            source_id: 0,
            start_byte: 0,
            end_byte: 1,
            start_line: 1,
            start_column: 1,
            end_line: 1,
            end_column: 2,
            instrumented: true,
            unsupported_reason: None,
        }
    }

    #[test]
    fn merging_is_worker_count_and_order_invariant() {
        let fragments = vec![
            CoverageFragment {
                program_revision: "program".into(),
                plan_revision: "unit-a".into(),
                sites: vec![site("a", 1)],
                counts: BTreeMap::from([(1, 2)]),
            },
            CoverageFragment {
                program_revision: "program".into(),
                plan_revision: "unit-a".into(),
                sites: vec![site("a", 1)],
                counts: BTreeMap::from([(1, 3)]),
            },
        ];
        let forward = merge_coverage(fragments.clone()).unwrap();
        let reverse = merge_coverage(fragments.into_iter().rev()).unwrap();
        assert_eq!(forward, reverse);
        assert_eq!(forward.counts.get("a"), Some(&5));
    }

    #[test]
    fn rejects_unknown_counters_and_collisions() {
        let unknown = CoverageFragment {
            program_revision: "program".into(),
            plan_revision: "unit".into(),
            sites: vec![site("a", 1)],
            counts: BTreeMap::from([(2, 1)]),
        };
        assert!(matches!(
            merge_coverage([unknown]),
            Err(CoverageMergeError::UnknownCounter { counter_key: 2 })
        ));

        let collision = [
            CoverageFragment {
                program_revision: "program".into(),
                plan_revision: "unit-a".into(),
                sites: vec![site("a", 1)],
                counts: BTreeMap::new(),
            },
            CoverageFragment {
                program_revision: "program".into(),
                plan_revision: "unit-b".into(),
                sites: vec![site("b", 1)],
                counts: BTreeMap::new(),
            },
        ];
        assert!(matches!(
            merge_coverage(collision),
            Err(CoverageMergeError::CounterCollision { .. })
        ));
    }
}
