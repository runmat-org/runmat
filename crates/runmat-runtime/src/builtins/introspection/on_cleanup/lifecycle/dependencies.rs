//! Order cleanup admission by actual callback-to-cleanup-target reachability.

use runmat_gc::RegisteredRootReachability;

/// One Active registration in the same order as the GC graph source and target
/// vectors. The ordinal is issued by the owning session, not a handle address.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ActiveCleanupIdentity {
    pub(crate) ordinal: u64,
}

/// Mark independently admissible source components without allocation during
/// the GC root-publication guard. A callback's registered
/// root is artificial ownership of its own target but captures of *other*
/// cleanup targets impose an execution dependency. Queuing just one target
/// causes its root to become real pending work before the next audit.
///
/// Mutual capture cycles have no total peer-live order. Choose one member of
/// a source strongly connected component deterministically; its queued root
/// keeps the remaining members alive until it completes or is abandoned.
pub(crate) fn mark_admissible_batch(
    active: &[ActiveCleanupIdentity],
    graph: &RegisteredRootReachability,
    selected: &mut [bool],
) -> bool {
    if active.len() != graph.target_count()
        || active.len() != graph.source_count()
        || active.len() != selected.len()
    {
        return false;
    }
    // The graph is transitively closed. A predecessor SCC blocks admission;
    // one deterministic member of each source SCC is selected. Independent
    // singleton components all enter Queued in this one guarded snapshot.
    let mut any = false;
    for candidate in 0..active.len() {
        let externally_owned_component = (0..active.len()).any(|source| {
            graph.outside_reaches(source)
                && (source == candidate
                    || (graph.source_reaches(source, candidate)
                        && graph.source_reaches(candidate, source)))
        });
        if externally_owned_component {
            continue;
        }
        let has_predecessor = (0..active.len()).any(|source| {
            source != candidate
                && graph.source_reaches(source, candidate)
                && !graph.source_reaches(candidate, source)
        });
        let earlier_cycle_member = (0..active.len()).any(|source| {
            source != candidate
                && active[source].ordinal < active[candidate].ordinal
                && graph.source_reaches(source, candidate)
                && graph.source_reaches(candidate, source)
        });
        if !has_predecessor && !earlier_cycle_member {
            selected[candidate] = true;
            any = true;
        }
    }
    any
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identities(count: usize) -> Vec<ActiveCleanupIdentity> {
        (0..count)
            .map(|ordinal| ActiveCleanupIdentity {
                ordinal: ordinal as u64,
            })
            .collect()
    }

    #[test]
    fn dependency_precedes_captured_target_regardless_of_registration_order() {
        // B has the lower ordinal, but A's callback captures B.
        let active = identities(2);
        let graph = RegisteredRootReachability::from_rows(
            vec![false, false],
            vec![vec![false, false], vec![true, false]],
        )
        .unwrap();
        let mut selected = vec![false; 2];
        assert!(mark_admissible_batch(&active, &graph, &mut selected));
        assert_eq!(selected, vec![false, true]);
    }

    #[test]
    fn source_cycle_precedes_downstream_cycle() {
        let active = identities(4);
        // 2<->3 is a source cycle and reaches the earlier-ordinal 0<->1 cycle.
        let graph = RegisteredRootReachability::from_rows(
            vec![false; 4],
            vec![
                vec![true, true, false, false],
                vec![true, true, false, false],
                vec![true, true, true, true],
                vec![true, true, true, true],
            ],
        )
        .unwrap();
        let mut selected = vec![false; 4];
        assert!(mark_admissible_batch(&active, &graph, &mut selected));
        assert_eq!(selected, vec![false, false, true, false]);
    }

    #[test]
    fn external_alias_defers_target_even_with_no_active_dependency() {
        let active = identities(2);
        let graph = RegisteredRootReachability::from_rows(
            vec![true, false],
            vec![vec![false; 2], vec![false; 2]],
        )
        .unwrap();
        let mut selected = vec![false; 2];
        assert!(mark_admissible_batch(&active, &graph, &mut selected));
        assert_eq!(selected, vec![false, true]);
    }

    #[test]
    fn independent_sources_share_one_admission_snapshot() {
        let active = identities(4);
        let graph =
            RegisteredRootReachability::from_rows(vec![false; 4], vec![vec![false; 4]; 4]).unwrap();
        let mut selected = vec![false; 4];
        assert!(mark_admissible_batch(&active, &graph, &mut selected));
        assert_eq!(selected, vec![true; 4]);
    }

    #[test]
    fn externally_owned_callback_capture_blocks_unowned_peer() {
        let active = identities(2);
        let graph = RegisteredRootReachability::from_rows(
            vec![false, true],
            vec![vec![false, false], vec![true, false]],
        )
        .unwrap();
        let mut selected = vec![false; 2];
        assert!(!mark_admissible_batch(&active, &graph, &mut selected));
        assert_eq!(selected, vec![false; 2]);
    }

    #[test]
    fn externally_owned_cycle_blocks_every_member() {
        let active = identities(2);
        let graph = RegisteredRootReachability::from_rows(
            vec![false, true],
            vec![vec![true, true], vec![true, true]],
        )
        .unwrap();
        let mut selected = vec![false; 2];
        assert!(!mark_admissible_batch(&active, &graph, &mut selected));
        assert_eq!(selected, vec![false; 2]);
    }
}
