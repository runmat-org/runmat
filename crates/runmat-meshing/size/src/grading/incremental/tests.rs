use super::*;

#[derive(Clone)]
struct Graph {
    raw: BTreeMap<usize, f64>,
    edges: BTreeMap<usize, BTreeSet<usize>>,
}
fn full(graph: &Graph, ratio: f64) -> BTreeMap<usize, f64> {
    let keys = graph.raw.keys().copied().collect::<Vec<_>>();
    let positions = keys
        .iter()
        .enumerate()
        .map(|(i, key)| (*key, i))
        .collect::<BTreeMap<_, _>>();
    let mut offsets = vec![0];
    let mut columns = vec![];
    for key in &keys {
        columns.extend(graph.edges[key].iter().map(|key| positions[key]));
        offsets.push(columns.len());
    }
    let mut result = graph.raw.clone();
    for (index, cap) in
        crate::grading::indexed::plan(
            ratio,
            &keys,
            &offsets,
            &columns,
            |i| Ok(graph.raw[&keys[i]]),
        )
        .unwrap()
    {
        result.insert(keys[index], cap);
    }
    result
}
struct Edit<'a> {
    old: &'a Graph,
    new: &'a Graph,
    caps: BTreeMap<usize, f64>,
}
impl MetricCapGraph<usize> for Edit<'_> {
    fn previous_cap(&self, key: usize) -> Option<f64> {
        self.caps.get(&key).copied()
    }
    fn raw_length(&self, key: usize) -> Result<Option<f64>, MetricContractError> {
        Ok(self.new.raw.get(&key).copied())
    }
    fn neighbors(&self, key: usize, previous: bool) -> Vec<usize> {
        let graph = if previous { self.old } else { self.new };
        graph
            .edges
            .get(&key)
            .into_iter()
            .flatten()
            .copied()
            .collect()
    }
}
fn changes(old: &Graph, new: &Graph) -> BTreeSet<usize> {
    old.raw
        .keys()
        .chain(new.raw.keys())
        .copied()
        .filter(|key| {
            old.raw.get(key) != new.raw.get(key) || old.edges.get(key) != new.edges.get(key)
        })
        .collect()
}
fn check(old: &Graph, new: &Graph, ratio: f64) -> MetricCapUpdate<usize> {
    let caps = full(old, ratio);
    let edit = Edit {
        old,
        new,
        caps: caps.clone(),
    };
    let plan = plan_metric_cap_update(ratio, &edit, changes(old, new), || false).unwrap();
    let mut observed = caps;
    observed.retain(|key, _| new.raw.contains_key(key));
    observed.extend(plan.caps.iter());
    assert_eq!(observed, full(new, ratio));
    plan
}
fn edge(graph: &mut Graph, a: usize, b: usize) {
    if !graph.edges.get_mut(&a).unwrap().remove(&b) {
        graph.edges.get_mut(&a).unwrap().insert(b);
        graph.edges.get_mut(&b).unwrap().insert(a);
    } else {
        graph.edges.get_mut(&b).unwrap().remove(&a);
    }
}
fn chain(count: usize) -> Graph {
    let mut graph = Graph {
        raw: (0..count)
            .map(|i| (i, if i == 0 { 1. } else { 10. }))
            .collect(),
        edges: (0..count).map(|i| (i, BTreeSet::new())).collect(),
    };
    for i in 1..count {
        edge(&mut graph, i - 1, i);
    }
    graph
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Graph edits compare exact caps with the complete indexed grading oracle across
//   cycles, ties, raw-length increases/decreases, removal/reinsertion and ratios from one through
//   the largest finite f64. The fixture family does not establish complete native mesh performance
//   or original-study acceptance.
#[test]
fn graph_edits_match_full_reference_across_cycles_ties_increases_and_deletions() {
    let mut random = 17_u64;
    let mut next = || {
        random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
        random as usize
    };
    for count in [2, 5, 13] {
        for ratio in [1., 1.01, 1.5, 2., f64::MAX] {
            let mut old = chain(count);
            for _ in 0..80 {
                let mut new = old.clone();
                let a = next() % count;
                let b = next() % count;
                if a != b {
                    edge(&mut new, a, b);
                }
                new.raw
                    .insert(a, [0.1, 0.2, 0.3, 1., 2., 4., 10.][next() % 7]);
                check(&old, &new, ratio);
                old = new;
            }
            let mut new = old.clone();
            let removed = count / 2;
            new.raw.remove(&removed);
            for neighbor in new.edges.remove(&removed).unwrap() {
                new.edges.get_mut(&neighbor).unwrap().remove(&removed);
            }
            check(&old, &new, ratio);
            check(&new, &old, ratio);
        }
    }
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A fixed source-cap increase on connected chains of 16, 128 and4096 vertices retains
//   at most five cap entries and reads fewer than40 neighbors while matching complete grading.
//   Actual dependency changes can legitimately propagate across the full graph; this is no
//   constant-work guarantee for arbitrary edits.
#[test]
fn changed_dependency_closure_does_not_scan_unrelated_connected_inventory() {
    for count in [16, 128, 4096] {
        let old = chain(count);
        let mut new = old.clone();
        new.raw.insert(0, 2.);
        let plan = check(&old, &new, 2.);
        assert!(plan.invalidated_vertices <= 5);
        assert!(
            plan.neighbor_reads < 40,
            "count={count}, reads={}",
            plan.neighbor_reads
        );
        assert!(plan.caps.len() <= 5);
    }
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Cancellation interrupts a borrowed cap-update plan without modifying its admitted
//   graph; invalid grading policy rejects, and an empty edit performs zero neighbor reads. Caller
//   topology admission and atomic application remain separate contracts.
#[test]
fn cancellation_and_invalid_policy_leave_the_graph_unmodified() {
    let graph = chain(8);
    let caps = full(&graph, 1.5);
    let edit = Edit {
        old: &graph,
        new: &graph,
        caps: caps.clone(),
    };
    let mut polls = 0;
    assert_eq!(
        plan_metric_cap_update(1.5, &edit, [0], || {
            polls += 1;
            polls > 3
        })
        .unwrap_err(),
        MetricCapUpdateError::Cancelled
    );
    assert_eq!(edit.caps, caps);
    assert!(plan_metric_cap_update(0.5, &edit, [0], || false).is_err());
    let plan = plan_metric_cap_update(1.5, &edit, [], || false).unwrap();
    assert!(plan.caps.is_empty());
    assert_eq!(plan.neighbor_reads, 0);
}
