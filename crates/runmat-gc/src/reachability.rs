//! Batched registration-scoped reachability for domain-owned lifecycle audits.

use std::collections::{HashMap, HashSet};

use crate::{GcHandle, RootId};

// Last-owner proof must never allocate without a domain-owned bound. A large
// or opaque session is indeterminate, not evidence that a target is dead.
const MAX_RELATION_CELLS: usize = 4_000_000;
const MAX_PROVENANCE_WORDS: usize = 1_000_000;
const MAX_TRACED_EDGES: usize = 2_000_000;

/// One root-publication snapshot. Source rows follow the supplied registered
/// root IDs; columns follow the supplied target handles. A `None` snapshot is
/// indeterminate and must never be interpreted as an unreachable target.
#[derive(Debug)]
pub struct RegisteredRootReachability {
    outside: Vec<bool>,
    sources: Vec<Vec<bool>>,
}

impl RegisteredRootReachability {
    /// Construct a projected relation, rejecting ragged rows. Domain policy
    /// tests can exercise ordering without allocating or collecting GC values.
    pub fn from_rows(outside: Vec<bool>, sources: Vec<Vec<bool>>) -> Option<Self> {
        sources
            .iter()
            .all(|row| row.len() == outside.len())
            .then_some(Self { outside, sources })
    }

    pub fn outside_reaches(&self, target: usize) -> bool {
        self.outside[target]
    }

    pub fn source_reaches(&self, source: usize, target: usize) -> bool {
        self.sources[source][target]
    }

    pub fn target_count(&self) -> usize {
        self.outside.len()
    }

    pub fn source_count(&self) -> usize {
        self.sources.len()
    }
}

/// Trace all supplied registrations and every other visible root under one
/// admission guard. `excluded_temporary` names only caller-owned temporary
/// registrations. A source registration is excluded from the outside column
/// by its ID, never by its handle identity. `decide` may only update already
/// rooted state; no root mutation, user code, or await is permitted in it.
pub fn gc_with_registered_root_reachability<R>(
    targets: &[GcHandle],
    source_registrations: &[RootId],
    excluded_temporary: &[RootId],
    decide: impl FnOnce(Option<RegisteredRootReachability>) -> R,
) -> R {
    let _admission = crate::ROOT_ADMISSION.read();
    decide(snapshot(targets, source_registrations, excluded_temporary))
}

fn snapshot(
    targets: &[GcHandle],
    sources: &[RootId],
    temporary: &[RootId],
) -> Option<RegisteredRootReachability> {
    use std::sync::atomic::Ordering;

    if crate::GC.collection_in_progress.load(Ordering::Acquire) {
        return None;
    }
    if sources.len().checked_mul(targets.len())? > MAX_RELATION_CELLS {
        return None;
    }
    let unique = sources.iter().copied().collect::<HashSet<_>>();
    if unique.len() != sources.len() || sources.iter().any(|id| temporary.contains(id)) {
        return None;
    }
    let mut excluded = sources.to_vec();
    excluded.extend_from_slice(temporary);
    let mut starts = vec![crate::external_roots()];
    starts[0].extend(crate::GC.root_ptrs.try_lock()?.keys().copied());
    if starts[0].len() > MAX_TRACED_EDGES {
        return None;
    }
    crate::gc_visit_retained_values(&excluded, &mut |value| {
        starts[0].extend(crate::gc_handles_in_value(value));
        (starts[0].len() <= MAX_TRACED_EDGES).then_some(())
    })?;
    for registration in sources {
        let mut handles = Vec::new();
        crate::ROOT_SCANNER.with(|scanner| {
            scanner.visit_registered_values(*registration, &mut |value| {
                handles.extend(crate::gc_handles_in_value(value));
                (handles.len() <= MAX_TRACED_EDGES).then_some(())
            })
        })?;
        starts.push(handles);
    }

    // The collector's Value graph is read once per handle. Provenance bits
    // then propagate through that shared graph, including cycles, until fixed.
    let words = starts.len().div_ceil(64);
    if words == 0 || words > MAX_PROVENANCE_WORDS {
        return None;
    }
    let mut provenance: HashMap<GcHandle, Vec<u64>> = HashMap::new();
    let mut adjacency: HashMap<GcHandle, Vec<GcHandle>> = HashMap::new();
    let mut pending = Vec::new();
    let mut edges = 0usize;
    for (origin, handles) in starts.into_iter().enumerate() {
        for handle in handles {
            if !provenance.contains_key(&handle)
                && (provenance.len() + 1).checked_mul(words)? > MAX_PROVENANCE_WORDS
            {
                return None;
            }
            let bits = provenance.entry(handle).or_insert_with(|| vec![0; words]);
            let mask = 1u64 << (origin % 64);
            if bits[origin / 64] & mask == 0 {
                bits[origin / 64] |= mask;
                pending.push(handle);
            }
        }
    }
    while let Some(handle) = pending.pop() {
        let incoming = provenance.get(&handle)?.clone();
        let children = if let Some(cached) = adjacency.get(&handle) {
            cached.clone()
        } else {
            let children = crate::gc_handles_in_allocation(&handle)?;
            edges = edges.checked_add(children.len())?;
            if edges > MAX_TRACED_EDGES {
                return None;
            }
            adjacency.insert(handle, children.clone());
            children
        };
        for child in children {
            if !provenance.contains_key(&child)
                && (provenance.len() + 1).checked_mul(words)? > MAX_PROVENANCE_WORDS
            {
                return None;
            }
            let bits = provenance.entry(child).or_insert_with(|| vec![0; words]);
            let mut changed = false;
            for (destination, source) in bits.iter_mut().zip(&incoming) {
                let next = *destination | *source;
                changed |= next != *destination;
                *destination = next;
            }
            if changed {
                pending.push(child);
            }
        }
    }
    let reaches = |target: GcHandle, source: usize| {
        provenance
            .get(&target)
            .is_some_and(|bits| bits[source / 64] & (1u64 << (source % 64)) != 0)
    };
    Some(RegisteredRootReachability {
        outside: targets.iter().map(|target| reaches(*target, 0)).collect(),
        sources: (0..sources.len())
            .map(|source| {
                targets
                    .iter()
                    .map(|target| reaches(*target, source + 1))
                    .collect()
            })
            .collect(),
    })
}
