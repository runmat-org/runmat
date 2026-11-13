//! Active intervals indexed by physical edge; one queued copy per live interval.
//! Retired split parents reuse their slot, so storage follows live leaves.
use super::*;
use std::collections::BTreeMap;

pub(super) struct PendingInterval {
    pub(super) constraint_index: usize,
    pub(super) interval: interval::ChildInterval,
    edge: [u32; 2],
    queued: bool,
}

#[derive(Default)]
pub(super) struct IntervalQueue {
    records: Vec<Option<PendingInterval>>,
    free: Vec<usize>,
    pending: VecDeque<usize>,
    edges: BTreeMap<[u32; 2], BTreeSet<usize>>,
}

impl IntervalQueue {
    pub(super) fn insert(
        &mut self,
        constraint_index: usize,
        interval: interval::ChildInterval,
        topology: &TopologyMutation,
    ) -> Result<(), DelaunaySegmentRecoveryError> {
        let edge = [interval.0, interval.1].map(|node| topology.node_index(node.identity));
        let [Some(left), Some(right)] = edge else {
            return Err(error(
                DelaunaySegmentRecoveryErrorKind::InvalidTopology,
                Some(constraint_index as u32),
                "queued segment endpoint is missing",
            ));
        };
        let mut edge = [left, right];
        edge.sort_unstable();
        let record = PendingInterval {
            constraint_index,
            interval,
            edge,
            queued: true,
        };
        let index = if let Some(index) = self.free.pop() {
            self.records[index] = Some(record);
            index
        } else {
            self.records.push(Some(record));
            self.records.len() - 1
        };
        self.edges.entry(edge).or_default().insert(index);
        self.pending.push_back(index);
        Ok(())
    }

    pub(super) fn pop(&mut self) -> Option<(usize, usize, interval::ChildInterval)> {
        let index = self.pending.pop_front()?;
        let record = self.records[index].as_mut().expect("queued live interval");
        record.queued = false;
        Some((index, record.constraint_index, record.interval))
    }

    pub(super) fn retire(&mut self, index: usize) {
        let record = self.records[index].take().expect("split live parent");
        let uses = self.edges.get_mut(&record.edge).expect("indexed interval");
        uses.remove(&index);
        if uses.is_empty() {
            self.edges.remove(&record.edge);
        }
        self.free.push(index);
    }

    pub(super) fn requeue_lost(
        &mut self,
        candidates: BTreeSet<[u32; 2]>,
        topology: &TopologyMutation,
        work: &mut RecoveryWork<'_>,
    ) -> Result<(), DelaunaySegmentRecoveryError> {
        for edge in candidates {
            let Some(uses) = self.edges.get(&edge) else {
                continue;
            };
            let first = self.records[*uses.first().expect("nonempty edge uses")]
                .as_ref()
                .unwrap();
            if edge_exists(
                topology,
                topology.nodes()[edge[0] as usize].identity,
                topology.nodes()[edge[1] as usize].identity,
                first.constraint_index as u32,
                work,
            )? {
                continue;
            }
            for &index in uses {
                let record = self.records[index].as_mut().expect("indexed live interval");
                if !record.queued {
                    record.queued = true;
                    self.pending.push_back(index);
                }
            }
        }
        Ok(())
    }

    pub(super) fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub(super) fn into_completed(self, constraints: usize) -> Vec<Vec<interval::ChildInterval>> {
        let mut completed = vec![Vec::new(); constraints];
        for record in self.records.into_iter().flatten() {
            debug_assert!(!record.queued);
            completed[record.constraint_index].push(record.interval);
        }
        completed
    }
}
