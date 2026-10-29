//! Match only the MIR ordering owner's private source-read snapshots.
//!
//! In `S .* exp(drift + scale .* Z)`, lowering captures S before preparing
//! exp's arguments and drift before preparing scale.*Z. These two pure reads
//! precede the same five semantic operations; named copies, other statements,
//! additional uses, and writes cannot be erased by this recognition.

use super::{assigned_local, source_binding};
use runmat_mir::{MirBody, MirLocalId, MirLocalKind, MirOperand, MirRvalue, MirStmt};

#[derive(Clone, Copy)]
struct Snapshot {
    target: MirLocalId,
    source: MirLocalId,
}

pub(super) struct SourceBody<'a> {
    pub(super) statements: [&'a MirStmt; 5],
    snapshots: [Option<Snapshot>; 2],
    consumed: u8,
    state_snapshot: Option<usize>,
}

impl<'a> SourceBody<'a> {
    pub(super) fn parse(body: &MirBody, statements: &'a [MirStmt]) -> Option<Self> {
        // Exact grammar: randn; [private source snapshots]; four arithmetic
        // statements. The detector still authenticates every semantic node.
        let (noise, tail) = statements.split_first()?;
        let snapshots_len = tail.len().checked_sub(4)?;
        if snapshots_len > 2 {
            return None;
        }
        let (copies, arithmetic) = tail.split_at(snapshots_len);
        let [scaled, argument, exponent, state] = arithmetic else {
            return None;
        };
        let mut snapshots = [None; 2];
        for (index, statement) in copies.iter().enumerate() {
            let (target, MirRvalue::Use(MirOperand::Local(source))) = assigned_local(statement)?
            else {
                return None;
            };
            if !body
                .locals
                .iter()
                .any(|local| local.id == target && matches!(local.kind, MirLocalKind::Temporary))
                || !source_binding(body, *source)
            {
                return None;
            }
            snapshots[index] = Some(Snapshot {
                target,
                source: *source,
            });
        }
        Some(Self {
            statements: [noise, scaled, argument, exponent, state],
            snapshots,
            consumed: 0,
            state_snapshot: None,
        })
    }

    pub(super) fn snapshot_targets(&self) -> impl Iterator<Item = MirLocalId> + '_ {
        self.snapshots.iter().flatten().map(|copy| copy.target)
    }

    pub(super) fn state_source(&mut self, state: MirLocalId) -> Option<MirLocalId> {
        let (source, index) = self.resolve(state)?;
        // If present, the state read must precede argument preparation.
        if index.is_some_and(|index| index != 0) {
            return None;
        }
        self.state_snapshot = index;
        Some(source)
    }

    pub(super) fn drift_source(&mut self, operand: &MirOperand) -> Option<MirOperand> {
        let MirOperand::Local(local) = operand else {
            return Some(operand.clone());
        };
        let (source, index) = self.resolve(*local)?;
        if index.is_some_and(|index| self.state_snapshot.is_some_and(|state| index <= state)) {
            return None;
        }
        Some(MirOperand::Local(source))
    }

    fn resolve(&mut self, local: MirLocalId) -> Option<(MirLocalId, Option<usize>)> {
        let Some((index, snapshot)) =
            self.snapshots.iter().enumerate().find_map(|(index, copy)| {
                copy.filter(|copy| copy.target == local)
                    .map(|copy| (index, copy))
            })
        else {
            return Some((local, None));
        };
        let bit = 1 << index;
        if self.consumed & bit != 0 {
            return None;
        }
        self.consumed |= bit;
        Some((snapshot.source, Some(index)))
    }

    pub(super) fn all_consumed(&self) -> bool {
        self.consumed.count_ones() as usize == self.snapshots.iter().flatten().count()
    }
}
