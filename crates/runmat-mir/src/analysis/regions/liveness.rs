use std::collections::{BTreeMap, BTreeSet};

use crate::{BasicBlockId, MirBody, MirLocalId};

use super::uses::{
    statement_uses_defs, successors, terminator_edge_defs, terminator_uses_defs, Locals,
};

pub(crate) struct BodyLiveness {
    pub before: BTreeMap<(BasicBlockId, usize), Locals>,
    pub after: BTreeMap<(BasicBlockId, usize), Locals>,
}

pub(crate) fn analyze_liveness(body: &MirBody) -> BodyLiveness {
    let block_index = body
        .blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.id, index))
        .collect::<BTreeMap<_, _>>();
    let mut live_in = vec![Locals::new(); body.blocks.len()];

    loop {
        let mut changed = false;
        for (index, block) in body.blocks.iter().enumerate().rev() {
            let mut next_in = live_before_terminator(block, &block_index, &live_in);
            for statement in block.statements.iter().rev() {
                let (uses, defs) = statement_uses_defs(statement);
                next_in = transfer_backwards(&next_in, &uses, &defs);
            }
            if live_in[index] != next_in {
                live_in[index] = next_in;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut before = BTreeMap::new();
    let mut after = BTreeMap::new();
    for block in &body.blocks {
        let mut live = live_before_terminator(block, &block_index, &live_in);
        for (position, statement) in block.statements.iter().enumerate().rev() {
            after.insert((block.id, position), live.clone());
            let (uses, defs) = statement_uses_defs(statement);
            live = transfer_backwards(&live, &uses, &defs);
            before.insert((block.id, position), live.clone());
        }
    }
    BodyLiveness { before, after }
}

fn live_before_terminator(
    block: &crate::BasicBlock,
    block_index: &BTreeMap<BasicBlockId, usize>,
    live_in: &[Locals],
) -> Locals {
    let mut live = Locals::new();
    for successor in successors(&block.terminator.kind) {
        let Some(index) = block_index.get(&successor).copied() else {
            continue;
        };
        let edge_defs = terminator_edge_defs(&block.terminator.kind, successor);
        live.extend(
            live_in[index]
                .iter()
                .copied()
                .filter(|local| !edge_defs.contains(local)),
        );
    }
    let (uses, _) = terminator_uses_defs(&block.terminator.kind);
    live.extend(uses);
    live
}

fn transfer_backwards(live: &Locals, uses: &Locals, defs: &Locals) -> BTreeSet<MirLocalId> {
    uses.iter()
        .copied()
        .chain(live.iter().copied().filter(|local| !defs.contains(local)))
        .collect()
}
