use std::collections::{BTreeSet, VecDeque};

use crate::{BasicBlockId, MirBody, MirLocalId, MirTerminatorKind};

use super::super::regions::{statement_uses_defs, successors, terminator_uses_defs};

#[derive(Default)]
pub(super) struct RegionAccess {
    pub reads: BTreeSet<MirLocalId>,
    pub writes: BTreeSet<MirLocalId>,
}

pub(super) fn body_blocks(
    body: &MirBody,
    header: BasicBlockId,
    first: BasicBlockId,
    exit: BasicBlockId,
) -> BTreeSet<BasicBlockId> {
    let mut pending = VecDeque::from([first]);
    let mut blocks = BTreeSet::new();
    while let Some(block) = pending.pop_front() {
        if block == header || block == exit || !blocks.insert(block) {
            continue;
        }
        if let Some(block) = body.blocks.iter().find(|candidate| candidate.id == block) {
            pending.extend(successors(&block.terminator.kind));
        }
    }
    blocks
}

pub(super) fn accesses(body: &MirBody, blocks: &BTreeSet<BasicBlockId>) -> RegionAccess {
    let mut result = RegionAccess::default();
    for block in body
        .blocks
        .iter()
        .filter(|block| blocks.contains(&block.id))
    {
        for statement in &block.statements {
            let (reads, writes) = statement_uses_defs(statement);
            result.reads.extend(reads);
            result.writes.extend(writes);
        }
        let (reads, writes) = terminator_uses_defs(&block.terminator.kind);
        result.reads.extend(reads);
        result.writes.extend(writes);
    }
    result
}

pub(super) fn contains_parallel_region(body: &MirBody, blocks: &BTreeSet<BasicBlockId>) -> bool {
    body.blocks.iter().any(|block| {
        blocks.contains(&block.id)
            && matches!(
                block.terminator.kind,
                MirTerminatorKind::ParFor { .. } | MirTerminatorKind::Spmd { .. }
            )
    })
}

pub(super) fn is_nested_parallel_header(body: &MirBody, candidate: BasicBlockId) -> bool {
    body.blocks.iter().any(|header| {
        let (body_block, exit_block) = match &header.terminator.kind {
            MirTerminatorKind::ParFor {
                body_block,
                exit_block,
                ..
            }
            | MirTerminatorKind::Spmd {
                body_block,
                exit_block,
                ..
            } => (*body_block, *exit_block),
            _ => return false,
        };
        header.id != candidate
            && body_blocks(body, header.id, body_block, exit_block).contains(&candidate)
    })
}
