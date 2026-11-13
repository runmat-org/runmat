//! Client exports and worker-to-client control-flow boundaries.
use std::collections::BTreeSet;

use runmat_types::{LabCount, ParallelRegionId, SpmdLabRequirement};

use crate::{BasicBlockId, MirBody, MirLocalId, MirLocalKind, MirTerminatorKind};

use super::super::region;

pub(in crate::analysis) struct SpmdExit {
    pub region: ParallelRegionId,
    blocks: BTreeSet<BasicBlockId>,
    exit: BasicBlockId,
    pub outputs: Vec<MirLocalId>,
    pub labs: Option<LabCount>,
}

impl SpmdExit {
    pub fn crosses(&self, predecessor: BasicBlockId, successor: BasicBlockId) -> bool {
        self.exit == successor && self.blocks.contains(&predecessor)
    }
}

pub(in crate::analysis) fn spmd_exit_projections(body: &MirBody) -> Vec<SpmdExit> {
    body.blocks
        .iter()
        .filter_map(|block| {
            let MirTerminatorKind::Spmd {
                region: id,
                header,
                body_block,
                exit_block,
            } = &block.terminator.kind
            else {
                return None;
            };
            let blocks = region::body_blocks(body, block.id, *body_block, *exit_block);
            let outputs = region::accesses(body, &blocks)
                .writes
                .into_iter()
                .filter(|local| body.locals[local.0].kind != MirLocalKind::Temporary)
                .collect();
            let labs = match super::lab_requirement(header) {
                SpmdLabRequirement::Exact { labs } if labs.0 > 0 => Some(labs),
                _ => None,
            };
            Some(SpmdExit {
                region: *id,
                blocks,
                exit: *exit_block,
                outputs,
                labs,
            })
        })
        .collect()
}
