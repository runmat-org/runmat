use std::collections::{HashMap, HashSet};

use runmat_mir::{BasicBlock, BasicBlockId, MirBody, MirTerminatorKind};

pub(super) struct ExceptionScopes {
    scopes: Vec<TryScope>,
    leaving_at: HashMap<BasicBlockId, Vec<usize>>,
}

struct TryScope {
    id: usize,
    protected_blocks: HashSet<BasicBlockId>,
}

impl ExceptionScopes {
    pub(super) fn analyze(body: &MirBody) -> Self {
        let blocks = body
            .blocks
            .iter()
            .map(|block| (block.id, block))
            .collect::<HashMap<_, _>>();
        let scopes = body
            .blocks
            .iter()
            .filter_map(|block| {
                let MirTerminatorKind::TryCatch {
                    try_block,
                    catch_block,
                    ..
                } = &block.terminator.kind
                else {
                    return None;
                };
                let mut protected_blocks = reachable_blocks(&blocks, *try_block);
                let catch_reachable = reachable_blocks(&blocks, *catch_block);
                protected_blocks.retain(|block| !catch_reachable.contains(block));
                Some(TryScope {
                    id: block.id.0,
                    protected_blocks,
                })
            })
            .collect::<Vec<_>>();
        let mut leaving_at: HashMap<BasicBlockId, Vec<usize>> = HashMap::new();
        for scope in &scopes {
            for protected in &scope.protected_blocks {
                let Some(block) = blocks.get(protected) else {
                    continue;
                };
                for successor in successors(&block.terminator.kind) {
                    if !scope.protected_blocks.contains(&successor) {
                        leaving_at.entry(successor).or_default().push(scope.id);
                    }
                }
            }
        }
        for scope_ids in leaving_at.values_mut() {
            scope_ids.sort_by_key(|scope_id| {
                let scope = scopes
                    .iter()
                    .find(|scope| scope.id == *scope_id)
                    .expect("leaving scope must be present");
                (scope.protected_blocks.len(), scope.id)
            });
            scope_ids.dedup();
        }
        Self { scopes, leaving_at }
    }

    pub(super) fn leaving_at(&self, block: BasicBlockId) -> &[usize] {
        self.leaving_at.get(&block).map_or(&[], Vec::as_slice)
    }

    pub(super) fn active_at(&self, block: BasicBlockId) -> Vec<usize> {
        let mut active = self
            .scopes
            .iter()
            .filter(|scope| scope.protected_blocks.contains(&block))
            .collect::<Vec<_>>();
        active.sort_by_key(|scope| (scope.protected_blocks.len(), scope.id));
        active.into_iter().map(|scope| scope.id).collect()
    }
}

fn reachable_blocks(
    blocks: &HashMap<BasicBlockId, &BasicBlock>,
    entry: BasicBlockId,
) -> HashSet<BasicBlockId> {
    let mut reachable = HashSet::new();
    let mut pending = vec![entry];
    while let Some(block_id) = pending.pop() {
        if !reachable.insert(block_id) {
            continue;
        }
        if let Some(block) = blocks.get(&block_id) {
            pending.extend(successors(&block.terminator.kind));
        }
    }
    reachable
}

fn successors(kind: &MirTerminatorKind) -> Vec<BasicBlockId> {
    match kind {
        MirTerminatorKind::Goto(target) => vec![*target],
        MirTerminatorKind::Branch {
            then_block,
            else_block,
            ..
        } => vec![*then_block, *else_block],
        MirTerminatorKind::Switch {
            cases, otherwise, ..
        } => cases
            .iter()
            .map(|(_, target)| *target)
            .chain(std::iter::once(*otherwise))
            .collect(),
        MirTerminatorKind::For {
            body_block,
            exit_block,
            ..
        }
        | MirTerminatorKind::ParFor {
            body_block,
            exit_block,
            ..
        }
        | MirTerminatorKind::Spmd {
            body_block,
            exit_block,
            ..
        } => vec![*body_block, *exit_block],
        MirTerminatorKind::TryCatch {
            try_block,
            catch_block,
            ..
        } => vec![*try_block, *catch_block],
        MirTerminatorKind::Await { resume, .. } => vec![*resume],
        MirTerminatorKind::Return(_) | MirTerminatorKind::Unreachable => Vec::new(),
    }
}
