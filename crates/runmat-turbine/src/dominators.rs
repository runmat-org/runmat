//! Dominator tree computation for SSA optimization passes
//!
//! Uses the simple iterative algorithm (sufficient for our small CFGs).
//! For larger CFGs, consider Lengauer-Tarjan.

use crate::ssa::{BlockId, SsaFunc};
use std::collections::{HashMap, HashSet};

/// Dominator tree for a function
#[derive(Debug, Clone)]
pub struct DomTree {
    /// Immediate dominator for each block (entry has no idom)
    idom: HashMap<BlockId, BlockId>,
    /// Dominance frontier for each block
    frontier: HashMap<BlockId, HashSet<BlockId>>,
    /// Preorder traversal of dominator tree
    preorder: Vec<BlockId>,
    /// Depth of each block in dominator tree
    depth: HashMap<BlockId, usize>,
}

impl DomTree {
    /// Compute dominator tree for a function
    pub fn compute(func: &SsaFunc) -> Self {
        let blocks: Vec<BlockId> = func.blocks.iter().map(|b| b.id).collect();
        let entry = func.entry;

        // Build predecessor map
        let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
        for block in &func.blocks {
            preds.entry(block.id).or_default();
            for succ in block.term.successors() {
                preds.entry(succ).or_default().push(block.id);
            }
        }

        // Compute reverse postorder (for efficient iteration)
        let rpo = reverse_postorder(func, entry);

        // Initialize: every block dominated by entry, except entry dominates itself
        let mut idom: HashMap<BlockId, BlockId> = HashMap::new();

        // Iterative dominator computation
        let mut changed = true;
        while changed {
            changed = false;
            for &block in &rpo {
                if block == entry {
                    continue;
                }
                let block_preds = preds.get(&block).cloned().unwrap_or_default();
                // Find first predecessor with computed idom
                let mut new_idom: Option<BlockId> = None;
                for &pred in &block_preds {
                    if pred == entry || idom.contains_key(&pred) {
                        new_idom = Some(pred);
                        break;
                    }
                }
                if let Some(mut current) = new_idom {
                    // Intersect with other predecessors
                    for &pred in &block_preds {
                        if Some(pred) == new_idom {
                            continue;
                        }
                        if pred == entry || idom.contains_key(&pred) {
                            current = intersect(&idom, &rpo, current, pred, entry);
                        }
                    }
                    if idom.get(&block) != Some(&current) {
                        idom.insert(block, current);
                        changed = true;
                    }
                }
            }
        }

        // Compute dominance frontier
        let mut frontier: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();
        for &block in &blocks {
            frontier.insert(block, HashSet::new());
        }
        for &block in &blocks {
            let block_preds = preds.get(&block).cloned().unwrap_or_default();
            if block_preds.len() >= 2 {
                for &pred in &block_preds {
                    let mut runner = pred;
                    while runner != entry && Some(&runner) != idom.get(&block) {
                        frontier.entry(runner).or_default().insert(block);
                        if let Some(&next) = idom.get(&runner) {
                            runner = next;
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        // Compute depth and preorder
        let (preorder, depth) = compute_preorder_and_depth(&idom, entry, &blocks);

        DomTree {
            idom,
            frontier,
            preorder,
            depth,
        }
    }

    /// Get immediate dominator of a block
    pub fn idom(&self, block: BlockId) -> Option<BlockId> {
        self.idom.get(&block).copied()
    }

    /// Check if `a` dominates `b`
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        if a == b {
            return true;
        }
        let mut current = b;
        while let Some(dom) = self.idom.get(&current) {
            if *dom == a {
                return true;
            }
            current = *dom;
        }
        false
    }

    /// Check if `a` strictly dominates `b` (a dominates b and a != b)
    pub fn strictly_dominates(&self, a: BlockId, b: BlockId) -> bool {
        a != b && self.dominates(a, b)
    }

    /// Get dominance frontier for a block
    pub fn frontier(&self, block: BlockId) -> &HashSet<BlockId> {
        // Use a lazy static for the empty set
        self.frontier.get(&block).unwrap_or_else(|| {
            // Return reference to empty set from frontier (create dummy entry if needed)
            static EMPTY: std::sync::OnceLock<HashSet<BlockId>> = std::sync::OnceLock::new();
            EMPTY.get_or_init(HashSet::new)
        })
    }

    /// Get blocks in dominator tree preorder (useful for CSE/GVN)
    pub fn preorder(&self) -> &[BlockId] {
        &self.preorder
    }

    /// Get depth of block in dominator tree
    pub fn depth(&self, block: BlockId) -> usize {
        self.depth.get(&block).copied().unwrap_or(0)
    }
}

/// Reverse postorder traversal (used for dominator computation)
fn reverse_postorder(func: &SsaFunc, entry: BlockId) -> Vec<BlockId> {
    let mut visited: HashSet<BlockId> = HashSet::new();
    let mut postorder: Vec<BlockId> = Vec::new();

    fn dfs(
        func: &SsaFunc,
        block: BlockId,
        visited: &mut HashSet<BlockId>,
        postorder: &mut Vec<BlockId>,
    ) {
        if visited.contains(&block) {
            return;
        }
        visited.insert(block);
        if let Some(b) = func.block(block) {
            for succ in b.term.successors() {
                dfs(func, succ, visited, postorder);
            }
        }
        postorder.push(block);
    }

    dfs(func, entry, &mut visited, &mut postorder);
    postorder.reverse();
    postorder
}

/// Find intersection of two dominators using finger algorithm
fn intersect(
    idom: &HashMap<BlockId, BlockId>,
    rpo: &[BlockId],
    mut a: BlockId,
    mut b: BlockId,
    entry: BlockId,
) -> BlockId {
    // Get RPO index for comparison
    let rpo_idx: HashMap<BlockId, usize> = rpo.iter().enumerate().map(|(i, &b)| (b, i)).collect();

    while a != b {
        while rpo_idx.get(&a).copied().unwrap_or(usize::MAX)
            > rpo_idx.get(&b).copied().unwrap_or(usize::MAX)
        {
            if a == entry {
                return entry;
            }
            a = *idom.get(&a).unwrap_or(&entry);
        }
        while rpo_idx.get(&b).copied().unwrap_or(usize::MAX)
            > rpo_idx.get(&a).copied().unwrap_or(usize::MAX)
        {
            if b == entry {
                return entry;
            }
            b = *idom.get(&b).unwrap_or(&entry);
        }
    }
    a
}

/// Compute preorder traversal and depth of dominator tree
fn compute_preorder_and_depth(
    idom: &HashMap<BlockId, BlockId>,
    entry: BlockId,
    blocks: &[BlockId],
) -> (Vec<BlockId>, HashMap<BlockId, usize>) {
    // Build children map
    let mut children: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    children.insert(entry, Vec::new());
    for (&block, &dom) in idom {
        children.entry(dom).or_default().push(block);
    }

    let mut preorder = Vec::new();
    let mut depth = HashMap::new();

    fn dfs(
        block: BlockId,
        d: usize,
        children: &HashMap<BlockId, Vec<BlockId>>,
        preorder: &mut Vec<BlockId>,
        depth: &mut HashMap<BlockId, usize>,
    ) {
        preorder.push(block);
        depth.insert(block, d);
        if let Some(kids) = children.get(&block) {
            for &child in kids {
                dfs(child, d + 1, children, preorder, depth);
            }
        }
    }

    dfs(entry, 0, &children, &mut preorder, &mut depth);

    // Add any unreachable blocks (shouldn't happen in valid SSA, but be safe)
    for &block in blocks {
        depth.entry(block).or_insert_with(|| {
            preorder.push(block);
            0
        });
    }

    (preorder, depth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssa::Terminator;

    fn make_diamond() -> SsaFunc {
        // CFG:
        //       entry(0)
        //       /    \
        //    then(1) else(2)
        //       \    /
        //       merge(3)
        let mut func = SsaFunc::new("diamond");

        let entry = func.new_block();
        let then_block = func.new_block();
        let else_block = func.new_block();
        let merge = func.new_block();

        func.entry = entry;

        let cond = func.new_value();

        // Entry block
        {
            let block = func.block_mut(entry).unwrap();
            block.instrs.push(crate::ssa::SsaInstr {
                dst: cond,
                op: crate::ssa::SsaOp::ConstBool(true),
                ty: crate::ssa::SsaType::Bool,
            });
            block.term = Terminator::Cbr {
                cond,
                then_block,
                then_args: vec![],
                else_block,
                else_args: vec![],
            };
        }

        // Then block
        {
            let block = func.block_mut(then_block).unwrap();
            block.term = Terminator::Br {
                target: merge,
                args: vec![],
            };
        }

        // Else block
        {
            let block = func.block_mut(else_block).unwrap();
            block.term = Terminator::Br {
                target: merge,
                args: vec![],
            };
        }

        // Merge block
        {
            let block = func.block_mut(merge).unwrap();
            block.term = Terminator::Ret(None);
        }

        func
    }

    #[test]
    fn test_dominator_tree_diamond() {
        let func = make_diamond();
        let dom = DomTree::compute(&func);

        let entry = BlockId(0);
        let then_block = BlockId(1);
        let else_block = BlockId(2);
        let merge = BlockId(3);

        // Entry dominates everything
        assert!(dom.dominates(entry, entry));
        assert!(dom.dominates(entry, then_block));
        assert!(dom.dominates(entry, else_block));
        assert!(dom.dominates(entry, merge));

        // Then/else don't dominate merge (both paths lead to it)
        assert!(!dom.strictly_dominates(then_block, merge));
        assert!(!dom.strictly_dominates(else_block, merge));

        // Immediate dominators
        assert_eq!(dom.idom(then_block), Some(entry));
        assert_eq!(dom.idom(else_block), Some(entry));
        assert_eq!(dom.idom(merge), Some(entry));
    }

    #[test]
    fn test_preorder() {
        let func = make_diamond();
        let dom = DomTree::compute(&func);

        let preorder = dom.preorder();
        // Entry should be first
        assert_eq!(preorder[0], BlockId(0));
        // All blocks should be present
        assert_eq!(preorder.len(), 4);
    }
}
