//! Loop Analysis for LICM
//!
//! Identifies natural loops in the CFG for loop-invariant code motion.

use crate::dominators::DomTree;
use crate::ssa::{BlockId, SsaFunc};
use std::collections::{HashMap, HashSet};

/// Information about a single loop
#[derive(Debug, Clone)]
pub struct Loop {
    /// The loop header (entry point, dominates all blocks in loop)
    pub header: BlockId,
    /// All blocks in the loop (including header)
    pub blocks: HashSet<BlockId>,
    /// Back edges into this loop (from latch to header)
    pub back_edges: Vec<(BlockId, BlockId)>,
    /// Preheader block (if exists or created)
    pub preheader: Option<BlockId>,
    /// Exit blocks (blocks outside loop with predecessor in loop)
    pub exits: HashSet<BlockId>,
    /// Nesting depth (0 = outermost)
    pub depth: usize,
}

/// Loop analysis result
#[derive(Debug, Clone)]
pub struct LoopInfo {
    /// All loops, keyed by header
    loops: HashMap<BlockId, Loop>,
    /// Map from block to innermost containing loop header
    block_to_loop: HashMap<BlockId, BlockId>,
}

impl LoopInfo {
    /// Analyze loops in a function
    pub fn compute(func: &SsaFunc, dom: &DomTree) -> Self {
        let mut loops: HashMap<BlockId, Loop> = HashMap::new();
        let mut block_to_loop: HashMap<BlockId, BlockId> = HashMap::new();

        // Find back edges: edge (a, b) is a back edge if b dominates a
        let back_edges = find_back_edges(func, dom);

        // For each back edge, compute the natural loop
        for (latch, header) in &back_edges {
            let loop_blocks = compute_natural_loop(func, *header, *latch);

            // Update or create loop
            let loop_entry = loops.entry(*header).or_insert_with(|| Loop {
                header: *header,
                blocks: HashSet::new(),
                back_edges: Vec::new(),
                preheader: None,
                exits: HashSet::new(),
                depth: 0,
            });

            loop_entry.blocks.extend(loop_blocks.iter().cloned());
            loop_entry.back_edges.push((*latch, *header));
        }

        // Compute exits for each loop
        for (_header, loop_data) in loops.iter_mut() {
            for &block in &loop_data.blocks {
                if let Some(b) = func.block(block) {
                    for succ in b.term.successors() {
                        if !loop_data.blocks.contains(&succ) {
                            loop_data.exits.insert(succ);
                        }
                    }
                }
            }
        }

        // Compute nesting depth
        let headers: Vec<BlockId> = loops.keys().copied().collect();
        for header in &headers {
            let mut depth = 0;
            for other_header in &headers {
                if header != other_header {
                    if let (Some(this_loop), Some(other_loop)) =
                        (loops.get(header), loops.get(other_header))
                    {
                        // If this loop is contained in other loop
                        if other_loop.blocks.contains(header)
                            && this_loop.blocks.is_subset(&other_loop.blocks)
                        {
                            depth += 1;
                        }
                    }
                }
            }
            if let Some(loop_data) = loops.get_mut(header) {
                loop_data.depth = depth;
            }
        }

        // Map blocks to innermost loop
        for (header, loop_data) in &loops {
            for &block in &loop_data.blocks {
                // Only update if this is a deeper loop
                let should_update = match block_to_loop.get(&block) {
                    None => true,
                    Some(&existing_header) => {
                        loops.get(&existing_header).map(|l| l.depth).unwrap_or(0) < loop_data.depth
                    }
                };
                if should_update {
                    block_to_loop.insert(block, *header);
                }
            }
        }

        LoopInfo {
            loops,
            block_to_loop,
        }
    }

    /// Get all loops
    pub fn loops(&self) -> impl Iterator<Item = &Loop> {
        self.loops.values()
    }

    /// Get loop containing a block (innermost if nested)
    pub fn loop_for_block(&self, block: BlockId) -> Option<&Loop> {
        self.block_to_loop
            .get(&block)
            .and_then(|h| self.loops.get(h))
    }

    /// Check if block is in any loop
    pub fn is_in_loop(&self, block: BlockId) -> bool {
        self.block_to_loop.contains_key(&block)
    }

    /// Get loop by header
    pub fn loop_by_header(&self, header: BlockId) -> Option<&Loop> {
        self.loops.get(&header)
    }

    /// Get loop nesting depth for a block (0 if not in loop)
    pub fn loop_depth(&self, block: BlockId) -> usize {
        self.loop_for_block(block).map(|l| l.depth + 1).unwrap_or(0)
    }
}

/// Find back edges in the CFG
fn find_back_edges(func: &SsaFunc, dom: &DomTree) -> Vec<(BlockId, BlockId)> {
    let mut back_edges = Vec::new();

    for block in &func.blocks {
        for succ in block.term.successors() {
            // Edge (block, succ) is a back edge if succ dominates block
            if dom.dominates(succ, block.id) {
                back_edges.push((block.id, succ));
            }
        }
    }

    back_edges
}

/// Compute natural loop for a back edge
fn compute_natural_loop(func: &SsaFunc, header: BlockId, latch: BlockId) -> HashSet<BlockId> {
    let mut loop_blocks = HashSet::new();
    loop_blocks.insert(header);

    if header == latch {
        return loop_blocks;
    }

    // Work backwards from latch
    let mut worklist = vec![latch];
    loop_blocks.insert(latch);

    while let Some(block) = worklist.pop() {
        // Add predecessors
        let preds = func.predecessors(block);
        for pred in preds {
            if !loop_blocks.contains(&pred) {
                loop_blocks.insert(pred);
                worklist.push(pred);
            }
        }
    }

    loop_blocks
}

/// Create preheader for a loop (modifies the function)
pub fn ensure_preheader(func: &mut SsaFunc, loop_info: &mut LoopInfo, header: BlockId) -> BlockId {
    if let Some(loop_data) = loop_info.loops.get(&header) {
        if let Some(preheader) = loop_data.preheader {
            return preheader;
        }
    }

    // Find predecessors of header that are NOT in the loop
    let loop_blocks = loop_info
        .loops
        .get(&header)
        .map(|l| l.blocks.clone())
        .unwrap_or_default();

    let outside_preds: Vec<BlockId> = func
        .predecessors(header)
        .into_iter()
        .filter(|p| !loop_blocks.contains(p))
        .collect();

    // If only one outside predecessor and it only goes to header, use it
    if outside_preds.len() == 1 {
        let pred = outside_preds[0];
        if let Some(block) = func.block(pred) {
            if block.term.successors() == vec![header] {
                // This predecessor is already a preheader
                if let Some(loop_data) = loop_info.loops.get_mut(&header) {
                    loop_data.preheader = Some(pred);
                }
                return pred;
            }
        }
    }

    // Need to create a new preheader block
    let preheader = func.new_block();

    // Set preheader terminator to branch to header
    if let Some(ph_block) = func.block_mut(preheader) {
        ph_block.term = crate::ssa::Terminator::Br {
            target: header,
            args: vec![],
        };
    }

    // Redirect outside predecessors to preheader
    for pred in &outside_preds {
        if let Some(block) = func.block_mut(*pred) {
            redirect_terminator(&mut block.term, header, preheader);
        }
    }

    // Update loop info
    if let Some(loop_data) = loop_info.loops.get_mut(&header) {
        loop_data.preheader = Some(preheader);
    }

    preheader
}

/// Redirect a terminator from old_target to new_target
fn redirect_terminator(term: &mut crate::ssa::Terminator, old: BlockId, new: BlockId) {
    match term {
        crate::ssa::Terminator::Br { target, .. } => {
            if *target == old {
                *target = new;
            }
        }
        crate::ssa::Terminator::Cbr {
            then_block,
            else_block,
            ..
        } => {
            if *then_block == old {
                *then_block = new;
            }
            if *else_block == old {
                *else_block = new;
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssa::{SsaInstr, SsaOp, SsaType, Terminator};

    fn make_simple_loop() -> SsaFunc {
        // CFG:
        //   entry(0)
        //     |
        //   header(1) <--+
        //     |          |
        //   body(2) -----+
        //     |
        //   exit(3)
        let mut func = SsaFunc::new("loop");

        let entry = func.new_block();
        let header = func.new_block();
        let body = func.new_block();
        let exit = func.new_block();

        func.entry = entry;

        let cond = func.new_value();

        // Entry -> header
        {
            let block = func.block_mut(entry).unwrap();
            block.term = Terminator::Br {
                target: header,
                args: vec![],
            };
        }

        // Header: cbr to body or exit
        {
            let block = func.block_mut(header).unwrap();
            block.instrs.push(SsaInstr {
                dst: cond,
                op: SsaOp::ConstBool(true),
                ty: SsaType::Bool,
            });
            block.term = Terminator::Cbr {
                cond,
                then_block: body,
                then_args: vec![],
                else_block: exit,
                else_args: vec![],
            };
        }

        // Body -> header (back edge)
        {
            let block = func.block_mut(body).unwrap();
            block.term = Terminator::Br {
                target: header,
                args: vec![],
            };
        }

        // Exit
        {
            let block = func.block_mut(exit).unwrap();
            block.term = Terminator::Ret(None);
        }

        func
    }

    #[test]
    fn test_loop_detection() {
        let func = make_simple_loop();
        let dom = DomTree::compute(&func);
        let loops = LoopInfo::compute(&func, &dom);

        // Should find one loop
        assert_eq!(loops.loops.len(), 1);

        // Header should be block 1
        let header = BlockId(1);
        let loop_data = loops.loop_by_header(header).unwrap();

        // Loop should contain header and body
        assert!(loop_data.blocks.contains(&BlockId(1))); // header
        assert!(loop_data.blocks.contains(&BlockId(2))); // body
        assert!(!loop_data.blocks.contains(&BlockId(0))); // entry not in loop
        assert!(!loop_data.blocks.contains(&BlockId(3))); // exit not in loop

        // Back edge: body -> header
        assert_eq!(loop_data.back_edges.len(), 1);
        assert_eq!(loop_data.back_edges[0], (BlockId(2), BlockId(1)));
    }

    #[test]
    fn test_loop_depth() {
        let func = make_simple_loop();
        let dom = DomTree::compute(&func);
        let loops = LoopInfo::compute(&func, &dom);

        assert_eq!(loops.loop_depth(BlockId(0)), 0); // entry
        assert_eq!(loops.loop_depth(BlockId(1)), 1); // header
        assert_eq!(loops.loop_depth(BlockId(2)), 1); // body
        assert_eq!(loops.loop_depth(BlockId(3)), 0); // exit
    }
}
