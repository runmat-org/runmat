//! Iterative postorder and actual edge-use counts for one immutable DAG.
//! Last-use transfer avoids retaining/cloning an intermediate array solely
//! for bookkeeping; shared numerical generations remain shared until needed.
use super::super::*;
use runmat_gc::GcHandle;
use std::collections::{HashMap, HashSet};
#[derive(Clone, Copy)]
enum Visit {
    Enter(GcHandle),
    Leave(GcHandle),
}
pub(in crate::builtins::math::optim::optimmodel) struct Plan {
    pub roots: Vec<GcHandle>,
    pub order: Vec<GcHandle>,
    pub nodes: HashMap<GcHandle, owner::OwnedNode>,
    pub uses: HashMap<GcHandle, usize>,
}
impl Plan {
    pub async fn prepare(
        root: GcHandle,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        Self::prepare_roots(std::slice::from_ref(&root), policy, cooperation).await
    }
    pub async fn prepare_roots(
        roots: &[GcHandle],
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut out = Self {
            roots: runmat_value::exact_capacity_vec(roots.len()).map_err(|e| policy.storage(e))?,
            order: Vec::new(),
            nodes: HashMap::new(),
            uses: HashMap::new(),
        };
        let mut active = HashSet::new();
        let mut done = HashSet::new();
        let mut pending =
            runmat_value::exact_capacity_vec(roots.len()).map_err(|e| policy.storage(e))?;
        out.nodes
            .try_reserve(roots.len())
            .map_err(|e| policy.reserve(e))?;
        out.uses
            .try_reserve(roots.len())
            .map_err(|e| policy.reserve(e))?;
        // Retain every requested output root before traversal can yield.
        for &root in roots {
            out.roots.push(root);
            if let std::collections::hash_map::Entry::Vacant(entry) = out.nodes.entry(root) {
                entry.insert(owner::OwnedNode::read(root, policy)?);
            }
            let count = out.uses.entry(root).or_insert(0usize);
            *count = count
                .checked_add(1)
                .ok_or_else(|| policy.invalid("expression root-use count overflows usize"))?;
            pending.push(Visit::Enter(root));
        }
        while let Some(visit) = pending.pop() {
            match visit {
                Visit::Leave(handle) => {
                    active.remove(&handle);
                    done.try_reserve(1)
                        .map_err(|source| policy.reserve(source))?;
                    done.insert(handle);
                    out.order
                        .try_reserve(1)
                        .map_err(|source| policy.reserve(source))?;
                    out.order.push(handle);
                }
                Visit::Enter(handle) => {
                    if active.contains(&handle) {
                        return Err(
                            policy.invalid("optimization expression graph contains a cycle")
                        );
                    }
                    if done.contains(&handle) {
                        continue;
                    }
                    let node = match out.nodes.remove(&handle) {
                        Some(node) => node,
                        None => owner::OwnedNode::read(handle, policy)?,
                    };
                    active
                        .try_reserve(1)
                        .map_err(|source| policy.reserve(source))?;
                    active.insert(handle);
                    pending
                        .try_reserve(1)
                        .map_err(|source| policy.reserve(source))?;
                    pending.push(Visit::Leave(handle));
                    let mut index = 0;
                    while let Some(child) = node.dependency_at(index) {
                        out.uses
                            .try_reserve(1)
                            .map_err(|source| policy.reserve(source))?;
                        let count = out.uses.entry(child).or_insert(0);
                        *count = count.checked_add(1).ok_or_else(|| {
                            policy.invalid("expression edge count overflows usize")
                        })?;
                        pending
                            .try_reserve(1)
                            .map_err(|source| policy.reserve(source))?;
                        pending.push(Visit::Enter(child));
                        index += 1;
                        cooperation.completed(1).await?;
                    }
                    out.nodes
                        .try_reserve(1)
                        .map_err(|source| policy.reserve(source))?;
                    out.nodes.insert(handle, node);
                }
            }
            cooperation.completed(1).await?;
        }
        Ok(out)
    }
}
