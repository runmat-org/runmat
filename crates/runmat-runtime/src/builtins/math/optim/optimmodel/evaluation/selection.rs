//! Selected ancestry over the ONE original source graph.
use super::*;
use runmat_gc::GcHandle;
use std::collections::{HashMap, HashSet};
pub(in crate::builtins::math::optim::optimmodel) struct Selection {
    pub nodes: HashSet<GcHandle>,
    pub roots: Vec<GcHandle>,
}
impl Selection {
    pub async fn prepare(
        source: &evaluation::Plan,
        roots: Vec<GcHandle>,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut nodes = HashSet::new();
        let mut pending =
            runmat_value::exact_capacity_vec(roots.len()).map_err(|e| policy.storage(e))?;
        pending.extend(roots.iter().copied());
        while let Some(handle) = pending.pop() {
            if nodes.contains(&handle) {
                c.completed(1).await?;
                continue;
            }
            let node = source
                .nodes
                .get(&handle)
                .ok_or_else(|| policy.invalid("selected original node is unavailable"))?;
            nodes.try_reserve(1).map_err(|e| policy.reserve(e))?;
            nodes.insert(handle);
            let mut edge = 0;
            while let Some(child) = node.dependency_at(edge) {
                pending.try_reserve(1).map_err(|e| policy.reserve(e))?;
                pending.push(child);
                edge += 1;
                c.completed(1).await?;
            }
            c.completed(1).await?;
        }
        Ok(Self { nodes, roots })
    }
    pub async fn uses(
        &self,
        source: &evaluation::Plan,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<HashMap<GcHandle, usize>> {
        let mut remaining = HashMap::new();
        remaining
            .try_reserve(self.nodes.len())
            .map_err(|e| policy.reserve(e))?;
        for &handle in &self.nodes {
            let node = source
                .nodes
                .get(&handle)
                .ok_or_else(|| policy.invalid("selected source node is unavailable"))?;
            let mut edge = 0;
            while let Some(child) = node.dependency_at(edge) {
                if !self.nodes.contains(&child) {
                    return Err(policy.invalid("selected source omits dependency"));
                }
                let uses = remaining.entry(child).or_insert(0usize);
                *uses = uses
                    .checked_add(1)
                    .ok_or_else(|| policy.invalid("selected edge-use count overflow"))?;
                edge += 1;
                c.completed(1).await?;
            }
            c.completed(1).await?;
        }
        for &root in &self.roots {
            if !self.nodes.contains(&root) {
                return Err(policy.invalid("selected source omits root"));
            }
            let uses = remaining.entry(root).or_insert(0usize);
            *uses = uses
                .checked_add(1)
                .ok_or_else(|| policy.invalid("selected root-use count overflow"))?;
            c.completed(1).await?;
        }
        Ok(remaining)
    }
}
