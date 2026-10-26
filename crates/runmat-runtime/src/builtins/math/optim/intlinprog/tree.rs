//! Persistent parent links, never per-node copies of a constraint matrix.
use super::*;
use common::conic::linear::ExactLowerBound;
use model::{Bound, Workspace};

pub(super) struct Node {
    parent: Option<usize>,
    change: Option<Bound>,
    pub lower: Option<ExactLowerBound>,
}
pub(super) struct Tree {
    nodes: Vec<Node>,
    pub open: Vec<usize>,
    cutoff_lower: Option<ExactLowerBound>,
}
fn reserve<T>(items: &mut Vec<T>, additional: usize) -> BuiltinResult<()> {
    items.try_reserve(additional).map_err(|source| {
        build_runtime_error("intlinprog: branch tree allocation failed")
            .with_builtin(NAME)
            .with_source(source)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    })
}
impl Tree {
    pub fn new() -> BuiltinResult<Self> {
        let mut result = Self {
            nodes: Vec::new(),
            open: Vec::new(),
            cutoff_lower: None,
        };
        reserve(&mut result.nodes, 1)?;
        reserve(&mut result.open, 1)?;
        result.nodes.push(Node {
            parent: None,
            change: None,
            lower: None,
        });
        result.open.push(0);
        Ok(result)
    }
    pub fn node(&self, id: usize) -> &Node {
        &self.nodes[id]
    }
    pub fn record_cutoff(&mut self, id: usize, cutoff: Option<&num_rational::BigRational>) {
        if let Some(bound) = self.nodes[id]
            .lower
            .as_ref()
            .filter(|b| cutoff.is_some_and(|c| b.value() > c))
        {
            if self
                .cutoff_lower
                .as_ref()
                .is_none_or(|old| bound.value() < old.value())
            {
                self.cutoff_lower = Some(bound.clone());
            }
        }
    }
    pub fn strengthen(&mut self, id: usize, proof: Option<ExactLowerBound>) {
        if let Some(proof) = proof {
            if self.nodes[id]
                .lower
                .as_ref()
                .is_none_or(|old| old.value() < proof.value())
            {
                self.nodes[id].lower = Some(proof);
            }
        }
    }
    pub fn restore_open(&mut self, id: usize) -> BuiltinResult<()> {
        reserve(&mut self.open, 1)?;
        self.open.push(id);
        Ok(())
    }
    pub fn branch(
        &mut self,
        parent: usize,
        column: usize,
        floor: f64,
        ceil: f64,
    ) -> BuiltinResult<()> {
        reserve(&mut self.nodes, 2)?;
        reserve(&mut self.open, 2)?;
        let first = self.nodes.len();
        let bound = self.nodes[parent].lower.clone();
        self.nodes.push(Node {
            parent: Some(parent),
            change: Some(Bound::Lower {
                column,
                value: ceil,
            }),
            lower: bound.clone(),
        });
        self.nodes.push(Node {
            parent: Some(parent),
            change: Some(Bound::Upper {
                column,
                value: floor,
            }),
            lower: bound,
        });
        self.open.push(first);
        self.open.push(first + 1);
        Ok(())
    }
    pub async fn materialize(
        &self,
        id: usize,
        workspace: &mut Workspace,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        workspace.reset(cooperation).await?;
        let mut cursor = Some(id);
        while let Some(id) = cursor {
            let node = &self.nodes[id];
            if let Some(change) = node.change {
                workspace.tighten(change);
            }
            cursor = node.parent;
            cooperation.completed(1).await?;
        }
        workspace.publish_bounds(cooperation).await
    }
    /// None means at least one live leaf has no finite lower certificate.
    pub async fn lower_bound(
        &self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<ExactLowerBound>> {
        self.lower_bound_including(None, cooperation).await
    }
    /// An observation cannot forget the active leaf while it is absent from
    /// the work stack; doing so could falsely strengthen the global bound.
    pub async fn lower_bound_including(
        &self,
        current: Option<usize>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<ExactLowerBound>> {
        let mut lower: Option<ExactLowerBound> = self.cutoff_lower.clone();
        for id in self.open.iter().copied().chain(current) {
            let Some(value) = &self.nodes[id].lower else {
                return Ok(None);
            };
            if lower.as_ref().is_none_or(|old| value.value() < old.value()) {
                lower = Some(value.clone());
            }
            cooperation.completed(1).await?;
        }
        Ok(lower)
    }
}
