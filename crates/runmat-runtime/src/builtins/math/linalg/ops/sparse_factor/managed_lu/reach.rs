//! Iterative sparse graph reach, adapted from faer 0.24.4 simplicial LU.
//! See FAER-LICENSE.txt. Each edge is visited, not each absent matrix row.
use super::{ClaimedVec, ManagedSparseLuError, Work};

pub(super) struct Reach {
    marked: ClaimedVec<usize>,
    nodes: ClaimedVec<usize>,
    edges: ClaimedVec<usize>,
    pub(super) pattern: ClaimedVec<usize>,
}
impl Reach {
    pub(super) async fn new(n: usize, work: &mut Work<'_>) -> Result<Self, ManagedSparseLuError> {
        Ok(Self {
            marked: ClaimedVec::zeroed(n, work).await?,
            nodes: ClaimedVec::zeroed(n, work).await?,
            edges: ClaimedVec::zeroed(n, work).await?,
            pattern: ClaimedVec::zeroed(n, work).await?,
        })
    }

    pub(super) async fn column(
        &mut self,
        input_rows: impl Iterator<Item = usize>,
        epoch: usize,
        inverse: &[usize],
        l_ptr: &[usize],
        l_rows: &[usize],
        work: &mut Work<'_>,
    ) -> Result<usize, ManagedSparseLuError> {
        let n = inverse.len();
        let mut tail = n;
        for root in input_rows {
            work.tick().await?;
            if self.marked.as_slice()[root] == epoch {
                continue;
            }
            let mut depth = 1;
            self.marked.as_mut_slice()[root] = epoch;
            self.nodes.as_mut_slice()[0] = root;
            let pivot = inverse[root];
            self.edges.as_mut_slice()[0] = if pivot < epoch - 1 {
                l_ptr[pivot] + 1
            } else {
                0
            };
            while depth != 0 {
                let top = depth - 1;
                let node = self.nodes.as_slice()[top];
                let pivot = inverse[node];
                let end = if pivot < epoch - 1 {
                    l_ptr[pivot + 1]
                } else {
                    0
                };
                let edge = self.edges.as_slice()[top];
                work.tick().await?;
                if edge < end {
                    self.edges.as_mut_slice()[top] = edge + 1;
                    let child = l_rows[edge];
                    if self.marked.as_slice()[child] != epoch {
                        self.marked.as_mut_slice()[child] = epoch;
                        self.nodes.as_mut_slice()[depth] = child;
                        let child_pivot = inverse[child];
                        self.edges.as_mut_slice()[depth] = if child_pivot < epoch - 1 {
                            l_ptr[child_pivot] + 1
                        } else {
                            0
                        };
                        depth += 1;
                    }
                } else {
                    depth -= 1;
                    tail -= 1;
                    self.pattern.as_mut_slice()[tail] = node;
                }
            }
        }
        Ok(tail)
    }
}
