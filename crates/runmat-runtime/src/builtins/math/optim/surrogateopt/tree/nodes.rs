//! Finite FIFO branch metadata. Parent links point strictly backwards; only
//! changed integer coordinates are retained, never per-node points or boxes.
use super::super::{invalid, NAME};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use std::collections::VecDeque;

#[derive(Clone, Copy)]
pub(super) struct Index(usize);
#[derive(Clone, Copy)]
enum Link {
    Root,
    Branch {
        parent: Index,
        column: usize,
        value: f64,
    },
}
struct Node {
    link: Link,
    depth: usize,
    fingerprint: u64,
}
pub(super) struct Nodes<'a> {
    anchor: &'a [f64],
    nodes: Vec<Node>,
    queue: VecDeque<Index>,
    capacity: usize,
}
impl<'a> Nodes<'a> {
    pub(super) fn new(anchor: &'a [f64], steps: usize) -> BuiltinResult<Self> {
        let capacity = steps
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| invalid("tree metadata extent overflow"))?;
        let mut nodes = runmat_value::exact_capacity_vec(capacity).map_err(allocation)?;
        let mut queue = VecDeque::new();
        queue.try_reserve_exact(capacity).map_err(|source| {
            crate::build_runtime_error("surrogateopt: tree FIFO allocation failed")
                .with_builtin(NAME)
                .with_source(source)
                .build()
        })?;
        nodes.push(Node {
            link: Link::Root,
            depth: 0,
            fingerprint: 0,
        });
        queue.push_back(Index(0));
        Ok(Self {
            anchor,
            nodes,
            queue,
            capacity,
        })
    }
    pub(super) fn next(&mut self) -> Option<Index> {
        self.queue.pop_front()
    }
    fn node(&self, index: Index) -> BuiltinResult<&Node> {
        self.nodes
            .get(index.0)
            .ok_or_else(|| invalid("invalid original tree node"))
    }
    pub(super) fn depth(&self, index: Index) -> BuiltinResult<usize> {
        Ok(self.node(index)?.depth)
    }
    pub(super) async fn value(
        &self,
        mut index: Index,
        column: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<f64> {
        if column >= self.anchor.len() {
            return Err(invalid("invalid tree coordinate"));
        }
        loop {
            cooperation.completed(1).await?;
            match self.node(index)?.link {
                Link::Root => return Ok(self.anchor[column]),
                Link::Branch {
                    parent,
                    column: changed,
                    value,
                } => {
                    if changed == column {
                        return Ok(value);
                    }
                    index = parent;
                }
            }
        }
    }
    pub(super) async fn point(
        &self,
        index: Index,
        integers: &[usize],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<f64>> {
        let mut point = runmat_value::exact_capacity_vec(self.anchor.len()).map_err(allocation)?;
        for &value in self.anchor {
            point.push(value);
            cooperation.completed(1).await?;
        }
        for &column in integers {
            point[column] = self.value(index, column, cooperation).await?;
        }
        Ok(point)
    }
    pub(super) async fn append(
        &mut self,
        parent: Index,
        column: usize,
        value: f64,
        integers: &[usize],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if self.nodes.len() == self.capacity {
            return Ok(());
        }
        if !value.is_finite() || value.fract() != 0.0 || integers.binary_search(&column).is_err() {
            return Err(invalid("invalid actual tree assignment"));
        }
        let old = self.value(parent, column, cooperation).await?;
        if old == value {
            return Ok(());
        }
        let original = self.node(parent)?;
        let depth = original
            .depth
            .checked_add(1)
            .ok_or_else(|| invalid("tree depth overflow"))?;
        let fingerprint = original.fingerprint ^ mix(column, old) ^ mix(column, value);
        for (row, node) in self.nodes.iter().enumerate() {
            cooperation.completed(1).await?;
            if node.fingerprint != fingerprint {
                continue;
            }
            let mut same = true;
            for &coordinate in integers {
                let proposed = if coordinate == column {
                    value
                } else {
                    self.value(parent, coordinate, cooperation).await?
                };
                same &= proposed == self.value(Index(row), coordinate, cooperation).await?;
                if !same {
                    break;
                }
            }
            // Fingerprints only prefilter; actual scalar assignments prove
            // equality. Collisions cannot suppress a different branch.
            if same {
                return Ok(());
            }
        }
        let index = Index(self.nodes.len());
        self.nodes.push(Node {
            link: Link::Branch {
                parent,
                column,
                value,
            },
            depth,
            fingerprint,
        });
        self.queue.push_back(index);
        Ok(())
    }
}
fn allocation(source: runmat_value::ExactHostNumericAllocationError) -> crate::RuntimeError {
    crate::build_runtime_error("surrogateopt: tree metadata/working allocation failed")
        .with_builtin(NAME)
        .with_source(source)
        .build()
}
fn mix(column: usize, value: f64) -> u64 {
    let bits = if value == 0.0 { 0 } else { value.to_bits() };
    let mut key = bits ^ (column as u64).wrapping_mul(0x9e3779b97f4a7c15);
    key = (key ^ (key >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    key = (key ^ (key >> 27)).wrapping_mul(0x94d049bb133111eb);
    key ^ (key >> 31)
}
