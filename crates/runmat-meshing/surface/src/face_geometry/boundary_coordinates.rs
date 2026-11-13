//! One borrowed coordinate authority for all shared curve nodes in a partition.
//! Exact surface samples remain separate from the realized discrete boundary.
use runmat_geometry_core::{
    GeometryAllocationReservation, GeometryEvaluationControl, GeometryEvaluationError,
    GeometryEvaluationErrorKind,
};
use runmat_meshing_core::StableDigest;
use runmat_meshing_curve::{SharedCurveMesh, SharedCurveNode};

pub struct ExactBoundaryNodeCoordinates<'a> {
    nodes: Vec<&'a SharedCurveNode>,
    _storage: GeometryAllocationReservation<'a>,
}
impl<'a> ExactBoundaryNodeCoordinates<'a> {
    pub fn prepare(
        curves: &'a SharedCurveMesh,
        control: &'a dyn GeometryEvaluationControl,
    ) -> Result<Self, GeometryEvaluationError> {
        let mut count = 0usize;
        for edge in &curves.edges {
            control.checkpoint()?;
            count = count
                .checked_add(edge.nodes.len())
                .ok_or_else(|| invalid("boundary node inventory overflow"))?;
        }
        let bytes = count
            .checked_mul(2 * std::mem::size_of::<&SharedCurveNode>())
            .ok_or_else(|| invalid("boundary coordinate index size overflow"))?;
        let storage = GeometryAllocationReservation::new(control, (bytes / 2) as u64)?;
        let mut nodes = Vec::with_capacity(count);
        for edge in &curves.edges {
            for node in &edge.nodes {
                control.checkpoint()?;
                control.consume_search_work(1)?;
                if node.node_id == StableDigest::ZERO
                    || node
                        .coordinates_m
                        .canonical_value()
                        .iter()
                        .any(|v| !v.is_finite())
                {
                    return Err(invalid("invalid shared boundary coordinate"));
                }
                nodes.push(node);
            }
        }
        if nodes.is_empty() {
            return Ok(Self {
                nodes,
                _storage: storage,
            });
        }
        // Small bounded sorts permit cancellation without unwinding a comparator.
        for chunk in nodes.chunks_mut(128) {
            control.checkpoint()?;
            let mut failure = None;
            chunk.sort_unstable_by(|a, b| {
                if failure.is_none() {
                    failure = control.consume_search_work(1).err();
                }
                a.node_id.cmp(&b.node_id)
            });
            if let Some(failure) = failure {
                return Err(failure);
            }
        }
        let scratch_storage = GeometryAllocationReservation::new(control, (bytes / 2) as u64)?;
        let mut scratch = vec![nodes[0]; count];
        let mut width = 128usize;
        while width < count {
            for start in (0..count).step_by(width.saturating_mul(2)) {
                let middle = start.saturating_add(width).min(count);
                let end = middle.saturating_add(width).min(count);
                let (mut left, mut right) = (start, middle);
                for destination in &mut scratch[start..end] {
                    control.consume_search_work(1)?;
                    control.checkpoint()?;
                    if right == end || left < middle && nodes[left].node_id <= nodes[right].node_id
                    {
                        *destination = nodes[left];
                        left += 1;
                    } else {
                        *destination = nodes[right];
                        right += 1;
                    }
                }
            }
            std::mem::swap(&mut nodes, &mut scratch);
            width = width.saturating_mul(2);
        }
        drop(scratch);
        drop(scratch_storage);
        for pair in nodes.windows(2) {
            control.checkpoint()?;
            if pair[0].node_id == pair[1].node_id
                && pair[0].coordinates_m.canonical_value()
                    != pair[1].coordinates_m.canonical_value()
            {
                return Err(invalid(
                    "one shared node identity has inconsistent original curve coordinates",
                ));
            }
        }
        nodes.dedup_by_key(|node| node.node_id);
        Ok(Self {
            nodes,
            _storage: storage,
        })
    }
    pub fn point(
        &self,
        id: StableDigest,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<Option<[f64; 3]>, GeometryEvaluationError> {
        control.checkpoint()?;
        control.consume_search_work((usize::BITS - self.nodes.len().leading_zeros()) as u64 + 1)?;
        Ok(self
            .nodes
            .binary_search_by_key(&id, |node| node.node_id)
            .ok()
            .map(|index| self.nodes[index].coordinates_m.canonical_value()))
    }
}
fn invalid(reason: &str) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::InconsistentGeometry, reason)
}
