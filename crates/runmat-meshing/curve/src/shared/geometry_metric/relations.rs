//! Borrowed incidence views: prepare once, then visit only an entity's own uses.
use runmat_geometry_core::model::PreparedOccurrenceTransforms;
use runmat_geometry_core::{
    ExactBRepTopology, ExactCoedge, ExactEdge, ExactFace, GeometryAllocationReservation,
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind,
    GeometryTransform, PersistentEntityId,
};

pub(super) struct TopologyRelations<'a> {
    topology: &'a ExactBRepTopology,
    edge_uses: UsesByOwner,
    face_uses: UsesByOwner,
    transforms: PreparedOccurrenceTransforms<'a>,
    _storage: GeometryAllocationReservation<'a>,
}

impl<'a> TopologyRelations<'a> {
    pub fn new(
        topology: &'a ExactBRepTopology,
        control: &'a dyn GeometryEvaluationControl,
    ) -> Result<Self, GeometryEvaluationError> {
        control.checkpoint()?;
        let slots = topology
            .edges
            .len()
            .checked_add(topology.faces.len())
            .and_then(|owners| owners.checked_add(topology.coedges.len()))
            .and_then(|count| count.checked_mul(2))
            .and_then(|count| count.checked_add(2))
            .and_then(|count| count.checked_mul(std::mem::size_of::<usize>()))
            .ok_or_else(|| invalid("curve relation allocation inventory overflows"))?;
        let storage = GeometryAllocationReservation::new(control, slots as u64)?;
        for pair in topology.edges.windows(2) {
            control.consume_iterations(1)?;
            if pair[0].id >= pair[1].id {
                return Err(invalid("curve relations require canonical edge order"));
            }
        }
        for pair in topology.faces.windows(2) {
            control.consume_iterations(1)?;
            if pair[0].id >= pair[1].id {
                return Err(invalid("curve relations require canonical face order"));
            }
        }
        let edge_uses = UsesByOwner::build(
            topology.edges.len(),
            topology.coedges.len(),
            |index| {
                topology
                    .edges
                    .binary_search_by(|edge| edge.id.cmp(&topology.coedges[index].edge_id))
                    .map_err(|_| invalid("coedge references an absent edge"))
            },
            control,
        )?;
        let face_uses = UsesByOwner::build(
            topology.faces.len(),
            topology.coedges.len(),
            |index| {
                topology
                    .faces
                    .binary_search_by(|face| face.id.cmp(&topology.coedges[index].face_id))
                    .map_err(|_| invalid("coedge references an absent face"))
            },
            control,
        )?;
        let transforms = PreparedOccurrenceTransforms::new(topology, control)?;
        Ok(Self {
            topology,
            edge_uses,
            face_uses,
            transforms,
            _storage: storage,
        })
    }

    pub fn edge(&self, id: &PersistentEntityId) -> &ExactEdge {
        &self.topology.edges[self
            .topology
            .edges
            .binary_search_by(|edge| edge.id.cmp(id))
            .expect("admitted edge reference")]
    }
    pub fn face(&self, id: &PersistentEntityId) -> &ExactFace {
        &self.topology.faces[self
            .topology
            .faces
            .binary_search_by(|face| face.id.cmp(id))
            .expect("admitted face reference")]
    }
    pub fn edge_coedges(
        &self,
        id: &PersistentEntityId,
    ) -> impl ExactSizeIterator<Item = &ExactCoedge> {
        let owner = self
            .topology
            .edges
            .binary_search_by(|edge| edge.id.cmp(id))
            .expect("admitted edge");
        self.edge_uses
            .row(owner)
            .iter()
            .map(|index| &self.topology.coedges[*index])
    }
    pub fn face_coedges(
        &self,
        id: &PersistentEntityId,
    ) -> impl ExactSizeIterator<Item = &ExactCoedge> {
        let owner = self
            .topology
            .faces
            .binary_search_by(|face| face.id.cmp(id))
            .expect("admitted face");
        self.face_uses
            .row(owner)
            .iter()
            .map(|index| &self.topology.coedges[*index])
    }
    pub fn world_transform_for(
        &self,
        id: &PersistentEntityId,
    ) -> Result<GeometryTransform, GeometryEvaluationError> {
        self.transforms.world_transform_for(id)
    }
}

struct UsesByOwner {
    offsets: Vec<usize>,
    order: Vec<usize>,
}
impl UsesByOwner {
    fn build(
        owners: usize,
        uses: usize,
        owner: impl Fn(usize) -> Result<usize, GeometryEvaluationError>,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<Self, GeometryEvaluationError> {
        let mut offsets = vec![0; owners + 1];
        for index in 0..uses {
            control.consume_iterations(1)?;
            let target = owner(index)?;
            if target >= owners {
                return Err(invalid("relation owner index is outside inventory"));
            }
            offsets[target + 1] += 1;
        }
        for index in 0..owners {
            control.consume_iterations(1)?;
            offsets[index + 1] += offsets[index];
        }
        let mut positions = offsets[..owners].to_vec();
        let mut order = vec![0; uses];
        for index in 0..uses {
            control.consume_iterations(1)?;
            let target = owner(index)?;
            order[positions[target]] = index;
            positions[target] += 1;
        }
        Ok(Self { offsets, order })
    }
    fn row(&self, owner: usize) -> &[usize] {
        &self.order[self.offsets[owner]..self.offsets[owner + 1]]
    }
}

fn invalid(reason: &str) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::InconsistentGeometry, reason)
}

#[cfg(test)]
mod tests;
