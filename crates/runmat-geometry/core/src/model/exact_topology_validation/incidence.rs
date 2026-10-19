//! Dense incidence over validated canonical identifiers, without cloned keys.
use super::super::validation_storage::{IdIndex, ValidationBuffer, ValidationWork};
use crate::{GeometryContractError, PersistentEntityId, TopologicalOrientation};
#[derive(Clone, Copy, Default)]
pub(super) struct FaceUses {
    pub count: usize,
    forward: bool,
    reverse: bool,
}
impl FaceUses {
    pub fn contains(&self, orientation: TopologicalOrientation) -> bool {
        match orientation {
            TopologicalOrientation::Forward => self.forward,
            TopologicalOrientation::Reversed => self.reverse,
        }
    }
}
pub(super) struct FaceIncidence<'index, 'data> {
    index: &'index IdIndex<'data>,
    uses: ValidationBuffer<'data, FaceUses>,
}
impl<'index, 'data> FaceIncidence<'index, 'data> {
    pub fn new(
        index: &'index IdIndex<'data>,
        work: ValidationWork<'data>,
    ) -> Result<Self, GeometryContractError> {
        let mut uses = work.buffer(index.len())?;
        for _ in 0..index.len() {
            uses.push(FaceUses::default());
        }
        Ok(Self { index, uses })
    }
    pub fn push(
        &mut self,
        id: &PersistentEntityId,
        orientation: TopologicalOrientation,
    ) -> Result<(), GeometryContractError> {
        let slot = self.index.index(id).ok_or_else(|| {
            GeometryContractError::invalid("shell face incidence", "face index is incomplete")
        })?;
        let uses = &mut self.uses[slot];
        uses.count += 1;
        match orientation {
            TopologicalOrientation::Forward => uses.forward = true,
            TopologicalOrientation::Reversed => uses.reverse = true,
        }
        Ok(())
    }
    pub fn get(&self, id: &PersistentEntityId) -> Option<&FaceUses> {
        self.index.index(id).map(|slot| &self.uses[slot])
    }
}
pub(super) struct WireOwners<'index, 'data> {
    index: &'index IdIndex<'data>,
    owners: ValidationBuffer<'data, Option<&'data PersistentEntityId>>,
    count: usize,
}
impl<'index, 'data> WireOwners<'index, 'data> {
    pub fn new(
        index: &'index IdIndex<'data>,
        work: ValidationWork<'data>,
    ) -> Result<Self, GeometryContractError> {
        let mut owners = work.buffer(index.len())?;
        for _ in 0..index.len() {
            owners.push(None);
        }
        Ok(Self {
            index,
            owners,
            count: 0,
        })
    }
    pub fn claim(
        &mut self,
        wire: &PersistentEntityId,
        face: &'data PersistentEntityId,
    ) -> Result<(), GeometryContractError> {
        let slot = self.index.index(wire).ok_or_else(|| {
            GeometryContractError::invalid("wire ownership", "wire index is incomplete")
        })?;
        if self.owners[slot].replace(face).is_some() {
            return Err(GeometryContractError::invalid(
                "wire ownership",
                "wire has multiple face owners",
            ));
        }
        self.count += 1;
        Ok(())
    }
    pub fn complete(&self) -> bool {
        self.count == self.index.len()
    }
    pub fn get(&self, wire: &PersistentEntityId) -> Option<&'data PersistentEntityId> {
        self.index.index(wire).and_then(|slot| self.owners[slot])
    }
}
