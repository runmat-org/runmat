//! Construction allocations belong to the consumer's evaluation scope. Charge
//! before allocation; scratch uses a child scope and retained data outlives its
//! charge until the execution owner transfers it into the output reservation.
use super::{error, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind, ProjectionInput};
use runmat_geometry_core::{GeometryEvaluationControl, PersistentEntityId};

#[derive(Clone, Copy)]
pub(super) struct Storage<'a>(pub Option<&'a dyn GeometryEvaluationControl>);
impl<'a> Storage<'a> {
    pub fn for_input(input: &ProjectionInput<'a>) -> Self {
        Self(input.exact_evaluation.map(|evaluation| evaluation.control))
    }
    pub fn charge(self, bytes: u64) -> Result<(), DelaunaySolverTopologyError> {
        if let Some(control) = self.0 {
            control.checkpoint().map_err(error::geometry)?;
            control
                .consume_allocation_bytes(bytes)
                .map_err(error::geometry)?;
        }
        Ok(())
    }
    pub fn slots<T>(self, count: usize) -> Result<(), DelaunaySolverTopologyError> {
        self.charge(bytes::<T>(count)?)
    }
    pub fn vector<T>(self, count: usize) -> Result<Vec<T>, DelaunaySolverTopologyError> {
        let mut result = Vec::new();
        self.reserve(&mut result, count)?;
        Ok(result)
    }
    pub fn collect<T>(
        self,
        values: impl Iterator<Item = T> + Clone,
    ) -> Result<Vec<T>, DelaunaySolverTopologyError> {
        let mut result = self.vector(values.clone().count())?;
        result.extend(values);
        Ok(result)
    }
    /// Reallocation admits old and new buffers together, then releases the old
    /// charge. No guessed allocator growth factor or silent over-reserve.
    pub fn reserve<T>(
        self,
        values: &mut Vec<T>,
        capacity: usize,
    ) -> Result<(), DelaunaySolverTopologyError> {
        if capacity <= values.capacity() {
            return Ok(());
        }
        let old_bytes = bytes::<T>(values.capacity())?;
        let new_bytes = bytes::<T>(capacity)?;
        self.charge(new_bytes)?;
        if values.try_reserve_exact(capacity - values.len()).is_err() {
            if let Some(control) = self.0 {
                control.release_allocation_bytes(new_bytes);
            }
            return Err(overflow());
        }
        if let Some(control) = self.0 {
            control.release_allocation_bytes(old_bytes);
        }
        Ok(())
    }
    pub fn push<T>(self, values: &mut Vec<T>, value: T) -> Result<(), DelaunaySolverTopologyError> {
        if values.len() == values.capacity() {
            let capacity = values.capacity().saturating_mul(2).max(4);
            self.reserve(values, capacity)?;
        }
        values.push(value);
        Ok(())
    }
    pub fn release_backing<T>(self, values: Vec<T>) -> Result<(), DelaunaySolverTopologyError> {
        let requested = bytes::<T>(values.capacity())?;
        drop(values);
        if let Some(control) = self.0 {
            control.release_allocation_bytes(requested);
        }
        Ok(())
    }
    pub fn entity(
        self,
        value: &PersistentEntityId,
    ) -> Result<PersistentEntityId, DelaunaySolverTopologyError> {
        // A clone is a handle to the shared identity allocation; its slot is charged
        // with the owning record.
        Ok(value.clone())
    }
    pub fn entities<'b>(
        self,
        values: impl Iterator<Item = &'b PersistentEntityId> + Clone,
    ) -> Result<Vec<PersistentEntityId>, DelaunaySolverTopologyError> {
        let mut result = self.vector(values.clone().count())?;
        for value in values {
            result.push(self.entity(value)?);
        }
        Ok(result)
    }
    pub fn parameter(
        self,
        value: &runmat_meshing_core::SolverNodeExactParameter,
    ) -> Result<runmat_meshing_core::SolverNodeExactParameter, DelaunaySolverTopologyError> {
        use runmat_meshing_core::SolverNodeExactParameter;
        Ok(match value {
            SolverNodeExactParameter::Curve {
                source_edge_id,
                parameter,
            } => SolverNodeExactParameter::Curve {
                source_edge_id: self.entity(source_edge_id)?,
                parameter: *parameter,
            },
            SolverNodeExactParameter::Surface {
                source_face_id,
                chart_id,
                evaluator_uv,
            } => SolverNodeExactParameter::Surface {
                source_face_id: self.entity(source_face_id)?,
                chart_id: *chart_id,
                evaluator_uv: *evaluator_uv,
            },
        })
    }
    pub fn parameters(
        self,
        values: &[runmat_meshing_core::SolverNodeExactParameter],
    ) -> Result<Vec<runmat_meshing_core::SolverNodeExactParameter>, DelaunaySolverTopologyError>
    {
        let mut result = self.vector(values.len())?;
        for value in values {
            result.push(self.parameter(value)?);
        }
        Ok(result)
    }
    pub fn node(
        self,
        value: &runmat_meshing_core::SolverMeshNode,
    ) -> Result<runmat_meshing_core::SolverMeshNode, DelaunaySolverTopologyError> {
        Ok(runmat_meshing_core::SolverMeshNode {
            node_id: value.node_id,
            stable_identity: value.stable_identity,
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(value.coordinates_m.canonical_value()),
            provenance: self.entities(value.provenance.iter())?,
            exact_parameters: self.parameters(&value.exact_parameters)?,
        })
    }
    pub fn string(self, value: &str) -> Result<String, DelaunaySolverTopologyError> {
        self.slots::<u8>(value.len())?;
        Ok(value.to_owned())
    }
}
fn bytes<T>(count: usize) -> Result<u64, DelaunaySolverTopologyError> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(overflow)
}
fn overflow() -> DelaunaySolverTopologyError {
    error::failure(
        DelaunaySolverTopologyErrorKind::MemoryBudgetExceeded,
        "solver construction storage exceeds its admitted address space",
    )
}

#[cfg(test)]
mod tests;

mod buffer;
pub(super) use buffer::ReservedBuffer;
