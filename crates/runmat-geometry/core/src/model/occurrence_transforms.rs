//! Prepared occurrence lookup borrows paths and composes each placement once.
use super::{
    ExactBRepTopology, GeometryAllocationReservation, GeometryEvaluationControl,
    GeometryEvaluationError, GeometryEvaluationErrorKind, GeometryTransform, PersistentEntityId,
};

pub struct PreparedOccurrenceTransforms<'a> {
    by_path: Vec<(&'a [String], GeometryTransform)>,
    // Includes the admitted construction peak conservatively until this index drops.
    _storage: GeometryAllocationReservation<'a>,
}

impl<'a> PreparedOccurrenceTransforms<'a> {
    /// Requested vector backing bytes, including construction temporaries;
    /// excludes allocator bookkeeping. Callers admit this before preparation.
    pub fn requested_allocation_bytes(topology: &ExactBRepTopology) -> Option<usize> {
        let per_assembly = std::mem::size_of::<(usize, &super::ExactAssembly)>()
            + std::mem::size_of::<Option<GeometryTransform>>()
            + std::mem::size_of::<bool>()
            + std::mem::size_of::<(usize, GeometryTransform)>()
            + std::mem::size_of::<(&[String], GeometryTransform)>();
        topology
            .assemblies
            .len()
            .checked_mul(per_assembly)?
            .checked_add(
                topology
                    .instances
                    .len()
                    .checked_mul(std::mem::size_of::<&super::ExactInstance>())?,
            )
    }

    pub fn new(
        topology: &'a ExactBRepTopology,
        control: &'a dyn GeometryEvaluationControl,
    ) -> Result<Self, GeometryEvaluationError> {
        control.checkpoint()?;
        let requested = Self::requested_allocation_bytes(topology)
            .ok_or_else(|| invalid("occurrence index allocation inventory overflows"))?;
        let storage = GeometryAllocationReservation::new(control, requested as u64)?;
        let mut assemblies = topology.assemblies.iter().enumerate().collect::<Vec<_>>();
        assemblies.sort_unstable_by(|a, b| a.1.id.cmp(&b.1.id));
        if assemblies
            .windows(2)
            .any(|pair| pair[0].1.id == pair[1].1.id)
        {
            return Err(invalid("duplicate occurrence assembly identity"));
        }
        let assembly_index = |id: &PersistentEntityId| {
            assemblies
                .binary_search_by(|(_, assembly)| assembly.id.cmp(id))
                .map(|position| assemblies[position].0)
                .map_err(|_| invalid("occurrence references an absent assembly"))
        };
        control.checkpoint()?;
        let root = assembly_index(&topology.root_assembly_id)?;
        let mut parents = topology.instances.iter().collect::<Vec<_>>();
        parents
            .sort_unstable_by(|a, b| a.instantiated_assembly_id.cmp(&b.instantiated_assembly_id));
        if parents
            .windows(2)
            .any(|pair| pair[0].instantiated_assembly_id == pair[1].instantiated_assembly_id)
        {
            return Err(invalid("occurrence assembly has more than one parent"));
        }
        control.checkpoint()?;
        let mut worlds = vec![None; topology.assemblies.len()];
        worlds[root] = Some(GeometryTransform::identity());
        let mut visiting = vec![false; topology.assemblies.len()];
        let mut chain = Vec::with_capacity(topology.assemblies.len());
        for index in 0..topology.assemblies.len() {
            control.consume_iterations(1)?;
            let mut current = index;
            while worlds[current].is_none() {
                control.checkpoint()?;
                if visiting[current] {
                    return Err(invalid("exact assembly occurrence graph contains a cycle"));
                }
                visiting[current] = true;
                let id = &topology.assemblies[current].id;
                let parent = parents
                    .binary_search_by(|instance| instance.instantiated_assembly_id.cmp(id))
                    .map(|position| parents[position])
                    .map_err(|_| {
                        invalid("entity occurrence is disconnected from the root assembly")
                    })?;
                chain.push((current, parent.transform));
                current = assembly_index(&parent.parent_assembly_id)?;
            }
            let mut world = worlds[current].expect("resolved ancestor");
            while let Some((child, local)) = chain.pop() {
                world = world.compose(&local);
                worlds[child] = Some(world);
                visiting[child] = false;
            }
        }
        let mut by_path = topology
            .assemblies
            .iter()
            .zip(worlds)
            .map(|(assembly, world)| {
                (
                    assembly.id.assembly_path.as_slice(),
                    world.expect("resolved occurrence"),
                )
            })
            .collect::<Vec<_>>();
        by_path.sort_unstable_by(|a, b| a.0.cmp(b.0));
        if by_path.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(invalid("occurrence assembly paths must be unique"));
        }
        control.checkpoint()?;
        Ok(Self {
            by_path,
            _storage: storage,
        })
    }

    pub fn world_transform_for(
        &self,
        entity: &PersistentEntityId,
    ) -> Result<GeometryTransform, GeometryEvaluationError> {
        self.by_path
            .binary_search_by(|(path, _)| path.cmp(&entity.assembly_path.as_slice()))
            .map(|index| self.by_path[index].1)
            .map_err(|_| invalid("entity occurrence path does not resolve to an exact assembly"))
    }
}

fn invalid(reason: &'static str) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::InconsistentGeometry, reason)
}

#[cfg(test)]
mod tests;
