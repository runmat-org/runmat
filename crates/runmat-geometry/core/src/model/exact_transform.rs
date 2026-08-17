use std::collections::BTreeMap;

use super::{ExactBRepTopology, GeometryContractError, GeometryTransform, PersistentEntityId};

impl GeometryTransform {
    pub const fn identity() -> Self {
        Self([
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// Returns the row-major matrix product `self * local`.
    pub fn compose(&self, local: &Self) -> Self {
        let mut product = [0.0; 16];
        for row in 0..4 {
            for column in 0..4 {
                product[row * 4 + column] = (0..4)
                    .map(|inner| self.0[row * 4 + inner] * local.0[inner * 4 + column])
                    .sum();
            }
        }
        Self(product)
    }

    pub fn transform_point(&self, point: [f64; 3]) -> [f64; 3] {
        [
            self.0[0] * point[0] + self.0[1] * point[1] + self.0[2] * point[2] + self.0[3],
            self.0[4] * point[0] + self.0[5] * point[1] + self.0[6] * point[2] + self.0[7],
            self.0[8] * point[0] + self.0[9] * point[1] + self.0[10] * point[2] + self.0[11],
        ]
    }

    pub fn transform_vector(&self, vector: [f64; 3]) -> [f64; 3] {
        [
            self.0[0] * vector[0] + self.0[1] * vector[1] + self.0[2] * vector[2],
            self.0[4] * vector[0] + self.0[5] * vector[1] + self.0[6] * vector[2],
            self.0[8] * vector[0] + self.0[9] * vector[1] + self.0[10] * vector[2],
        ]
    }
}

impl ExactBRepTopology {
    /// Resolves the accumulated root-to-occurrence transform for an entity's assembly path.
    pub fn world_transform_for(
        &self,
        entity: &PersistentEntityId,
    ) -> Result<GeometryTransform, GeometryContractError> {
        let assembly = self
            .assemblies
            .iter()
            .find(|assembly| assembly.id.assembly_path == entity.assembly_path)
            .ok_or_else(|| {
                invalid("entity occurrence path does not resolve to an exact assembly")
            })?;
        let parent_by_child = self
            .instances
            .iter()
            .map(|instance| (&instance.instantiated_assembly_id, instance))
            .collect::<BTreeMap<_, _>>();
        let mut current = &assembly.id;
        let mut local_chain = Vec::new();
        while current != &self.root_assembly_id {
            let instance = parent_by_child.get(current).ok_or_else(|| {
                invalid("entity occurrence is disconnected from the root assembly")
            })?;
            local_chain.push(instance.transform);
            current = &instance.parent_assembly_id;
            if local_chain.len() > self.instances.len() {
                return Err(invalid("exact assembly occurrence graph contains a cycle"));
            }
        }
        Ok(local_chain
            .iter()
            .rev()
            .fold(GeometryTransform::identity(), |world, local| {
                world.compose(local)
            }))
    }
}

fn invalid(reason: &'static str) -> GeometryContractError {
    GeometryContractError::invalid("exact occurrence transform", reason)
}
