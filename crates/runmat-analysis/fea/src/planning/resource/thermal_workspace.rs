//! Thermal host workspace lifetimes; excludes sparse payloads and retained field histories.
use crate::topology::AnalysisProjection;

use super::{ByteCount, ResourcePlanningError, TopologyResourceFacts};

/// Logical vector payloads plus the shared topology resident estimate, not measured RSS.
/// Builders, allocation capacity/headers, dynamic evidence, model/property maps, observations,
/// serialization, and provider-owned storage need separate admission.
#[derive(Clone, Debug)]
pub struct ThermalWorkspaceResourceFacts {
    pub solver_vector_bytes: ByteCount,
    pub assembly_vector_bytes: ByteCount,
    pub recovery_scratch_bytes: ByteCount,
    pub excitation_scratch_bytes: ByteCount,
    pub projection_estimated_bytes: ByteCount,
}

impl ThermalWorkspaceResourceFacts {
    pub(crate) fn derive(projection: &AnalysisProjection) -> Result<Self, ResourcePlanningError> {
        let topology = TopologyResourceFacts::from_projection(projection)?;
        Self::from_counts(
            topology.nodes.0,
            topology.elements.0,
            topology.estimated_resident_bytes,
        )
    }

    fn from_counts(
        nodes: u64,
        elements: u64,
        projection: ByteCount,
    ) -> Result<Self, ResourcePlanningError> {
        // Jacobi-PCG: diagonal + x/r/z/p + ap + replacement z = 7N doubles.
        // Derived/prepared RHS = 2N; driver initial/previous/prior solution = 3N.
        // Two system masks plus a possible accelerator prepared-mask clone = 3N bools.
        // Accelerator host iteration uses fewer vectors; device/provider storage is excluded.
        let solver_vector_bytes = add(
            scale(nodes, 12 * 8)?,
            scale(nodes, 3 * std::mem::size_of::<bool>() as u64)?,
        )?;
        // Initial assembly remains live during step assembly: each has source, Robin weight,
        // prescribed Option<f64>, conductivity and volumetric capacity. Construction also
        // retains specific-heat and density arrays for the new assembly (two extra E doubles).
        let assembly_vector_bytes = add(
            scale(nodes, 4 * 8 + 2 * std::mem::size_of::<Option<f64>>() as u64)?,
            scale(elements, 6 * 8)?,
        )?;
        // Recovery can hold vector-valued gradients/flux and a flattened destination together.
        // Energy-balance's 3N doubles fit inside the separately reserved solver vector envelope.
        let recovery_scratch_bytes = ByteCount(scale(elements, 6 * 8)?);
        // Scalar source evidence uses one visited-node mask, not an allocation-per-node tree.
        // Temperature-field conflict checking borrows identifiers from the realized inventory;
        // its regional values/owners and selected-node mask coexist with the output vector
        // already included in assembly_vector_bytes. Reserve both phases conservatively.
        let excitation_scratch_bytes = ByteCount(scale(
            nodes,
            2 * std::mem::size_of::<bool>() as u64
                + std::mem::size_of::<Option<(f64, &str)>>() as u64,
        )?);
        Ok(Self {
            solver_vector_bytes: ByteCount(solver_vector_bytes),
            assembly_vector_bytes: ByteCount(assembly_vector_bytes),
            recovery_scratch_bytes,
            excitation_scratch_bytes,
            projection_estimated_bytes: ByteCount(scale(projection.0, 2)?),
        })
    }

    pub fn reservation_bytes(&self) -> Result<ByteCount, ResourcePlanningError> {
        [
            self.solver_vector_bytes,
            self.assembly_vector_bytes,
            self.recovery_scratch_bytes,
            self.excitation_scratch_bytes,
            self.projection_estimated_bytes,
        ]
        .into_iter()
        .try_fold(0, |total, bytes| add(total, bytes.0))
        .map(ByteCount)
    }
}

fn scale(count: u64, width: u64) -> Result<u64, ResourcePlanningError> {
    count
        .checked_mul(width)
        .ok_or(ResourcePlanningError::Overflow("thermal host workspace"))
}

fn add(left: u64, right: u64) -> Result<u64, ResourcePlanningError> {
    left.checked_add(right)
        .ok_or(ResourcePlanningError::Overflow("thermal host workspace"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_counts_live_arrays_and_checks_overflow() {
        let facts = ThermalWorkspaceResourceFacts::from_counts(10, 2, ByteCount(100)).unwrap();
        assert_eq!(facts.solver_vector_bytes.0, 10 * (12 * 8 + 3));
        assert_eq!(
            facts.assembly_vector_bytes.0,
            10 * (32 + 2 * std::mem::size_of::<Option<f64>>() as u64) + 2 * 48
        );
        assert_eq!(facts.recovery_scratch_bytes.0, 96);
        assert_eq!(
            facts.excitation_scratch_bytes.0,
            10 * (2 + std::mem::size_of::<Option<(f64, &str)>>() as u64)
        );
        assert_eq!(facts.projection_estimated_bytes.0, 200);
        for (nodes, elements, projection) in [(u64::MAX, 1, 1), (1, u64::MAX, 1), (1, 1, u64::MAX)]
        {
            assert!(ThermalWorkspaceResourceFacts::from_counts(
                nodes,
                elements,
                ByteCount(projection)
            )
            .is_err());
        }
        let mut overflow = facts;
        overflow.projection_estimated_bytes = ByteCount(u64::MAX);
        assert!(overflow.reservation_bytes().is_err());
    }
}
