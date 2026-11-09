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
    pub steady_reference_scratch_bytes: ByteCount,
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
        // Reserve another 2N for steady nodal references/shifted initial guess, or the
        // transient constant nodal reference and retained departure coordinates.
        // Absolute snapshots are retained history, not extra solver scratch; reconstruction
        // writes directly into that destination after energy/flux/storage recovery.
        // Two system masks plus a possible accelerator prepared-mask clone = 3N bools.
        // Accelerator host iteration uses fewer vectors; device/provider storage is excluded.
        // Radiation Newton additionally retains current/trial departure, old/new physical
        // residuals, conductivity product, relative source, capacity rate/product, tangent
        // anchor weights and absolute steady recovery coordinates. Reserve all ten N-vectors
        // conservatively, including linear procedures; no resident ceiling is increased.
        let solver_vector_bytes = add(
            scale(nodes, 24 * 8)?,
            scale(nodes, 3 * std::mem::size_of::<bool>() as u64)?,
        )?;
        // Initial assembly remains live during step assembly: each has applied source,
        // Robin weight,
        // prescribed Option<f64>, conductivity and volumetric capacity. Construction also
        // retains specific-heat and density arrays for the new assembly (two extra E doubles).
        let assembly_vector_bytes = add(
            scale(nodes, 4 * 8 + 2 * std::mem::size_of::<Option<f64>>() as u64)?,
            scale(elements, 6 * 8)?,
        )?;
        // Recovery can hold vector-valued gradients/flux and a flattened destination together.
        // Energy-balance's rate/storage 2N doubles fit in the solver vector envelope.
        // Reserve another N for nodal prescribed reactions and their physical gross scale;
        // boundary/conductive recovery otherwise streams quadrature without retained descriptors.
        let recovery_scratch_bytes = ByteCount(add(scale(elements, 6 * 8)?, scale(nodes, 8)?)?);
        // Scalar source evidence uses one visited-node mask, not an allocation-per-node tree.
        // Temperature-field conflict checking borrows identifiers from the realized inventory;
        // its regional values/owners and selected-node mask coexist with the output vector
        // already included in assembly_vector_bytes. Reserve both phases conservatively.
        let excitation_scratch_bytes = ByteCount(scale(
            nodes,
            2 * std::mem::size_of::<bool>() as u64
                + std::mem::size_of::<Option<(f64, &str)>>() as u64,
        )?);
        // Shared component traversal: seen mask, queue and component node indices, plus
        // at most N component Vec headers. Reference selection adds a node/component map
        // and at most N optional component references; conservatively reserve all together.
        let steady_reference_scratch_bytes = ByteCount(scale(
            nodes,
            std::mem::size_of::<bool>() as u64
                + 3 * std::mem::size_of::<usize>() as u64
                + std::mem::size_of::<Vec<usize>>() as u64
                + std::mem::size_of::<Option<f64>>() as u64,
        )?);
        Ok(Self {
            solver_vector_bytes: ByteCount(solver_vector_bytes),
            assembly_vector_bytes: ByteCount(assembly_vector_bytes),
            recovery_scratch_bytes,
            excitation_scratch_bytes,
            steady_reference_scratch_bytes,
            // The assembly retains one projection: a finite-conductance split rewrites it in place.
            projection_estimated_bytes: projection,
        })
    }

    pub fn reservation_bytes(&self) -> Result<ByteCount, ResourcePlanningError> {
        [
            self.solver_vector_bytes,
            self.assembly_vector_bytes,
            self.recovery_scratch_bytes,
            self.excitation_scratch_bytes,
            self.steady_reference_scratch_bytes,
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
        assert_eq!(facts.solver_vector_bytes.0, 10 * (24 * 8 + 3));
        assert_eq!(
            facts.assembly_vector_bytes.0,
            10 * (32 + 2 * std::mem::size_of::<Option<f64>>() as u64) + 2 * 48
        );
        assert_eq!(facts.recovery_scratch_bytes.0, 96 + 10 * 8);
        assert_eq!(
            facts.excitation_scratch_bytes.0,
            10 * (2 + std::mem::size_of::<Option<(f64, &str)>>() as u64)
        );
        assert_eq!(facts.projection_estimated_bytes.0, 100);
        assert_eq!(
            facts.steady_reference_scratch_bytes.0,
            10 * (std::mem::size_of::<bool>() as u64
                + 3 * std::mem::size_of::<usize>() as u64
                + std::mem::size_of::<Vec<usize>>() as u64
                + std::mem::size_of::<Option<f64>>() as u64)
        );
        for (nodes, elements, projection) in [(u64::MAX, 1, 1), (1, u64::MAX, 1)] {
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
