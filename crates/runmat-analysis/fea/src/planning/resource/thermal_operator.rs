//! Preassembly scalar thermal support and shared exact-solver work reservation.
use std::collections::BTreeSet;

use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    algebra::{RealSparseMatrix, SparsePattern},
    topology::AnalysisProjection,
};

use super::{
    symbolic::pattern_from_rows, ByteCount, ResourcePlanningError, SolverAdmissionResourcePolicy,
    ThermalHistoryResourceFacts, ThermalWorkspaceResourceFacts,
};

mod preparation;

/// Material-independent support, including every volume and boundary coupling and each diagonal.
/// This is numerical component accounting, not a worker or whole-process resource plan.
#[derive(Clone, Debug)]
pub struct ThermalOperatorResourceFacts {
    topology_digest: Digest,
    support: SparsePattern,
    admission: SolverAdmissionResourcePolicy,
    csr_payload_bytes: ByteCount,
    support_payload_bytes: ByteCount,
    workspace: ThermalWorkspaceResourceFacts,
    planning_resident_bytes: ByteCount,
}

impl ThermalOperatorResourceFacts {
    /// Thermal interfaces with a finite conductance give their interface nodes a second-side
    /// dof; the support is that of the split projection plus the conductance couplings.
    pub fn derive(
        mesh: &SolverMeshArtifact,
        interfaces: &[runmat_analysis_core::AnalysisInterface],
    ) -> Result<Self, ResourcePlanningError> {
        let planning_resident_bytes = Self::derivation_resident_bytes_estimate(mesh)?;
        let projection = AnalysisProjection::derive(mesh)
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        let (plans, dofs) =
            crate::physics::thermal_interface::solid_conduction(&projection, interfaces)
                .map_err(ResourcePlanningError::Invalid)?;
        let projection = dofs.split_projection(&projection, &plans);
        let workspace = ThermalWorkspaceResourceFacts::derive(&projection)?;
        let mut rows = (0..projection.nodes.len())
            .map(|index| BTreeSet::from([index]))
            .collect::<Vec<_>>();
        for nodes in projection
            .elements
            .iter()
            .map(|element| element.node_indices.as_slice())
            .chain(
                projection
                    .boundary_faces
                    .iter()
                    .map(|face| face.node_indices.as_slice()),
            )
        {
            for &row in nodes {
                rows[row].extend(nodes.iter().copied());
            }
        }
        dofs.extend_support(&projection, &plans, &mut rows);
        let support = pattern_from_rows(rows, projection.nodes.len())?;
        // Thermal assembly captures no local-energy source, so coercivity is always proved by the
        // sparse Cholesky factor that then serves the solve. The admission is the factor law that
        // structural statics uses; the graph-count local-energy budget is not reserved.
        let admission = super::preflight::factor_admission(&support)?;
        let rows = u64::try_from(support.rows())
            .map_err(|_| ResourcePlanningError::Overflow("thermal rows"))?;
        let entries = u64::try_from(support.column_indices().len())
            .map_err(|_| ResourcePlanningError::Overflow("thermal sparse entries"))?;
        // Two live assemblies (K,C each), derived effective K and cloned C, prepared K,
        // plus a possible accelerator prepared-system clone and the explicitly stored diagonal
        // zero damping matrix. This excludes provider-owned buffers. Steady/CPU use fewer copies.
        let diagonal_bytes = csr_bytes(rows, rows)?;
        // Nonlinear radiation also constructs the shared-support tangent assembler and
        // finalized Jacobian while the conductivity/capacity assemblies remain live.
        let csr_payload_bytes = csr_bytes(rows, entries)?
            .checked_mul(10)
            .and_then(|bytes| bytes.checked_add(diagonal_bytes))
            .ok_or(ResourcePlanningError::Overflow(
                "thermal simultaneous CSR payloads",
            ))?;
        Ok(Self {
            topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
            support,
            admission,
            csr_payload_bytes: ByteCount(csr_payload_bytes),
            support_payload_bytes: ByteCount(pattern_bytes(rows, entries)?),
            workspace,
            planning_resident_bytes,
        })
    }

    pub fn topology_digest(&self) -> Digest {
        self.topology_digest
    }
    pub fn support(&self) -> &SparsePattern {
        &self.support
    }
    pub fn solver_admission(&self) -> &SolverAdmissionResourcePolicy {
        &self.admission
    }
    pub fn workspace(&self) -> &ThermalWorkspaceResourceFacts {
        &self.workspace
    }
    /// Logical row-offset, column-index and value lengths, not allocation capacities.
    /// Sparse-builder storage is an additional allocation.
    pub fn csr_payload_bytes_upper_bound(&self) -> ByteCount {
        self.csr_payload_bytes
    }

    /// Successful proof work is reserved from the shared graph-count envelope, not from a
    /// symbolic factorization of the superset (whose ordering could underestimate exact fill).
    pub fn numerical_reservation_bytes(
        &self,
        history: &ThermalHistoryResourceFacts,
    ) -> Result<ByteCount, ResourcePlanningError> {
        let workspace = self.workspace.reservation_bytes()?.0;
        let history_bytes = history.reservation_bytes()?.0;
        // Support derivation finishes (and frees its row sets and symbolic analysis) before any
        // operator is assembled or factored; only the derived support and history persist.
        let numerical = self
            .csr_payload_bytes
            .0
            .checked_add(self.admission.resident_bytes_upper_bound.0)
            .and_then(|bytes| bytes.checked_add(workspace));
        tracing::info!(
            dimension = self.admission.dimension.0,
            support_entries = self.support.column_indices().len(),
            csr_payload_bytes = self.csr_payload_bytes.0,
            factor_nonzeros = self.admission.factor_nonzero_upper_bound.0,
            factor_resident_bytes = self.admission.resident_bytes_upper_bound.0,
            workspace_bytes = workspace,
            planning_resident_bytes = self.planning_resident_bytes.0,
            support_payload_bytes = self.support_payload_bytes.0,
            history_bytes,
            "thermal numerical reservation terms"
        );
        numerical
            .map(|bytes| bytes.max(self.planning_resident_bytes.0))
            .and_then(|bytes| bytes.checked_add(self.support_payload_bytes.0))
            .and_then(|bytes| bytes.checked_add(history_bytes))
            .map(ByteCount)
            .ok_or(ResourcePlanningError::Overflow(
                "thermal numerical reservation",
            ))
    }

    pub(crate) fn require_fits(
        &self,
        history: &ThermalHistoryResourceFacts,
        maximum: u64,
    ) -> Result<(), ResourcePlanningError> {
        let required = self.numerical_reservation_bytes(history)?.0;
        if required > maximum {
            return Err(ResourcePlanningError::AccuracyCeiling {
                resource: "thermal numerical and topology reservation",
                required,
                maximum,
            });
        }
        Ok(())
    }

    pub(crate) fn verify_matrix(
        &self,
        matrix: &RealSparseMatrix,
    ) -> Result<(), ResourcePlanningError> {
        if !matches!(
            self.support.first_uncovered_entry(matrix.pattern()),
            Ok(None)
        ) {
            return Err(ResourcePlanningError::Invalid(
                "assembled thermal operator exceeds preassembly support".into(),
            ));
        }
        Ok(())
    }
}

fn csr_bytes(rows: u64, entries: u64) -> Result<u64, ResourcePlanningError> {
    pattern_bytes(rows, entries)?
        .checked_add(
            entries
                .checked_mul(8)
                .ok_or(ResourcePlanningError::Overflow("thermal CSR values"))?,
        )
        .ok_or(ResourcePlanningError::Overflow("thermal CSR payload"))
}

fn pattern_bytes(rows: u64, entries: u64) -> Result<u64, ResourcePlanningError> {
    rows.checked_add(1)
        .and_then(|count| count.checked_mul(8))
        .and_then(|offsets| {
            entries
                .checked_mul(8)
                .and_then(|values| offsets.checked_add(values))
        })
        .ok_or(ResourcePlanningError::Overflow("thermal CSR payload"))
}

#[cfg(test)]
mod tests;
