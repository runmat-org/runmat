//! Preassembly scalar thermal support and shared exact-solver work reservation.
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    algebra::{RealSparseMatrix, SparsePattern},
    topology::AnalysisProjection,
};

use super::{
    ByteCount, ResourcePlanningError, SolverAdmissionResourcePolicy,
    ThermalHistoryResourceFacts, ThermalWorkspaceResourceFacts,
};

mod preparation;
mod support;

/// How a thermal solve proves its operator coercive and solves it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThermalSolvePath {
    /// The sparse Cholesky factor proves coercivity and solves.
    ProvingFactorization,
    /// Structural evidence proves the steady conduction operator positive definite: positive
    /// element conductivity, nonnegative convection coefficients, and a prescribed temperature
    /// or positive convection on every connected component of the operator graph (see
    /// [`crate::algebra::ScalarConductionProof`]). AMG-preconditioned CG solves; the normwise
    /// backward error of the returned temperature accepts it. No global factor is held.
    ConductionMultigrid,
}

/// Material-independent support, including every volume and boundary coupling and each diagonal.
/// This is numerical component accounting, not a worker or whole-process resource plan.
#[derive(Clone, Debug)]
pub struct ThermalOperatorResourceFacts {
    topology_digest: Digest,
    support: SparsePattern,
    path: ThermalSolvePath,
    /// The proving factor's admission, kept on both paths: it is the factor estimate the path
    /// selection compares with the run's budget.
    factor_admission: SolverAdmissionResourcePolicy,
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
        let projection = AnalysisProjection::derive(mesh)
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        let (plans, dofs) =
            crate::physics::thermal_interface::solid_conduction(&projection, interfaces)
                .map_err(ResourcePlanningError::Invalid)?;
        // A nodal interface inventory leaves the projection unchanged; only a finite
        // conductance splits it into a second copy.
        let split;
        let projection = if dofs.is_nodal() {
            &projection
        } else {
            split = dofs.split_projection(&projection, &plans);
            &split
        };
        let workspace = ThermalWorkspaceResourceFacts::derive(projection)?;
        let extra = dofs.support_cliques(projection, &plans);
        let elements = projection.elements.len();
        let faces = projection.boundary_faces.len();
        let support = support::build(
            projection.nodes.len(),
            elements + faces + extra.len(),
            |clique| {
                if clique < elements {
                    projection.elements[clique].node_indices.as_slice()
                } else if clique < elements + faces {
                    projection.boundary_faces[clique - elements]
                        .node_indices
                        .as_slice()
                } else {
                    extra[clique - elements - faces].as_slice()
                }
            },
        )?;
        drop(extra);
        let planning_resident_bytes =
            Self::derivation_resident_bytes(mesh, Some(support.column_indices().len() as u64))?;
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
            path: ThermalSolvePath::ProvingFactorization,
            factor_admission: admission.clone(),
            admission,
            csr_payload_bytes: ByteCount(csr_payload_bytes),
            support_payload_bytes: ByteCount(pattern_bytes(rows, entries)?),
            workspace,
            planning_resident_bytes,
        })
    }

    /// Requested-heap bound of the support derivation at the support's exact entry count.
    pub fn planning_resident_bytes(&self) -> ByteCount {
        self.planning_resident_bytes
    }
    pub fn topology_digest(&self) -> Digest {
        self.topology_digest
    }
    pub fn path(&self) -> ThermalSolvePath {
        self.path
    }
    /// The proving factor's admission on either path.
    pub fn factor_admission(&self) -> &SolverAdmissionResourcePolicy {
        &self.factor_admission
    }

    /// The same facts on `path`. The conduction multigrid path admits the structural proof's
    /// inventory in place of the factor.
    pub fn on_path(mut self, path: ThermalSolvePath) -> Self {
        self.admission = match path {
            ThermalSolvePath::ProvingFactorization => self.factor_admission.clone(),
            ThermalSolvePath::ConductionMultigrid => {
                crate::algebra::ScalarConductionProof::admission(&self.support)
            }
        };
        self.path = path;
        self
    }

    /// Resident bytes of the selected solver beyond the operator payloads: the proving factor,
    /// or the conduction proof inventory plus the AMG hierarchy and the PCG and backward-error
    /// vectors. This is the one law both the plan and the solve-time admission read.
    pub fn solver_resident_bytes(&self) -> Result<ByteCount, ResourcePlanningError> {
        match self.path {
            ThermalSolvePath::ProvingFactorization => Ok(self.admission.resident_bytes_upper_bound),
            ThermalSolvePath::ConductionMultigrid => {
                let rows = self.support.rows();
                let entries = self.support.column_indices().len();
                runmat_sparse_solvers::estimate::amg_bytes(rows, entries, 1)
                    .checked_add(
                        (rows as u64)
                            .checked_mul(CONDUCTION_MULTIGRID_VECTOR_COUNT * 8)
                            .ok_or(ResourcePlanningError::Overflow("thermal PCG vectors"))?,
                    )
                    .and_then(|bytes| bytes.checked_add(self.admission.resident_bytes_upper_bound.0))
                    .map(ByteCount)
                    .ok_or(ResourcePlanningError::Overflow(
                        "thermal conduction multigrid storage",
                    ))
            }
        }
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
        let solver = self.solver_resident_bytes()?.0;
        let numerical = self
            .csr_payload_bytes
            .0
            .checked_add(solver)
            .and_then(|bytes| bytes.checked_add(workspace));
        tracing::info!(
            path = ?self.path,
            solver_resident_bytes = solver,
            dimension = self.admission.dimension.0,
            support_entries = self.support.column_indices().len(),
            csr_payload_bytes = self.csr_payload_bytes.0,
            factor_nonzeros = self.factor_admission.factor_nonzero_upper_bound.0,
            factor_resident_bytes = self.factor_admission.resident_bytes_upper_bound.0,
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

/// Full-length f64 vectors the conduction multigrid solve holds besides the operators: PCG's
/// best iterate, residual, direction and operation output; the prepared right-hand side and
/// iterate; the preconditioned residual and operator output of the spectral estimate; the
/// backward-error residual; and the initial guess.
const CONDUCTION_MULTIGRID_VECTOR_COUNT: u64 = 10;

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
pub(crate) mod tests;
