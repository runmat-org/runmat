use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::Serialize;
use thiserror::Error;

use super::{
    MatrixSymmetry, NullspaceEliminationEvidence, PreparedNullspaceEvidence, RealSparseMatrix,
};

const CERTIFICATE_PREFIX: &[u8] = b"runmat-fea-coercivity-certificate/v2\0";
const CERTIFICATE_DOMAIN: &str = "fea-coercivity-certificate";
const PREPARED_DOMAIN: &str = "fea-prepared-solver-admissibility";
const RHS_DOMAIN: &str = "fea-coercivity-rhs";
const LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);
const SYMMETRY_RELATIVE_TOLERANCE: f64 = 1.0e-12;

mod elastic;
mod energy;
mod symbolic;

pub(crate) use symbolic::CoercivityWorkEnvelope;

/// Capacity is acquired before this exact symbolic phase runs. The direct solver's
/// own reservation remains the only completed-work owner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NumericalCapacityPhase {
    SolidAssemblyBlock { elements: std::ops::Range<usize> },
    Tet4AssemblyElement { element: usize },
    Tet10AssemblyElement { element: usize },
    ExplicitAssemblyElement { element: usize },
    ExplicitAssemblyScatter { elements: std::ops::Range<usize> },
    ThermoelasticLoadElement { element: usize },
    CoercivityFactorization,
    AffineRhsReduction,
    AffineOperatorScatter,
    AffineExpansion,
    AffineRejectedNorm,
    PcgPreconditionerSetup,
    PcgBatch,
    StaticBalance,
    ConstraintRecovery,
    FieldRecovery,
    SolidRecoveryElement { element: usize },
    SolvePrimitive,
    DirectSolve(runmat_sparse_solvers::direct::DirectSolvePhase),
}

/// Keep an awaited host refusal separate from a numerical proof rejection.
pub enum NumericalCapacityError<E, A> {
    Numerical(E),
    Admission(A),
}

impl<E, A> From<E> for NumericalCapacityError<E, A> {
    fn from(error: E) -> Self {
        Self::Numerical(error)
    }
}

impl<E, A> NumericalCapacityError<E, A> {
    pub fn map_numerical<F>(self, map: impl FnOnce(E) -> F) -> NumericalCapacityError<F, A> {
        match self {
            Self::Numerical(error) => NumericalCapacityError::Numerical(map(error)),
            Self::Admission(error) => NumericalCapacityError::Admission(error),
        }
    }
}

/// A numeric Cholesky factorization of exactly the certified operator, shared between the
/// coercivity proof and the direct solve (one factorization serves both).
#[derive(Clone)]
pub(crate) struct SharedFactor(
    pub std::sync::Arc<runmat_sparse_solvers::direct::DirectSolver<f64>>,
);

/// Factors are derived data: the certificate binds the exact operator, so two factors of the
/// same certified operator compare equal.
impl PartialEq for SharedFactor {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
            || (self.0.kind() == other.0.kind()
                && self.0.symbolic().fingerprint() == other.0.symbolic().fingerprint())
    }
}

impl std::fmt::Debug for SharedFactor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("SharedFactor")
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum CoercivityError {
    #[error("coercivity requires a nonempty square symmetric operator")]
    IncompatibleOperator,
    #[error("operator values do not satisfy the declared symmetry")]
    AsymmetricValues,
    #[error("operator is not numerically coercive at pivot {pivot}")]
    NonPositivePivot { pivot: usize },
    #[error("coercivity proof arithmetic overflowed")]
    WorkOverflow,
    #[error("coercivity proof exceeded its bounded sparse envelope at pivot {pivot}: factor entries {factor_nonzero_count}/{factor_nonzero_limit}, updates {update_count}/{update_limit}")]
    ResourceLimit {
        pivot: usize,
        factor_nonzero_count: u64,
        factor_nonzero_limit: u64,
        update_count: u64,
        update_limit: u64,
    },
    #[error("coercivity factor needs {required_bytes} numerical bytes; admitted {budget_bytes}")]
    FactorAdmission {
        required_bytes: u64,
        budget_bytes: u64,
    },
    #[error("coercivity dimension {required} differs from admitted {admitted}")]
    AdmissionDimension { required: u64, admitted: u64 },
    #[error("coercivity symbolic ordering could not cover the complete operator graph")]
    SymbolicOrdering,
    #[error("coercivity proof was cancelled")]
    Cancelled,
    #[error("coercivity evidence does not match its prepared operator or right-hand side")]
    BindingMismatch,
    #[error("coercivity evidence identity encoding failed: {0}")]
    Encoding(String),
    #[error(transparent)]
    Energy(#[from] super::EnergyError),
    #[error(transparent)]
    ElasticRigidity(#[from] super::ElasticRigidityError),
    #[error("anchored local energy did not establish coercivity: graph lower bound {}, error upper bound {}", f64::from_bits(*.graph_lower_bits), f64::from_bits(*.error_upper_bits))]
    EnergyIndeterminate {
        graph_lower_bits: u64,
        error_upper_bits: u64,
    },
}

/// A process-local numerical certificate that the exact sparse operator admitted to an SPD
/// solver has strictly positive inertia. The factor itself is intentionally discarded; the
/// certificate retains enough immutable evidence to bind the completed proof to the operator.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CoercivityCertificate {
    pattern_identity: Digest,
    values_identity: Digest,
    dimension: usize,
    proof: CoercivityProof,
    identity: Digest,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CoercivityProof {
    /// Supernodal Cholesky with nested dissection: every scaled pivot is strictly positive.
    SparseCholesky {
        minimum_scaled_pivot_bits: u64,
        factor_nonzero_count: u64,
        flop_count: u64,
        resident_bytes: u64,
    },
    AnchoredLocalEnergy {
        lower_bound_bits: u64,
        assembly_error_upper_bits: u64,
        scalar_operations: u64,
        resident_bytes: u64,
    },
    /// Element rigid-motion kernels glued by face adjacency and fixed by anchors
    /// ([`super::elastic_rigidity`]): positive definite in exact arithmetic, with no eigenvalue
    /// bound. Solution accuracy is accepted by the true-residual certificate.
    ElasticRigidity {
        elements: u64,
        components: u64,
        element_margin_rounding_factor_bits: u64,
        anchor_rank_ratio_bits: u64,
        scalar_operations: u64,
        resident_bytes: u64,
    },
}

impl CoercivityCertificate {
    /// Whether the sparse Cholesky factorization (rather than the anchored local-energy bound)
    /// certified this operator.
    pub(crate) fn proved_by_factorization(&self) -> bool {
        matches!(self.proof, CoercivityProof::SparseCholesky { .. })
    }
}

/// Binds algebraic coercivity, the exact prepared right-hand side, and optional physics-owned
/// null-space attribution. This evidence is never accepted from serialized callers.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PreparedSolverAdmissibility {
    coercivity: CoercivityCertificate,
    source_nullspace_identity: Option<Digest>,
    prepared_rhs_identity: Digest,
    identity: Digest,
    /// The proving factorization when the proof was a sparse Cholesky (process-local).
    #[serde(skip)]
    factor: Option<SharedFactor>,
}

impl CoercivityCertificate {
    /// Global sparse factor entries are zero for the local-energy method; local
    /// factor scratch and the retained source are included in resident bytes.
    pub(crate) fn work_counts(&self) -> (u64, u64, u64, u64) {
        let dimension = u64::try_from(self.dimension).expect("proved dimension fits u64");
        match self.proof {
            CoercivityProof::SparseCholesky {
                factor_nonzero_count,
                flop_count,
                resident_bytes,
                ..
            } => (dimension, factor_nonzero_count, flop_count, resident_bytes),
            CoercivityProof::AnchoredLocalEnergy {
                scalar_operations,
                resident_bytes,
                ..
            }
            | CoercivityProof::ElasticRigidity {
                scalar_operations,
                resident_bytes,
                ..
            } => (dimension, 0, scalar_operations, resident_bytes),
        }
    }

    /// Proves strict positivity of the inertia by a sparse Cholesky factorization of the
    /// diagonally scaled operator and returns the factorization for reuse by the solve.
    #[cfg(test)]
    pub(super) fn prove_admitted(
        matrix: &RealSparseMatrix,
        admission: Option<&crate::planning::resource::SolverAdmissionResourcePolicy>,
    ) -> Result<(Self, SharedFactor), CoercivityError> {
        let future = Self::prove_admitted_with_owner(matrix, admission, |_, _| {
            std::future::ready(Ok::<(), std::convert::Infallible>(()))
        });
        let mut future = std::pin::pin!(future);
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        match std::future::Future::poll(future.as_mut(), &mut context) {
            std::task::Poll::Ready(result) => result.map_err(|error| match error {
                NumericalCapacityError::Numerical(error) => error,
                NumericalCapacityError::Admission(never) => match never {},
            }),
            std::task::Poll::Pending => unreachable!("immediate factor owner cannot suspend"),
        }
    }

    pub(super) async fn prove_admitted_with_owner<O, F, A>(
        matrix: &RealSparseMatrix,
        admission: Option<&crate::planning::resource::SolverAdmissionResourcePolicy>,
        mut before_factor: O,
    ) -> Result<(Self, SharedFactor), NumericalCapacityError<CoercivityError, A>>
    where
        O: FnMut(NumericalCapacityPhase, runmat_simulation_usage::OperationCount) -> F,
        F: std::future::Future<Output = Result<(), A>>,
    {
        use runmat_sparse_solvers::direct::{
            DirectOptions, DirectSolver, DirectWorkspacePolicy, FactorKind, SymbolicFactorization,
        };
        let dimension = matrix.rows();
        if dimension == 0
            || matrix.columns() != dimension
            || matrix.properties().symmetry != MatrixSymmetry::Symmetric
        {
            return Err(CoercivityError::IncompatibleOperator.into());
        }
        let diagonal = matrix.diagonal_values();
        if let Some(pivot) = diagonal
            .iter()
            .position(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(CoercivityError::NonPositivePivot { pivot }.into());
        }
        for row in 0..dimension {
            for entry in matrix.pattern().row_range(row) {
                let column = matrix.pattern().column(entry);
                let value = matrix.values()[entry];
                let transpose = matrix.get(column, row);
                // An entry of a symmetric positive operator is bounded by sqrt(a_ii a_jj); the
                // roundoff of an assembled or reduced entry scales with that bound, not with the
                // entry, which cancellation can make arbitrarily small.
                let symmetry_scale = value
                    .abs()
                    .max(transpose.abs())
                    .max((diagonal[row] * diagonal[column]).sqrt());
                if (value - transpose).abs() > symmetry_scale * SYMMETRY_RELATIVE_TOLERANCE {
                    return Err(CoercivityError::AsymmetricValues.into());
                }
            }
        }
        let arrays = crate::solve::shared_sparse::CsrArrays::of(matrix);
        let options = DirectOptions {
            refinement_steps: 2,
            workspace: if admission.is_some() {
                DirectWorkspacePolicy::CallerOwned
            } else {
                DirectWorkspacePolicy::BackendManaged
            },
            ..DirectOptions::new(FactorKind::Cholesky)
        };
        let control = crate::solve::shared_sparse::control();
        let symbolic =
            SymbolicFactorization::analyze::<f64>(arrays.view().into(), &options, control)
                .map_err(|_| CoercivityError::SymbolicOrdering)?;
        let estimate = symbolic.estimate();
        if let Some(admission) = admission {
            if admission.dimension.0 != dimension as u64 {
                return Err(CoercivityError::AdmissionDimension {
                    required: dimension as u64,
                    admitted: admission.dimension.0,
                }
                .into());
            }
            let updates = symbolic::flop_count(&estimate);
            if estimate.factor_entries > admission.factor_nonzero_upper_bound.0
                || updates > admission.update_count_upper_bound.0
            {
                return Err(CoercivityError::ResourceLimit {
                    pivot: 0,
                    factor_nonzero_count: estimate.factor_entries,
                    factor_nonzero_limit: admission.factor_nonzero_upper_bound.0,
                    update_count: updates,
                    update_limit: admission.update_count_upper_bound.0,
                }
                .into());
            }
            if estimate.peak_bytes() > admission.resident_bytes_upper_bound.0 {
                return Err(CoercivityError::FactorAdmission {
                    required_bytes: estimate.peak_bytes(),
                    budget_bytes: admission.resident_bytes_upper_bound.0,
                }
                .into());
            }
        }
        let factor_work = symbolic.reference_work().map_err(|error| match error {
            runmat_sparse_solvers::direct::ReferenceWorkError::Overflow(_) => {
                CoercivityError::WorkOverflow
            }
            _ => CoercivityError::SymbolicOrdering,
        })?;
        before_factor(
            NumericalCapacityPhase::CoercivityFactorization,
            factor_work
                .factorization_operations()
                .map_err(|_| CoercivityError::WorkOverflow)?,
        )
        .await
        .map_err(NumericalCapacityError::Admission)?;
        let solver = DirectSolver::<f64>::factorize(
            symbolic,
            arrays.view().into(),
            options,
            admission.map(|policy| policy.resident_bytes_upper_bound.0),
            control,
        )
        .map_err(|error| match error {
            runmat_sparse_solvers::SolverError::NotPositiveDefinite { pivot }
            | runmat_sparse_solvers::SolverError::Singular { pivot } => {
                CoercivityError::NonPositivePivot { pivot }
            }
            runmat_sparse_solvers::SolverError::OutOfMemory { .. }
            | runmat_sparse_solvers::SolverError::OverBudget { .. } => {
                CoercivityError::ResourceLimit {
                    pivot: 0,
                    factor_nonzero_count: 0,
                    factor_nonzero_limit: 0,
                    update_count: 0,
                    update_limit: 0,
                }
            }
            runmat_sparse_solvers::SolverError::Cancelled => CoercivityError::Cancelled,
            _ => CoercivityError::SymbolicOrdering,
        })?;
        // Scaled pivots l_jj² / a_jj are the pivots of D^{-1/2} A D^{-1/2}; a tiny one is a
        // numerical mechanism even when it is positive.
        let pivots = solver.pivots().ok_or(CoercivityError::SymbolicOrdering)?;
        let tolerance = f64::EPSILON * (dimension.max(1) as f64) * 64.0;
        let mut minimum_scaled_pivot = f64::INFINITY;
        for (index, (&pivot, &scale)) in pivots.iter().zip(&diagonal).enumerate() {
            let scaled = pivot / scale;
            if !scaled.is_finite() || scaled <= tolerance {
                return Err(CoercivityError::NonPositivePivot { pivot: index }.into());
            }
            minimum_scaled_pivot = minimum_scaled_pivot.min(scaled);
        }
        let estimate = solver.symbolic().estimate();
        let pattern_identity = matrix
            .pattern_identity()
            .map_err(|error| CoercivityError::Encoding(error.to_string()))?;
        let values_identity = matrix
            .values_identity()
            .map_err(|error| CoercivityError::Encoding(error.to_string()))?;
        let mut certificate = Self {
            pattern_identity,
            values_identity,
            dimension,
            proof: CoercivityProof::SparseCholesky {
                minimum_scaled_pivot_bits: minimum_scaled_pivot.to_bits(),
                factor_nonzero_count: estimate.factor_entries,
                flop_count: symbolic::flop_count(&estimate),
                resident_bytes: estimate.peak_bytes(),
            },
            identity: Digest::sha256([]),
        };
        certificate.identity = certificate.expected_identity()?;
        Ok((certificate, SharedFactor(std::sync::Arc::new(solver))))
    }

    pub(crate) fn validate_for(&self, matrix: &RealSparseMatrix) -> Result<(), CoercivityError> {
        if self.dimension != matrix.rows()
            || self.pattern_identity
                != matrix
                    .pattern_identity()
                    .map_err(|error| CoercivityError::Encoding(error.to_string()))?
            || self.values_identity
                != matrix
                    .values_identity()
                    .map_err(|error| CoercivityError::Encoding(error.to_string()))?
            || !self.proof_is_positive()
            || self.identity != self.expected_identity()?
        {
            return Err(CoercivityError::BindingMismatch);
        }
        Ok(())
    }

    fn proof_is_positive(&self) -> bool {
        let bits = match self.proof {
            CoercivityProof::SparseCholesky {
                minimum_scaled_pivot_bits,
                ..
            } => minimum_scaled_pivot_bits,
            CoercivityProof::AnchoredLocalEnergy {
                lower_bound_bits,
                assembly_error_upper_bits,
                ..
            } => {
                let error = f64::from_bits(assembly_error_upper_bits);
                if !error.is_finite() || error < 0. {
                    return false;
                }
                lower_bound_bits
            }
            CoercivityProof::ElasticRigidity {
                element_margin_rounding_factor_bits,
                anchor_rank_ratio_bits,
                ..
            } => {
                if !(f64::from_bits(anchor_rank_ratio_bits) > 0.) {
                    return false;
                }
                element_margin_rounding_factor_bits
            }
        };
        f64::from_bits(bits).is_finite() && f64::from_bits(bits) > 0.
    }

    fn expected_identity(&self) -> Result<Digest, CoercivityError> {
        digest(
            CERTIFICATE_DOMAIN,
            &(
                self.pattern_identity,
                self.values_identity,
                self.dimension,
                &self.proof,
            ),
        )
    }
}

impl PreparedSolverAdmissibility {
    #[cfg(test)]
    pub(super) fn prove_admitted(
        matrix: &RealSparseMatrix,
        rhs: &[f64],
        source: Option<&NullspaceEliminationEvidence>,
        admission: Option<&crate::planning::resource::SolverAdmissionResourcePolicy>,
    ) -> Result<Self, CoercivityError> {
        let future = Self::prove_admitted_with_owner(matrix, rhs, source, admission, |_, _| {
            std::future::ready(Ok::<(), std::convert::Infallible>(()))
        });
        let mut future = std::pin::pin!(future);
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        match std::future::Future::poll(future.as_mut(), &mut context) {
            std::task::Poll::Ready(result) => result.map_err(|error| match error {
                NumericalCapacityError::Numerical(error) => error,
                NumericalCapacityError::Admission(never) => match never {},
            }),
            std::task::Poll::Pending => unreachable!("immediate proof owner cannot suspend"),
        }
    }

    pub(super) async fn prove_admitted_with_owner<O, F, A>(
        matrix: &RealSparseMatrix,
        rhs: &[f64],
        source: Option<&NullspaceEliminationEvidence>,
        admission: Option<&crate::planning::resource::SolverAdmissionResourcePolicy>,
        before_factor: O,
    ) -> Result<Self, NumericalCapacityError<CoercivityError, A>>
    where
        O: FnMut(NumericalCapacityPhase, runmat_simulation_usage::OperationCount) -> F,
        F: std::future::Future<Output = Result<(), A>>,
    {
        let source_prepared = source
            .map(|source| source.bind_prepared(matrix, rhs))
            .transpose()
            .map_err(|error| CoercivityError::Encoding(error.to_string()))?;
        let (coercivity, factor) =
            CoercivityCertificate::prove_admitted_with_owner(matrix, admission, before_factor)
                .await?;
        Ok(Self::bind(
            coercivity,
            rhs,
            source_prepared
                .as_ref()
                .map(PreparedNullspaceEvidence::identity),
            Some(factor),
        )?)
    }

    /// The Cholesky factorization that proved coercivity, when available.
    pub(crate) fn factor(&self) -> Option<&SharedFactor> {
        self.factor.as_ref()
    }

    pub(super) fn validate_for(
        &self,
        matrix: &RealSparseMatrix,
        rhs: &[f64],
    ) -> Result<(), CoercivityError> {
        self.coercivity.validate_for(matrix)?;
        if self.prepared_rhs_identity != digest(RHS_DOMAIN, &rhs)?
            || self.identity != self.expected_identity()?
        {
            return Err(CoercivityError::BindingMismatch);
        }
        Ok(())
    }

    pub(crate) fn coercivity(&self) -> &CoercivityCertificate {
        &self.coercivity
    }

    pub(crate) fn bind_existing(
        coercivity: CoercivityCertificate,
        matrix: &RealSparseMatrix,
        rhs: &[f64],
    ) -> Result<Self, CoercivityError> {
        Self::bind_existing_with_source(coercivity, None, matrix, rhs, None)
    }

    pub(super) fn bind_existing_with_source(
        coercivity: CoercivityCertificate,
        factor: Option<SharedFactor>,
        matrix: &RealSparseMatrix,
        rhs: &[f64],
        source: Option<&NullspaceEliminationEvidence>,
    ) -> Result<Self, CoercivityError> {
        coercivity.validate_for(matrix)?;
        let source_prepared = source
            .map(|source| source.bind_prepared(matrix, rhs))
            .transpose()
            .map_err(|error| CoercivityError::Encoding(error.to_string()))?;
        Self::bind(
            coercivity,
            rhs,
            source_prepared
                .as_ref()
                .map(PreparedNullspaceEvidence::identity),
            factor,
        )
    }

    fn bind(
        coercivity: CoercivityCertificate,
        rhs: &[f64],
        source_nullspace_identity: Option<Digest>,
        factor: Option<SharedFactor>,
    ) -> Result<Self, CoercivityError> {
        let prepared_rhs_identity = digest(RHS_DOMAIN, &rhs)?;
        let mut evidence = Self {
            coercivity,
            source_nullspace_identity,
            prepared_rhs_identity,
            identity: Digest::sha256([]),
            factor,
        };
        evidence.identity = evidence.expected_identity()?;
        Ok(evidence)
    }

    fn expected_identity(&self) -> Result<Digest, CoercivityError> {
        digest(
            PREPARED_DOMAIN,
            &(
                self.coercivity.identity,
                self.source_nullspace_identity,
                self.prepared_rhs_identity,
            ),
        )
    }
}

fn digest<T: Serialize>(domain: &'static str, value: &T) -> Result<Digest, CoercivityError> {
    encode_contract(CERTIFICATE_PREFIX, domain, value, LIMITS)
        .map(Digest::sha256)
        .map_err(|error| CoercivityError::Encoding(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{MatrixProperties, OperatorRole, RealSparseAssembler};

    fn matrix(values: [[f64; 3]; 3]) -> RealSparseMatrix {
        let mut assembler = RealSparseAssembler::square(
            3,
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        );
        for (row, values) in values.into_iter().enumerate() {
            for (column, value) in values.into_iter().enumerate() {
                assembler.add(row, column, value);
            }
        }
        assembler.finish().unwrap()
    }

    #[test]
    fn finite_coercivity_consumes_planned_scratch_quote_before_numeric_factorization() {
        use crate::planning::resource::{ByteCount, Count, SolverAdmissionResourcePolicy};
        let operator = matrix([[4.0, 1.0, 1.0], [1.0, 3.0, 0.0], [1.0, 0.0, 2.0]]);
        let estimate = symbolic::analyze(operator.pattern()).unwrap();
        let mut admission = SolverAdmissionResourcePolicy {
            dimension: Count(3),
            factor_nonzero_upper_bound: Count(estimate.factor_entries),
            update_count_upper_bound: Count(symbolic::flop_count(&estimate)),
            resident_bytes_upper_bound: ByteCount(estimate.peak_bytes()),
        };
        let (certificate, factor) =
            CoercivityCertificate::prove_admitted(&operator, Some(&admission)).unwrap();
        assert!(admission.admits_certificate(&certificate));
        assert_eq!(
            factor.0.symbolic().workspace_policy(),
            runmat_sparse_solvers::direct::DirectWorkspacePolicy::CallerOwned
        );
        admission.resident_bytes_upper_bound.0 -= 1;
        assert!(matches!(
            CoercivityCertificate::prove_admitted(&operator, Some(&admission)),
            Err(CoercivityError::FactorAdmission { .. })
        ));
        admission.resident_bytes_upper_bound.0 += 1;
        admission.factor_nonzero_upper_bound.0 -= 1;
        assert!(matches!(
            CoercivityCertificate::prove_admitted(&operator, Some(&admission)),
            Err(CoercivityError::ResourceLimit { .. })
        ));
    }

    #[test]
    fn sparse_inertia_proves_fill_requiring_spd_operator_and_binds_rhs() {
        let operator = matrix([[4.0, 1.0, 1.0], [1.0, 3.0, 0.0], [1.0, 0.0, 2.0]]);
        let evidence =
            PreparedSolverAdmissibility::prove_admitted(&operator, &[1.0, 2.0, 3.0], None, None)
                .unwrap();
        let estimate = symbolic::analyze(operator.pattern()).unwrap();
        assert_eq!(evidence.coercivity.work_counts().1, estimate.factor_entries);
        assert_eq!(
            evidence.coercivity.work_counts().2,
            symbolic::flop_count(&estimate)
        );
        assert!(evidence.factor().is_some());
        evidence.validate_for(&operator, &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(
            evidence.validate_for(&operator, &[1.0, 2.0, 4.0]),
            Err(CoercivityError::BindingMismatch)
        );
    }

    #[test]
    fn positive_definite_metadata_cannot_hide_an_extra_mechanism_or_negative_mode() {
        let mechanism = matrix([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        let indefinite = matrix([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert!(matches!(
            CoercivityCertificate::prove_admitted(&mechanism, None).map(|(c, _)| c),
            Err(CoercivityError::NonPositivePivot { .. })
        ));
        assert!(matches!(
            CoercivityCertificate::prove_admitted(&indefinite, None).map(|(c, _)| c),
            Err(CoercivityError::NonPositivePivot { .. })
        ));
    }

    #[test]
    fn roundoff_asymmetry_is_measured_against_the_coupled_diagonals() {
        // A reduced coupling entry that cancels to almost zero keeps the roundoff of the
        // 1e11-scale terms it was formed from; that is not an asymmetric operator.
        let reduced = matrix([
            [4.0e11, 1.0e-6, 0.0],
            [-1.0e-6, 4.0e11, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        assert!(CoercivityCertificate::prove_admitted(&reduced, None).is_ok());
    }

    #[test]
    fn symmetric_metadata_cannot_hide_asymmetric_values() {
        let asymmetric = matrix([[2.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_eq!(
            CoercivityCertificate::prove_admitted(&asymmetric, None).map(|(c, _)| c),
            Err(CoercivityError::AsymmetricValues)
        );
    }
}
