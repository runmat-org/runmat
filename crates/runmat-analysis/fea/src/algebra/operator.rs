use serde::{Deserialize, Serialize};
mod linear_reuse;
pub(crate) use linear_reuse::LinearPreparationReuse;

use crate::constraints::AffineConstraintReduction;

use super::{
    spmv, CoercivityError, NullspaceEliminationEvidence, NullspaceEvidenceError, OperatorRole,
    PreparedSolverAdmissibility, RealSparseMatrix,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "CanonicalRealOperatorsData")]
pub struct CanonicalRealOperators {
    stiffness: RealSparseMatrix,
    mass: RealSparseMatrix,
    damping: RealSparseMatrix,
}

#[derive(Deserialize)]
struct CanonicalRealOperatorsData {
    stiffness: RealSparseMatrix,
    mass: RealSparseMatrix,
    damping: RealSparseMatrix,
}

impl TryFrom<CanonicalRealOperatorsData> for CanonicalRealOperators {
    type Error = RealSystemError;

    fn try_from(data: CanonicalRealOperatorsData) -> Result<Self, Self::Error> {
        Self::try_new(data.stiffness, data.mass, data.damping)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RealSystemError {
    ShapeMismatch,
    RoleMismatch,
    NonFiniteLoad { index: usize },
    NullspaceEvidence(NullspaceEvidenceError),
    Coercivity(CoercivityError),
}

impl std::fmt::Display for RealSystemError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "invalid canonical real system: {self:?}")
    }
}

impl std::error::Error for RealSystemError {}

impl CanonicalRealOperators {
    pub fn try_new(
        stiffness: RealSparseMatrix,
        mass: RealSparseMatrix,
        damping: RealSparseMatrix,
    ) -> Result<Self, RealSystemError> {
        let n = stiffness.rows();
        if stiffness.columns() != n
            || mass.rows() != n
            || mass.columns() != n
            || damping.rows() != n
            || damping.columns() != n
        {
            return Err(RealSystemError::ShapeMismatch);
        }
        if !matches!(
            stiffness.role(),
            OperatorRole::Stiffness | OperatorRole::Conductivity | OperatorRole::EffectiveTangent
        ) || !matches!(mass.role(), OperatorRole::Mass | OperatorRole::Capacity)
            || damping.role() != OperatorRole::Damping
        {
            return Err(RealSystemError::RoleMismatch);
        }
        Ok(Self {
            stiffness,
            mass,
            damping,
        })
    }
    pub fn dof_count(&self) -> usize {
        self.stiffness.rows()
    }
    pub fn stiffness(&self) -> &RealSparseMatrix {
        &self.stiffness
    }
    pub fn mass(&self) -> &RealSparseMatrix {
        &self.mass
    }
    pub fn damping(&self) -> &RealSparseMatrix {
        &self.damping
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "AssembledRealSystemData")]
pub struct AssembledRealSystem {
    operators: CanonicalRealOperators,
    constrained_dofs: Vec<bool>,
    load: Vec<f64>,
    /// Admission evidence is regenerated from authoritative topology, operator, and constraint
    /// inputs in the executing process. It is deliberately not a portable caller assertion.
    #[serde(skip)]
    nullspace_evidence: Option<NullspaceEliminationEvidence>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AssembledRealSystemData {
    operators: CanonicalRealOperators,
    constrained_dofs: Vec<bool>,
    load: Vec<f64>,
}

impl TryFrom<AssembledRealSystemData> for AssembledRealSystem {
    type Error = RealSystemError;

    fn try_from(data: AssembledRealSystemData) -> Result<Self, Self::Error> {
        Self::try_new_with_nullspace_evidence(
            data.operators,
            data.constrained_dofs,
            data.load,
            None,
        )
    }
}

impl AssembledRealSystem {
    pub fn try_new(
        operators: CanonicalRealOperators,
        constrained_dofs: Vec<bool>,
        load: Vec<f64>,
    ) -> Result<Self, RealSystemError> {
        Self::try_new_with_nullspace_evidence(operators, constrained_dofs, load, None)
    }

    pub fn try_new_with_nullspace_evidence(
        operators: CanonicalRealOperators,
        constrained_dofs: Vec<bool>,
        load: Vec<f64>,
        nullspace_evidence: Option<NullspaceEliminationEvidence>,
    ) -> Result<Self, RealSystemError> {
        let n = operators.dof_count();
        if constrained_dofs.len() != n || load.len() != n {
            return Err(RealSystemError::ShapeMismatch);
        }
        if let Some(index) = load.iter().position(|value| !value.is_finite()) {
            return Err(RealSystemError::NonFiniteLoad { index });
        }
        if let Some(evidence) = &nullspace_evidence {
            evidence
                .validate_for_system(operators.stiffness(), &constrained_dofs)
                .map_err(RealSystemError::NullspaceEvidence)?;
        }
        Ok(Self {
            operators,
            constrained_dofs,
            load,
            nullspace_evidence,
        })
    }
    pub fn dof_count(&self) -> usize {
        self.operators.dof_count()
    }
    pub fn operators(&self) -> &CanonicalRealOperators {
        &self.operators
    }
    pub fn constrained_dofs(&self) -> &[bool] {
        &self.constrained_dofs
    }
    pub fn load(&self) -> &[f64] {
        &self.load
    }
    pub fn nullspace_evidence(&self) -> Option<&NullspaceEliminationEvidence> {
        self.nullspace_evidence.as_ref()
    }
    pub fn with_load(&self, load: Vec<f64>) -> Result<Self, RealSystemError> {
        Self::try_new_with_nullspace_evidence(
            self.operators.clone(),
            self.constrained_dofs.clone(),
            load,
            self.nullspace_evidence.clone(),
        )
    }
    /// Canonical Dirichlet-lifted RHS used for both preparation and residual scaling.
    /// Does not allocate a constrained matrix or perform coercivity work.
    pub fn linear_rhs(&self) -> Vec<f64> {
        let stiffness = self.operators.stiffness();
        let mut rhs = self.load.clone();
        for (row, rhs_value) in rhs.iter_mut().enumerate() {
            if self.constrained_dofs[row] {
                continue;
            }
            for entry in stiffness.pattern().row_range(row) {
                let column = stiffness.pattern().column(entry);
                if self.constrained_dofs[column] {
                    *rhs_value -= stiffness.values()[entry] * self.load[column];
                }
            }
        }
        rhs
    }
    pub fn prepare_linear(&self) -> Result<PreparedRealLinearSystem, RealSystemError> {
        self.prepare_linear_with_reuse(None, None)
    }
    pub(crate) fn prepare_linear_reusing(
        &self,
        reuse: &mut LinearPreparationReuse,
    ) -> Result<PreparedRealLinearSystem, RealSystemError> {
        self.prepare_linear_with_reuse(Some(reuse), None)
    }
    pub(crate) fn prepare_linear_with_reuse(
        &self,
        reuse: Option<&mut LinearPreparationReuse>,
        admission: Option<&crate::planning::resource::SolverAdmissionResourcePolicy>,
    ) -> Result<PreparedRealLinearSystem, RealSystemError> {
        let future = self.prepare_linear_with_reuse_owned(reuse, admission, |_, _| {
            std::future::ready(Ok::<(), std::convert::Infallible>(()))
        });
        let mut future = std::pin::pin!(future);
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        match std::future::Future::poll(future.as_mut(), &mut context) {
            std::task::Poll::Ready(result) => result.map_err(|error| match error {
                super::coercivity::NumericalCapacityError::Numerical(error) => error,
                super::coercivity::NumericalCapacityError::Admission(never) => match never {},
            }),
            std::task::Poll::Pending => unreachable!("immediate linear-preparation owner cannot suspend"),
        }
    }

    pub(crate) async fn prepare_linear_with_reuse_owned<O, F, A>(
        &self,
        reuse: Option<&mut LinearPreparationReuse>,
        admission: Option<&crate::planning::resource::SolverAdmissionResourcePolicy>,
        before_factor: O,
    ) -> Result<PreparedRealLinearSystem, super::coercivity::NumericalCapacityError<RealSystemError, A>>
    where
        O: FnMut(super::coercivity::NumericalCapacityPhase, runmat_simulation_usage::OperationCount) -> F,
        F: std::future::Future<Output = Result<(), A>>,
    {
        let stiffness = self.operators.stiffness();
        let rhs = self.linear_rhs();
        let matrix = stiffness
            .with_identity_constraints(&self.constrained_dofs)
            .map_err(|_| RealSystemError::ShapeMismatch)?;
        PreparedRealLinearSystem::try_new_with_reuse_owned(
            matrix,
            rhs,
            self.constrained_dofs.clone(),
            self.nullspace_evidence.as_ref(),
            reuse,
            admission,
            before_factor,
        ).await
    }

    pub fn prepare_modal(&self) -> Result<PreparedModalSystem, RealSystemError> {
        let free_dofs = self
            .constrained_dofs
            .iter()
            .enumerate()
            .filter_map(|(index, constrained)| (!constrained).then_some(index))
            .collect::<Vec<_>>();
        let free_columns = free_dofs
            .iter()
            .enumerate()
            .map(|(column, dof)| (*dof, column))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut reduction = AffineConstraintReduction {
            full_dof_count: self.dof_count(),
            reduced_dof_count: free_dofs.len(),
            free_dof_indices: free_dofs,
            pivot_dof_indices: self
                .constrained_dofs
                .iter()
                .enumerate()
                .filter_map(|(dof, constrained)| constrained.then_some(dof))
                .collect(),
            recovery_row_offsets: vec![0],
            recovery_column_indices: Vec::with_capacity(free_columns.len()),
            recovery_coefficients: Vec::with_capacity(free_columns.len()),
            offset: vec![0.0; self.dof_count()],
        };
        for dof in 0..self.dof_count() {
            if let Some(column) = free_columns.get(&dof) {
                reduction.recovery_column_indices.push(*column);
                reduction.recovery_coefficients.push(1.0);
            }
            reduction
                .recovery_row_offsets
                .push(reduction.recovery_column_indices.len());
        }
        self.prepare_modal_with_reduction(reduction)
    }

    pub fn prepare_modal_with_reduction(
        &self,
        reduction: AffineConstraintReduction,
    ) -> Result<PreparedModalSystem, RealSystemError> {
        reduction
            .validate()
            .map_err(|_| RealSystemError::ShapeMismatch)?;
        if reduction.full_dof_count != self.dof_count()
            || reduction.offset.iter().any(|value| *value != 0.0)
        {
            return Err(RealSystemError::ShapeMismatch);
        }
        let operators = CanonicalRealOperators::try_new(
            reduction
                .reduce_sparse_operator(self.operators.stiffness())
                .map_err(|_| RealSystemError::ShapeMismatch)?,
            reduction
                .reduce_sparse_operator(self.operators.mass())
                .map_err(|_| RealSystemError::ShapeMismatch)?,
            reduction
                .reduce_sparse_operator(self.operators.damping())
                .map_err(|_| RealSystemError::ShapeMismatch)?,
        )?;
        Ok(PreparedModalSystem {
            operators,
            reduction,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "PreparedModalSystemData")]
pub struct PreparedModalSystem {
    operators: CanonicalRealOperators,
    reduction: AffineConstraintReduction,
}

#[derive(Deserialize)]
struct PreparedModalSystemData {
    operators: CanonicalRealOperators,
    reduction: AffineConstraintReduction,
}

impl TryFrom<PreparedModalSystemData> for PreparedModalSystem {
    type Error = RealSystemError;

    fn try_from(data: PreparedModalSystemData) -> Result<Self, Self::Error> {
        if data.reduction.validate().is_err()
            || data.operators.dof_count() != data.reduction.reduced_dof_count
            || data.reduction.offset.iter().any(|value| *value != 0.0)
        {
            return Err(RealSystemError::ShapeMismatch);
        }
        Ok(Self {
            operators: data.operators,
            reduction: data.reduction,
        })
    }
}

impl PreparedModalSystem {
    pub fn operators(&self) -> &CanonicalRealOperators {
        &self.operators
    }

    pub fn reduction(&self) -> &AffineConstraintReduction {
        &self.reduction
    }

    pub fn as_assembled(&self) -> Result<AssembledRealSystem, RealSystemError> {
        let dof_count = self.operators.dof_count();
        AssembledRealSystem::try_new(
            self.operators.clone(),
            vec![false; dof_count],
            vec![0.0; dof_count],
        )
    }

    pub fn free_dofs(&self) -> &[usize] {
        &self.reduction.free_dof_indices
    }

    pub fn expand(&self, reduced: &[f64]) -> Result<Vec<f64>, RealSystemError> {
        self.reduction
            .expand_linear(reduced)
            .map_err(|_| RealSystemError::ShapeMismatch)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "PreparedRealLinearSystemData")]
pub struct PreparedRealLinearSystem {
    matrix: RealSparseMatrix,
    rhs: Vec<f64>,
    constrained_dofs: Vec<bool>,
    /// Derived admission evidence is process-local and must be rebuilt after decoding.
    #[serde(skip)]
    admissibility: Option<PreparedSolverAdmissibility>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PreparedRealLinearSystemData {
    matrix: RealSparseMatrix,
    rhs: Vec<f64>,
    constrained_dofs: Vec<bool>,
}

impl TryFrom<PreparedRealLinearSystemData> for PreparedRealLinearSystem {
    type Error = RealSystemError;

    fn try_from(data: PreparedRealLinearSystemData) -> Result<Self, Self::Error> {
        Self::try_new(data.matrix, data.rhs, data.constrained_dofs)
    }
}

impl PreparedRealLinearSystem {
    pub fn try_new(
        matrix: RealSparseMatrix,
        rhs: Vec<f64>,
        constrained_dofs: Vec<bool>,
    ) -> Result<Self, RealSystemError> {
        Self::try_new_with_source(matrix, rhs, constrained_dofs, None)
    }

    fn try_new_with_source(
        matrix: RealSparseMatrix,
        rhs: Vec<f64>,
        constrained_dofs: Vec<bool>,
        source: Option<&NullspaceEliminationEvidence>,
    ) -> Result<Self, RealSystemError> {
        Self::try_new_with_reuse(matrix, rhs, constrained_dofs, source, None, None)
    }
    pub(crate) async fn try_new_admitted_with_owner<O, F, A>(
        matrix: RealSparseMatrix,
        rhs: Vec<f64>,
        constrained_dofs: Vec<bool>,
        reuse: Option<&mut LinearPreparationReuse>,
        admission: Option<&crate::planning::resource::SolverAdmissionResourcePolicy>,
        before_factor: O,
    ) -> Result<Self, super::coercivity::NumericalCapacityError<RealSystemError, A>>
    where
        O: FnMut(super::coercivity::NumericalCapacityPhase, runmat_simulation_usage::OperationCount) -> F,
        F: std::future::Future<Output = Result<(), A>>,
    {
        Self::try_new_with_reuse_owned(
            matrix, rhs, constrained_dofs, None, reuse, admission, before_factor,
        ).await
    }
    fn try_new_with_reuse(
        matrix: RealSparseMatrix,
        rhs: Vec<f64>,
        constrained_dofs: Vec<bool>,
        source: Option<&NullspaceEliminationEvidence>,
        reuse: Option<&mut LinearPreparationReuse>,
        admission: Option<&crate::planning::resource::SolverAdmissionResourcePolicy>,
    ) -> Result<Self, RealSystemError> {
        let future = Self::try_new_with_reuse_owned(
            matrix, rhs, constrained_dofs, source, reuse, admission,
            |_, _| std::future::ready(Ok::<(), std::convert::Infallible>(())),
        );
        let mut future = std::pin::pin!(future);
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        match std::future::Future::poll(future.as_mut(), &mut context) {
            std::task::Poll::Ready(result) => result.map_err(|error| match error {
                super::coercivity::NumericalCapacityError::Numerical(error) => error,
                super::coercivity::NumericalCapacityError::Admission(never) => match never {},
            }),
            std::task::Poll::Pending => unreachable!("immediate prepared-system owner cannot suspend"),
        }
    }

    async fn try_new_with_reuse_owned<O, F, A>(
        matrix: RealSparseMatrix,
        rhs: Vec<f64>,
        constrained_dofs: Vec<bool>,
        source: Option<&NullspaceEliminationEvidence>,
        reuse: Option<&mut LinearPreparationReuse>,
        admission: Option<&crate::planning::resource::SolverAdmissionResourcePolicy>,
        before_factor: O,
    ) -> Result<Self, super::coercivity::NumericalCapacityError<RealSystemError, A>>
    where
        O: FnMut(super::coercivity::NumericalCapacityPhase, runmat_simulation_usage::OperationCount) -> F,
        F: std::future::Future<Output = Result<(), A>>,
    {
        if matrix.rows() != matrix.columns()
            || rhs.len() != matrix.rows()
            || constrained_dofs.len() != matrix.rows()
        {
            return Err(RealSystemError::ShapeMismatch.into());
        }
        if let Some(index) = rhs.iter().position(|value| !value.is_finite()) {
            return Err(RealSystemError::NonFiniteLoad { index }.into());
        }
        let admissibility = Some(
            match reuse {
                Some(reuse) => reuse.prepare_admitted_with_owner(
                    &matrix, &rhs, source, admission, before_factor,
                ).await,
                None => {
                    PreparedSolverAdmissibility::prove_admitted_with_owner(
                        &matrix, &rhs, source, admission, before_factor,
                    ).await
                }
            }
            .map_err(|error| error.map_numerical(RealSystemError::Coercivity))?,
        );
        Ok(Self {
            matrix,
            rhs,
            constrained_dofs,
            admissibility,
        })
    }
    pub fn dof_count(&self) -> usize {
        self.matrix.rows()
    }
    pub fn matrix(&self) -> &RealSparseMatrix {
        &self.matrix
    }
    pub fn rhs(&self) -> &[f64] {
        &self.rhs
    }
    pub fn constrained_dofs(&self) -> &[bool] {
        &self.constrained_dofs
    }
    pub fn admissibility(&self) -> Option<&PreparedSolverAdmissibility> {
        self.admissibility.as_ref()
    }
}

pub fn apply_k(system: &AssembledRealSystem, x: &[f64]) -> Vec<f64> {
    spmv(system.operators().stiffness(), x).expect("validated stiffness dimensions")
}
pub fn apply_m(system: &AssembledRealSystem, x: &[f64]) -> Vec<f64> {
    spmv(system.operators().mass(), x).expect("validated mass dimensions")
}
pub fn apply_c(system: &AssembledRealSystem, x: &[f64]) -> Vec<f64> {
    spmv(system.operators().damping(), x).expect("validated damping dimensions")
}
pub fn apply_prepared(system: &PreparedRealLinearSystem, x: &[f64]) -> Vec<f64> {
    spmv(system.matrix(), x).expect("validated prepared solve dimensions")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::MatrixProperties;

    fn physical_system() -> AssembledRealSystem {
        let stiffness = RealSparseMatrix::from_dense(
            &[4.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 4.0],
            3,
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .unwrap();
        let mass = RealSparseMatrix::from_dense(
            &[2.0, 0.5, 0.0, 0.5, 2.0, 0.5, 0.0, 0.5, 2.0],
            3,
            OperatorRole::Mass,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .unwrap();
        let damping = RealSparseMatrix::diagonal(
            vec![0.0; 3],
            OperatorRole::Damping,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .unwrap();
        AssembledRealSystem::try_new(
            CanonicalRealOperators::try_new(stiffness, mass, damping).unwrap(),
            vec![true, false, false],
            vec![2.0, 7.0, 8.0],
        )
        .unwrap()
    }

    #[test]
    fn linear_preparation_preserves_raw_stiffness_and_mass() {
        let system = physical_system();
        let raw_stiffness = system.operators().stiffness().clone();
        let raw_mass = system.operators().mass().clone();
        let prepared = system.prepare_linear().unwrap();

        assert_eq!(system.operators().stiffness(), &raw_stiffness);
        assert_eq!(system.operators().mass(), &raw_mass);
        assert_eq!(
            prepared.matrix().to_dense(),
            vec![1.0, 0.0, 0.0, 0.0, 4.0, -1.0, 0.0, -1.0, 4.0]
        );
        assert_eq!(prepared.rhs(), &[2.0, 9.0, 8.0]);
    }

    #[test]
    fn modal_preparation_reduces_both_k_and_m_without_identity_rows() {
        let system = physical_system();
        let raw_mass = system.operators().mass().clone();
        let modal = system.prepare_modal().unwrap();
        let reduced = modal.as_assembled().unwrap();

        assert_eq!(modal.free_dofs(), &[1, 2]);
        assert_eq!(
            reduced.operators().stiffness().to_dense(),
            vec![4.0, -1.0, -1.0, 4.0]
        );
        assert_eq!(
            reduced.operators().mass().to_dense(),
            vec![2.0, 0.5, 0.5, 2.0]
        );
        assert_eq!(modal.expand(&[3.0, 4.0]).unwrap(), vec![0.0, 3.0, 4.0]);
        assert_eq!(system.operators().mass(), &raw_mass);
    }

    #[test]
    fn deserialization_cannot_bypass_system_shape_validation() {
        let system = physical_system();
        let mut encoded = serde_json::to_value(system).unwrap();
        encoded["load"] = serde_json::json!([1.0]);
        assert!(serde_json::from_value::<AssembledRealSystem>(encoded).is_err());
    }
}
