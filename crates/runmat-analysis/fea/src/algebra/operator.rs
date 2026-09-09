use serde::{Deserialize, Serialize};

use super::{spmv, OperatorRole, RealSparseMatrix};

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
}

#[derive(Deserialize)]
struct AssembledRealSystemData {
    operators: CanonicalRealOperators,
    constrained_dofs: Vec<bool>,
    load: Vec<f64>,
}

impl TryFrom<AssembledRealSystemData> for AssembledRealSystem {
    type Error = RealSystemError;

    fn try_from(data: AssembledRealSystemData) -> Result<Self, Self::Error> {
        Self::try_new(data.operators, data.constrained_dofs, data.load)
    }
}

impl AssembledRealSystem {
    pub fn try_new(
        operators: CanonicalRealOperators,
        constrained_dofs: Vec<bool>,
        load: Vec<f64>,
    ) -> Result<Self, RealSystemError> {
        let n = operators.dof_count();
        if constrained_dofs.len() != n || load.len() != n {
            return Err(RealSystemError::ShapeMismatch);
        }
        if let Some(index) = load.iter().position(|value| !value.is_finite()) {
            return Err(RealSystemError::NonFiniteLoad { index });
        }
        Ok(Self {
            operators,
            constrained_dofs,
            load,
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
    pub fn with_load(&self, load: Vec<f64>) -> Result<Self, RealSystemError> {
        Self::try_new(self.operators.clone(), self.constrained_dofs.clone(), load)
    }
    pub fn prepare_linear(&self) -> Result<PreparedRealLinearSystem, RealSystemError> {
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
        let matrix = stiffness
            .with_identity_constraints(&self.constrained_dofs)
            .map_err(|_| RealSystemError::ShapeMismatch)?;
        PreparedRealLinearSystem::try_new(matrix, rhs, self.constrained_dofs.clone())
    }

    pub fn prepare_modal(&self) -> Result<PreparedModalSystem, RealSystemError> {
        let free_dofs = self
            .constrained_dofs
            .iter()
            .enumerate()
            .filter_map(|(index, constrained)| (!constrained).then_some(index))
            .collect::<Vec<_>>();
        let reduce = |matrix: &RealSparseMatrix| {
            matrix
                .principal_submatrix(&free_dofs)
                .map_err(|_| RealSystemError::ShapeMismatch)
        };
        let operators = CanonicalRealOperators::try_new(
            reduce(self.operators.stiffness())?,
            reduce(self.operators.mass())?,
            reduce(self.operators.damping())?,
        )?;
        Ok(PreparedModalSystem {
            operators,
            free_dofs,
            full_dof_count: self.dof_count(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "PreparedModalSystemData")]
pub struct PreparedModalSystem {
    operators: CanonicalRealOperators,
    free_dofs: Vec<usize>,
    full_dof_count: usize,
}

#[derive(Deserialize)]
struct PreparedModalSystemData {
    operators: CanonicalRealOperators,
    free_dofs: Vec<usize>,
    full_dof_count: usize,
}

impl TryFrom<PreparedModalSystemData> for PreparedModalSystem {
    type Error = RealSystemError;

    fn try_from(data: PreparedModalSystemData) -> Result<Self, Self::Error> {
        if data.operators.dof_count() != data.free_dofs.len()
            || data
                .free_dofs
                .iter()
                .any(|index| *index >= data.full_dof_count)
            || data.free_dofs.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(RealSystemError::ShapeMismatch);
        }
        Ok(Self {
            operators: data.operators,
            free_dofs: data.free_dofs,
            full_dof_count: data.full_dof_count,
        })
    }
}

impl PreparedModalSystem {
    pub fn as_assembled(&self) -> Result<AssembledRealSystem, RealSystemError> {
        let dof_count = self.operators.dof_count();
        AssembledRealSystem::try_new(
            self.operators.clone(),
            vec![false; dof_count],
            vec![0.0; dof_count],
        )
    }

    pub fn free_dofs(&self) -> &[usize] {
        &self.free_dofs
    }

    pub fn expand(&self, reduced: &[f64]) -> Result<Vec<f64>, RealSystemError> {
        if reduced.len() != self.free_dofs.len() {
            return Err(RealSystemError::ShapeMismatch);
        }
        let mut full = vec![0.0; self.full_dof_count];
        for (value, original) in reduced.iter().zip(&self.free_dofs) {
            full[*original] = *value;
        }
        Ok(full)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "PreparedRealLinearSystemData")]
pub struct PreparedRealLinearSystem {
    matrix: RealSparseMatrix,
    rhs: Vec<f64>,
    constrained_dofs: Vec<bool>,
}

#[derive(Deserialize)]
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
        if matrix.rows() != matrix.columns()
            || rhs.len() != matrix.rows()
            || constrained_dofs.len() != matrix.rows()
        {
            return Err(RealSystemError::ShapeMismatch);
        }
        if let Some(index) = rhs.iter().position(|value| !value.is_finite()) {
            return Err(RealSystemError::NonFiniteLoad { index });
        }
        Ok(Self {
            matrix,
            rhs,
            constrained_dofs,
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
