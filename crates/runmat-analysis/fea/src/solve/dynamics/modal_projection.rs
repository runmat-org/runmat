use runmat_execution::Digest;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    algebra::{spmv, RealSparseMatrix},
    assembly::AssemblySummary,
    pipeline::modal_basis::{ModalBasisArtifact, ModalBasisIdentity},
};

const SYMMETRY_TOLERANCE: f64 = 1.0e-11;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum ModalProjectionError {
    #[error("the modal basis is incompatible with the structural operator: {0}")]
    IncompatibleBasis(String),
    #[error("the modal projection dimensions are inconsistent")]
    DimensionMismatch,
    #[error("the modal projection contains a nonfinite value")]
    NonFinite,
    #[error("the modal mass projection is not positive definite")]
    NonPositiveMass,
    #[error("the projected structural operators are not symmetric")]
    AsymmetricOperator,
}

/// Recomputed evidence for the exact full-order operators and modal basis used by a response step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalProjectionEvidence {
    pub basis: ModalBasisIdentity,
    pub operator_digest: Digest,
    pub full_dof_count: usize,
    pub mode_count: usize,
    pub maximum_mass_off_diagonal_ratio: f64,
    pub maximum_stiffness_off_diagonal_ratio: f64,
    pub maximum_damping_asymmetry_ratio: f64,
}

impl ModalProjectionEvidence {
    /// Validate evidence against the exact physical operator and full-order coordinate space.
    pub fn validate_for(
        &self,
        operator_digest: Digest,
        full_dof_count: usize,
    ) -> Result<(), ModalProjectionError> {
        if self.operator_digest != operator_digest
            || self.full_dof_count != full_dof_count
            || self.mode_count == 0
            || self.mode_count > full_dof_count
        {
            return Err(ModalProjectionError::DimensionMismatch);
        }
        if [
            self.maximum_mass_off_diagonal_ratio,
            self.maximum_stiffness_off_diagonal_ratio,
            self.maximum_damping_asymmetry_ratio,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(ModalProjectionError::NonFinite);
        }
        Ok(())
    }
}

/// Dense reduced K/M/C operators derived from one immutable modal basis and full-order assembly.
///
/// Reduced systems are deliberately dense: modal coupling is small enough to retain every term,
/// and discarding off-diagonal damping would silently assume proportional damping.
#[derive(Clone, Debug, PartialEq)]
pub struct ModalReducedSystem {
    evidence: ModalProjectionEvidence,
    mode_shapes: Vec<Vec<f64>>,
    mass_applied_mode_shapes: Vec<Vec<f64>>,
    stiffness: Vec<f64>,
    mass: Vec<f64>,
    damping: Vec<f64>,
}

impl ModalReducedSystem {
    pub fn project(
        summary: &AssemblySummary,
        basis: &ModalBasisArtifact,
    ) -> Result<Self, ModalProjectionError> {
        basis
            .validate_for(summary)
            .map_err(|error| ModalProjectionError::IncompatibleBasis(error.to_string()))?;
        let mode_shapes = basis
            .modes()
            .iter()
            .map(|mode| mode.dof_shape().to_vec())
            .collect::<Vec<_>>();
        let stiffness = project_operator(summary.operator.operators().stiffness(), &mode_shapes)?;
        let mass = project_operator(summary.operator.operators().mass(), &mode_shapes)?;
        let mass_applied_mode_shapes =
            apply_operator(summary.operator.operators().mass(), &mode_shapes)?;
        let damping = project_operator(summary.operator.operators().damping(), &mode_shapes)?;
        let mode_count = mode_shapes.len();
        if mass
            .iter()
            .step_by(mode_count.saturating_add(1))
            .any(|value| *value <= 0.0)
        {
            return Err(ModalProjectionError::NonPositiveMass);
        }
        if !is_symmetric(&stiffness, mode_count)
            || !is_symmetric(&mass, mode_count)
            || !is_symmetric(&damping, mode_count)
        {
            return Err(ModalProjectionError::AsymmetricOperator);
        }
        let evidence = ModalProjectionEvidence {
            basis: basis.identity(),
            operator_digest: basis.operator_digest(),
            full_dof_count: summary.dof_count,
            mode_count,
            maximum_mass_off_diagonal_ratio: maximum_off_diagonal_ratio(&mass, mode_count),
            maximum_stiffness_off_diagonal_ratio: maximum_off_diagonal_ratio(
                &stiffness, mode_count,
            ),
            maximum_damping_asymmetry_ratio: maximum_asymmetry_ratio(&damping, mode_count),
        };
        let reduced = Self {
            evidence,
            mode_shapes,
            mass_applied_mode_shapes,
            stiffness,
            mass,
            damping,
        };
        reduced.validate()?;
        Ok(reduced)
    }

    pub fn evidence(&self) -> &ModalProjectionEvidence {
        &self.evidence
    }

    pub fn mode_count(&self) -> usize {
        self.evidence.mode_count
    }

    pub fn full_dof_count(&self) -> usize {
        self.evidence.full_dof_count
    }

    pub fn stiffness(&self) -> &[f64] {
        &self.stiffness
    }

    pub fn mass(&self) -> &[f64] {
        &self.mass
    }

    pub fn damping(&self) -> &[f64] {
        &self.damping
    }

    pub fn project_vector(&self, full: &[f64]) -> Result<Vec<f64>, ModalProjectionError> {
        if full.len() != self.full_dof_count() || full.iter().any(|value| !value.is_finite()) {
            return Err(if full.len() == self.full_dof_count() {
                ModalProjectionError::NonFinite
            } else {
                ModalProjectionError::DimensionMismatch
            });
        }
        Ok(self
            .mode_shapes
            .iter()
            .map(|shape| dot(shape, full))
            .collect())
    }

    /// Project a physical displacement or velocity using the modal mass inner product.
    ///
    /// The generalized coordinates solve `M_r q = Phi' M x`; using `Phi' x` is only valid for
    /// an identity physical mass matrix and would give incorrect initial conditions in general.
    pub fn project_physical_state(&self, full: &[f64]) -> Result<Vec<f64>, ModalProjectionError> {
        if full.len() != self.full_dof_count() || full.iter().any(|value| !value.is_finite()) {
            return Err(if full.len() == self.full_dof_count() {
                ModalProjectionError::NonFinite
            } else {
                ModalProjectionError::DimensionMismatch
            });
        }
        let right_hand_side = self
            .mass_applied_mode_shapes
            .iter()
            .map(|mass_applied_shape| dot(mass_applied_shape, full))
            .collect::<Vec<_>>();
        super::dense::solve_square(self.mass(), &right_hand_side, self.mode_count()).map_err(
            |error| match error {
                super::dense::DenseSolveError::DimensionMismatch => {
                    ModalProjectionError::DimensionMismatch
                }
                super::dense::DenseSolveError::NonFinite => ModalProjectionError::NonFinite,
                super::dense::DenseSolveError::Singular => ModalProjectionError::NonPositiveMass,
            },
        )
    }

    pub fn expand_vector(&self, reduced: &[f64]) -> Result<Vec<f64>, ModalProjectionError> {
        expand_vector(&self.mode_shapes, self.full_dof_count(), reduced)
    }

    fn validate(&self) -> Result<(), ModalProjectionError> {
        self.evidence
            .validate_for(self.evidence.operator_digest, self.evidence.full_dof_count)?;
        let mode_count = self.mode_count();
        let matrix_size = mode_count
            .checked_mul(mode_count)
            .ok_or(ModalProjectionError::DimensionMismatch)?;
        if mode_count == 0
            || self.full_dof_count() == 0
            || self.mode_shapes.len() != mode_count
            || self.mass_applied_mode_shapes.len() != mode_count
            || self
                .mode_shapes
                .iter()
                .any(|shape| shape.len() != self.full_dof_count())
            || self
                .mass_applied_mode_shapes
                .iter()
                .any(|shape| shape.len() != self.full_dof_count())
            || self.stiffness.len() != matrix_size
            || self.mass.len() != matrix_size
            || self.damping.len() != matrix_size
        {
            return Err(ModalProjectionError::DimensionMismatch);
        }
        if self
            .mode_shapes
            .iter()
            .flatten()
            .chain(self.mass_applied_mode_shapes.iter().flatten())
            .chain(&self.stiffness)
            .chain(&self.mass)
            .chain(&self.damping)
            .any(|value| !value.is_finite())
        {
            return Err(ModalProjectionError::NonFinite);
        }
        Ok(())
    }
}

fn project_operator(
    operator: &RealSparseMatrix,
    mode_shapes: &[Vec<f64>],
) -> Result<Vec<f64>, ModalProjectionError> {
    if mode_shapes.is_empty()
        || mode_shapes
            .iter()
            .any(|shape| shape.len() != operator.columns())
    {
        return Err(ModalProjectionError::DimensionMismatch);
    }
    let applied = apply_operator(operator, mode_shapes)?;
    let mut projected = vec![0.0; mode_shapes.len() * mode_shapes.len()];
    for (row, left) in mode_shapes.iter().enumerate() {
        for (column, right_applied) in applied.iter().enumerate() {
            projected[row * mode_shapes.len() + column] = dot(left, right_applied);
        }
    }
    if projected.iter().any(|value| !value.is_finite()) {
        return Err(ModalProjectionError::NonFinite);
    }
    Ok(projected)
}

fn apply_operator(
    operator: &RealSparseMatrix,
    mode_shapes: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, ModalProjectionError> {
    if mode_shapes.is_empty()
        || mode_shapes
            .iter()
            .any(|shape| shape.len() != operator.columns())
    {
        return Err(ModalProjectionError::DimensionMismatch);
    }
    mode_shapes
        .iter()
        .map(|shape| spmv(operator, shape).map_err(|_| ModalProjectionError::DimensionMismatch))
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn expand_vector(
    mode_shapes: &[Vec<f64>],
    full_dof_count: usize,
    reduced: &[f64],
) -> Result<Vec<f64>, ModalProjectionError> {
    if reduced.len() != mode_shapes.len() || reduced.iter().any(|value| !value.is_finite()) {
        return Err(if reduced.len() == mode_shapes.len() {
            ModalProjectionError::NonFinite
        } else {
            ModalProjectionError::DimensionMismatch
        });
    }
    let mut full = vec![0.0; full_dof_count];
    for (coefficient, shape) in reduced.iter().zip(mode_shapes) {
        if shape.len() != full_dof_count {
            return Err(ModalProjectionError::DimensionMismatch);
        }
        for (value, component) in full.iter_mut().zip(shape) {
            *value += coefficient * component;
        }
    }
    Ok(full)
}

fn is_symmetric(matrix: &[f64], size: usize) -> bool {
    maximum_asymmetry_ratio(matrix, size) <= SYMMETRY_TOLERANCE
}

fn maximum_asymmetry_ratio(matrix: &[f64], size: usize) -> f64 {
    let scale = matrix
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    (0..size)
        .flat_map(|row| (row + 1..size).map(move |column| (row, column)))
        .map(|(row, column)| {
            (matrix[row * size + column] - matrix[column * size + row]).abs() / scale
        })
        .fold(0.0_f64, f64::max)
}

fn maximum_off_diagonal_ratio(matrix: &[f64], size: usize) -> f64 {
    let diagonal_scale = (0..size)
        .map(|index| matrix[index * size + index].abs())
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    (0..size)
        .flat_map(|row| (0..size).map(move |column| (row, column)))
        .filter(|(row, column)| row != column)
        .map(|(row, column)| matrix[row * size + column].abs() / diagonal_scale)
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{MatrixProperties, OperatorRole};
    use runmat_analysis_core::{FormulationRequirement, ProcedureStep};

    #[test]
    fn exact_projection_retains_coupling_and_reconstructs_vectors() {
        let operator = RealSparseMatrix::from_dense(
            &[4.0, 1.0, 1.0, 3.0],
            2,
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .unwrap();
        let shapes = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(
            project_operator(&operator, &shapes).unwrap(),
            operator.to_dense()
        );

        assert_eq!(
            expand_vector(&shapes, 2, &[2.0, -1.0]).unwrap(),
            vec![2.0, -1.0]
        );
        assert_eq!(maximum_asymmetry_ratio(&[0.2, 0.1, 0.1, 0.3], 2), 0.0);
    }

    #[test]
    fn physical_modal_basis_reprojects_against_its_exact_assembly() {
        let mut model =
            crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
        model.sequence.steps = vec![ProcedureStep::structural_modal("projection")];
        let modal = crate::run_modal_with_options(
            &model,
            crate::ComputeBackend::Cpu,
            crate::ModalSolveOptions {
                formulation_requirement: FormulationRequirement::Experimental,
                ..crate::ModalSolveOptions::default()
            },
        )
        .unwrap();
        let step = &model.sequence.steps[0];
        let excitations = crate::excitation::realize_step_excitations(
            &model,
            step,
            crate::excitation::ExcitationSamplePoint::Static,
        )
        .unwrap();
        let summary = crate::assembly::assemble_structural_system(
            &model,
            &excitations,
            None,
            crate::assembly::StructuralFormulationRequirement::ExperimentalSyntheticOperator,
        )
        .unwrap();
        let reduced = ModalReducedSystem::project(&summary, &modal.basis).unwrap();

        assert_eq!(reduced.mode_count(), modal.basis.modes().len());
        assert_eq!(reduced.mass(), modal.basis.mass_orthogonality());
        for (mode_index, mode) in modal.basis.modes().iter().enumerate() {
            let mut coordinate = vec![0.0; reduced.mode_count()];
            coordinate[mode_index] = 1.0;
            assert_eq!(
                reduced.expand_vector(&coordinate).unwrap(),
                mode.dof_shape()
            );
            assert!(
                (reduced.stiffness()[mode_index * reduced.mode_count() + mode_index]
                    - mode.modal_stiffness())
                .abs()
                    <= 1.0e-10 * mode.modal_stiffness().abs().max(1.0)
            );
        }
        assert_eq!(reduced.evidence().basis, modal.basis.identity());
        assert_eq!(
            reduced.evidence().operator_digest,
            modal.basis.operator_digest()
        );
        for (mode_index, mode) in modal.basis.modes().iter().enumerate() {
            let projected = reduced.project_physical_state(mode.dof_shape()).unwrap();
            for (coordinate_index, coordinate) in projected.iter().enumerate() {
                let expected = usize::from(mode_index == coordinate_index) as f64;
                assert!((coordinate - expected).abs() <= 1.0e-10);
            }
        }
    }
}
