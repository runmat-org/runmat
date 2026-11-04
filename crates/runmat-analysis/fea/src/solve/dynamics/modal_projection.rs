use thiserror::Error;

use crate::{
    algebra::{spmv, RealSparseMatrix},
    assembly::AssemblySummary,
    execution::ModalProjectionEvidence,
    pipeline::modal_basis::ModalBasisArtifact,
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

/// Dense reduced K/M/C operators derived from one immutable modal basis and full-order assembly.
///
/// Reduced systems are deliberately dense: modal coupling is small enough to retain every term,
/// and discarding off-diagonal damping would silently assume proportional damping.
#[derive(Clone, Debug, PartialEq)]
pub struct ModalReducedSystem {
    evidence: ModalProjectionEvidence,
    coordinates: std::sync::Arc<crate::pipeline::modal_basis::ModalBasisCoordinates>,
    mode_shapes: Vec<Vec<f64>>,
    mass_applied_mode_shapes: Vec<Vec<f64>>,
    stiffness: runmat_types::physical::Quantity<runmat_types::physical::kind::Stiffness, Vec<f64>>,
    mass: runmat_types::physical::Quantity<runmat_types::physical::kind::Mass, Vec<f64>>,
    damping: runmat_types::physical::Quantity<runmat_types::physical::kind::Damping, Vec<f64>>,
}

impl ModalReducedSystem {
    /// Project onto the retained eigenmodes of `basis` only.
    pub fn project(
        summary: &AssemblySummary,
        basis: &ModalBasisArtifact,
    ) -> Result<Self, ModalProjectionError> {
        let mode_shapes = basis
            .modes()
            .iter()
            .map(|mode| mode.dof_shape().to_vec())
            .collect::<Vec<_>>();
        Self::project_shapes(summary, basis, mode_shapes)
    }

    /// Project onto the retained eigenmodes, then the basis residual vectors, then
    /// `static_corrections` (mass-orthonormal static responses appended by the caller).
    pub fn project_enriched(
        summary: &AssemblySummary,
        basis: &ModalBasisArtifact,
        static_corrections: Vec<Vec<f64>>,
    ) -> Result<Self, ModalProjectionError> {
        let mode_shapes = basis
            .modes()
            .iter()
            .map(|mode| mode.dof_shape().to_vec())
            .chain(
                basis
                    .residual_vectors()
                    .iter()
                    .map(|residual| residual.dof_shape().to_vec()),
            )
            .chain(static_corrections)
            .collect::<Vec<_>>();
        Self::project_shapes(summary, basis, mode_shapes)
    }

    fn project_shapes(
        summary: &AssemblySummary,
        basis: &ModalBasisArtifact,
        mode_shapes: Vec<Vec<f64>>,
    ) -> Result<Self, ModalProjectionError> {
        basis
            .validate_for(summary)
            .map_err(|error| ModalProjectionError::IncompatibleBasis(error.to_string()))?;
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
            coordinates: basis.shared_coordinates(),
            mode_shapes,
            mass_applied_mode_shapes,
            stiffness: runmat_types::physical::Quantity::try_new(stiffness)
                .map_err(|_| ModalProjectionError::NonFinite)?,
            mass: runmat_types::physical::Quantity::try_new(mass)
                .map_err(|_| ModalProjectionError::NonFinite)?,
            damping: runmat_types::physical::Quantity::try_new(damping)
                .map_err(|_| ModalProjectionError::NonFinite)?,
        };
        reduced.validate()?;
        Ok(reduced)
    }

    pub fn coordinates(&self) -> &crate::pipeline::modal_basis::ModalBasisCoordinates {
        &self.coordinates
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

    pub fn stiffness(
        &self,
    ) -> runmat_types::physical::Quantity<runmat_types::physical::kind::Stiffness, &[f64]> {
        runmat_types::physical::Quantity::new(self.stiffness.canonical_values().as_slice())
    }

    pub fn mass(
        &self,
    ) -> runmat_types::physical::Quantity<runmat_types::physical::kind::Mass, &[f64]> {
        runmat_types::physical::Quantity::new(self.mass.canonical_values().as_slice())
    }

    pub fn damping(
        &self,
    ) -> runmat_types::physical::Quantity<runmat_types::physical::kind::Damping, &[f64]> {
        runmat_types::physical::Quantity::new(self.damping.canonical_values().as_slice())
    }

    pub fn load_field<'a>(
        &'a self,
        values: &'a [f64],
    ) -> Result<
        runmat_types::physical::QuantityField<
            crate::assembly::dofs::StructuralQuantityLayout<'a>,
            &'a [f64],
        >,
        ModalProjectionError,
    > {
        self.coordinates
            .full_dof_layout()
            .physical_field(crate::assembly::dofs::StructuralQuantityLane::Load, values)
            .map_err(|_| ModalProjectionError::DimensionMismatch)
    }

    pub fn project_load<S: AsRef<[f64]>>(
        &self,
        full: &runmat_types::physical::QuantityField<
            crate::assembly::dofs::StructuralQuantityLayout<'_>,
            S,
        >,
    ) -> Result<
        runmat_types::physical::Quantity<runmat_types::physical::kind::Force, Vec<f64>>,
        ModalProjectionError,
    > {
        if full.layout().dofs != self.coordinates.full_dof_layout()
            || full.layout().lane != crate::assembly::dofs::StructuralQuantityLane::Load
        {
            return Err(ModalProjectionError::DimensionMismatch);
        }
        // Phi is the retained work-conjugate transformation: full forces and
        // moments project to force conjugate to the declared modal length.
        let values = full.with_canonical_storage(|values| {
            self.mode_shapes
                .iter()
                .map(|shape| dot(shape, values))
                .collect::<Vec<_>>()
        });
        runmat_types::physical::Quantity::try_new(values)
            .map_err(|_| ModalProjectionError::NonFinite)
    }

    /// Project a physical state through its retained mass metric. A raw array
    /// cannot choose translation/rotation or derivative roles by its shape.
    pub fn project_state<K: ModalCoordinateQuantity, S: AsRef<[f64]>>(
        &self,
        full: &runmat_types::physical::QuantityField<
            crate::assembly::dofs::StructuralQuantityLayout<'_>,
            S,
        >,
    ) -> Result<runmat_types::physical::Quantity<K, Vec<f64>>, ModalProjectionError> {
        if full.layout().dofs != self.coordinates.full_dof_layout()
            || full.layout().lane
                != (crate::assembly::dofs::StructuralQuantityLane::State {
                    derivative_order: K::DERIVATIVE_ORDER,
                })
        {
            return Err(ModalProjectionError::DimensionMismatch);
        }
        let right_hand_side = full.with_canonical_storage(|values| {
            self.mass_applied_mode_shapes
                .iter()
                .map(|shape| dot(shape, values))
                .collect::<Vec<_>>()
        });
        let values = super::dense::solve_square(
            self.mass().canonical_value(),
            &right_hand_side,
            self.mode_count(),
        )
        .map_err(|error| match error {
            super::dense::DenseSolveError::DimensionMismatch => {
                ModalProjectionError::DimensionMismatch
            }
            super::dense::DenseSolveError::NonFinite => ModalProjectionError::NonFinite,
            super::dense::DenseSolveError::Singular => ModalProjectionError::NonPositiveMass,
        })?;
        runmat_types::physical::Quantity::try_new(values)
            .map_err(|_| ModalProjectionError::NonFinite)
    }

    pub fn state_field<'a, K: ModalCoordinateQuantity>(
        &'a self,
        values: &'a [f64],
    ) -> Result<
        runmat_types::physical::QuantityField<
            crate::assembly::dofs::StructuralQuantityLayout<'a>,
            &'a [f64],
        >,
        ModalProjectionError,
    > {
        self.coordinates
            .full_dof_layout()
            .physical_field(
                crate::assembly::dofs::StructuralQuantityLane::State {
                    derivative_order: K::DERIVATIVE_ORDER,
                },
                values,
            )
            .map_err(|_| ModalProjectionError::DimensionMismatch)
    }

    pub fn expand_state<K: ModalCoordinateQuantity, S: AsRef<[f64]>>(
        &self,
        reduced: &runmat_types::physical::Quantity<K, S>,
    ) -> Result<
        runmat_types::physical::QuantityField<
            crate::assembly::dofs::StructuralQuantityLayout<'_>,
            Vec<f64>,
        >,
        ModalProjectionError,
    > {
        let values = expand_vector(
            &self.mode_shapes,
            self.full_dof_count(),
            reduced.canonical_values().as_ref(),
        )?;
        runmat_types::physical::QuantityField::try_new(
            crate::assembly::dofs::StructuralQuantityLayout {
                dofs: self.coordinates.full_dof_layout(),
                lane: crate::assembly::dofs::StructuralQuantityLane::State {
                    derivative_order: K::DERIVATIVE_ORDER,
                },
            },
            values,
        )
        .map_err(|_| ModalProjectionError::NonFinite)
    }

    fn validate(&self) -> Result<(), ModalProjectionError> {
        self.coordinates
            .validate()
            .map_err(|error| ModalProjectionError::IncompatibleBasis(error.into()))?;
        if self.coordinates.full_dof_layout().total_dof_count() != self.full_dof_count() {
            return Err(ModalProjectionError::DimensionMismatch);
        }

        self.evidence
            .validate_for(self.evidence.operator_digest, self.evidence.full_dof_count)
            .map_err(|error| ModalProjectionError::IncompatibleBasis(error.to_string()))?;
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
            || self.stiffness.canonical_values().len() != matrix_size
            || self.mass.canonical_values().len() != matrix_size
            || self.damping.canonical_values().len() != matrix_size
        {
            return Err(ModalProjectionError::DimensionMismatch);
        }
        if self
            .mode_shapes
            .iter()
            .flatten()
            .chain(self.mass_applied_mode_shapes.iter().flatten())
            .chain(self.stiffness.canonical_values())
            .chain(self.mass.canonical_values())
            .chain(self.damping.canonical_values())
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
    use runmat_analysis_core::ProcedureStep;

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
            crate::assembly::StructuralFormulationRequirement::SyntheticOperator,
        )
        .unwrap();
        let reduced = ModalReducedSystem::project(&summary, &modal.basis).unwrap();

        assert_eq!(reduced.mode_count(), modal.basis.modes().len());
        assert_eq!(
            reduced.mass().canonical_value(),
            modal.basis.mass_orthogonality()
        );
        for (mode_index, mode) in modal.basis.modes().iter().enumerate() {
            let mut coordinate = vec![0.0; reduced.mode_count()];
            coordinate[mode_index] = 1.0;
            // Source millimeters are converted once, before entering a generalized
            // length coordinate; reconstruction does not silently reinterpret them.
            let millimeters = runmat_types::physical::Unit::<
                runmat_types::physical::kind::Displacement,
            >::try_from_definition(
                runmat_types::physical::LengthUnit::Millimeter
                    .definition()
                    .clone(),
            )
            .unwrap();
            let source_amplitude = millimeters.quantity(2.5).unwrap();
            let mut source_coordinate = vec![0.0; reduced.mode_count()];
            source_coordinate[mode_index] = source_amplitude.canonical_value();
            let source_state = reduced
                .expand_state(&runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Displacement,
                    _,
                >::new(source_coordinate))
                .unwrap();
            let roundtrip = reduced
                .project_state::<runmat_types::physical::kind::Displacement, _>(&source_state)
                .unwrap();
            assert!((roundtrip.canonical_values()[mode_index] - 0.0025).abs() <= 1.0e-12);
            source_state.with_canonical_storage(|values| {
                for (actual, shape) in values.iter().zip(mode.dof_shape()) {
                    assert!((actual - shape * 0.0025).abs() <= 1.0e-12);
                }
            });
            assert_eq!(
                reduced
                    .expand_state(&runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Displacement,
                        _,
                    >::new(&coordinate))
                    .unwrap()
                    .into_canonical_storage(),
                mode.dof_shape()
            );
            assert!(
                (reduced.stiffness().canonical_value()
                    [mode_index * reduced.mode_count() + mode_index]
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
            let state = reduced
                .state_field::<runmat_types::physical::kind::Displacement>(mode.dof_shape())
                .unwrap();
            let projected = reduced
                .project_state::<runmat_types::physical::kind::Displacement, _>(&state)
                .unwrap();
            for (coordinate_index, coordinate) in projected.canonical_values().iter().enumerate() {
                let expected = usize::from(mode_index == coordinate_index) as f64;
                assert!((coordinate - expected).abs() <= 1.0e-10);
            }
        }
    }
    /// Full-space relative balance of a modal-reduced harmonic solve at `omega` (rad/s).
    fn reduced_balance(summary: &AssemblySummary, reduced: &ModalReducedSystem, omega: f64) -> f64 {
        let zeros = vec![0.0; summary.dof_count];
        let load = super::super::ModalHarmonicLoad {
            real: reduced
                .project_load(&reduced.load_field(summary.operator.load()).unwrap())
                .unwrap(),
            imaginary: reduced
                .project_load(&reduced.load_field(&zeros).unwrap())
                .unwrap(),
        };
        let response = super::super::solve_modal_harmonic(reduced, omega, &load).unwrap();
        assert!(response.relative_residual <= 1.0e-12);
        let real = reduced
            .expand_state(&response.displacement_real)
            .unwrap()
            .into_canonical_storage();
        let imaginary = reduced
            .expand_state(&response.displacement_imaginary)
            .unwrap()
            .into_canonical_storage();
        let physical = super::super::harmonic_balance::evaluate_harmonic_physical_state(
            summary,
            omega,
            &real,
            &imaginary,
            summary.operator.load(),
            &zeros,
        )
        .unwrap();
        physical
            .balance
            .closure_metrics(&physical.reaction)
            .unwrap()
            .1
    }

    // witness: FEA-DYNAMICS-RESPONSE-001/direct_modal_procedures @ linear_structural_dynamics/tet4/numerical_kernel; coverage: partial; boundary: applied-load static correction cuts the full-space truncation balance of a modal-reduced harmonic solve below the first mode
    #[test]
    fn applied_load_static_correction_collapses_the_modal_truncation_balance() {
        let mut model =
            crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
        model.sequence.steps = vec![ProcedureStep::structural_modal("projection")];
        let modal = crate::run_modal_with_options(
            &model,
            crate::ComputeBackend::Cpu,
            crate::ModalSolveOptions::default(),
        )
        .unwrap();
        let excitations = crate::excitation::realize_step_excitations(
            &model,
            &model.sequence.steps[0],
            crate::excitation::ExcitationSamplePoint::Static,
        )
        .unwrap();
        let summary = crate::assembly::assemble_structural_system(
            &model,
            &excitations,
            None,
            crate::assembly::StructuralFormulationRequirement::SyntheticOperator,
        )
        .unwrap();
        assert!(summary.operator.load().iter().any(|value| *value != 0.0));
        let omega =
            0.25 * 2.0 * std::f64::consts::PI * modal.basis.modes()[0].frequency_hz();

        let modes_only = ModalReducedSystem::project(&summary, &modal.basis).unwrap();
        let truncated = reduced_balance(&summary, &modes_only, omega);

        let retained = modal
            .basis
            .modes()
            .iter()
            .map(|mode| mode.dof_shape())
            .collect::<Vec<_>>();
        let corrections = crate::pipeline::modal_residual::applied_load_static_corrections(
            &summary,
            &retained,
            &[summary.operator.load()],
        )
        .unwrap();
        assert_eq!(corrections.len(), 1);
        let corrected = ModalReducedSystem::project_enriched(&summary, &modal.basis, corrections)
            .unwrap();
        assert_eq!(corrected.mode_count(), modes_only.mode_count() + 1);
        let enriched = reduced_balance(&summary, &corrected, omega);

        assert!(truncated > 1.0e-4, "modes-only truncation balance={truncated}");
        assert!(
            enriched <= 1.0e-2 * truncated,
            "static correction must cut the truncation balance by 100x: {truncated} -> {enriched}"
        );
    }
}

/// Closed semantic roles admitted by modal state projection/reconstruction.
pub trait ModalCoordinateQuantity: runmat_types::physical::QuantityMarker {
    const DERIVATIVE_ORDER: u8;
}
impl ModalCoordinateQuantity for runmat_types::physical::kind::Displacement {
    const DERIVATIVE_ORDER: u8 = 0;
}
impl ModalCoordinateQuantity for runmat_types::physical::kind::Velocity {
    const DERIVATIVE_ORDER: u8 = 1;
}
impl ModalCoordinateQuantity for runmat_types::physical::kind::Acceleration {
    const DERIVATIVE_ORDER: u8 = 2;

}
