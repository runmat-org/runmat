use runmat_analysis_core::RandomExcitationSpectrum;

use crate::{
    assembly::{dofs::StructuralDofKind, AssemblySummary},
    pipeline::modal_basis::ModalBasisArtifact,
};

#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    real: f64,
    imaginary: f64,
}

impl Complex {
    const fn new(real: f64, imaginary: f64) -> Self {
        Self { real, imaginary }
    }

    fn norm_squared(self) -> f64 {
        self.real * self.real + self.imaginary * self.imaginary
    }

    fn conjugate(self) -> Self {
        Self::new(self.real, -self.imaginary)
    }
}

impl std::ops::Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.real + rhs.real, self.imaginary + rhs.imaginary)
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.real - rhs.real, self.imaginary - rhs.imaginary)
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.real * rhs.real - self.imaginary * rhs.imaginary,
            self.real * rhs.imaginary + self.imaginary * rhs.real,
        )
    }
}

impl std::ops::Div for Complex {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        let denominator = rhs.norm_squared();
        self * Self::new(rhs.real / denominator, -rhs.imaginary / denominator)
    }
}

pub(super) fn integrate_dense_response(
    summary: &AssemblySummary,
    basis: &ModalBasisArtifact,
    excitation: &RandomExcitationSpectrum,
    damping_ratio: f64,
    panel_count: usize,
) -> Vec<f64> {
    assert_eq!(panel_count % 2, 0, "composite Simpson panels must be even");
    let system = DenseReferenceSystem::new(summary, basis, excitation, damping_ratio);
    let lower = excitation.frequencies_hz.canonical_values()[0];
    let upper = *excitation.frequencies_hz.canonical_values().last().unwrap();
    let step = (upper - lower) / panel_count as f64;
    let mut integral = vec![0.0; summary.dof_count];
    for panel in 0..=panel_count {
        let frequency = lower + panel as f64 * step;
        let weight = if panel == 0 || panel == panel_count {
            1.0
        } else if panel % 2 == 0 {
            2.0
        } else {
            4.0
        };
        for (sum, density) in integral.iter_mut().zip(system.density(frequency)) {
            *sum += weight * density;
        }
    }
    for value in &mut integral {
        *value *= step / 3.0;
    }
    integral
}

struct DenseReferenceSystem {
    full_dof_count: usize,
    transform: Vec<f64>,
    reduced_dof_count: usize,
    stiffness: Vec<f64>,
    mass: Vec<f64>,
    damping: Vec<f64>,
    loads: Vec<Vec<f64>>,
    spectral_density: Vec<Complex>,
}

impl DenseReferenceSystem {
    fn new(
        summary: &AssemblySummary,
        basis: &ModalBasisArtifact,
        excitation: &RandomExcitationSpectrum,
        damping_ratio: f64,
    ) -> Self {
        let full = summary.dof_count;
        let (transform, free_dof_indices, reduced) = constraint_transform(summary);
        let stiffness = reduce_matrix(
            &summary.operator.operators().stiffness().to_dense(),
            &transform,
            full,
            reduced,
        );
        let mass = reduce_matrix(
            &summary.operator.operators().mass().to_dense(),
            &transform,
            full,
            reduced,
        );
        let mut damping = vec![0.0; reduced * reduced];
        for mode in basis.modes() {
            let reduced_shape = free_dof_indices
                .iter()
                .map(|dof| mode.dof_shape()[*dof])
                .collect::<Vec<_>>();
            let mass_shape = multiply_real_matrix_vector(&mass, &reduced_shape, reduced);
            let coefficient = 2.0 * damping_ratio * std::f64::consts::TAU * mode.frequency_hz()
                / mode.modal_mass();
            for row in 0..reduced {
                for column in 0..reduced {
                    damping[row * reduced + column] +=
                        coefficient * mass_shape[row] * mass_shape[column];
                }
            }
        }
        let loads = excitation
            .channels
            .iter()
            .map(|channel| {
                let direction = normalize(channel.direction.vector);
                let influence = influence_vector(summary, direction);
                let reduced_influence = transpose_multiply(&transform, &influence, full, reduced);
                multiply_real_matrix_vector(&mass, &reduced_influence, reduced)
                    .into_iter()
                    .map(|value| -value)
                    .collect()
            })
            .collect();
        let spectral_density = excitation.resolved_spectral_density().unwrap()[0]
            .values
            .canonical_values()
            .iter()
            .map(|value| Complex::new(value.real, value.imaginary))
            .collect();
        Self {
            full_dof_count: full,
            transform,
            reduced_dof_count: reduced,
            stiffness,
            mass,
            damping,
            loads,
            spectral_density,
        }
    }

    fn density(&self, frequency_hz: f64) -> Vec<f64> {
        let omega = std::f64::consts::TAU * frequency_hz;
        let dynamic = self
            .stiffness
            .iter()
            .zip(&self.mass)
            .zip(&self.damping)
            .map(|((stiffness, mass), damping)| {
                Complex::new(stiffness - omega * omega * mass, omega * damping)
            })
            .collect::<Vec<_>>();
        let transfers = self
            .loads
            .iter()
            .map(|load| {
                let reduced = solve_complex(
                    dynamic.clone(),
                    load.iter().map(|value| Complex::new(*value, 0.0)).collect(),
                    self.reduced_dof_count,
                );
                multiply_complex_transform(
                    &self.transform,
                    &reduced,
                    self.full_dof_count,
                    self.reduced_dof_count,
                )
            })
            .collect::<Vec<_>>();
        (0..self.full_dof_count)
            .map(|dof| {
                let mut density = Complex::default();
                for row in 0..transfers.len() {
                    for column in 0..transfers.len() {
                        density = density
                            + transfers[row][dof]
                                * self.spectral_density[row * transfers.len() + column]
                                * transfers[column][dof].conjugate();
                    }
                }
                assert!(density.imaginary.abs() <= 1.0e-8 * density.real.abs().max(1.0e-30));
                density.real.max(0.0)
            })
            .collect()
    }
}

fn constraint_transform(summary: &AssemblySummary) -> (Vec<f64>, Vec<usize>, usize) {
    let full = summary.dof_count;
    if let Some(reduction) = &summary.affine_constraint_reduction {
        assert!(reduction
            .offset
            .iter()
            .all(|value| value.abs() <= f64::EPSILON));
        assert_eq!(reduction.full_dof_count, full);
        let reduced = reduction.reduced_dof_count;
        let mut transform = vec![0.0; full * reduced];
        for row in 0..full {
            for entry in
                reduction.recovery_row_offsets[row]..reduction.recovery_row_offsets[row + 1]
            {
                transform[row * reduced + reduction.recovery_column_indices[entry]] =
                    reduction.recovery_coefficients[entry];
            }
        }
        return (transform, reduction.free_dof_indices.clone(), reduced);
    }

    let free_dof_indices = summary
        .operator
        .constrained_dofs()
        .iter()
        .enumerate()
        .filter_map(|(dof, constrained)| (!constrained).then_some(dof))
        .collect::<Vec<_>>();
    let reduced = free_dof_indices.len();
    let mut transform = vec![0.0; full * reduced];
    for (column, row) in free_dof_indices.iter().copied().enumerate() {
        transform[row * reduced + column] = 1.0;
    }
    (transform, free_dof_indices, reduced)
}

fn reduce_matrix(full_matrix: &[f64], transform: &[f64], full: usize, reduced: usize) -> Vec<f64> {
    let mut result = vec![0.0; reduced * reduced];
    for row in 0..reduced {
        for column in 0..reduced {
            for left in 0..full {
                for right in 0..full {
                    result[row * reduced + column] += transform[left * reduced + row]
                        * full_matrix[left * full + right]
                        * transform[right * reduced + column];
                }
            }
        }
    }
    result
}

fn transpose_multiply(matrix: &[f64], vector: &[f64], rows: usize, columns: usize) -> Vec<f64> {
    (0..columns)
        .map(|column| {
            (0..rows)
                .map(|row| matrix[row * columns + column] * vector[row])
                .sum()
        })
        .collect()
}

fn multiply_real_matrix_vector(matrix: &[f64], vector: &[f64], size: usize) -> Vec<f64> {
    (0..size)
        .map(|row| {
            (0..size)
                .map(|column| matrix[row * size + column] * vector[column])
                .sum()
        })
        .collect()
}

fn multiply_complex_transform(
    matrix: &[f64],
    vector: &[Complex],
    rows: usize,
    columns: usize,
) -> Vec<Complex> {
    (0..rows)
        .map(|row| {
            (0..columns).fold(Complex::default(), |sum, column| {
                sum + Complex::new(matrix[row * columns + column], 0.0) * vector[column]
            })
        })
        .collect()
}

fn solve_complex(mut matrix: Vec<Complex>, mut rhs: Vec<Complex>, size: usize) -> Vec<Complex> {
    for pivot in 0..size {
        let best = (pivot..size)
            .max_by(|left, right| {
                matrix[*left * size + pivot]
                    .norm_squared()
                    .total_cmp(&matrix[*right * size + pivot].norm_squared())
            })
            .unwrap();
        assert!(matrix[best * size + pivot].norm_squared() > f64::MIN_POSITIVE);
        if best != pivot {
            for column in 0..size {
                matrix.swap(pivot * size + column, best * size + column);
            }
            rhs.swap(pivot, best);
        }
        for row in pivot + 1..size {
            let factor = matrix[row * size + pivot] / matrix[pivot * size + pivot];
            matrix[row * size + pivot] = Complex::default();
            for column in pivot + 1..size {
                matrix[row * size + column] =
                    matrix[row * size + column] - factor * matrix[pivot * size + column];
            }
            rhs[row] = rhs[row] - factor * rhs[pivot];
        }
    }
    let mut solution = vec![Complex::default(); size];
    for row in (0..size).rev() {
        let remainder = (row + 1..size).fold(Complex::default(), |sum, column| {
            sum + matrix[row * size + column] * solution[column]
        });
        solution[row] = (rhs[row] - remainder) / matrix[row * size + row];
    }
    solution
}

fn normalize(direction: [f64; 3]) -> [f64; 3] {
    let norm = direction
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    direction.map(|value| value / norm)
}

fn influence_vector(summary: &AssemblySummary, direction: [f64; 3]) -> Vec<f64> {
    (0..summary.dof_count)
        .map(|row| {
            if summary.operator.constrained_dofs()[row] {
                return 0.0;
            }
            match summary
                .structural_dof_layout
                .address(row)
                .map(|address| address.kind)
            {
                Some(StructuralDofKind::Ux) => direction[0],
                Some(StructuralDofKind::Uy) => direction[1],
                Some(StructuralDofKind::Uz) => direction[2],
                _ => 0.0,
            }
        })
        .collect()
}

pub(super) fn maximum_relative_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .filter(|(_, expected)| **expected > 1.0e-24)
        .map(|(actual, expected)| (actual - expected).abs() / expected)
        .fold(0.0, f64::max)
}
