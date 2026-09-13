use thiserror::Error;

use super::{dense::solve_square, ModalReducedSystem};

const BETA: f64 = 0.25;
const GAMMA: f64 = 0.5;

#[derive(Clone, Debug, PartialEq)]
pub struct ModalTransientInitialState {
    pub displacement: Vec<f64>,
    pub velocity: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModalTransientHistory {
    pub time_points_s: Vec<f64>,
    pub displacement_snapshots: Vec<Vec<f64>>,
    pub velocity_snapshots: Vec<Vec<f64>>,
    pub acceleration_snapshots: Vec<Vec<f64>>,
    pub relative_residuals: Vec<f64>,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum ModalTransientError {
    #[error("modal transient inputs have inconsistent dimensions")]
    DimensionMismatch,
    #[error("modal transient time points must be finite and strictly increasing")]
    InvalidTimeDomain,
    #[error("modal transient inputs contain a nonfinite value")]
    NonFinite,
    #[error("modal transient effective system is singular")]
    Singular,
    #[error("modal transient solve was cancelled")]
    Cancelled,
}

pub fn solve_modal_transient(
    system: &ModalReducedSystem,
    time_points_s: &[f64],
    projected_loads: &[Vec<f64>],
    initial: ModalTransientInitialState,
) -> Result<ModalTransientHistory, ModalTransientError> {
    integrate(
        DenseSecondOrderOperators {
            size: system.mode_count(),
            stiffness: system.stiffness(),
            mass: system.mass(),
            damping: system.damping(),
        },
        time_points_s,
        projected_loads,
        initial,
    )
}

#[derive(Clone, Copy)]
struct DenseSecondOrderOperators<'a> {
    size: usize,
    stiffness: &'a [f64],
    mass: &'a [f64],
    damping: &'a [f64],
}

fn integrate(
    operators: DenseSecondOrderOperators<'_>,
    time_points_s: &[f64],
    loads: &[Vec<f64>],
    initial: ModalTransientInitialState,
) -> Result<ModalTransientHistory, ModalTransientError> {
    validate_inputs(operators, time_points_s, loads, &initial)?;
    let size = operators.size;
    let mut displacement = initial.displacement;
    let mut velocity = initial.velocity;
    let initial_right_hand_side = subtract_three(
        &loads[0],
        &multiply(operators.damping, size, &velocity),
        &multiply(operators.stiffness, size, &displacement),
    );
    let mut acceleration = solve_square(operators.mass, &initial_right_hand_side, size)
        .map_err(|_| ModalTransientError::Singular)?;
    let mut history = ModalTransientHistory {
        time_points_s: time_points_s.to_vec(),
        displacement_snapshots: vec![displacement.clone()],
        velocity_snapshots: vec![velocity.clone()],
        acceleration_snapshots: vec![acceleration.clone()],
        relative_residuals: vec![relative_residual(
            operators,
            &displacement,
            &velocity,
            &acceleration,
            &loads[0],
        )],
    };

    for index in 1..time_points_s.len() {
        if crate::is_fea_cancelled() {
            return Err(ModalTransientError::Cancelled);
        }
        let time_step_s = time_points_s[index] - time_points_s[index - 1];
        let predicted_displacement = (0..size)
            .map(|row| {
                displacement[row]
                    + time_step_s * velocity[row]
                    + time_step_s.powi(2) * (0.5 - BETA) * acceleration[row]
            })
            .collect::<Vec<_>>();
        let predicted_velocity = (0..size)
            .map(|row| velocity[row] + time_step_s * (1.0 - GAMMA) * acceleration[row])
            .collect::<Vec<_>>();
        let right_hand_side = subtract_three(
            &loads[index],
            &multiply(operators.damping, size, &predicted_velocity),
            &multiply(operators.stiffness, size, &predicted_displacement),
        );
        let effective = linear_combination(
            operators.mass,
            GAMMA * time_step_s,
            operators.damping,
            BETA * time_step_s.powi(2),
            operators.stiffness,
        );
        acceleration = solve_square(&effective, &right_hand_side, size)
            .map_err(|_| ModalTransientError::Singular)?;
        displacement = predicted_displacement
            .iter()
            .zip(&acceleration)
            .map(|(predicted, acceleration)| predicted + BETA * time_step_s.powi(2) * acceleration)
            .collect();
        velocity = predicted_velocity
            .iter()
            .zip(&acceleration)
            .map(|(predicted, acceleration)| predicted + GAMMA * time_step_s * acceleration)
            .collect();
        history.displacement_snapshots.push(displacement.clone());
        history.velocity_snapshots.push(velocity.clone());
        history.acceleration_snapshots.push(acceleration.clone());
        history.relative_residuals.push(relative_residual(
            operators,
            &displacement,
            &velocity,
            &acceleration,
            &loads[index],
        ));
    }
    Ok(history)
}

fn validate_inputs(
    operators: DenseSecondOrderOperators<'_>,
    time_points_s: &[f64],
    loads: &[Vec<f64>],
    initial: &ModalTransientInitialState,
) -> Result<(), ModalTransientError> {
    let matrix_size = operators
        .size
        .checked_mul(operators.size)
        .ok_or(ModalTransientError::DimensionMismatch)?;
    if operators.size == 0
        || operators.stiffness.len() != matrix_size
        || operators.mass.len() != matrix_size
        || operators.damping.len() != matrix_size
        || time_points_s.len() < 2
        || loads.len() != time_points_s.len()
        || loads.iter().any(|load| load.len() != operators.size)
        || initial.displacement.len() != operators.size
        || initial.velocity.len() != operators.size
    {
        return Err(ModalTransientError::DimensionMismatch);
    }
    if time_points_s.iter().any(|time| !time.is_finite())
        || time_points_s.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(ModalTransientError::InvalidTimeDomain);
    }
    if operators
        .stiffness
        .iter()
        .chain(operators.mass)
        .chain(operators.damping)
        .chain(loads.iter().flatten())
        .chain(&initial.displacement)
        .chain(&initial.velocity)
        .any(|value| !value.is_finite())
    {
        return Err(ModalTransientError::NonFinite);
    }
    Ok(())
}

fn multiply(matrix: &[f64], size: usize, vector: &[f64]) -> Vec<f64> {
    (0..size)
        .map(|row| {
            (0..size)
                .map(|column| matrix[row * size + column] * vector[column])
                .sum()
        })
        .collect()
}

fn subtract_three(left: &[f64], middle: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter()
        .zip(middle)
        .zip(right)
        .map(|((left, middle), right)| left - middle - right)
        .collect()
}

fn linear_combination(
    mass: &[f64],
    damping_scale: f64,
    damping: &[f64],
    stiffness_scale: f64,
    stiffness: &[f64],
) -> Vec<f64> {
    mass.iter()
        .zip(damping)
        .zip(stiffness)
        .map(|((mass, damping), stiffness)| {
            mass + damping_scale * damping + stiffness_scale * stiffness
        })
        .collect()
}

fn relative_residual(
    operators: DenseSecondOrderOperators<'_>,
    displacement: &[f64],
    velocity: &[f64],
    acceleration: &[f64],
    load: &[f64],
) -> f64 {
    let inertia = multiply(operators.mass, operators.size, acceleration);
    let damping = multiply(operators.damping, operators.size, velocity);
    let elastic = multiply(operators.stiffness, operators.size, displacement);
    let residual = load
        .iter()
        .zip(&inertia)
        .zip(&damping)
        .zip(&elastic)
        .map(|(((load, inertia), damping), elastic)| load - inertia - damping - elastic)
        .collect::<Vec<_>>();
    norm(&residual)
        / (norm(load) + norm(&inertia) + norm(&damping) + norm(&elastic)).max(f64::MIN_POSITIVE)
}

fn norm(vector: &[f64]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn free_oscillator(interval_count: usize) -> ModalTransientHistory {
        let omega = 2.0;
        let operators = DenseSecondOrderOperators {
            size: 1,
            stiffness: &[omega * omega],
            mass: &[1.0],
            damping: &[0.0],
        };
        let times = (0..=interval_count)
            .map(|index| index as f64 / interval_count as f64)
            .collect::<Vec<_>>();
        integrate(
            operators,
            &times,
            &vec![vec![0.0]; times.len()],
            ModalTransientInitialState {
                displacement: vec![1.0],
                velocity: vec![0.0],
            },
        )
        .unwrap()
    }

    #[test]
    fn average_acceleration_modal_integration_is_second_order() {
        let exact = 2.0_f64.cos();
        let error = |interval_count| {
            let history = free_oscillator(interval_count);
            (history.displacement_snapshots.last().unwrap()[0] - exact).abs()
        };
        let coarse = error(10);
        let medium = error(20);
        let fine = error(40);
        assert!(coarse / medium > 3.8);
        assert!(medium / fine > 3.8);
        assert!(free_oscillator(20)
            .relative_residuals
            .iter()
            .all(|residual| *residual <= 2.0e-15));
    }

    #[test]
    fn coupled_modal_damping_is_retained_in_the_effective_system() {
        let operators = DenseSecondOrderOperators {
            size: 2,
            stiffness: &[4.0, 0.0, 0.0, 9.0],
            mass: &[1.0, 0.0, 0.0, 1.0],
            damping: &[0.2, 0.05, 0.05, 0.3],
        };
        let history = integrate(
            operators,
            &[0.0, 0.1, 0.2],
            &[vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0]],
            ModalTransientInitialState {
                displacement: vec![0.0; 2],
                velocity: vec![0.0; 2],
            },
        )
        .unwrap();
        assert_ne!(history.displacement_snapshots[1][1], 0.0);
        assert!(history
            .relative_residuals
            .iter()
            .all(|residual| *residual <= 2.0e-15));
    }
}
