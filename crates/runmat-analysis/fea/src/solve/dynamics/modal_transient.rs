use crate::execution::ModalTransientCoordinateState;

use super::{dense::solve_square, ModalReducedSystem};

mod algebra;
mod contract;

use algebra::{
    linear_combination, multiply, relative_residual, subtract_three, DenseSecondOrderOperators,
};
pub use contract::{
    ModalTransientError, ModalTransientHistory, ModalTransientInitialState, ModalTransientSample,
    ModalTransientStep,
};

const BETA: f64 = 0.25;
const GAMMA: f64 = 0.5;

fn dense_error(error: super::dense::DenseSolveError) -> ModalTransientError {
    match error {
        super::dense::DenseSolveError::Usage(error) => ModalTransientError::Usage(error),
        _ => ModalTransientError::Singular,
    }
}

pub fn solve_modal_transient(
    system: &ModalReducedSystem,
    time_points_s: &[f64],
    projected_loads: &[Vec<f64>],
    initial: ModalTransientInitialState,
) -> Result<ModalTransientHistory, ModalTransientError> {
    integrate_initial(
        DenseSecondOrderOperators {
            size: system.mode_count(),
            stiffness: system.stiffness().canonical_value(),
            mass: system.mass().canonical_value(),
            damping: system.damping().canonical_value(),
        },
        time_points_s,
        projected_loads,
        initial,
    )
}

/// Continue a modal transient from explicit generalized displacement, velocity, and acceleration.
///
/// The first returned sample is the supplied continuation boundary. Persisting all three vectors
/// avoids silently recomputing a different reduced state after a durable or remote restart.
pub fn continue_modal_transient(
    system: &ModalReducedSystem,
    time_points_s: &[f64],
    projected_loads: &[Vec<f64>],
    continuation: ModalTransientCoordinateState,
) -> Result<ModalTransientHistory, ModalTransientError> {
    integrate_continuation(
        DenseSecondOrderOperators {
            size: system.mode_count(),
            stiffness: system.stiffness().canonical_value(),
            mass: system.mass().canonical_value(),
            damping: system.damping().canonical_value(),
        },
        time_points_s,
        projected_loads,
        continuation,
    )
}

/// Establish the reduced acceleration that closes the governing equation at one sample.
pub fn initialize_modal_transient_state(
    system: &ModalReducedSystem,
    projected_load: &[f64],
    initial: ModalTransientInitialState,
) -> Result<ModalTransientStep, ModalTransientError> {
    let operators = operators(system);
    validate_operators_and_load(operators, projected_load)?;
    if initial.displacement.canonical_values().len() != operators.size
        || initial.velocity.canonical_values().len() != operators.size
        || initial
            .displacement
            .canonical_values()
            .iter()
            .chain(initial.velocity.canonical_values())
            .any(|value| !value.is_finite())
    {
        return Err(
            if initial.displacement.canonical_values().len() == operators.size
                && initial.velocity.canonical_values().len() == operators.size
            {
                ModalTransientError::NonFinite
            } else {
                ModalTransientError::DimensionMismatch
            },
        );
    }
    refresh_modal_transient_state(
        system,
        projected_load,
        initial.displacement.into_canonical_values(),
        initial.velocity.into_canonical_values(),
    )
}

/// Re-establish acceleration after an instantaneous load-side transition without changing q or qdot.
pub fn refresh_modal_transient_state(
    system: &ModalReducedSystem,
    projected_load: &[f64],
    displacement: Vec<f64>,
    velocity: Vec<f64>,
) -> Result<ModalTransientStep, ModalTransientError> {
    let operators = operators(system);
    validate_operators_and_load(operators, projected_load)?;
    if displacement.len() != operators.size
        || velocity.len() != operators.size
        || displacement
            .iter()
            .chain(&velocity)
            .any(|value| !value.is_finite())
    {
        return Err(
            if displacement.len() == operators.size && velocity.len() == operators.size {
                ModalTransientError::NonFinite
            } else {
                ModalTransientError::DimensionMismatch
            },
        );
    }
    let right_hand_side = subtract_three(
        projected_load,
        &multiply(operators.damping, operators.size, &velocity),
        &multiply(operators.stiffness, operators.size, &displacement),
    );
    let acceleration =
        solve_square(operators.mass, &right_hand_side, operators.size).map_err(dense_error)?;
    let coordinates = ModalTransientCoordinateState {
        displacement: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Displacement,
            _,
        >::new(displacement),
        velocity:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Velocity, _>::new(
                velocity,
            ),
        acceleration: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Acceleration,
            _,
        >::new(acceleration),
    };
    Ok(ModalTransientStep {
        relative_residual: relative_residual(
            operators,
            coordinates.displacement.canonical_values(),
            coordinates.velocity.canonical_values(),
            coordinates.acceleration.canonical_values(),
            projected_load,
        ),
        coordinates,
    })
}

/// Advance one strictly positive interval using the complete coupled reduced K/M/C system.
pub fn advance_modal_transient_state(
    system: &ModalReducedSystem,
    time_step_s: f64,
    projected_load: &[f64],
    continuation: ModalTransientCoordinateState,
) -> Result<ModalTransientStep, ModalTransientError> {
    let operators = operators(system);
    validate_operators_and_load(operators, projected_load)?;
    validate_coordinate_state(operators.size, &continuation)?;
    if !time_step_s.is_finite() || time_step_s <= 0.0 {
        return Err(ModalTransientError::InvalidTimeDomain);
    }
    if crate::is_fea_cancelled() {
        return Err(ModalTransientError::Cancelled);
    }
    advance_dense_modal_state(operators, time_step_s, projected_load, continuation)
}

fn operators(system: &ModalReducedSystem) -> DenseSecondOrderOperators<'_> {
    DenseSecondOrderOperators {
        size: system.mode_count(),
        stiffness: system.stiffness().canonical_value(),
        mass: system.mass().canonical_value(),
        damping: system.damping().canonical_value(),
    }
}

fn validate_operators_and_load(
    operators: DenseSecondOrderOperators<'_>,
    load: &[f64],
) -> Result<(), ModalTransientError> {
    let matrix_size = operators
        .size
        .checked_mul(operators.size)
        .ok_or(ModalTransientError::DimensionMismatch)?;
    if operators.size == 0
        || operators.stiffness.len() != matrix_size
        || operators.mass.len() != matrix_size
        || operators.damping.len() != matrix_size
        || load.len() != operators.size
    {
        return Err(ModalTransientError::DimensionMismatch);
    }
    if operators
        .stiffness
        .iter()
        .chain(operators.mass)
        .chain(operators.damping)
        .chain(load)
        .any(|value| !value.is_finite())
    {
        return Err(ModalTransientError::NonFinite);
    }
    Ok(())
}

fn integrate_initial(
    operators: DenseSecondOrderOperators<'_>,
    time_points_s: &[f64],
    loads: &[Vec<f64>],
    initial: ModalTransientInitialState,
) -> Result<ModalTransientHistory, ModalTransientError> {
    validate_common_inputs(operators, time_points_s, loads)?;
    if initial.displacement.canonical_values().len() != operators.size
        || initial.velocity.canonical_values().len() != operators.size
        || initial
            .displacement
            .canonical_values()
            .iter()
            .chain(initial.velocity.canonical_values())
            .any(|value| !value.is_finite())
    {
        return Err(
            if initial.displacement.canonical_values().len() == operators.size
                && initial.velocity.canonical_values().len() == operators.size
            {
                ModalTransientError::NonFinite
            } else {
                ModalTransientError::DimensionMismatch
            },
        );
    }
    let displacement = initial.displacement.into_canonical_values();
    let velocity = initial.velocity.into_canonical_values();
    let initial_right_hand_side = subtract_three(
        &loads[0],
        &multiply(operators.damping, operators.size, &velocity),
        &multiply(operators.stiffness, operators.size, &displacement),
    );
    let acceleration = solve_square(operators.mass, &initial_right_hand_side, operators.size)
        .map_err(dense_error)?;
    integrate_from_state(
        operators,
        time_points_s,
        loads,
        ModalTransientCoordinateState {
            displacement: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Displacement,
                _,
            >::new(displacement),
            velocity:
                runmat_types::physical::Quantity::<runmat_types::physical::kind::Velocity, _>::new(
                    velocity,
                ),
            acceleration: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Acceleration,
                _,
            >::new(acceleration),
        },
    )
}

fn integrate_continuation(
    operators: DenseSecondOrderOperators<'_>,
    time_points_s: &[f64],
    loads: &[Vec<f64>],
    continuation: ModalTransientCoordinateState,
) -> Result<ModalTransientHistory, ModalTransientError> {
    validate_common_inputs(operators, time_points_s, loads)?;
    validate_coordinate_state(operators.size, &continuation)?;
    integrate_from_state(operators, time_points_s, loads, continuation)
}

fn integrate_from_state(
    operators: DenseSecondOrderOperators<'_>,
    time_points_s: &[f64],
    loads: &[Vec<f64>],
    initial: ModalTransientCoordinateState,
) -> Result<ModalTransientHistory, ModalTransientError> {
    let mut coordinates = initial;
    let mut history = ModalTransientHistory {
        samples: vec![ModalTransientSample {
            time_s: runmat_types::physical::Time::new(time_points_s[0]),
            coordinates: coordinates.clone(),
            relative_residual: relative_residual(
                operators,
                coordinates.displacement.canonical_values(),
                coordinates.velocity.canonical_values(),
                coordinates.acceleration.canonical_values(),
                &loads[0],
            ),
        }],
    };

    for index in 1..time_points_s.len() {
        if crate::is_fea_cancelled() {
            return Err(ModalTransientError::Cancelled);
        }
        let step = advance_dense_modal_state(
            operators,
            time_points_s[index] - time_points_s[index - 1],
            &loads[index],
            coordinates,
        )?;
        coordinates = step.coordinates;
        history.samples.push(ModalTransientSample {
            time_s: runmat_types::physical::Time::new(time_points_s[index]),
            coordinates: coordinates.clone(),
            relative_residual: step.relative_residual,
        });
    }
    Ok(history)
}

fn advance_dense_modal_state(
    operators: DenseSecondOrderOperators<'_>,
    time_step_s: f64,
    projected_load: &[f64],
    continuation: ModalTransientCoordinateState,
) -> Result<ModalTransientStep, ModalTransientError> {
    let predicted_displacement = (0..operators.size)
        .map(|row| {
            continuation.displacement.canonical_values()[row]
                + time_step_s * continuation.velocity.canonical_values()[row]
                + time_step_s.powi(2)
                    * (0.5 - BETA)
                    * continuation.acceleration.canonical_values()[row]
        })
        .collect::<Vec<_>>();
    let predicted_velocity = (0..operators.size)
        .map(|row| {
            continuation.velocity.canonical_values()[row]
                + time_step_s * (1.0 - GAMMA) * continuation.acceleration.canonical_values()[row]
        })
        .collect::<Vec<_>>();
    let right_hand_side = subtract_three(
        projected_load,
        &multiply(operators.damping, operators.size, &predicted_velocity),
        &multiply(operators.stiffness, operators.size, &predicted_displacement),
    );
    let effective = linear_combination(
        operators.mass,
        GAMMA * time_step_s,
        operators.damping,
        BETA * time_step_s.powi(2),
        operators.stiffness,
    );
    let acceleration =
        solve_square(&effective, &right_hand_side, operators.size).map_err(dense_error)?;
    let displacement = predicted_displacement
        .iter()
        .zip(&acceleration)
        .map(|(predicted, acceleration)| predicted + BETA * time_step_s.powi(2) * acceleration)
        .collect::<Vec<_>>();
    let velocity = predicted_velocity
        .iter()
        .zip(&acceleration)
        .map(|(predicted, acceleration)| predicted + GAMMA * time_step_s * acceleration)
        .collect::<Vec<_>>();
    let coordinates = ModalTransientCoordinateState {
        displacement: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Displacement,
            _,
        >::new(displacement),
        velocity:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Velocity, _>::new(
                velocity,
            ),
        acceleration: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Acceleration,
            _,
        >::new(acceleration),
    };
    Ok(ModalTransientStep {
        relative_residual: relative_residual(
            operators,
            coordinates.displacement.canonical_values(),
            coordinates.velocity.canonical_values(),
            coordinates.acceleration.canonical_values(),
            projected_load,
        ),
        coordinates,
    })
}

fn validate_common_inputs(
    operators: DenseSecondOrderOperators<'_>,
    time_points_s: &[f64],
    loads: &[Vec<f64>],
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
        .any(|value| !value.is_finite())
    {
        return Err(ModalTransientError::NonFinite);
    }
    Ok(())
}

fn validate_coordinate_state(
    size: usize,
    state: &ModalTransientCoordinateState,
) -> Result<(), ModalTransientError> {
    if state.displacement.canonical_values().len() != size
        || state.velocity.canonical_values().len() != size
        || state.acceleration.canonical_values().len() != size
    {
        return Err(ModalTransientError::DimensionMismatch);
    }
    if state
        .displacement
        .canonical_values()
        .iter()
        .chain(state.velocity.canonical_values())
        .chain(state.acceleration.canonical_values())
        .any(|value| !value.is_finite())
    {
        return Err(ModalTransientError::NonFinite);
    }
    Ok(())
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
        integrate_initial(
            operators,
            &times,
            &vec![vec![0.0]; times.len()],
            ModalTransientInitialState {
                displacement: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Displacement,
                    _,
                >::new(vec![1.0]),
                velocity: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Velocity,
                    _,
                >::new(vec![0.0]),
            },
        )
        .unwrap()
    }

    // witness: FEA-DYNAMICS-RESPONSE-001/second_order_equilibrium_excitation @
    //   linear_structural_dynamics/mesh_independent/numerical_kernel; coverage: partial;
    //   check: free-vibration-time-integration; boundary: Average-acceleration modal integration is second order.
    #[test]
    fn average_acceleration_modal_integration_is_second_order() {
        let exact = 2.0_f64.cos();
        let error = |interval_count| {
            let history = free_oscillator(interval_count);
            (history
                .samples
                .last()
                .unwrap()
                .coordinates
                .displacement
                .canonical_values()[0]
                - exact)
                .abs()
        };
        let coarse = error(10);
        let medium = error(20);
        let fine = error(40);
        assert!(coarse / medium > 3.8);
        assert!(medium / fine > 3.8);
        assert!(free_oscillator(20)
            .samples
            .iter()
            .all(|sample| sample.relative_residual <= 2.0e-15));
    }

    #[test]
    fn coupled_modal_damping_is_retained_in_the_effective_system() {
        let operators = DenseSecondOrderOperators {
            size: 2,
            stiffness: &[4.0, 0.0, 0.0, 9.0],
            mass: &[1.0, 0.0, 0.0, 1.0],
            damping: &[0.2, 0.05, 0.05, 0.3],
        };
        let history = integrate_initial(
            operators,
            &[0.0, 0.1, 0.2],
            &[vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0]],
            ModalTransientInitialState {
                displacement: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Displacement,
                    _,
                >::new(vec![0.0; 2]),
                velocity: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Velocity,
                    _,
                >::new(vec![0.0; 2]),
            },
        )
        .unwrap();
        assert_ne!(
            history.samples[1]
                .coordinates
                .displacement
                .canonical_values()[1],
            0.0
        );
        assert!(history
            .samples
            .iter()
            .all(|sample| sample.relative_residual <= 2.0e-15));
    }

    #[test]
    fn explicit_modal_coordinates_restart_without_changing_the_solution() {
        let operators = DenseSecondOrderOperators {
            size: 2,
            stiffness: &[4.0, 0.0, 0.0, 9.0],
            mass: &[1.0, 0.0, 0.0, 1.0],
            damping: &[0.2, 0.05, 0.05, 0.3],
        };
        let times = [0.0, 0.05, 0.1, 0.15, 0.2];
        let loads = [
            vec![1.0, 0.0],
            vec![0.8, 0.1],
            vec![0.4, 0.2],
            vec![0.0, 0.1],
            vec![-0.2, 0.0],
        ];
        let initial = ModalTransientInitialState {
            displacement: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Displacement,
                _,
            >::new(vec![0.0; 2]),
            velocity:
                runmat_types::physical::Quantity::<runmat_types::physical::kind::Velocity, _>::new(
                    vec![0.0; 2],
                ),
        };
        let uninterrupted = integrate_initial(operators, &times, &loads, initial.clone()).unwrap();
        let first = integrate_initial(operators, &times[..=2], &loads[..=2], initial).unwrap();
        let second = integrate_continuation(
            operators,
            &times[2..],
            &loads[2..],
            first.terminal_coordinates().clone(),
        )
        .unwrap();
        let mut restarted = first.samples;
        restarted.extend(second.samples.into_iter().skip(1));

        assert_eq!(restarted, uninterrupted.samples);
    }

    #[test]
    fn continuation_rejects_missing_or_nonfinite_reduced_coordinates() {
        let operators = DenseSecondOrderOperators {
            size: 1,
            stiffness: &[4.0],
            mass: &[1.0],
            damping: &[0.0],
        };
        let invalid_shape = ModalTransientCoordinateState {
            displacement: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Displacement,
                _,
            >::new(vec![]),
            velocity:
                runmat_types::physical::Quantity::<runmat_types::physical::kind::Velocity, _>::new(
                    vec![0.0],
                ),
            acceleration: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Acceleration,
                _,
            >::new(vec![0.0]),
        };
        assert_eq!(
            integrate_continuation(
                operators,
                &[0.0, 0.1],
                &[vec![0.0], vec![0.0]],
                invalid_shape
            ),
            Err(ModalTransientError::DimensionMismatch)
        );
        let invalid_value = ModalTransientCoordinateState {
            displacement: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Displacement,
                _,
            >::new(vec![0.0]),
            velocity:
                runmat_types::physical::Quantity::<runmat_types::physical::kind::Velocity, _>::new(
                    vec![0.0],
                ),
            acceleration: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Acceleration,
                _,
            >::new(vec![f64::NAN]),
        };
        assert_eq!(
            integrate_continuation(
                operators,
                &[0.0, 0.1],
                &[vec![0.0], vec![0.0]],
                invalid_value
            ),
            Err(ModalTransientError::NonFinite)
        );
    }
}
