use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SecondOrderState {
    displacement: Vec<f64>,
    velocity: Vec<f64>,
    acceleration: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub(crate) enum SecondOrderStateError {
    #[error("a structural dynamics state must contain at least one degree of freedom")]
    Empty,
    #[error("structural dynamics state vectors must have identical dimensions")]
    DimensionMismatch,
    #[error("structural dynamics state contains a nonfinite value")]
    NonFinite,
}

impl SecondOrderState {
    pub(crate) fn try_new(
        displacement: Vec<f64>,
        velocity: Vec<f64>,
        acceleration: Vec<f64>,
    ) -> Result<Self, SecondOrderStateError> {
        if displacement.is_empty() {
            return Err(SecondOrderStateError::Empty);
        }
        if velocity.len() != displacement.len() || acceleration.len() != displacement.len() {
            return Err(SecondOrderStateError::DimensionMismatch);
        }
        if displacement
            .iter()
            .chain(&velocity)
            .chain(&acceleration)
            .any(|value| !value.is_finite())
        {
            return Err(SecondOrderStateError::NonFinite);
        }
        Ok(Self {
            displacement,
            velocity,
            acceleration,
        })
    }

    pub(crate) fn dof_count(&self) -> usize {
        self.displacement.len()
    }

    pub(crate) fn displacement(&self) -> &[f64] {
        &self.displacement
    }

    pub(crate) fn velocity(&self) -> &[f64] {
        &self.velocity
    }

    pub(crate) fn acceleration(&self) -> &[f64] {
        &self.acceleration
    }

    pub(super) fn into_parts(self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        (self.displacement, self.velocity, self.acceleration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_rejects_empty_mismatched_and_nonfinite_vectors() {
        assert_eq!(
            SecondOrderState::try_new(Vec::new(), Vec::new(), Vec::new()),
            Err(SecondOrderStateError::Empty)
        );
        assert_eq!(
            SecondOrderState::try_new(vec![0.0], vec![0.0, 0.0], vec![0.0]),
            Err(SecondOrderStateError::DimensionMismatch)
        );
        assert_eq!(
            SecondOrderState::try_new(vec![0.0], vec![f64::NAN], vec![0.0]),
            Err(SecondOrderStateError::NonFinite)
        );
    }

    #[test]
    fn state_preserves_each_physical_coordinate_vector() {
        let state =
            SecondOrderState::try_new(vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]).unwrap();
        assert_eq!(state.dof_count(), 2);
        assert_eq!(state.displacement(), &[1.0, 2.0]);
        assert_eq!(state.velocity(), &[3.0, 4.0]);
        assert_eq!(state.acceleration(), &[5.0, 6.0]);
    }
}
