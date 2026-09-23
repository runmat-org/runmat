use thiserror::Error;

use crate::execution::ModalTransientCoordinateState;

#[derive(Clone, Debug, PartialEq)]
pub struct ModalTransientInitialState {
    pub displacement: Vec<f64>,
    pub velocity: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModalTransientSample {
    pub time_s: f64,
    pub coordinates: ModalTransientCoordinateState,
    pub relative_residual: f64,
}

/// One accepted average-acceleration update in modal coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct ModalTransientStep {
    pub coordinates: ModalTransientCoordinateState,
    pub relative_residual: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModalTransientHistory {
    pub samples: Vec<ModalTransientSample>,
}

impl ModalTransientHistory {
    pub fn terminal_coordinates(&self) -> &ModalTransientCoordinateState {
        &self
            .samples
            .last()
            .expect("a validated modal transient history is nonempty")
            .coordinates
    }
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
