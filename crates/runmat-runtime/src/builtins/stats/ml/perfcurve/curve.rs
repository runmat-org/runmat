use super::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum ThresholdSpec {
    AboveAll(f64),
    Value(f64),
}

#[derive(Clone, Debug)]
pub(super) struct Options {
    pub(super) xcrit: Criterion,
    pub(super) ycrit: Criterion,
    pub(super) neg_class: Option<Vec<Label>>,
    pub(super) weights: Option<Vec<f64>>,
    pub(super) prior: PriorOption,
    pub(super) cost: [[f64; 2]; 2],
    pub(super) tvals: Option<Vec<f64>>,
    pub(super) xvals: Option<Vec<f64>>,
    pub(super) use_nearest: bool,
    pub(super) process_nan: ProcessNaN,
}

#[derive(Clone, Debug)]
pub(super) enum PriorOption {
    Empirical,
    Uniform,
    Values(f64, f64),
}

#[derive(Clone, Debug)]
pub(super) struct Observation {
    pub(super) score: f64,
    pub(super) weight: f64,
    pub(super) is_positive: bool,
    pub(super) neg_index: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Counts {
    pub(super) tp: f64,
    pub(super) fp: f64,
    pub(super) tn: f64,
    pub(super) fn_: f64,
}

impl Counts {
    pub(super) fn total(self) -> f64 {
        self.tp + self.fp + self.tn + self.fn_
    }
}

#[derive(Clone, Debug)]
pub(super) struct Curve {
    pub(super) x: Vec<f64>,
    pub(super) y: Vec<f64>,
    pub(super) thresholds: Vec<f64>,
    pub(super) counts: Vec<Counts>,
    pub(super) auc: f64,
}

mod core;
mod interpolation;
mod numerics;
mod observations;
mod subclasses;
#[cfg(test)]
mod tests;

pub(super) use core::{compute_curve, thresholds_for_scores};
use core::{counts_for_thresholds, criterion_value, threshold_output_value};
pub(super) use interpolation::{apply_xvals, auc, auc_interval, optimal_point};
use interpolation::{interpolate_at_x, nearest_at_x};
pub(super) use observations::prepare_observations;
pub(super) use subclasses::suby_output;
