//! Shared typed metric admission and host computation for pairwise-distance consumers.

mod compute;
mod geometry;
mod numeric;
mod options;

use std::fmt;

use runmat_builtins::{
    BuiltinErrorDescriptor, DISTANCE_ERROR_INTERNAL, DISTANCE_ERROR_INVALID_ARGUMENT,
    LINKAGE_ERROR_INTERNAL, LINKAGE_ERROR_INVALID_ARGUMENT,
};

use crate::{build_runtime_error, RuntimeError};

pub(super) use compute::{
    condensed_distances, distance_matrix, observation_matrix_dimensions, row_distance, row_value,
};
pub(super) use numeric::{DirectionError, DirectionProgress, DirectionWork};
pub(super) use options::{parse_metric_kind, parse_metric_name, scalar_parameter};

const EPS: f64 = 1.0e-12;

#[derive(Clone, Debug)]
pub(super) enum DistanceMetric {
    Euclidean,
    SquareEuclidean,
    Cityblock,
    Chebychev,
    Minkowski(f64),
    Seuclidean(Vec<f64>),
    Mahalanobis(Vec<f64>),
    Cosine,
    Correlation,
    Hamming,
    Jaccard,
    Spearman,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum MetricCaller {
    Pdist,
    Pdist2,
    Knnsearch,
    Linkage,
}

impl MetricCaller {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Pdist => "pdist",
            Self::Pdist2 => "pdist2",
            Self::Knnsearch => "knnsearch",
            Self::Linkage => "linkage",
        }
    }

    const fn invalid_descriptor(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Pdist | Self::Pdist2 | Self::Knnsearch => &DISTANCE_ERROR_INVALID_ARGUMENT,
            Self::Linkage => &LINKAGE_ERROR_INVALID_ARGUMENT,
        }
    }

    const fn internal_descriptor(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Pdist | Self::Pdist2 | Self::Knnsearch => &DISTANCE_ERROR_INTERNAL,
            Self::Linkage => &LINKAGE_ERROR_INTERNAL,
        }
    }
}

impl fmt::Display for MetricCaller {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

fn metric_error(
    caller: MetricCaller,
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(caller.name());
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn distance_error(caller: MetricCaller, message: impl Into<String>) -> RuntimeError {
    metric_error(caller, message, caller.invalid_descriptor())
}

fn internal_error(caller: MetricCaller, message: impl Into<String>) -> RuntimeError {
    metric_error(caller, message, caller.internal_descriptor())
}
