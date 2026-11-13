//! Probability distribution compatibility helpers.

use runmat_accelerate_api::{GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{BuiltinExtensionDescriptor, DistributionKind, DistributionTail};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, NumericScalar, Tensor, Value};

use crate::builtins::common::{broadcast, gpu_helpers, tensor};
use crate::builtins::math::elementwise::error_functions::erfcinv::erfcinv_scalar;
use crate::builtins::stats::summary::distribution_math;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

mod admission;
mod geometry;
mod numerics;
mod preflight;

use admission::*;
use geometry::*;
use numerics::*;
use preflight::{numeric_inputs, NumericShapePolicy};
pub mod normpdf;

pub mod normcdf;

pub mod norminv;

pub mod tpdf;

pub mod tcdf;

pub mod tinv;

pub mod binocdf;

pub mod chi2cdf;

pub mod wblinv;

pub mod icdf;

#[cfg(test)]
mod tests;
