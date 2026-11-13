//! Canonical decomposition ingress and errors, retaining original coefficients.
use super::*;
use crate::builtins::math::linalg::ops::sparse_factor::{
    self, ManagedSparseCholeskyError, ManagedSparseLuError, NativeMatrixFacts, RetainedFactorError,
    SparseCompleteQrError,
};
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use runmat_value::NumericScalar;

#[path = "native/errors.rs"]
mod errors;
pub(super) use errors::{
    allocation, dense_error, publish_value, retained_error, selected_runtime, terminal,
};

#[path = "native/ingress.rs"]
mod ingress;
pub(super) use ingress::{
    checked_double, coefficient, exact_integer, normalize_selected, prepare_factor_matrix,
};

#[path = "native/inspection.rs"]
mod inspection;
pub(super) use inspection::{check_factor_recipe, inspect_original, selected, selected_real};
