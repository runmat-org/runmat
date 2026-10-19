//! Typed geometric, harmonic and root-mean-square reductions.
mod data;
mod host;
pub(super) mod numerics;
mod options;
mod resident;

pub(crate) mod geomean;
pub(crate) mod harmmean;
pub(crate) mod rms;
pub(crate) mod rmse;

#[cfg(test)]
mod tests;

use options::{invalid, ReductionKind};
use runmat_value::Value;

fn admit_data(kind: ReductionKind, value: &Value) -> crate::BuiltinResult<()> {
    let admitted = match value {
        Value::Num(_) | Value::Int(_) | Value::Tensor(_) => true,
        Value::GpuTensor(handle) => {
            let real = runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::Real;
            let logical = runmat_accelerate_api::handle_is_logical(handle);
            let numeric = runmat_accelerate_api::handle_precision(handle).is_some()
                || runmat_accelerate_api::handle_integer_type(handle).is_some();
            numeric
                && match kind {
                    ReductionKind::Geometric | ReductionKind::Harmonic => real && !logical,
                    ReductionKind::RootMeanSquare => true,
                    ReductionKind::RootMeanSquaredError => !logical,
                }
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            matches!(
                kind,
                ReductionKind::RootMeanSquare | ReductionKind::RootMeanSquaredError
            )
        }
        Value::Bool(_) | Value::LogicalArray(_) | Value::CharArray(_) => {
            kind == ReductionKind::RootMeanSquare
        }
        _ => false,
    };
    admitted
        .then_some(())
        .ok_or_else(|| invalid(kind, "unsupported input class"))
}
