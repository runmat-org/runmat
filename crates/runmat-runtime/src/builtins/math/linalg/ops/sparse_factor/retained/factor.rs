//! Native factor selection under the actual construction owner.
use super::super::managed_storage::Work;
use super::super::{ManagedSparseCholesky, PublicSparseLu, SparseCompleteQr, SparseLuOptions};
use super::{input, validate_data, RetainedFactorError, RetainedFactorScalar};
use runmat_value::{ComplexElement, NumericFactorCandidate, NumericFactorData, SparseTensor};
pub(in crate::builtins::math::linalg) async fn factor_sparse(
    sparse: &SparseTensor,
    kind: runmat_value::NumericFactorKind,
    rank_tolerance: Option<f64>,
    lower: bool,
    runtime: &crate::context::RuntimeContext,
) -> Result<NumericFactorData, RetainedFactorError> {
    if sparse.as_f32_slice().is_some() {
        factor_typed::<f32>(sparse, kind, rank_tolerance, lower, runtime).await
    } else if sparse.as_f64_slice().is_some() {
        factor_typed::<f64>(sparse, kind, rank_tolerance, lower, runtime).await
    } else if sparse.as_complex_f32_slice().is_some() {
        factor_typed::<ComplexElement<f32>>(sparse, kind, rank_tolerance, lower, runtime).await
    } else if sparse.as_complex_f64_slice().is_some() {
        factor_typed::<ComplexElement<f64>>(sparse, kind, rank_tolerance, lower, runtime).await
    } else {
        Err(RetainedFactorError::UnsupportedClass)
    }
}

async fn factor_typed<T: RetainedFactorScalar>(
    sparse: &SparseTensor,
    kind: runmat_value::NumericFactorKind,
    tolerance: Option<f64>,
    lower: bool,
    runtime: &crate::context::RuntimeContext,
) -> Result<NumericFactorData, RetainedFactorError> {
    use runmat_value::NumericFactorKind;
    if sparse.rows == 0 || sparse.cols == 0 {
        use crate::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
        let receipt = HostAllocationReceipt::admit(Some(runtime), HostAllocationPlan::new(0, 0)?)?;
        let mut work = Work::for_policy(runtime, receipt.as_ref())?;
        input::validate_sparse(sparse, &mut work).await?;
        return validate_data(
            NumericFactorCandidate::Empty {
                kind,
                rows: sparse.rows,
                columns: sparse.cols,
                class: T::storage_class(),
            },
            &mut work,
        )
        .await;
    }
    match kind {
        NumericFactorKind::Lu => {
            PublicSparseLu::<T>::factor(
                sparse,
                SparseLuOptions {
                    column_ordering: true,
                    row_scaling: false,
                    unpermute_lower: false,
                    threshold: [0.1, 0.001],
                },
                runtime,
            )
            .await?
            .into_factor_data(runtime)
            .await
        }
        NumericFactorKind::Cholesky => {
            ManagedSparseCholesky::<T>::factor(sparse, true, lower, runtime)
                .await?
                .into_factor_data(runtime)
                .await
        }
        NumericFactorKind::Qr | NumericFactorKind::Cod => {
            let complete = kind == NumericFactorKind::Cod;
            SparseCompleteQr::<T>::factor_for_decomposition(sparse, tolerance, complete, runtime)
                .await?
                .into_factor_data(complete, runtime)
                .await
        }
    }
}
