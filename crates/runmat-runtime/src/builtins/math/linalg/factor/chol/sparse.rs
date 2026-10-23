use super::*;
use crate::builtins::math::linalg::ops::sparse_factor::{
    ManagedCholeskyScalar, ManagedSparseCholesky, ManagedSparseCholeskyError, SparsePublication,
    SparsePublicationError,
};
use crate::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
use crate::context::RuntimeContext;
use runmat_value::{ExactHostSparseClass, ExactHostTensorAllocation, SparseTensor};

pub(super) async fn evaluate(
    sparse: SparseTensor,
    triangle: CholTriangle,
    output_form: SparsePermutationOutput,
    use_ordering: bool,
    publish_permutation_output: bool,
    runtime: &RuntimeContext,
) -> BuiltinResult<CholEval> {
    match sparse.numeric_dtype() {
        Some(NumericDType::F32) if sparse.as_complex_f32_slice().is_some() => {
            evaluate_typed::<faer::c32>(
                sparse,
                triangle,
                output_form,
                use_ordering,
                publish_permutation_output,
                runtime,
            )
            .await
        }
        Some(NumericDType::F64) if sparse.as_complex_f64_slice().is_some() => {
            evaluate_typed::<faer::c64>(
                sparse,
                triangle,
                output_form,
                use_ordering,
                publish_permutation_output,
                runtime,
            )
            .await
        }
        Some(NumericDType::F32) => {
            evaluate_typed::<f32>(
                sparse,
                triangle,
                output_form,
                use_ordering,
                publish_permutation_output,
                runtime,
            )
            .await
        }
        Some(NumericDType::F64) => {
            evaluate_typed::<f64>(
                sparse,
                triangle,
                output_form,
                use_ordering,
                publish_permutation_output,
                runtime,
            )
            .await
        }
        _ => Err(chol_invalid_input(
            "chol: sparse input must be single or double",
        )),
    }
}

async fn evaluate_typed<T: ManagedCholeskyScalar>(
    sparse: SparseTensor,
    triangle: CholTriangle,
    output_form: SparsePermutationOutput,
    use_ordering: bool,
    publish_permutation_output: bool,
    runtime: &RuntimeContext,
) -> BuiltinResult<CholEval> {
    let factor = ManagedSparseCholesky::<T>::factor(
        &sparse,
        use_ordering,
        matches!(triangle, CholTriangle::Lower),
        runtime,
    )
    .await
    .map_err(|error| factor_error(error, runtime))?;
    let factor_value = publish_factor(&factor, &sparse, triangle, runtime).await?;
    let permutation = if publish_permutation_output {
        Some(publish_permutation(&factor, output_form, runtime).await?)
    } else {
        None
    };
    Ok(CholEval {
        factor: factor_value,
        flag: if factor.completed_columns() == factor.order() {
            0
        } else {
            factor.completed_columns() + 1
        },
        triangle,
        permutation,
    })
}

pub(in crate::builtins::math::linalg) fn factor_error(
    error: ManagedSparseCholeskyError,
    runtime: &RuntimeContext,
) -> RuntimeError {
    match error {
        ManagedSparseCholeskyError::Admission(source) => {
            crate::context::host_allocation::host_allocation_runtime_error(
                Some(runtime),
                "RunMat:chol:Internal",
                "chol: sparse factor allocation",
                source,
            )
        }
        ManagedSparseCholeskyError::Execution(source) => {
            source.into_runtime_error_in_context(Some(runtime))
        }
        other => chol_internal_error(other.to_string()),
    }
}

fn sparse_class(source: &SparseTensor) -> BuiltinResult<ExactHostSparseClass> {
    Ok(if source.as_complex_f32_slice().is_some() {
        ExactHostSparseClass::Complex(NumericDType::F32)
    } else if source.as_complex_f64_slice().is_some() {
        ExactHostSparseClass::Complex(NumericDType::F64)
    } else {
        ExactHostSparseClass::Numeric(
            source
                .numeric_dtype()
                .ok_or_else(|| chol_invalid_input("chol: sparse input must be single or double"))?,
        )
    })
}

pub(in crate::builtins::math::linalg) async fn publish_factor<T: ManagedCholeskyScalar>(
    factor: &ManagedSparseCholesky<T>,
    source: &SparseTensor,
    triangle: CholTriangle,
    runtime: &RuntimeContext,
) -> BuiltinResult<Value> {
    factor.check_owner(runtime).map_err(|e| {
        crate::context::host_allocation::host_allocation_runtime_error(
            Some(runtime),
            "RunMat:chol:Internal",
            "chol: sparse factor owner",
            e,
        )
    })?;
    let n = factor.order();
    let q = factor.completed_columns();
    let nnz = factor.lower_values().len();
    let (rows, cols) = if matches!(triangle, CholTriangle::Lower) {
        (n, q)
    } else {
        (q, n)
    };
    let class = sparse_class(source)?;
    let mut output = SparsePublication::new(runtime, class, rows, cols, nnz)
        .map_err(|e| publication_error(e, runtime))?;
    if matches!(triangle, CholTriangle::Lower) {
        let mut written = 0usize;
        output
            .pointer(0, 0)
            .map_err(|e| publication_error(e, runtime))?;
        for column in 0..cols {
            for at in factor.lower_col_ptrs()[column]..factor.lower_col_ptrs()[column + 1] {
                output
                    .entry(
                        written,
                        factor.lower_rows()[at],
                        factor.lower_values()[at].parts(),
                    )
                    .map_err(|e| publication_error(e, runtime))?;
                written += 1;
            }
            output
                .pointer(column + 1, written)
                .map_err(|e| publication_error(e, runtime))?;
            output
                .checkpoint(runtime)
                .map_err(|e| publication_error(e, runtime))?;
        }
    } else {
        let projection = factor
            .upper_projection(runtime)
            .await
            .map_err(|error| factor_error(error, runtime))?;
        for (column, &pointer) in projection.col_ptrs().iter().enumerate() {
            output
                .pointer(column, pointer)
                .map_err(|e| publication_error(e, runtime))?;
        }
        for index in 0..projection.values().len() {
            output
                .entry(
                    index,
                    projection.rows()[index],
                    projection.values()[index].parts(),
                )
                .map_err(|e| publication_error(e, runtime))?;
            if index % 256 == 255 {
                output
                    .checkpoint(runtime)
                    .map_err(|e| publication_error(e, runtime))?;
            }
        }
        output
            .checkpoint(runtime)
            .map_err(|e| publication_error(e, runtime))?;
    }
    output
        .finish(runtime)
        .await
        .map_err(|e| publication_error(e, runtime))
}

fn publication_error(error: SparsePublicationError, runtime: &RuntimeContext) -> RuntimeError {
    match error {
        SparsePublicationError::Admission(source) => {
            crate::context::host_allocation::host_allocation_runtime_error(
                Some(runtime),
                "RunMat:chol:Internal",
                "chol: sparse output allocation",
                source,
            )
        }
        SparsePublicationError::Execution(source) => {
            source.into_runtime_error_in_context(Some(runtime))
        }
        other => chol_internal_error(other.to_string()),
    }
}

async fn publish_permutation<T: ManagedCholeskyScalar>(
    factor: &ManagedSparseCholesky<T>,
    form: SparsePermutationOutput,
    runtime: &RuntimeContext,
) -> BuiltinResult<Value> {
    let n = factor.order();
    match form {
        SparsePermutationOutput::Vector => {
            let shape = [1, n];
            let retained = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &shape)
                .map_err(|e| chol_internal_error(e.to_string()))?;
            let receipt = HostAllocationReceipt::admit(
                Some(runtime),
                HostAllocationPlan::new(0, retained)
                    .map_err(|e| chol_internal_error(e.to_string()))?,
            )
            .map_err(|e| {
                crate::context::host_allocation::host_allocation_runtime_error(
                    Some(runtime),
                    "RunMat:chol:Internal",
                    "chol: permutation allocation",
                    e,
                )
            })?;
            if let Some(receipt) = receipt {
                let mut allocation = receipt
                    .with_output_allocation(runtime, |permit| {
                        ExactHostTensorAllocation::new(NumericDType::F64, &shape, permit)
                    })
                    .map_err(|e| chol_internal_error(e.to_string()))?;
                for (slot, &original) in allocation
                    .as_f64_slice_mut()
                    .unwrap()
                    .iter_mut()
                    .zip(factor.permutation())
                {
                    *slot = (original + 1) as f64;
                }
                return allocation
                    .into_tensor()
                    .map(Value::Tensor)
                    .map_err(|e| chol_internal_error(e.to_string()));
            }
            Tensor::new(
                factor
                    .permutation()
                    .iter()
                    .map(|&v| (v + 1) as f64)
                    .collect(),
                shape.to_vec(),
            )
            .map(Value::Tensor)
            .map_err(chol_internal_error)
        }
        SparsePermutationOutput::Matrix => {
            let class = ExactHostSparseClass::Numeric(NumericDType::F64);
            let mut output = SparsePublication::new(runtime, class, n, n, n)
                .map_err(|e| publication_error(e, runtime))?;
            for column in 0..=n {
                output
                    .pointer(column, column)
                    .map_err(|e| publication_error(e, runtime))?;
            }
            for (column, &original) in factor.permutation().iter().enumerate() {
                output
                    .entry(column, original, (1.0, 0.0))
                    .map_err(|e| publication_error(e, runtime))?;
            }
            output
                .finish(runtime)
                .await
                .map_err(|e| publication_error(e, runtime))
        }
    }
}
