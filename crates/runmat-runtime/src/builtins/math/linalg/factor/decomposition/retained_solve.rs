//! Public decomposition solves borrow retained factors and original RHS owners.
use super::*;
use crate::builtins::math::linalg::ops::{
    dense_factor::{self, DenseScalar},
    linear_operator::{CooperativeOperatorError, OperatorWork},
    sparse_factor::{
        self, FactorRhsGeometry, FactorSolveOptions, RetainedSolveStatus, SingularSolvePolicy,
    },
};
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use runmat_value::{ExactHostSparseClass, NativeDenseFactorMatrix, NumericFactorCandidate};
#[path = "retained_solve/input.rs"]
mod input;
#[path = "retained_solve/resident.rs"]
mod resident;
#[path = "retained_solve/structural.rs"]
mod structural;

fn work_error(error: CooperativeOperatorError, runtime: &RuntimeContext) -> RuntimeError {
    use crate::builtins::math::linalg::ops::sparse_operator::SparseOperatorError;
    native::terminal(match error {
        CooperativeOperatorError::Operator(SparseOperatorError::Admission(e)) => {
            native::allocation(e, runtime)
        }
        CooperativeOperatorError::Execution(e) => e.into_runtime_error_in_context(Some(runtime)),
        CooperativeOperatorError::Operator(e) => {
            invalid(format!("decomposition: numerical work failed: {e:?}"))
        }
    })
}

pub(super) async fn evaluate(
    spec: &DecompositionSpec,
    operand: &Value,
    right: bool,
    runtime: &RuntimeContext,
) -> BuiltinResult<Value> {
    if spec.decomp_type == DecompositionType::Qr && (right || spec.is_conjugate_transposed) {
        return Err(invalid("decomposition: QR supports only normal left solves; use COD for transpose or right solves"));
    }
    ensure_gpu_operand_enabled(operand)?;
    let (m, n) = matrix_dims(&spec.matrix);
    if !is_nonfloating_coefficient_matrix(&spec.matrix) && is_integer_value(operand) {
        if crate::builtins::common::validation::is_typed_complex_integer(operand) {
            return Err(invalid("decomposition: complex integer RHS arithmetic is not supported with floating coefficients"));
        }
        if m != 1 || n != 1 {
            return Err(invalid(
                "decomposition: integer RHS inputs require a scalar floating coefficient",
            ));
        }
        if !input::complex(&spec.matrix) && spec.scale.1 == 0.0 {
            if input::dtype(&spec.matrix) != NumericDType::F64 {
                return Err(invalid("decomposition: integer arrays can only be combined with scalar double coefficients"));
            }
            let pair = native::coefficient(&spec.matrix, 0)?;
            let scalar = native::checked_double(pair.0)? * spec.scale.0;
            // Scalar integer division is not a matrix factor route. Its
            // canonical owner retains wide-integer/saturation and resident
            // integer output semantics without refactorization.
            return crate::call_builtin_async_in_context(
                runtime,
                "rdivide",
                &[operand.clone(), Value::Num(scalar)],
            )
            .await
            .map(|root| root.value().clone());
        }
        return Err(invalid(
            "decomposition: integer RHS inputs require a real scalar double coefficient",
        ));
    }
    let source = if let Value::GpuTensor(handle) = operand {
        Some(handle)
    } else {
        None
    };
    let host = crate::dispatcher::gather_if_needed_async(operand).await?;
    let (host, _) = native::normalize_selected(host, runtime)?;
    input::validate(&host, runtime).await?;
    if spec.factor_data.is_none()
        && matches!(
            spec.decomp_type,
            DecompositionType::Diagonal | DecompositionType::Triangular
        )
    {
        // One current-owner proof covers all real/imaginary RHS channels;
        // immutable recipe storage remains borrowed through every solve.
        native::inspect_original(&spec.matrix, false, runtime).await?;
        native::check_factor_recipe(&spec.matrix, spec.triangle, runtime).await?;
    }
    let (m, n) = matrix_dims(&spec.matrix);
    let (a_rows, a_cols) = if spec.is_conjugate_transposed {
        (n, m)
    } else {
        (m, n)
    };
    let (rows, columns) = matrix_dims(&host);
    let scalar_operator = m == 1 && n == 1;
    let (shape, solve_columns) = if scalar_operator {
        (
            [rows, columns],
            rows.checked_mul(columns)
                .ok_or_else(|| native::allocation(HostAllocationError::Overflow, runtime))?,
        )
    } else if right {
        if columns != a_cols {
            return Err(invalid(
                "decomposition: right-hand operand dimensions must agree",
            ));
        }
        ([rows, a_rows], rows)
    } else {
        if rows != a_rows {
            return Err(invalid(
                "decomposition: right-hand operand dimensions must agree",
            ));
        }
        ([a_cols, columns], columns)
    };
    let factor_class = if let Some(data) = &spec.factor_data {
        data.storage_class()
    } else {
        persistence::expected_factor_class(&spec.matrix)?
    };
    let dtype = if input::dtype(&host) == NumericDType::F32
        || matches!(
            factor_class,
            ExactHostSparseClass::Numeric(NumericDType::F32)
                | ExactHostSparseClass::Complex(NumericDType::F32)
        ) {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    let complex = matches!(factor_class, ExactHostSparseClass::Complex(_))
        || input::complex(&host)
        || spec.scale.1 != 0.0;
    let input = input::Input {
        value: &host,
        rows,
        columns,
        right,
        scalar_operator,
        scale: spec.scale,
    };
    match (dtype, complex) {
        (NumericDType::F32, false) => {
            output::<f32>(
                spec,
                &input,
                shape,
                solve_columns,
                m == 0 || n == 0,
                source,
                runtime,
            )
            .await
        }
        (NumericDType::F32, true) => {
            output::<ComplexElement<f32>>(
                spec,
                &input,
                shape,
                solve_columns,
                m == 0 || n == 0,
                source,
                runtime,
            )
            .await
        }
        (_, false) => {
            output::<f64>(
                spec,
                &input,
                shape,
                solve_columns,
                m == 0 || n == 0,
                source,
                runtime,
            )
            .await
        }
        (_, true) => {
            output::<ComplexElement<f64>>(
                spec,
                &input,
                shape,
                solve_columns,
                m == 0 || n == 0,
                source,
                runtime,
            )
            .await
        }
    }
}

async fn output<O: DenseScalar>(
    spec: &DecompositionSpec,
    input: &input::Input<'_>,
    shape: [usize; 2],
    columns: usize,
    empty: bool,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
    runtime: &RuntimeContext,
) -> BuiltinResult<Value> {
    if !spec.scale.0.is_finite() || !spec.scale.1.is_finite() {
        return Err(invalid("decomposition: invalid scaled operator"));
    }
    let bytes = if O::COMPLEX {
        HostAllocationPlan::retained_complex_tensor_bytes(O::DTYPE, &shape)
    } else {
        HostAllocationPlan::retained_tensor_bytes(O::DTYPE, &shape)
    }
    .map_err(|e| native::allocation(e, runtime))?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, bytes).map_err(|e| native::allocation(e, runtime))?,
    )
    .map_err(|e| native::allocation(e, runtime))?;
    let mut result = O::allocate(shape, receipt.as_ref(), runtime)
        .await
        .map_err(|e| native::dense_error(e, runtime))?;
    let mut work = OperatorWork::new(runtime).map_err(|e| work_error(e, runtime))?;
    work.flush().await.map_err(|e| work_error(e, runtime))?;
    let output_rows = if input.right && !input.scalar_operator {
        shape[1]
    } else {
        shape[0]
    };
    let count = shape[0]
        .checked_mul(shape[1])
        .ok_or_else(|| native::allocation(HostAllocationError::Overflow, runtime))?;
    let mut status = RetainedSolveStatus::default();
    if !empty && count != 0 && spec.scale == (0.0, 0.0) {
        let minimum_norm = matches!(
            spec.decomp_type,
            DecompositionType::Qr | DecompositionType::Cod
        );
        let values = O::elements_mut(&mut result)
            .ok_or_else(|| internal("decomposition output lost unique native storage"))?;
        if !minimum_norm {
            for index in 0..count {
                let (real, imag) = input
                    .parts(index)
                    .ok_or_else(|| invalid("decomposition: invalid scaled RHS"))?;
                let target = input.output_index(index, output_rows);
                values[target] = O::from_ieee_parts(real, if input.right { -imag } else { imag });
                work.tick().await.map_err(|e| work_error(e, runtime))?;
            }
        }
        status = RetainedSolveStatus {
            singular: !minimum_norm,
            deficient_rank: minimum_norm.then_some(0),
        };
    } else if !empty && count != 0 {
        let selected_scale = if input.right {
            complex_conj(spec.scale)
        } else {
            spec.scale
        };
        let class = if let Some(data) = &spec.factor_data {
            data.storage_class()
        } else {
            persistence::expected_factor_class(&spec.matrix)?
        };
        macro_rules! run {
            ($factor:ty) => {
                channels::<$factor, O>(
                    spec,
                    input,
                    columns,
                    output_rows,
                    selected_scale,
                    &mut result,
                    runtime,
                )
                .await?
            };
        }
        status = match class {
            ExactHostSparseClass::Numeric(NumericDType::F32) if selected_scale.1 != 0.0 => {
                run!(ComplexElement<f32>)
            }
            ExactHostSparseClass::Numeric(NumericDType::F64) if selected_scale.1 != 0.0 => {
                run!(ComplexElement<f64>)
            }
            ExactHostSparseClass::Numeric(NumericDType::F32) => run!(f32),
            ExactHostSparseClass::Numeric(NumericDType::F64) => run!(f64),
            ExactHostSparseClass::Complex(NumericDType::F32) => run!(ComplexElement<f32>),
            ExactHostSparseClass::Complex(NumericDType::F64) => run!(ComplexElement<f64>),
            _ => return Err(invalid("decomposition: unsupported retained factor class")),
        };
    }
    work.flush().await.map_err(|e| work_error(e, runtime))?;
    if spec.check_condition {
        let warning = if status.singular {
            Some((
                "RunMat:decomposition:SingularMatrix",
                "Coefficient matrix is singular; the returned solution may contain Inf or NaN.",
            ))
        } else if status.deficient_rank.is_some() {
            Some(("RunMat:decomposition:RankDeficientMatrix","Coefficient matrix is rank deficient; the selected QR or COD solution uses its retained rank."))
        } else {
            None
        };
        if let Some((identifier, message)) = warning {
            crate::warnings::emit(crate::warnings::WarningRequest {
                builtin: NAME,
                identifier,
                message,
                show_identifier: true,
            })?;
        }
    }
    if let Some(receipt) = &receipt {
        receipt
            .checkpoint(runtime)
            .map_err(|e| native::allocation(e, runtime))?;
    }
    work.flush().await.map_err(|e| work_error(e, runtime))?;
    if let Some(source) = source {
        return resident::restore(&result, source, runtime).await;
    }
    Ok(match result {
        NativeDenseFactorMatrix::Real(t) => tensor::tensor_into_value(t),
        NativeDenseFactorMatrix::Complex(t) => {
            if t.len() == 1 && t.numeric_dtype() == NumericDType::F64 {
                let value = t.as_f64_slice().expect("native complex double")[0];
                Value::Complex(value.0, value.1)
            } else {
                Value::ComplexTensor(t)
            }
        }
    })
}

async fn channels<
    F: DenseScalar + crate::builtins::math::linalg::ops::triangular_operator::TriangularScalar,
    O: DenseScalar,
>(
    spec: &DecompositionSpec,
    input: &input::Input<'_>,
    columns: usize,
    output_rows: usize,
    scale: (f64, f64),
    result: &mut NativeDenseFactorMatrix,
    runtime: &RuntimeContext,
) -> BuiltinResult<RetainedSolveStatus> {
    let (m, n) = matrix_dims(&spec.matrix);
    let adjoint = spec.is_conjugate_transposed ^ input.right;
    let input_rows = if adjoint { n } else { m };
    let len = input_rows
        .checked_mul(columns)
        .ok_or_else(|| native::allocation(HostAllocationError::Overflow, runtime))?;
    let geometry = FactorRhsGeometry { len, columns };
    let options = FactorSolveOptions {
        adjoint,
        singular: SingularSolvePolicy::PublicDecompositionIeee,
    };
    let mut status = RetainedSolveStatus::default();
    for channel in 0..if !F::COMPLEX && O::COMPLEX { 2 } else { 1 } {
        let read = |index| {
            let (real, imag) = input.raw_parts(index)?;
            F::from_parts(
                if channel == 1 { imag } else { real },
                if F::COMPLEX { imag } else { 0.0 },
            )
        };
        let values = O::elements_mut(result)
            .ok_or_else(|| internal("decomposition output lost unique native storage"))?;
        let sink = |index, value: F| {
            let (real, mut imag) = value.parts();
            if input.right {
                imag = -imag;
            }
            let index = input.output_index(index, output_rows);
            if channel == 1 {
                let (saved, _) = values[index].parts();
                values[index] = O::from_ieee_parts(saved, if input.right { -real } else { real });
            } else {
                values[index] = O::from_ieee_parts(real, imag);
            }
            Ok(())
        };
        let next = if let Some(data) = &spec.factor_data {
            if matches!(
                data.parts(),
                NumericFactorCandidate::DenseLu { .. }
                    | NumericFactorCandidate::DenseCholesky { .. }
                    | NumericFactorCandidate::DenseQr(_)
                    | NumericFactorCandidate::DenseCod { .. }
            ) {
                dense_factor::solve_dense_indexed::<F>(
                    data, geometry, options, scale, runtime, read, sink,
                )
                .await
                .map_err(|e| native::dense_error(e, runtime))?
            } else if scale == (1.0, 0.0) && F::storage_class() == data.storage_class() {
                sparse_factor::solve_factor_indexed::<F>(
                    data, geometry, options, runtime, read, sink,
                )
                .await
                .map_err(|e| native::retained_error(e, runtime))?
            } else {
                sparse_factor::solve_factor_indexed_scaled::<F>(
                    data, geometry, options, scale, runtime, read, sink,
                )
                .await
                .map_err(|e| native::retained_error(e, runtime))?
            }
        } else if matches!(
            spec.decomp_type,
            DecompositionType::Diagonal | DecompositionType::Triangular
        ) {
            structural::solve::<F>(spec, geometry, adjoint, scale, runtime, read, sink).await?
        } else {
            return Err(invalid("decomposition: retained factor DATA is missing; reconstruct at its import boundary"));
        };
        status.singular |= next.singular;
        status.deficient_rank = next.deficient_rank;
    }
    Ok(status)
}
