//! MATLAB-compatible `chol` builtin with upper/lower forms and failure flag.
//!
//! This implementation matches MATLAB semantics for dense matrices, including
//! the two-output form that reports the leading minor index when a matrix fails
//! the positive-definiteness test. GPU execution is delegated to acceleration
//! providers when available, with automatic host fallbacks.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, random_args, tensor};
#[cfg(test)]
use crate::builtins::math::linalg::type_resolvers::matrix_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, ProviderCholResult};
use runmat_builtins::BuiltinErrorDescriptor;
#[cfg(test)]
use runmat_builtins::BuiltinIntegerOutputClassRule;
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, NumericDType, NumericScalar, Tensor, Value};

pub(in crate::builtins::math::linalg) mod sparse;

const BUILTIN_NAME: &str = "chol";

#[cfg(test)]
use runmat_builtins::CHOL_DESCRIPTOR;
use runmat_builtins::{
    CHOL_ERROR_INTERNAL, CHOL_ERROR_INVALID_ARGUMENT, CHOL_ERROR_INVALID_INPUT,
    CHOL_ERROR_NOT_POSITIVE_DEFINITE, CHOL_INTEGER_INPUT_EXTENSION, CHOL_LOGICAL_INPUT_EXTENSION,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::factor::chol")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "chol",
    op_kind: GpuOpKind::Custom("chol-factor"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("chol")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Uses the provider 'chol' hook when present; otherwise gathers to the host implementation.",
};

fn chol_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    chol_error_with_message(error.message, error)
}

#[cfg(test)]
mod sparse_public_tests {
    use super::*;
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeContext, RuntimeContextGuard, RuntimeParallelResources,
        RuntimeParallelService, RuntimeServicePorts,
    };
    use crate::execution::RuntimeExecutionService;
    use futures::executor::block_on;
    use runmat_value::SparseTensor;
    use std::rc::Rc;

    struct Parallel(ManagedHostEffectLedger);
    impl RuntimeParallelService for Parallel {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let state = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(state.free_bytes()),
                epoch: state.epoch,
                ..Default::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    fn runtime() -> (RuntimeContext, ManagedHostEffectLedger) {
        let ledger = ManagedHostEffectLedger::new(4_000_000);
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(
                RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
            );
        (runtime, ledger)
    }
    fn matrix() -> SparseTensor {
        SparseTensor::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0],
        )
        .unwrap()
    }
    fn invoke(value: Value, args: Vec<Value>) -> BuiltinResult<Value> {
        let args = crate::execution::RootedValues::new(args).map_err(|e| e.into_runtime_error())?;
        block_on(super::chol_builtin(
            crate::context::legacy::active(),
            value,
            args.into(),
        ))
    }

    #[test]
    fn sparse_public_three_output_preserves_sparse_class_and_permutation_identity() {
        let (runtime, ledger) = runtime();
        let _active = RuntimeContextGuard::enter(runtime);
        let _outputs = crate::output_count::push_output_count(Some(3)).unwrap();
        let result = invoke(Value::SparseTensor(matrix()), vec![Value::from("vector")]).unwrap();
        let Value::OutputList(outputs) = result else {
            panic!("expected three outputs")
        };
        assert_eq!(outputs.len(), 3);
        let Value::SparseTensor(factor) = &outputs[0] else {
            panic!("factor must stay sparse")
        };
        assert_eq!((factor.rows, factor.cols), (3, 3));
        for column in 0..factor.cols {
            assert!(
                factor.row_indices[factor.col_ptrs[column]..factor.col_ptrs[column + 1]]
                    .iter()
                    .all(|&row| row <= column)
            );
        }
        assert_eq!(outputs[1], Value::Num(0.0));
        let Value::Tensor(permutation) = &outputs[2] else {
            panic!("vector form")
        };
        let mut values = permutation.materialize_f64();
        values.sort_by(f64::total_cmp);
        assert_eq!(&*values, &[1.0, 2.0, 3.0]);
        assert!(ledger.snapshot().live_claimed_bytes > 0);
    }

    #[test]
    fn sparse_public_partial_failure_returns_q_by_n_factor_and_flag() {
        let (runtime, _) = runtime();
        let _active = RuntimeContextGuard::enter(runtime);
        let _outputs = crate::output_count::push_output_count(Some(2)).unwrap();
        let sparse = SparseTensor::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 2.0, 1.0],
        )
        .unwrap();
        let result = invoke(Value::SparseTensor(sparse), Vec::new()).unwrap();
        let Value::OutputList(outputs) = result else {
            panic!("two outputs")
        };
        let Value::SparseTensor(factor) = &outputs[0] else {
            panic!("partial sparse factor")
        };
        assert_eq!((factor.rows, factor.cols), (1, 2));
        assert_eq!(outputs[1], Value::Num(2.0));
    }

    #[test]
    fn sparse_factor_without_third_output_does_not_publish_permutation() {
        let (runtime, _) = runtime();
        let eval = block_on(sparse::evaluate(
            matrix(),
            CholTriangle::Upper,
            SparsePermutationOutput::Matrix,
            false,
            false,
            &runtime,
        ))
        .unwrap();
        assert!(eval.permutation.is_none());
    }
}

fn chol_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn chol_invalid_argument(message: impl Into<String>) -> RuntimeError {
    chol_error_with_message(message, &CHOL_ERROR_INVALID_ARGUMENT)
}

fn chol_invalid_input(message: impl Into<String>) -> RuntimeError {
    chol_error_with_message(message, &CHOL_ERROR_INVALID_INPUT)
}

fn chol_provider_error(error: runmat_accelerate_api::ProviderExecutionError) -> RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        chol_internal_error(error.to_string()),
        error,
    )
}

fn chol_internal_error(message: impl Into<String>) -> RuntimeError {
    chol_error_with_message(message, &CHOL_ERROR_INTERNAL)
}

fn with_chol_context(mut error: RuntimeError) -> RuntimeError {
    if error.context.builtin.is_none() {
        error.context = error.context.with_builtin(BUILTIN_NAME);
    }
    error
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::factor::chol")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "chol",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Factorisation executes eagerly and does not participate in expression fusion.",
};

#[runtime_builtin(
    name = "chol",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::chol"
)]
async fn chol_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 3)
        || (crate::output_count::current_output_count() == Some(3)
            && !matches!(value, Value::SparseTensor(_)))
    {
        return Err(chol_error(&CHOL_ERROR_INVALID_ARGUMENT));
    }
    let eval = evaluate_in_context(runtime.as_ref(), value, rest.view()).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        if out_count == 1 {
            if !eval.is_positive_definite() {
                return Err(chol_error(&CHOL_ERROR_NOT_POSITIVE_DEFINITE));
            }
            return Ok(Value::output_list(vec![eval.factor()]));
        }
        if out_count == 2 {
            return Ok(Value::output_list(vec![eval.factor(), eval.flag()]));
        }
        if out_count == 3 {
            return eval
                .permutation
                .clone()
                .map(|permutation| {
                    Value::output_list(vec![eval.factor(), eval.flag(), permutation])
                })
                .ok_or_else(|| chol_error(&CHOL_ERROR_INVALID_ARGUMENT));
        }
        return Err(chol_error(&CHOL_ERROR_INVALID_ARGUMENT));
    }
    if !eval.is_positive_definite() {
        return Err(chol_error(&CHOL_ERROR_NOT_POSITIVE_DEFINITE));
    }
    Ok(eval.factor())
}

/// Evaluate `chol` while keeping both the factor and the failure index available.
#[derive(Clone)]
pub struct CholEval {
    factor: Value,
    flag: usize,
    triangle: CholTriangle,
    permutation: Option<Value>,
}

impl CholEval {
    /// The factor (`R` or `L`) requested by the caller.
    pub fn factor(&self) -> Value {
        self.factor.clone()
    }

    /// MATLAB-compatible failure index (0 indicates success).
    pub fn flag(&self) -> Value {
        Value::Num(self.flag as f64)
    }

    /// Zero-based flag value (0 indicates success).
    pub fn flag_index(&self) -> usize {
        self.flag
    }

    /// The triangle variant that was requested.
    pub fn triangle(&self) -> CholTriangle {
        self.triangle
    }

    /// Returns true when the input matrix was positive definite.
    pub fn is_positive_definite(&self) -> bool {
        self.flag == 0
    }

    fn from_components(
        components: CholComponents,
        triangle: CholTriangle,
        output_dtype: NumericDType,
    ) -> BuiltinResult<Self> {
        let factor_matrix = match triangle {
            CholTriangle::Upper => components.upper.clone(),
            CholTriangle::Lower => components.upper.conjugate_transpose(),
        };
        let factor = matrix_to_value("chol", &factor_matrix, output_dtype)?;
        Ok(Self {
            factor,
            flag: components.info,
            triangle,
            permutation: None,
        })
    }

    fn from_provider(result: ProviderCholResult, triangle: CholTriangle) -> Self {
        Self {
            factor: Value::GpuTensor(result.factor),
            flag: result.info as usize,
            triangle,
            permutation: None,
        }
    }
}

/// Triangle variant for the Cholesky factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CholTriangle {
    Upper,
    Lower,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparsePermutationOutput {
    Matrix,
    Vector,
}

/// Compute the Cholesky factorization for the given value and option list.
pub async fn evaluate(
    value: Value,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<CholEval> {
    let runtime = crate::context::legacy::active();
    evaluate_in_context(runtime.as_ref(), value, args).await
}

async fn evaluate_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    value: Value,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<CholEval> {
    let options = parse_options(args)?;
    let triangle = options.triangle;
    if options.explicit_output_form && !matches!(value, Value::SparseTensor(_)) {
        return Err(chol_invalid_argument(
            "chol: permutation output form requires sparse input",
        ));
    }
    crate::builtins::common::validation::reject_typed_complex_integer(&value, BUILTIN_NAME)?;
    let integer_input = is_typed_integer_value(&value);
    if integer_input {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CHOL_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if is_logical_value(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CHOL_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    match value {
        Value::SparseTensor(sparse_value) => {
            let runtime = runtime.ok_or_else(|| {
                chol_internal_error("chol: sparse factorization requires an active runtime owner")
            })?;
            sparse::evaluate(
                sparse_value,
                triangle,
                options.output_form,
                options.explicit_output_form
                    || crate::output_count::current_output_count() == Some(3),
                crate::output_count::current_output_count() == Some(3),
                runtime,
            )
            .await
        }
        Value::GpuTensor(handle) => {
            let provider = gpu_helpers::require_resident_compute_owner(&handle, || {
                chol_internal_error("chol: no exact owner for resident input")
            })?;
            if !integer_input {
                if let Some(eval) = evaluate_gpu(&handle, triangle).await? {
                    return Ok(eval);
                }
            }
            let host =
                gpu_helpers::download_value_preserving_residency_async(provider, &handle).await?;
            ensure_exact_integer_boundary(&host)?;
            let eval = evaluate_host_value(host, triangle).await?;
            reupload_factor(eval, provider, &handle)
        }
        other => {
            ensure_exact_integer_boundary(&other)?;
            evaluate_host_value(other, triangle).await
        }
    }
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

fn ensure_exact_integer_boundary(value: &Value) -> BuiltinResult<()> {
    const MAX_EXACT_INTEGER: i128 = 1_i128 << 53;
    let inexact = match value {
        Value::Int(value) => {
            let value = NumericScalar::from(value.clone());
            numeric_scalar_as_i128(value)
                .is_some_and(|value| !(-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&value))
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            (0..tensor.len()).any(|index| {
                tensor
                    .numeric_value_at(index)
                    .and_then(numeric_scalar_as_i128)
                    .is_some_and(|value| !(-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&value))
            })
        }
        _ => false,
    };
    if inexact {
        return Err(chol_invalid_input(
            "chol: integer input values must be exactly representable as double",
        ));
    }
    Ok(())
}

fn numeric_scalar_as_i128(value: NumericScalar) -> Option<i128> {
    match value {
        NumericScalar::I8(value) => Some(i128::from(value)),
        NumericScalar::I16(value) => Some(i128::from(value)),
        NumericScalar::I32(value) => Some(i128::from(value)),
        NumericScalar::I64(value) => Some(i128::from(value)),
        NumericScalar::U8(value) => Some(i128::from(value)),
        NumericScalar::U16(value) => Some(i128::from(value)),
        NumericScalar::U32(value) => Some(i128::from(value)),
        NumericScalar::U64(value) => Some(i128::from(value)),
        NumericScalar::F32(_) | NumericScalar::F64(_) => None,
    }
}

fn reupload_factor(
    mut eval: CholEval,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
) -> BuiltinResult<CholEval> {
    let uploaded = match &eval.factor {
        Value::Tensor(tensor) => gpu_helpers::upload_tensor_protecting(provider, tensor, &[input]),
        Value::Num(value) => {
            let tensor = Tensor::new(vec![*value], vec![1, 1])
                .map_err(|error| chol_internal_error(error.to_string()))?;
            gpu_helpers::upload_tensor_protecting(provider, &tensor, &[input])
        }
        Value::ComplexTensor(tensor) => {
            gpu_helpers::upload_complex_tensor_protecting(provider, tensor, &[input])
        }
        Value::Complex(real, imag) => {
            let tensor = ComplexTensor::new(vec![(*real, *imag)], vec![1, 1])
                .map_err(|error| chol_internal_error(error.to_string()))?;
            gpu_helpers::upload_complex_tensor_protecting(provider, &tensor, &[input])
        }
        other => {
            return Err(chol_internal_error(format!(
                "chol: cannot restore GPU factor of type {other:?}"
            )))
        }
    }?;
    let uploaded = gpu_helpers::propagate_owned_output_provenance(uploaded, [input])?;
    eval.factor = gpu_helpers::resident_gpu_value(uploaded)?;
    Ok(eval)
}

async fn evaluate_host_value(value: Value, triangle: CholTriangle) -> BuiltinResult<CholEval> {
    let output_dtype = floating_output_dtype(&value);
    let matrix = extract_matrix(value).await?;
    if matrix.rows != matrix.cols {
        return Err(chol_invalid_input("chol: input matrix must be square"));
    }
    let components = chol_factor(matrix.hermitian_from_triangle(triangle))?;
    CholEval::from_components(components, triangle, output_dtype)
}

fn floating_output_dtype(value: &Value) -> NumericDType {
    match value {
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => NumericDType::F32,
        Value::ComplexTensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            NumericDType::F32
        }
        _ => NumericDType::F64,
    }
}

async fn evaluate_gpu(
    handle: &GpuTensorHandle,
    triangle: CholTriangle,
) -> BuiltinResult<Option<CholEval>> {
    if let Some(provider) = gpu_helpers::exact_provider_for_handle(handle) {
        if runmat_accelerate_api::handle_storage(handle)
            != runmat_accelerate_api::GpuTensorStorage::Real
            || runmat_accelerate_api::handle_is_logical(handle)
            || runmat_accelerate_api::handle_precision(handle) != Some(provider.precision())
        {
            return Ok(None);
        }
        let _residency = gpu_helpers::ResidencyRestoreGuard::new(handle);
        let lower = matches!(triangle, CholTriangle::Lower);
        match provider.chol(handle, lower).await {
            Ok(mut result) => {
                let valid = result.factor.shape() == handle.shape()
                    && result.info as usize <= handle.shape().first().copied().unwrap_or(1)
                    && !result.factor.shares_allocation_with(handle)
                    && !runmat_accelerate_api::handle_is_logical(&result.factor)
                    && !runmat_accelerate_api::handle_is_transposed(&result.factor)
                    && result.factor.physical().element_type() == handle.physical().element_type()
                    && result.factor.physical().storage() == handle.physical().storage()
                    && gpu_helpers::exact_provider_for_handle(&result.factor).is_some_and(
                        |owner| runmat_accelerate_api::same_provider_instance(owner, provider),
                    );
                if !valid {
                    return Err(gpu_helpers::reject_gpu_output(
                        result.factor,
                        chol_internal_error("chol: invalid provider factor output"),
                    ));
                }
                gpu_helpers::propagate_output_provenance(&mut result.factor, [handle])?;
                return Ok(Some(CholEval::from_provider(result, triangle)));
            }
            Err(error) if error.is_unsupported() => {}
            Err(error) => return Err(chol_provider_error(error)),
        }
    }
    Ok(None)
}

struct CholOptions {
    triangle: CholTriangle,
    output_form: SparsePermutationOutput,
    explicit_output_form: bool,
}

fn parse_options(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<CholOptions> {
    if args.is_empty() {
        return Ok(CholOptions {
            triangle: CholTriangle::Upper,
            output_form: SparsePermutationOutput::Matrix,
            explicit_output_form: false,
        });
    }
    if args.len() > 2 {
        return Err(chol_invalid_argument("chol: too many option arguments"));
    }
    let mut result = CholOptions {
        triangle: CholTriangle::Upper,
        output_form: SparsePermutationOutput::Matrix,
        explicit_output_form: false,
    };
    for argument in args.iter() {
        let Some(option) = tensor::value_to_string(argument) else {
            return Err(chol_invalid_argument(
                "chol: options must be strings or character vectors",
            ));
        };
        match option.trim().to_ascii_lowercase().as_str() {
            "upper" => result.triangle = CholTriangle::Upper,
            "lower" => result.triangle = CholTriangle::Lower,
            "matrix" => {
                result.output_form = SparsePermutationOutput::Matrix;
                result.explicit_output_form = true;
            }
            "vector" => {
                result.output_form = SparsePermutationOutput::Vector;
                result.explicit_output_form = true;
            }
            other => Err(chol_invalid_argument(format!(
                "chol: unknown option '{other}'"
            )))?,
        }
    }
    Ok(result)
}

const EPS: f64 = 1.0e-12;

fn chol_factor(matrix: RowMajorMatrix) -> BuiltinResult<CholComponents> {
    let factor = super::cholesky::factor_upper(matrix.rows, |row, col| matrix.get(row, col))
        .map_err(|error| {
            chol_internal_error(format!("chol: factor allocation failed: {error:?}"))
        })?;
    Ok(CholComponents {
        upper: RowMajorMatrix {
            rows: matrix.rows,
            cols: matrix.cols,
            data: factor.upper,
        },
        info: factor.info,
    })
}

async fn extract_matrix(value: Value) -> BuiltinResult<RowMajorMatrix> {
    match value {
        Value::Tensor(tensor) => RowMajorMatrix::from_tensor(&tensor, "chol"),
        Value::ComplexTensor(ct) => RowMajorMatrix::from_complex_tensor(&ct, "chol"),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|err| chol_invalid_input(format!("chol: {err}")))?;
            RowMajorMatrix::from_tensor(&tensor, "chol")
        }
        Value::Num(n) => Ok(RowMajorMatrix::from_scalar(Complex64::new(n, 0.0))),
        Value::Int(i) => Ok(RowMajorMatrix::from_scalar(Complex64::new(i.to_f64(), 0.0))),
        Value::Bool(b) => Ok(RowMajorMatrix::from_scalar(Complex64::new(
            if b { 1.0 } else { 0.0 },
            0.0,
        ))),
        Value::Complex(re, im) => Ok(RowMajorMatrix::from_scalar(Complex64::new(re, im))),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(with_chol_context)?;
            RowMajorMatrix::from_tensor(&tensor, "chol")
        }
        other => Err(chol_invalid_input(format!(
            "chol: unsupported input type {:?}; expected numeric or logical values",
            other
        ))),
    }
}

fn matrix_to_value(
    label: &str,
    matrix: &RowMajorMatrix,
    output_dtype: NumericDType,
) -> BuiltinResult<Value> {
    let mut has_imag = false;
    for val in &matrix.data {
        if val.im.abs() > EPS {
            has_imag = true;
            break;
        }
    }
    if has_imag {
        let mut data = Vec::with_capacity(matrix.rows * matrix.cols);
        for col in 0..matrix.cols {
            for row in 0..matrix.rows {
                let idx = row * matrix.cols + col;
                let v = matrix.data[idx];
                data.push((v.re, v.im));
            }
        }
        let tensor = ComplexTensor::from_f64_values_with_dtype(
            data,
            vec![matrix.rows, matrix.cols],
            output_dtype,
        )
        .map_err(|e| chol_internal_error(format!("{label}: {e}")))?;
        Ok(random_args::complex_tensor_into_value(tensor))
    } else {
        let mut data = Vec::with_capacity(matrix.rows * matrix.cols);
        for col in 0..matrix.cols {
            for row in 0..matrix.rows {
                let idx = row * matrix.cols + col;
                data.push(matrix.data[idx].re);
            }
        }
        let tensor = if output_dtype == NumericDType::F32 {
            Tensor::from_f32(
                data.into_iter().map(|value| value as f32).collect(),
                vec![matrix.rows, matrix.cols],
            )
        } else {
            Tensor::new(data, vec![matrix.rows, matrix.cols])
        }
        .map_err(|e| chol_internal_error(format!("{label}: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

struct CholComponents {
    upper: RowMajorMatrix,
    info: usize,
}

#[derive(Clone)]
struct RowMajorMatrix {
    rows: usize,
    cols: usize,
    data: Vec<Complex64>,
}

impl RowMajorMatrix {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Complex64::new(0.0, 0.0); rows.saturating_mul(cols)],
        }
    }

    fn from_scalar(value: Complex64) -> Self {
        Self {
            rows: 1,
            cols: 1,
            data: vec![value],
        }
    }

    fn hermitian_from_triangle(&self, triangle: CholTriangle) -> Self {
        let mut out = Self::zeros(self.rows, self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                let value = match triangle {
                    CholTriangle::Upper if row <= col => self.get(row, col),
                    CholTriangle::Upper => self.get(col, row).conj(),
                    CholTriangle::Lower if row >= col => self.get(row, col),
                    CholTriangle::Lower => self.get(col, row).conj(),
                };
                out.set(row, col, value);
            }
        }
        out
    }

    fn from_tensor(tensor: &Tensor, label: &str) -> BuiltinResult<Self> {
        if tensor.shape().len() > 2 {
            return Err(chol_invalid_input(format!("{label}: input must be 2-D")));
        }
        let rows = tensor.rows();
        let cols = tensor.cols();
        let values = tensor::tensor_values_f64_cow(tensor);
        let mut data = vec![Complex64::new(0.0, 0.0); rows.saturating_mul(cols)];
        for col in 0..cols {
            for row in 0..rows {
                let idx_col_major = row + col * rows;
                let idx_row_major = row * cols + col;
                data[idx_row_major] = Complex64::new(values[idx_col_major], 0.0);
            }
        }
        Ok(Self { rows, cols, data })
    }

    fn from_complex_tensor(tensor: &ComplexTensor, label: &str) -> BuiltinResult<Self> {
        if tensor.shape().len() > 2 {
            return Err(chol_invalid_input(format!("{label}: input must be 2-D")));
        }
        let rows = tensor.rows();
        let cols = tensor.cols();
        let mut data = vec![Complex64::new(0.0, 0.0); rows.saturating_mul(cols)];
        for col in 0..cols {
            for row in 0..rows {
                let idx_col_major = row + col * rows;
                let idx_row_major = row * cols + col;
                let (re, im) = tensor.materialize_f64()[idx_col_major];
                data[idx_row_major] = Complex64::new(re, im);
            }
        }
        Ok(Self { rows, cols, data })
    }

    fn get(&self, row: usize, col: usize) -> Complex64 {
        self.data[row * self.cols + col]
    }

    fn set(&mut self, row: usize, col: usize, value: Complex64) {
        self.data[row * self.cols + col] = value;
    }

    fn conjugate_transpose(&self) -> Self {
        let mut out = RowMajorMatrix::zeros(self.cols, self.rows);
        for row in 0..self.rows {
            for col in row..self.cols {
                let value = self.get(row, col);
                out.set(col, row, value.conj());
            }
        }
        out
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{IntegerStorage, LogicalArray, Tensor as Matrix};

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    fn tensor_from_value(value: Value) -> Matrix {
        match value {
            Value::Tensor(t) => t,
            Value::Num(n) => Matrix::new(vec![n], vec![1, 1]).expect("tensor"),
            other => panic!("expected tensor value, got {other:?}"),
        }
    }

    #[test]
    fn chol_type_preserves_matrix_shape() {
        let out = matrix_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(3), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(3)])
            }
        );
    }

    #[test]
    fn chol_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = CHOL_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"R = chol(A)"));
        assert!(labels.contains(&"R = chol(A, triangle)"));
        assert!(labels.contains(&"[R, p] = chol(A)"));
        assert!(labels.contains(&"[R, p] = chol(A, triangle)"));
    }

    #[test]
    fn chol_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = CHOL_DESCRIPTOR.errors.iter().map(|err| err.code).collect();
        assert!(codes.contains(&"RM.CHOL.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.CHOL.INVALID_INPUT"));
        assert!(codes.contains(&"RM.CHOL.NOT_POSITIVE_DEFINITE"));
        assert!(codes.contains(&"RM.CHOL.INTERNAL"));
    }

    #[test]
    fn chol_registers_integer_extension_and_capability() {
        let builtin = runmat_builtins::builtin_catalog_entry_by_name(BUILTIN_NAME)
            .expect("chol catalog entry");
        assert_eq!(builtin.extensions.len(), 2);
        assert_eq!(builtin.extensions[0].id, "chol-integer-input");
        assert_eq!(builtin.extensions[1].id, "chol-logical-input");
        assert_eq!(builtin.integer_capabilities.len(), 1);
        assert_eq!(builtin.integer_capabilities[0].inputs[0].classes.len(), 8);
        assert_eq!(
            builtin.integer_capabilities[0].output_class,
            BuiltinIntegerOutputClassRule::Double
        );
    }

    #[test]
    fn chol_matrix_conversion_reads_typed_integer_storage_exactly() {
        let tensor = Matrix::new_integer(IntegerStorage::I16(vec![4, 2, 2, 3]), vec![2, 2])
            .expect("typed integer tensor");

        let matrix = RowMajorMatrix::from_tensor(&tensor, "chol").expect("matrix");
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);
        assert_eq!(
            matrix.data,
            vec![
                Complex64::new(4.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ]
        );
    }

    fn reconstruct_from_upper(matrix: &Matrix) -> Matrix {
        let rows = matrix.rows();
        let cols = matrix.cols();
        assert_eq!(rows, cols, "expected square matrix");
        let mut data = vec![0.0; rows * cols];
        // Compute R' * R for validation (column-major input)
        for i in 0..rows {
            for j in 0..rows {
                let mut sum = 0.0;
                for k in 0..rows {
                    let rik = if k <= i {
                        matrix.materialize_f64()[k + i * rows]
                    } else {
                        0.0
                    };
                    let rjk = if k <= j {
                        matrix.materialize_f64()[k + j * rows]
                    } else {
                        0.0
                    };
                    sum += rik * rjk;
                }
                data[i + j * rows] = sum;
            }
        }
        Matrix::new(data, vec![rows, rows]).expect("matrix")
    }

    fn reconstruct_from_lower(matrix: &Matrix) -> Matrix {
        let rows = matrix.rows();
        let cols = matrix.cols();
        assert_eq!(rows, cols, "expected square matrix");
        let mut data = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..rows {
                let mut sum = 0.0;
                for k in 0..rows {
                    let lik = if i >= k {
                        matrix.materialize_f64()[i + k * rows]
                    } else {
                        0.0
                    };
                    let ljk = if j >= k {
                        matrix.materialize_f64()[j + k * rows]
                    } else {
                        0.0
                    };
                    sum += lik * ljk;
                }
                data[i + j * rows] = sum;
            }
        }
        Matrix::new(data, vec![rows, rows]).expect("matrix")
    }

    fn tensor_close(lhs: &Matrix, rhs: &Matrix, tol: f64) {
        assert_eq!(lhs.shape(), rhs.shape(), "shape mismatch");
        for (a, b) in lhs
            .materialize_f64()
            .iter()
            .zip(rhs.materialize_f64().iter())
        {
            assert!(
                (a - b).abs() <= tol,
                "tensors differ: {a} vs {b} (tol {tol})"
            );
        }
    }

    fn complex_tensor_from_value(value: Value) -> ComplexTensor {
        match value {
            Value::ComplexTensor(ct) => ct,
            Value::Complex(re, im) => {
                ComplexTensor::new(vec![(re, im)], vec![1, 1]).expect("complex tensor")
            }
            Value::Tensor(t) => {
                let data: Vec<(f64, f64)> = t.materialize_f64().iter().map(|&v| (v, 0.0)).collect();
                ComplexTensor::new(data, t.shape().to_vec()).expect("complex tensor")
            }
            Value::Num(n) => {
                ComplexTensor::new(vec![(n, 0.0)], vec![1, 1]).expect("complex tensor")
            }
            other => panic!("expected complex-capable value, got {other:?}"),
        }
    }

    fn reconstruct_complex_upper(matrix: &ComplexTensor) -> ComplexTensor {
        let rows = matrix.rows();
        let cols = matrix.cols();
        assert_eq!(rows, cols, "expected square matrix");
        let mut data = vec![(0.0, 0.0); rows * rows];
        for i in 0..rows {
            for j in 0..rows {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..rows {
                    let rik = if k <= i {
                        let (re, im) = matrix.materialize_f64()[k + i * rows];
                        Complex64::new(re, im)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    let rjk = if k <= j {
                        let (re, im) = matrix.materialize_f64()[k + j * rows];
                        Complex64::new(re, im)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    sum += rik.conj() * rjk;
                }
                data[i + j * rows] = (sum.re, sum.im);
            }
        }
        ComplexTensor::new(data, vec![rows, rows]).expect("complex tensor")
    }

    fn reconstruct_complex_lower(matrix: &ComplexTensor) -> ComplexTensor {
        let rows = matrix.rows();
        let cols = matrix.cols();
        assert_eq!(rows, cols, "expected square matrix");
        let mut data = vec![(0.0, 0.0); rows * rows];
        for i in 0..rows {
            for j in 0..rows {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..rows {
                    let lik = if i >= k {
                        let (re, im) = matrix.materialize_f64()[i + k * rows];
                        Complex64::new(re, im)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    let ljk = if j >= k {
                        let (re, im) = matrix.materialize_f64()[j + k * rows];
                        Complex64::new(re, im)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    sum += lik * ljk.conj();
                }
                data[i + j * rows] = (sum.re, sum.im);
            }
        }
        ComplexTensor::new(data, vec![rows, rows]).expect("complex tensor")
    }

    fn complex_tensor_close(lhs: &ComplexTensor, rhs: &ComplexTensor, tol: f64) {
        assert_eq!(lhs.shape(), rhs.shape(), "shape mismatch");
        for ((ar, ai), (br, bi)) in lhs
            .materialize_f64()
            .iter()
            .zip(rhs.materialize_f64().iter())
        {
            let a = Complex64::new(*ar, *ai);
            let b = Complex64::new(*br, *bi);
            assert!(
                (a - b).norm() <= tol,
                "tensors differ: {a:?} vs {b:?} (tol {tol})"
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_upper_factor_matches_reference() {
        let a = Matrix::new(
            vec![
                4.0, 12.0, -16.0, //
                12.0, 37.0, -43.0, //
                -16.0, -43.0, 98.0,
            ],
            vec![3, 3],
        )
        .unwrap();
        let r = chol_builtin(Value::Tensor(a.clone()), Vec::new()).expect("chol");
        let r_tensor = tensor_from_value(r);
        assert_eq!(r_tensor.shape(), [3, 3]);
        for diag in 0..3 {
            let value = r_tensor.materialize_f64()[diag + diag * 3];
            assert!(value > 0.0, "Cholesky diagonal must be positive");
        }
        let recon = reconstruct_from_upper(&r_tensor);
        tensor_close(&recon, &a, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_upper_option_matches_default() {
        let a = Matrix::new(
            vec![
                7.0, 2.0, 1.0, //
                2.0, 5.0, 2.0, //
                1.0, 2.0, 3.0,
            ],
            vec![3, 3],
        )
        .unwrap();
        let default = chol_builtin(Value::Tensor(a.clone()), Vec::new()).expect("chol");
        let explicit =
            chol_builtin(Value::Tensor(a.clone()), vec![Value::from("upper")]).expect("chol upper");
        let default_tensor = tensor_from_value(default);
        let explicit_tensor = tensor_from_value(explicit);
        tensor_close(&default_tensor, &explicit_tensor, 1e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_lower_option_returns_lower_factor() {
        let a = Matrix::new(
            vec![
                25.0, 15.0, -5.0, //
                15.0, 18.0, 0.0, //
                -5.0, 0.0, 11.0,
            ],
            vec![3, 3],
        )
        .unwrap();
        let result =
            chol_builtin(Value::Tensor(a.clone()), vec![Value::from("lower")]).expect("chol");
        let l = tensor_from_value(result);
        assert_eq!(l.shape(), [3, 3]);
        for diag in 0..3 {
            let value = l.materialize_f64()[diag + diag * 3];
            assert!(value > 0.0, "Cholesky diagonal must be positive");
        }
        let recon = reconstruct_from_lower(&l);
        tensor_close(&recon, &a, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_two_output_lower_variant() {
        let a = Matrix::new(
            vec![
                9.0, 3.0, 3.0, //
                3.0, 5.0, 1.0, //
                3.0, 1.0, 7.0,
            ],
            vec![3, 3],
        )
        .unwrap();
        let eval = evaluate(Value::Tensor(a.clone()), &[Value::from("lower")]).expect("chol eval");
        assert_eq!(eval.flag_index(), 0);
        assert_eq!(eval.triangle(), CholTriangle::Lower);
        let factor = tensor_from_value(eval.factor());
        let recon = reconstruct_from_lower(&factor);
        tensor_close(&recon, &a, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_two_output_reports_failure() {
        let a = Matrix::new(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2]).expect("matrix");
        let eval = evaluate(Value::Tensor(a), &[]).expect("chol eval");
        assert_eq!(eval.flag_index(), 2);
        let factor = tensor_from_value(eval.factor());
        assert_eq!(factor.shape(), [2, 2]);
        assert!((factor.materialize_f64()[0] - 1.0).abs() < 1e-12);
        assert!((factor.materialize_f64()[1] - 0.0).abs() < 1e-12);
        assert!((factor.materialize_f64()[2] - 2.0).abs() < 1e-12);
        assert!((factor.materialize_f64()[3] - 0.0).abs() < 1e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_single_output_errors_on_failure() {
        let a = Matrix::new(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2]).expect("matrix");
        let err = chol_builtin(Value::Tensor(a), Vec::new()).unwrap_err();
        assert_eq!(
            err.identifier(),
            CHOL_ERROR_NOT_POSITIVE_DEFINITE.identifier
        );
        let err = error_message(err);
        assert!(err.contains("positive definite"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_invalid_option_errors() {
        let a = Matrix::new(vec![4.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap();
        let err = chol_builtin(Value::Tensor(a), vec![Value::from("diagonal")]).unwrap_err();
        assert_eq!(err.identifier(), CHOL_ERROR_INVALID_ARGUMENT.identifier);
        let err = error_message(err);
        assert!(err.to_ascii_lowercase().contains("unknown option"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_non_square_errors() {
        let a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let err = chol_builtin(Value::Tensor(a), Vec::new()).unwrap_err();
        assert_eq!(err.identifier(), CHOL_ERROR_INVALID_INPUT.identifier);
        let err = error_message(err);
        assert!(err.to_ascii_lowercase().contains("square"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_empty_matrix_returns_empty() {
        let empty = Matrix::new(Vec::<f64>::new(), vec![0, 0]).unwrap();
        let eval = evaluate(Value::Tensor(empty.clone()), &[]).expect("chol eval");
        assert_eq!(eval.flag_index(), 0);
        let factor = tensor_from_value(eval.factor());
        assert_eq!(factor.shape(), [0, 0]);
        assert!(factor.materialize_f64().is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_uses_only_the_selected_triangle() {
        let upper = Matrix::new(vec![4.0, 999.0, 1.0, 3.0], vec![2, 2]).expect("matrix");
        let upper_eval = evaluate(Value::Tensor(upper), &[]).expect("upper chol");
        assert_eq!(upper_eval.flag_index(), 0);
        let upper_factor = tensor_from_value(upper_eval.factor());
        let expected = Matrix::new(vec![4.0, 1.0, 1.0, 3.0], vec![2, 2]).expect("expected");
        tensor_close(&reconstruct_from_upper(&upper_factor), &expected, 1e-10);

        let lower = Matrix::new(vec![4.0, 1.0, 999.0, 3.0], vec![2, 2]).expect("matrix");
        let lower_eval =
            evaluate(Value::Tensor(lower), &[Value::from("lower")]).expect("lower chol");
        assert_eq!(lower_eval.flag_index(), 0);
        let lower_factor = tensor_from_value(lower_eval.factor());
        tensor_close(&reconstruct_from_lower(&lower_factor), &expected, 1e-10);
    }

    #[test]
    fn chol_integer_extension_covers_all_classes_and_guards_wide_values() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let storages = vec![
            IntegerStorage::I8(vec![4, 0, 0, 9]),
            IntegerStorage::I16(vec![4, 0, 0, 9]),
            IntegerStorage::I32(vec![4, 0, 0, 9]),
            IntegerStorage::I64(vec![4, 0, 0, 9]),
            IntegerStorage::U8(vec![4, 0, 0, 9]),
            IntegerStorage::U16(vec![4, 0, 0, 9]),
            IntegerStorage::U32(vec![4, 0, 0, 9]),
            IntegerStorage::U64(vec![4, 0, 0, 9]),
        ];
        for storage in storages {
            let input = Matrix::new_integer(storage, vec![2, 2]).expect("integer matrix");
            let factor = chol_builtin(Value::Tensor(input), Vec::new()).expect("integer chol");
            let factor = tensor_from_value(factor);
            assert_eq!(factor.numeric_dtype(), NumericDType::F64);
            assert_eq!(factor.materialize_f64(), vec![2.0, 0.0, 0.0, 3.0]);
        }

        let wide = Matrix::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1])
            .expect("wide integer");
        let err = chol_builtin(Value::Tensor(wide), Vec::new()).expect_err("wide must reject");
        assert_eq!(err.identifier(), CHOL_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("exactly representable"));
    }

    #[test]
    fn chol_integer_extension_rejects_in_matlab_mode() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let input =
            Matrix::new_integer(IntegerStorage::U8(vec![4]), vec![1, 1]).expect("integer scalar");
        let err = chol_builtin(Value::Tensor(input), Vec::new()).expect_err("extension gate");
        assert_eq!(
            err.identifier(),
            CHOL_INTEGER_INPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn chol_preserves_native_single_output() {
        let input = Matrix::from_f32(vec![4.0, 0.0, 0.0, 9.0], vec![2, 2]).expect("single");
        let factor = chol_builtin(Value::Tensor(input), Vec::new()).expect("single chol");
        let factor = tensor_from_value(factor);
        assert_eq!(factor.numeric_dtype(), NumericDType::F32);
        assert_eq!(factor.as_f32_slice(), Some(&[2.0, 0.0, 0.0, 3.0][..]));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_logical_input_factorizes() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).expect("logical array");
        let result = chol_builtin(Value::LogicalArray(logical), Vec::new()).expect("chol");
        let factor = tensor_from_value(result);
        let recon = reconstruct_from_upper(&factor);
        let identity = Matrix::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        tensor_close(&recon, &identity, 1e-12);
    }

    #[test]
    fn chol_logical_extension_rejects_before_resident_provider_access() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).expect("logical array");
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let err = chol_builtin(Value::LogicalArray(logical.clone()), Vec::new())
                .expect_err("logical host input must be gated");
            assert_eq!(
                err.identifier(),
                CHOL_LOGICAL_INPUT_EXTENSION.error_identifier
            );
        }
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &logical.data,
                    shape: &logical.shape,
                })
                .expect("logical upload");
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let err = match evaluate(Value::GpuTensor(handle.clone()), &[]) {
                Err(err) => err,
                Ok(_) => panic!("resident logical input must gate before provider dispatch"),
            };
            assert_eq!(
                err.identifier(),
                CHOL_LOGICAL_INPUT_EXTENSION.error_identifier
            );
            assert!(runmat_accelerate_api::handle_is_logical(&handle));
            drop(handle);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_complex_positive_definite() {
        let complex = ComplexTensor::new(
            vec![(5.0, 0.0), (1.0, 2.0), (1.0, -2.0), (4.0, 0.0)],
            vec![2, 2],
        )
        .unwrap();
        let eval = evaluate(Value::ComplexTensor(complex.clone()), &[]).expect("chol eval");
        assert_eq!(eval.flag_index(), 0);
        let factor = complex_tensor_from_value(eval.factor());
        let recon = reconstruct_complex_upper(&factor);
        complex_tensor_close(&recon, &complex, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_complex_lower_variant() {
        let complex = ComplexTensor::new(
            vec![(5.0, 0.0), (1.0, 2.0), (1.0, -2.0), (4.0, 0.0)],
            vec![2, 2],
        )
        .unwrap();
        let eval = evaluate(
            Value::ComplexTensor(complex.clone()),
            &[Value::from("lower")],
        )
        .expect("chol eval");
        assert_eq!(eval.flag_index(), 0);
        let factor = complex_tensor_from_value(eval.factor());
        let recon = reconstruct_complex_lower(&factor);
        complex_tensor_close(&recon, &complex, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let a = Matrix::new(vec![6.0, 2.0, 2.0, 5.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &a.materialize_f64(),
                shape: a.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = chol_builtin(Value::GpuTensor(handle), Vec::new()).expect("chol");
            let gathered = test_support::gather(result).expect("gather");
            let recon = reconstruct_from_upper(&gathered);
            tensor_close(&recon, &a, 1e-10);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_gpu_failure_flag() {
        test_support::with_test_provider(|provider| {
            let a = Matrix::new(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &a.materialize_f64(),
                shape: a.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let eval = evaluate(Value::GpuTensor(handle), &[]).expect("chol eval");
            assert_eq!(eval.flag_index(), 2);
            let factor = eval.factor();
            assert!(matches!(factor, Value::GpuTensor(_)));
            let gathered = test_support::gather(factor).expect("gather factor");
            assert!((gathered.materialize_f64()[0] - 1.0).abs() < 1e-12);
            assert!((gathered.materialize_f64()[1] - 0.0).abs() < 1e-12);
            assert!((gathered.materialize_f64()[2] - 2.0).abs() < 1e-12);
            assert!((gathered.materialize_f64()[3] - 0.0).abs() < 1e-12);
        });
    }

    #[test]
    fn chol_resident_integer_gates_before_provider_and_falls_back_to_double() {
        test_support::with_test_provider(|provider| {
            let input = Matrix::new_integer(IntegerStorage::U16(vec![4, 0, 0, 9]), vec![2, 2])
                .expect("integer matrix");
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("integer upload");
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let err = match evaluate(Value::GpuTensor(handle.clone()), &[]) {
                    Err(err) => err,
                    Ok(_) => panic!("strict mode must reject before provider dispatch"),
                };
                assert_eq!(
                    err.identifier(),
                    CHOL_INTEGER_INPUT_EXTENSION.error_identifier
                );
            }
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let eval = evaluate(Value::GpuTensor(handle), &[]).expect("resident integer chol");
            let Value::GpuTensor(output) = eval.factor() else {
                panic!("resident fallback must restore factor residency");
            };
            assert_eq!(runmat_accelerate_api::handle_integer_type(&output), None);
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(runmat_accelerate_api::ProviderPrecision::F64)
            );
            let gathered = test_support::gather(Value::GpuTensor(output)).expect("gather output");
            assert_eq!(gathered.materialize_f64(), vec![2.0, 0.0, 0.0, 3.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn chol_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("register wgpu provider");

        let tol = match runmat_accelerate_api::provider()
            .expect("provider")
            .precision()
        {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };

        let tensor = Matrix::new(
            vec![
                10.0, 2.0, 3.0, //
                2.0, 9.0, 1.0, //
                3.0, 1.0, 7.0,
            ],
            vec![3, 3],
        )
        .unwrap();

        let host_eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("host eval");
        let host_factor = tensor_from_value(host_eval.factor());

        let provider = runmat_accelerate_api::provider().expect("provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");

        let gpu_eval = evaluate(Value::GpuTensor(handle), &[]).expect("gpu eval");
        assert_eq!(gpu_eval.flag_index(), 0, "gpu chol should succeed");
        let gpu_factor = test_support::gather(gpu_eval.factor()).expect("gather factor");

        tensor_close(&gpu_factor, &host_factor, tol);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_accepts_scalar() {
        let result = chol_builtin(Value::Num(9.0), Vec::new()).expect("chol");
        match result {
            Value::Num(n) => assert!((n - 3.0).abs() < 1e-12),
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 1]);
                assert!((t.materialize_f64()[0] - 3.0).abs() < 1e-12);
            }
            other => panic!("expected scalar-like, got {other:?}"),
        }
    }

    fn chol_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let rest = crate::execution::RootedValues::new(rest)
            .map_err(|error| error.into_runtime_error())?;
        block_on(super::chol_builtin(
            crate::context::legacy::active(),
            value,
            rest.into(),
        ))
    }

    fn evaluate(value: Value, args: &[Value]) -> BuiltinResult<CholEval> {
        block_on(super::evaluate(
            value,
            crate::call::FunctionArgumentView::from_slice(args),
        ))
    }
}
