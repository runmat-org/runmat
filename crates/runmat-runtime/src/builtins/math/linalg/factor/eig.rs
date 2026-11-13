//! MATLAB-compatible `eig` builtin with host and GPU-aware semantics.
//!
//! Implements the dense eigenvalue decomposition for real and complex matrices,
//! including the vector-only form and general complex `eig(A,B)` pencils,
//! including singular `B` with infinite or indeterminate eigenvalues,
//! the `[V,D]` factorisation, and the three-output `[V,D,W]` variant that
//! returns left eigenvectors. GPU inputs are currently gathered back to the host
//! unless their owning provider implements the reserved `eig` hook; host
//! fallback outputs are restored to that owner when possible.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg(test)]
use crate::builtins::math::linalg::type_resolvers::eig_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
#[cfg(test)]
use nalgebra::DMatrix;
use num_complex::Complex64;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{BuiltinErrorDescriptor, FactorOption};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::{ComplexTensor, Tensor};
use runmat_value::{IntValue, NumericDType, NumericScalar, Value};

#[path = "eig_publication.rs"]
mod publication;
use publication::upload_eig_eval_to_owner_in_context;
#[path = "eig_resident.rs"]
mod resident;
use resident::gather_eig_resident_value_in_context;
#[path = "eig_host.rs"]
pub(super) mod host;
#[path = "eig_work.rs"]
pub(super) mod work;
use crate::context::RuntimeContext;

const BUILTIN_NAME: &str = "eig";

#[cfg(test)]
use runmat_builtins::{EIG_DESCRIPTOR, EIG_EXTENSIONS};
use runmat_builtins::{
    EIG_ERROR_INTERNAL, EIG_ERROR_INVALID_ARGUMENT, EIG_ERROR_INVALID_INPUT,
    EIG_NONFLOATING_COEFFICIENT_EXTENSION,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::factor::eig")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "eig",
    op_kind: GpuOpKind::Custom("eig-factor"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("eig")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Genuine provider eig hooks may execute supported calls. WGPU and unsupported forms use the canonical contextual host EVD and exact-owner restoration of real or complex factors in the source floating class.",
};

fn eig_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    eig_error_with_message(error.message, error)
}

fn eig_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn eig_invalid_argument(message: impl Into<String>) -> RuntimeError {
    eig_error_with_message(message, &EIG_ERROR_INVALID_ARGUMENT)
}

fn eig_invalid_input(message: impl Into<String>) -> RuntimeError {
    eig_error_with_message(message, &EIG_ERROR_INVALID_INPUT)
}

fn eig_internal_error(message: impl Into<String>) -> RuntimeError {
    eig_error_with_message(message, &EIG_ERROR_INTERNAL)
}

fn with_eig_context(mut error: RuntimeError) -> RuntimeError {
    if error.message() == "interaction pending..." {
        return build_runtime_error("interaction pending...")
            .with_builtin(BUILTIN_NAME)
            .build();
    }
    if error.context.builtin.is_none() {
        error.context = error.context.with_builtin(BUILTIN_NAME);
    }
    error
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::factor::eig")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "eig",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Eigenvalue decomposition executes eagerly and never participates in fusion.",
};

#[runtime_builtin(
    name = "eig",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::eig",
    context = true
)]
async fn eig_builtin_context(
    runtime: Option<RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let requested = crate::output_count::current_output_count();
    let eval = evaluate_in_context(
        runtime.as_ref(),
        value,
        rest.view(),
        requested.is_some_and(|n| n >= 3),
    )
    .await?;
    match requested {
        None => Ok(eval.eigenvalues()),
        Some(0) => work::output_list(runtime.as_ref(), [None; 0], 0).await,
        Some(1) => work::output_list(runtime.as_ref(), [Some(eval.eigenvalues())], 1).await,
        Some(2) => {
            work::output_list(
                runtime.as_ref(),
                [Some(eval.right()), Some(eval.diagonal())],
                2,
            )
            .await
        }
        Some(3) => {
            work::output_list(
                runtime.as_ref(),
                [
                    Some(eval.right()),
                    Some(eval.diagonal()),
                    Some(eval.left()?),
                ],
                3,
            )
            .await
        }
        Some(_) => Err(eig_error(&EIG_ERROR_INVALID_ARGUMENT)),
    }
}

#[derive(Clone, Debug)]
pub struct EigEval {
    eigenvalues: Value,
    diagonal_matrix: Value,
    diagonal_output: Value,
    right: Value,
    left: Option<Value>,
}

impl EigEval {
    pub fn eigenvalues(&self) -> Value {
        self.eigenvalues.clone()
    }

    pub fn diagonal(&self) -> Value {
        self.diagonal_output.clone()
    }

    pub fn diagonal_matrix(&self) -> Value {
        self.diagonal_matrix.clone()
    }

    pub fn right(&self) -> Value {
        self.right.clone()
    }

    pub fn left(&self) -> BuiltinResult<Value> {
        self.left.clone().ok_or_else(|| {
            eig_internal_error(
                "eig: left eigenvectors are not available from the active acceleration provider",
            )
        })
    }

    fn from_provider(
        mut result: runmat_accelerate_api::ProviderEigResult,
        input: &GpuTensorHandle,
    ) -> BuiltinResult<Self> {
        let adoption = (|| {
            for output in [
                &mut result.eigenvalues,
                &mut result.diagonal,
                &mut result.right,
            ]
            .into_iter()
            .chain(result.left.iter_mut())
            {
                gpu_helpers::propagate_output_provenance(output, [input])?;
                output.allocation_lease().validate_live().map_err(|error| {
                    build_runtime_error(format!("eig: cannot adopt provider output: {error}"))
                        .with_source(error)
                        .build()
                })?;
                runmat_accelerate_api::mark_residency(output);
            }
            Ok::<(), RuntimeError>(())
        })();
        if let Err(error) = adoption {
            return Err(reject_provider_eig_result(result, error));
        }
        let runmat_accelerate_api::ProviderEigResult {
            eigenvalues,
            diagonal,
            right,
            left,
        } = result;
        Ok(EigEval {
            eigenvalues: Value::GpuTensor(eigenvalues),
            diagonal_matrix: Value::GpuTensor(diagonal.clone()),
            diagonal_output: Value::GpuTensor(diagonal),
            right: Value::GpuTensor(right),
            left: left.map(Value::GpuTensor),
        })
    }
}

#[derive(Clone, Copy)]
struct EigOptions {
    balance: bool,
    vector_output: bool,
}

impl Default for EigOptions {
    fn default() -> Self {
        Self {
            balance: true,
            vector_output: false,
        }
    }
}

#[derive(Clone)]
struct EigRequest {
    b: Option<Value>,
    options: EigOptions,
}

pub async fn evaluate(
    value: Value,
    args: crate::call::FunctionArgumentView<'_>,
    require_left: bool,
) -> BuiltinResult<EigEval> {
    evaluate_in_context(None, value, args, require_left).await
}

/// Explicit execution authority for Runtime-domain callers. The embedding
/// wrapper above grants no ambient finite-allocation authority.
pub async fn evaluate_in_context(
    runtime: Option<&RuntimeContext>,
    value: Value,
    args: crate::call::FunctionArgumentView<'_>,
    require_left: bool,
) -> BuiltinResult<EigEval> {
    if let Some(runtime) = runtime {
        runtime
            .cpu_cooperation()
            .checkpoint(0)
            .await
            .map_err(|e| e.into_runtime_error_in_context(Some(runtime)))?;
    }
    let request = parse_request(runtime, args)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, BUILTIN_NAME)?;
    ensure_eig_coefficient_extension_enabled(&value)?;
    if let Some(b) = request.b {
        crate::builtins::common::validation::reject_typed_complex_integer(&b, BUILTIN_NAME)?;
        ensure_eig_coefficient_extension_enabled(&b)?;
        gpu_helpers::require_resident_compute_values(&[&value, &b], false, || {
            eig_internal_error("eig: no acceleration provider owns the input handle")
        })?;
        return evaluate_generalized(runtime, value, b, request.options, require_left).await;
    }
    let options = request.options;
    match value {
        Value::GpuTensor(handle) => {
            gpu_helpers::require_resident_compute_owner(&handle, || {
                eig_internal_error("eig: no acceleration provider owns the input handle")
            })?;
            if is_nonfloating_coefficient(&Value::GpuTensor(handle.clone())) {
                let gathered = gather_eig_resident_value_in_context(runtime, &handle).await?;
                ensure_exact_eig_integer_boundary(&gathered, "A")?;
                let eval = evaluate_host(runtime, gathered, options, require_left).await?;
                return upload_eig_eval_to_owner_in_context(
                    runtime,
                    eval,
                    &handle,
                    options.vector_output,
                    &[],
                )
                .await;
            }
            if let Some(eval) = evaluate_gpu(runtime, &handle, options, require_left).await? {
                return Ok(eval);
            }
            let gathered = gather_eig_resident_value_in_context(runtime, &handle).await?;
            let eval = evaluate_host(runtime, gathered, options, require_left).await?;
            upload_eig_eval_to_owner_in_context(runtime, eval, &handle, options.vector_output, &[])
                .await
        }
        other => {
            ensure_exact_eig_integer_boundary(&other, "A")?;
            evaluate_host(runtime, other, options, require_left).await
        }
    }
}

async fn evaluate_gpu(
    runtime: Option<&RuntimeContext>,
    handle: &GpuTensorHandle,
    options: EigOptions,
    require_left: bool,
) -> BuiltinResult<Option<EigEval>> {
    if options.vector_output {
        return Ok(None);
    }
    if !options.balance {
        return Ok(None);
    }
    use crate::context::host_allocation::{HostAllocationPolicy, HostAllocationReceipt};
    if HostAllocationReceipt::policy(runtime).map_err(|source| {
        with_eig_context(
            crate::context::host_allocation::host_allocation_runtime_error(
                runtime,
                "RunMat:eig:Workspace",
                "eig: provider hook allocation policy",
                source,
            ),
        )
    })? == HostAllocationPolicy::Managed
    {
        // The opaque legacy provider continuation has no finite allocation
        // quote. Decline before creating it; canonical contextual fallback
        // retains genuine owner/class semantics without inventing a grant.
        return Ok(None);
    }
    let provider = match resolved_actual_eig_owner(handle) {
        Some(p) => p,
        None => return Ok(None),
    };
    let Some(order) = eig_gpu_order(handle) else {
        return Ok(None);
    };
    let _residency = gpu_helpers::ResidencyRestoreGuard::new(handle);
    match provider.eig(handle, require_left).await {
        Ok(result) => {
            if valid_provider_eig_result(&result, provider, handle, order, require_left) {
                Ok(Some(EigEval::from_provider(result, handle)?))
            } else {
                Err(reject_provider_eig_result(
                    result,
                    eig_internal_error("eig: provider returned invalid factor outputs"),
                ))
            }
        }
        Err(error) if error.is_unsupported() => Ok(None),
        Err(source) => Err(crate::runtime_error::with_provider_execution_source(
            eig_internal_error(source.to_string()),
            source,
        )),
    }
}

fn eig_gpu_order(handle: &GpuTensorHandle) -> Option<usize> {
    match handle.shape() {
        [rows, cols] if rows == cols => Some(*rows),
        _ => None,
    }
}

fn resolved_actual_eig_owner(
    handle: &GpuTensorHandle,
) -> Option<&'static dyn runmat_accelerate_api::AccelProvider> {
    gpu_helpers::exact_provider_for_handle(handle)
}

fn gpu_handles_alias(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    lhs.shares_allocation_with(rhs)
}

fn valid_provider_eig_result(
    result: &runmat_accelerate_api::ProviderEigResult,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    order: usize,
    require_left: bool,
) -> bool {
    if require_left && result.left.is_none() {
        return false;
    }
    let Some(expected_precision) = runmat_accelerate_api::handle_precision(input) else {
        return false;
    };
    let expected_class = match expected_precision {
        runmat_accelerate_api::ProviderPrecision::F32 => runmat_types::standard::SINGLE,
        runmat_accelerate_api::ProviderPrecision::F64 => runmat_types::standard::DOUBLE,
    };
    let handles = [
        Some(&result.eigenvalues),
        Some(&result.diagonal),
        Some(&result.right),
        result.left.as_ref(),
    ];
    let expected_shapes = [[order, 1], [order, order], [order, order], [order, order]];
    for (index, (handle, expected_shape)) in handles.iter().zip(expected_shapes).enumerate() {
        let Some(handle) = handle else {
            continue;
        };
        let storage = runmat_accelerate_api::handle_storage(handle);
        if gpu_handles_alias(handle, input)
            || handles[..index].iter().flatten().any(|prior| {
                prior.device_id() == handle.device_id() && prior.buffer_id() == handle.buffer_id()
            })
            || handle.device_id() != input.device_id()
            || handle.shape() != expected_shape
            || runmat_accelerate_api::handle_is_transposed(handle)
            || resolved_actual_eig_owner(handle)
                .is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
            || !matches!(
                storage,
                runmat_accelerate_api::GpuTensorStorage::Real
                    | runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            )
            || runmat_accelerate_api::handle_precision(handle) != Some(expected_precision)
            || runmat_accelerate_api::handle_class_identity(handle)
                .is_some_and(|class_identity| !class_identity.is(expected_class))
            || runmat_accelerate_api::handle_integer_type(handle).is_some()
            || runmat_accelerate_api::handle_is_logical(handle)
        {
            return false;
        }
    }
    true
}

fn reject_provider_eig_result(
    result: runmat_accelerate_api::ProviderEigResult,
    mut error: RuntimeError,
) -> RuntimeError {
    for handle in [result.eigenvalues, result.diagonal, result.right]
        .into_iter()
        .chain(result.left)
    {
        error = gpu_helpers::reject_gpu_output(handle, error);
    }
    error
}

async fn evaluate_host(
    runtime: Option<&RuntimeContext>,
    value: Value,
    options: EigOptions,
    require_left: bool,
) -> BuiltinResult<EigEval> {
    let precision = value_precision(&value);
    let pairs = host::compute(runtime, &value, None, require_left, false).await?;
    host::publish(runtime, &pairs, precision, options).await
}

async fn evaluate_generalized(
    runtime: Option<&RuntimeContext>,
    a: Value,
    b: Value,
    options: EigOptions,
    require_left: bool,
) -> BuiltinResult<EigEval> {
    let sources = [&a, &b].map(|value| match value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    });
    let a = gather_eig_coefficient(runtime, a).await?;
    let b = gather_eig_coefficient(runtime, b).await?;
    ensure_exact_eig_integer_boundary(&a, "A")?;
    ensure_exact_eig_integer_boundary(&b, "B")?;
    let precision = combined_value_precision(&a, &b);
    let pairs = host::compute(runtime, &a, Some(&b), require_left, false).await?;
    let eval = host::publish(runtime, &pairs, precision, options).await?;
    match sources {
        [Some(a), Some(b)] => {
            upload_eig_eval_to_owner_in_context(
                runtime,
                eval,
                &a,
                options.vector_output,
                &[a.clone(), b],
            )
            .await
        }
        [Some(source), None] | [None, Some(source)] => {
            upload_eig_eval_to_owner_in_context(
                runtime,
                eval,
                &source,
                options.vector_output,
                std::slice::from_ref(&source),
            )
            .await
        }
        [None, None] => Ok(eval),
    }
}

async fn gather_eig_coefficient(
    runtime: Option<&RuntimeContext>,
    value: Value,
) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => gather_eig_resident_value_in_context(runtime, &handle).await,
        other => Ok(other),
    }
}

fn ensure_eig_coefficient_extension_enabled(value: &Value) -> BuiltinResult<()> {
    if is_nonfloating_coefficient(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EIG_NONFLOATING_COEFFICIENT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn is_nonfloating_coefficient(value: &Value) -> bool {
    match value {
        Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_integer_type(handle).is_some()
                || runmat_accelerate_api::handle_is_logical(handle)
        }
        _ => false,
    }
}

fn ensure_exact_eig_integer_boundary(value: &Value, role: &str) -> BuiltinResult<()> {
    const MAX_EXACT_INTEGER: i128 = 1_i128 << 53;
    let check = |exact: i128| {
        if (-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&exact) {
            Ok(())
        } else {
            Err(eig_invalid_input(format!(
                "eig: integer {role} coefficients must be exactly representable as double"
            )))
        }
    };
    match value {
        Value::Int(value) => check(int_value_i128(value)),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            for index in 0..tensor.len() {
                if let Some(exact) = tensor.numeric_value_at(index).and_then(numeric_scalar_i128) {
                    check(exact)?;
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn int_value_i128(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => i128::from(*value),
        IntValue::I16(value) => i128::from(*value),
        IntValue::I32(value) => i128::from(*value),
        IntValue::I64(value) => i128::from(*value),
        IntValue::U8(value) => i128::from(*value),
        IntValue::U16(value) => i128::from(*value),
        IntValue::U32(value) => i128::from(*value),
        IntValue::U64(value) => i128::from(*value),
    }
}

fn numeric_scalar_i128(value: NumericScalar) -> Option<i128> {
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

fn value_precision(value: &Value) -> NumericDType {
    match value {
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => NumericDType::F32,
        Value::ComplexTensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            NumericDType::F32
        }
        Value::GpuTensor(handle) => match runmat_accelerate_api::handle_precision(handle) {
            Some(runmat_accelerate_api::ProviderPrecision::F32) => NumericDType::F32,
            _ => NumericDType::F64,
        },
        _ => NumericDType::F64,
    }
}

fn combined_value_precision(a: &Value, b: &Value) -> NumericDType {
    if value_precision(a) == NumericDType::F32 && value_precision(b) == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    }
}

fn parse_request(
    runtime: Option<&RuntimeContext>,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<EigRequest> {
    let mut opts = EigOptions::default();
    let mut b = None;
    for (idx, arg) in args.iter().enumerate() {
        if let Some(text) = option_text(runtime, arg)? {
            match FactorOption::parse(&text) {
                Some(FactorOption::Balance) => opts.balance = true,
                Some(FactorOption::NoBalance) => opts.balance = false,
                Some(FactorOption::Vector) => opts.vector_output = true,
                Some(FactorOption::Matrix) => opts.vector_output = false,
                _ => {
                    return Err(eig_invalid_argument(format!(
                        "eig: unknown option '{}'",
                        text.trim().to_ascii_lowercase()
                    )));
                }
            }
        } else if idx == 0 {
            b = Some(arg.clone());
        } else {
            return Err(eig_invalid_argument(
                "eig: option arguments must be character vectors or string scalars",
            ));
        }
    }
    Ok(EigRequest { b, options: opts })
}

fn option_text(
    runtime: Option<&RuntimeContext>,
    value: &Value,
) -> BuiltinResult<Option<runmat_value::Utf8Text>> {
    match value {
        Value::String(text) => Ok(text.text_ref().cloned()),
        Value::StringArray(array) if array.data().len() == 1 => {
            Ok(array.data()[0].text_ref().cloned())
        }
        Value::StringArray(_) => Err(eig_invalid_argument(
            "eig: option arguments must be character vectors or string scalars",
        )),
        Value::CharArray(chars) if chars.rows <= 1 => {
            crate::call::argument_text::copy_chars(chars.data(), runtime)
                .map(Some)
                .map_err(|mut error| {
                    if error
                        .source
                        .as_ref()
                        .is_some_and(|source| source.is::<runmat_value::Utf16DecodeError>())
                    {
                        let source = error.source.take().expect("checked UTF16 cause");
                        let mut error = eig_invalid_argument(source.to_string());
                        error.source = Some(source);
                        error
                    } else {
                        with_eig_context(error)
                    }
                })
        }
        Value::CharArray(_) => Err(eig_invalid_argument(
            "eig: option arguments must be character vectors or string scalars",
        )),
        _ => Ok(None),
    }
}

#[cfg(test)]
fn tensor_to_matrix(value: &Tensor) -> BuiltinResult<DMatrix<Complex64>> {
    host::dimensions(&Value::Tensor(value.clone()), false)?;
    Ok(DMatrix::from_fn(value.rows(), value.cols(), |r, c| {
        Complex64::new(
            value
                .numeric_value_at(r + c * value.rows())
                .expect("test coefficient")
                .materialize_f64(),
            0.0,
        )
    }))
}
#[cfg(test)]
fn complex_tensor_to_matrix(value: &ComplexTensor) -> BuiltinResult<DMatrix<Complex64>> {
    host::dimensions(&Value::ComplexTensor(value.clone()), false)?;
    Ok(DMatrix::from_fn(value.rows(), value.cols(), |r, c| {
        let (re, im) = value
            .numeric_value_at(r + c * value.rows())
            .expect("test coefficient");
        Complex64::new(re.materialize_f64(), im.materialize_f64())
    }))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{builtin_catalog_entry_by_name, ResolveContext, Type};
    use runmat_value::{IntValue, IntegerStorage};

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    fn matrix_from_value(value: Value) -> DMatrix<Complex64> {
        match value {
            Value::Tensor(t) => tensor_to_matrix(&t).expect("matrix"),
            Value::ComplexTensor(ct) => complex_tensor_to_matrix(&ct).expect("matrix"),
            other => panic!("expected matrix value, got {other:?}"),
        }
    }

    #[test]
    fn eig_type_returns_eigenvalue_vector() {
        let out = eig_type(
            &[Type::Tensor {
                shape: Some(vec![Some(3), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(1)])
            }
        );
    }

    #[test]
    fn eig_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = EIG_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"d = eig(A)"));
        assert!(labels.contains(&"d = eig(A, options...)"));
        assert!(labels.contains(&"[V, D] = eig(A)"));
        assert!(labels.contains(&"[V, D] = eig(A, options...)"));
        assert!(labels.contains(&"[V, D, W] = eig(A)"));
        assert!(labels.contains(&"[V, D, W] = eig(A, options...)"));
        assert!(labels.contains(&"d = eig(A, B)"));
        assert!(labels.contains(&"d = eig(A, B, options...)"));
        assert!(labels.contains(&"[V, D] = eig(A, B)"));
        assert!(labels.contains(&"[V, D] = eig(A, B, options...)"));
        assert!(labels.contains(&"[V, D, W] = eig(A, B)"));
        assert!(labels.contains(&"[V, D, W] = eig(A, B, options...)"));
    }

    #[test]
    fn eig_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = EIG_DESCRIPTOR.errors.iter().map(|err| err.code).collect();
        assert!(codes.contains(&"RM.EIG.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.EIG.INVALID_INPUT"));
        assert!(codes.contains(&"RM.EIG.INTERNAL"));
    }

    #[test]
    fn eig_provider_results_require_owner_shape_precision_storage_and_unique_handles() {
        use runmat_accelerate_api::{HostTensorView, ProviderEigResult};

        test_support::with_test_provider(|provider| {
            let upload = |data: &[f64], shape: &[usize]| {
                provider
                    .upload(&HostTensorView { data, shape })
                    .expect("upload")
            };
            let input = upload(&[1.0, 0.0, 0.0, 2.0], &[2, 2]);
            let eigenvalues = upload(&[1.0, 2.0], &[2, 1]);
            let diagonal = upload(&[1.0, 0.0, 0.0, 2.0], &[2, 2]);
            let right = upload(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
            let left = upload(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
            let valid = ProviderEigResult {
                eigenvalues: eigenvalues.clone(),
                diagonal: diagonal.clone(),
                right: right.clone(),
                left: Some(left.clone()),
            };
            assert!(valid_provider_eig_result(&valid, provider, &input, 2, true));

            let mut duplicate = valid.clone();
            duplicate.right = duplicate.diagonal.clone();
            assert!(!valid_provider_eig_result(
                &duplicate, provider, &input, 2, true
            ));

            let mut wrong_shape = valid.clone();
            wrong_shape.eigenvalues = wrong_shape
                .eigenvalues
                .with_view(runmat_accelerate_api::GpuTensorView::contiguous(vec![1, 2]))
                .unwrap();
            assert!(!valid_provider_eig_result(
                &wrong_shape,
                provider,
                &input,
                2,
                true
            ));

            let mut wrong_class = valid.clone();
            wrong_class.right = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &[1, 0, 0, 1],
                    shape: &[2, 2],
                })
                .unwrap();
            assert!(!valid_provider_eig_result(
                &wrong_class,
                provider,
                &input,
                2,
                true
            ));
            wrong_class.right = provider
                .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                    data: runmat_accelerate_api::HostNumericDataView::U8(&[1, 0, 0, 1]),
                    shape: &[2, 2],
                    storage: runmat_accelerate_api::GpuTensorStorage::Real,
                })
                .unwrap();
            assert!(!valid_provider_eig_result(
                &wrong_class,
                provider,
                &input,
                2,
                true
            ));
            assert!(valid_provider_eig_result(&valid, provider, &input, 2, true));

            let alias = upload(&[3.0, 4.0], &[2, 1]);
            let cleanup = ProviderEigResult {
                eigenvalues: input.clone(),
                diagonal: alias.clone(),
                right: alias.clone(),
                left: None,
            };
            let error = reject_provider_eig_result(cleanup, eig_internal_error("invalid output"));
            assert!(error.message().contains("invalid output"));
            assert!(block_on(provider.download(&input)).is_ok());
            assert!(
                block_on(provider.download(&alias)).is_ok(),
                "rejecting result references cannot retire another retained owner"
            );
        });
    }

    #[test]
    fn eig_matrix_conversion_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3, 4, 5, 6]), vec![3, 2])
            .expect("typed integer tensor");

        let matrix = tensor_to_matrix(&tensor).expect("matrix");
        assert_eq!(matrix.nrows(), 3);
        assert_eq!(matrix.ncols(), 2);
        assert_eq!(matrix[(0, 0)], Complex64::new(1.0, 0.0));
        assert_eq!(matrix[(1, 0)], Complex64::new(2.0, 0.0));
        assert_eq!(matrix[(2, 0)], Complex64::new(3.0, 0.0));
        assert_eq!(matrix[(0, 1)], Complex64::new(4.0, 0.0));
        assert_eq!(matrix[(1, 1)], Complex64::new(5.0, 0.0));
        assert_eq!(matrix[(2, 1)], Complex64::new(6.0, 0.0));
    }

    #[test]
    fn eig_nonfloating_coefficients_follow_strict_mode_and_exact_boundary() {
        let integer_matrix = || {
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![1, 0, 0, 2]), vec![2, 2])
                    .expect("integer matrix"),
            )
        };
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = evaluate(integer_matrix(), &[], false).expect_err("integer A must gate");
            assert_eq!(
                error.identifier(),
                EIG_NONFLOATING_COEFFICIENT_EXTENSION.error_identifier
            );
            let error = evaluate(Value::Num(2.0), &[Value::Int(IntValue::I16(1))], false)
                .expect_err("integer B must gate");
            assert_eq!(
                error.identifier(),
                EIG_NONFLOATING_COEFFICIENT_EXTENSION.error_identifier
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            assert!(evaluate(integer_matrix(), &[], false).is_ok());
            let wide = Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1])
                    .expect("wide matrix"),
            );
            let error = evaluate(wide, &[], false).expect_err("wide integer must reject");
            assert!(error.message().contains("exactly representable as double"));
        }
    }

    #[test]
    fn eig_admits_all_eight_small_integer_classes_with_double_outputs() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let storages = [
            IntegerStorage::I8(vec![2]),
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I64(vec![2]),
            IntegerStorage::U8(vec![2]),
            IntegerStorage::U16(vec![2]),
            IntegerStorage::U32(vec![2]),
            IntegerStorage::U64(vec![2]),
        ];
        for storage in storages {
            let input = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).unwrap());
            let output = evaluate(input, &[], false).unwrap().eigenvalues();
            assert!(matches!(output, Value::Num(_) | Value::Tensor(_)));
        }
    }

    #[test]
    fn eig_classifies_resident_integer_coefficients_before_provider_access() {
        let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::U16,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle;
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = evaluate(Value::GpuTensor(handle.clone()), &[], false)
            .expect_err("resident integer A must gate before provider access");
        assert_eq!(
            error.identifier(),
            EIG_NONFLOATING_COEFFICIENT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn eig_preserves_single_host_outputs_and_descriptor_policy() {
        let input = Value::Tensor(
            Tensor::from_f32(vec![2.0, 0.0, 0.0, 3.0], vec![2, 2]).expect("single matrix"),
        );
        let eval = evaluate(input, &[], false).expect("single eig");
        let Value::Tensor(values) = eval.eigenvalues() else {
            panic!("expected real single eigenvalues");
        };
        assert_eq!(values.numeric_dtype(), NumericDType::F32);
        let Value::Tensor(vectors) = eval.right() else {
            panic!("expected real single vectors");
        };
        assert_eq!(vectors.numeric_dtype(), NumericDType::F32);
        let scalar = evaluate(
            Value::Tensor(Tensor::from_f32(vec![8.], vec![1, 1]).unwrap()),
            &[],
            false,
        )
        .expect("single singleton eig");
        let Value::Tensor(singleton) = scalar.eigenvalues() else {
            panic!("single eigenvalue must not become double scalar")
        };
        assert_eq!(singleton.numeric_dtype(), NumericDType::F32);
        assert_eq!(singleton.shape(), [1, 1]);
        assert_eq!(singleton.as_f32_slice(), Some(&[8.][..]));
        let builtin =
            builtin_catalog_entry_by_name(BUILTIN_NAME).expect("registered eig catalog entry");
        assert_eq!(builtin.extensions, &EIG_EXTENSIONS);
        assert_eq!(builtin.integer_capabilities.len(), 2);
    }

    fn column_vector_from_value(value: Value) -> Vec<Complex64> {
        match value {
            Value::Num(v) => vec![Complex64::new(v, 0.0)],
            Value::Int(v) => vec![Complex64::new(v.to_f64(), 0.0)],
            Value::Complex(re, im) => vec![Complex64::new(re, im)],
            Value::Tensor(t) => t
                .materialize_f64()
                .iter()
                .map(|&v| Complex64::new(v, 0.0))
                .collect::<Vec<_>>(),
            Value::ComplexTensor(ct) => ct
                .materialize_f64()
                .iter()
                .map(|&(re, im)| Complex64::new(re, im))
                .collect::<Vec<_>>(),
            other => panic!("expected tensor for eigenvalues, got {other:?}"),
        }
    }

    fn assert_matrix_close(a: &DMatrix<Complex64>, b: &DMatrix<Complex64>, tol: f64) {
        assert_eq!(a.nrows(), b.nrows(), "row mismatch");
        assert_eq!(a.ncols(), b.ncols(), "column mismatch");
        for r in 0..a.nrows() {
            for c in 0..a.ncols() {
                let diff = (a[(r, c)] - b[(r, c)]).norm();
                assert!(
                    diff <= tol,
                    "matrix mismatch at ({r},{c}): {} vs {} (diff {diff})",
                    a[(r, c)],
                    b[(r, c)]
                );
            }
        }
    }

    fn assert_values_close_unordered(mut actual: Vec<Complex64>, mut expected: Vec<Complex64>) {
        actual.sort_by(|a, b| a.re.total_cmp(&b.re));
        expected.sort_by(|a, b| a.re.total_cmp(&b.re));
        assert_eq!(actual.len(), expected.len());
        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            assert!(
                (lhs - rhs).norm() <= 1e-10,
                "expected eigenvalue {rhs}, got {lhs}"
            );
        }
    }

    #[cfg(feature = "wgpu")]
    fn assert_tensor_close(a: &Tensor, b: &Tensor, tol: f64) {
        assert_eq!(
            a.shape(),
            b.shape(),
            "shape mismatch: {:?} vs {:?}",
            a.shape(),
            b.shape()
        );
        for (idx, (lhs, rhs)) in a
            .materialize_f64()
            .iter()
            .zip(b.materialize_f64().iter())
            .enumerate()
        {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= tol,
                "tensor mismatch at index {}: {} vs {} (diff {diff})",
                idx,
                lhs,
                rhs
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_scalar_real() {
        let result = eig_builtin(Value::Num(5.0), Vec::new()).expect("eig");
        match result {
            Value::Num(v) => assert!((v - 5.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_two_outputs_reconstruct() {
        let tensor = Tensor::new(vec![0.0, -2.0, 1.0, -3.0], vec![2, 2]).unwrap();
        let eval = evaluate(Value::Tensor(tensor.clone()), &[], false).expect("evaluate");
        let v = matrix_from_value(eval.right());
        let d = matrix_from_value(eval.diagonal_matrix());
        let a = matrix_from_value(Value::Tensor(tensor));
        let recon = &v * &d * v.try_inverse().unwrap();
        assert_matrix_close(&a, &recon, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_three_outputs_biorthogonality() {
        let tensor = Tensor::new(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[], true).expect("evaluate");
        let v = matrix_from_value(eval.right());
        let d = matrix_from_value(eval.diagonal_matrix());
        let w = matrix_from_value(eval.left().expect("left eigenvectors"));
        let vw = w.adjoint() * &v;
        let identity = DMatrix::<Complex64>::identity(v.ncols(), v.ncols());
        assert_matrix_close(&vw, &identity, 1e-10);
        let tensor = Tensor::new(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let a = matrix_from_value(Value::Tensor(tensor));
        let lhs = w.adjoint() * &a;
        let rhs = &d * w.adjoint();
        assert_matrix_close(&lhs, &rhs, 1e-10);

        // Non-real, nonnormal witnesses distinguish Hermitian pairing from
        // the old unconjugated dot product for both standard and pencil EVD.
        for generalized in [false, true] {
            let a = Value::ComplexTensor(
                ComplexTensor::new(vec![(1., 2.), (0., 0.), (2., -1.), (4., -3.)], vec![2, 2])
                    .unwrap(),
            );
            let b = Value::ComplexTensor(
                ComplexTensor::new(vec![(2., 1.), (0., 0.), (0., 0.), (3., -0.5)], vec![2, 2])
                    .unwrap(),
            );
            let args = if generalized {
                vec![b.clone()]
            } else {
                Vec::new()
            };
            let eval = evaluate(a.clone(), &args, true).expect("complex left eig");
            let a = matrix_from_value(a);
            let b = if generalized {
                matrix_from_value(b)
            } else {
                DMatrix::<Complex64>::identity(2, 2)
            };
            let v = matrix_from_value(eval.right());
            let d = matrix_from_value(eval.diagonal_matrix());
            let w = matrix_from_value(eval.left().expect("complex left factors"));
            assert_matrix_close(&(&a * &v), &(&b * &v * &d), 1e-10);
            assert_matrix_close(&(w.adjoint() * &a), &(&d * w.adjoint() * &b), 1e-10);
            assert_matrix_close(
                &(w.adjoint() * &b * &v),
                &DMatrix::<Complex64>::identity(2, 2),
                1e-10,
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_complex_matrix() {
        let data = vec![(1.0, 2.0), (0.0, 0.0), (2.0, -1.0), (0.0, -3.0)];
        let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let result = eig_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("eig");
        let values = column_vector_from_value(result);
        assert_eq!(values.len(), 2);
        assert!((values[0] - Complex64::new(1.0, 2.0)).norm() < 1e-10);
        assert!((values[1] - Complex64::new(0.0, -3.0)).norm() < 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_errors_on_non_square() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = eig_builtin(Value::Tensor(tensor), Vec::new()).unwrap_err();
        assert_eq!(err.identifier(), EIG_ERROR_INVALID_INPUT.identifier);
        let err = error_message(err);
        assert!(err.contains("square"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_accepts_nobalance_option() {
        let tensor = Tensor::new(vec![1.0, 1.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let base = eig_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("eig");
        let opt = eig_builtin(
            Value::Tensor(tensor),
            vec![Value::String("nobalance".into())],
        )
        .expect("eig with option");
        let base_vals = column_vector_from_value(base);
        let opt_vals = column_vector_from_value(opt);
        for (a, b) in base_vals.iter().zip(opt_vals.iter()) {
            assert!((a - b).norm() < 1e-10);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_vector_option_returns_column_vector() {
        let tensor = Tensor::new(vec![0.0, 1.0, -2.0, -3.0], vec![2, 2]).unwrap();
        let eval =
            evaluate(Value::Tensor(tensor), &[Value::from("vector")], false).expect("evaluate");
        match eval.diagonal() {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [2, 1]);
            }
            other => panic!("expected vector second output, got {other:?}"),
        }
        // Matrix form stays available for reconstruction.
        let _ = matrix_from_value(eval.diagonal_matrix());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_vector_option_allows_left_eigenvectors() {
        let tensor = Tensor::new(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let eval =
            evaluate(Value::Tensor(tensor), &[Value::from("vector")], true).expect("evaluate");
        match eval.diagonal() {
            Value::Tensor(t) => assert_eq!(t.shape(), [2, 1]),
            other => panic!("expected vector second output, got {other:?}"),
        }
        let left = eval.left().expect("left eigenvectors");
        match left {
            Value::Tensor(t) => assert_eq!(t.shape(), [2, 2]),
            Value::ComplexTensor(ct) => assert_eq!(ct.shape(), [2, 2]),
            other => panic!("unexpected type for left eigenvectors: {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_generalized_identity_matches_standard() {
        let a = Tensor::new(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let result = eig_builtin(Value::Tensor(a), vec![Value::Tensor(b)]).expect("eig(A,B)");
        let values = column_vector_from_value(result);
        assert_values_close_unordered(
            values,
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_generalized_diagonal_pair() {
        let a = Tensor::new(vec![2.0, 0.0, 0.0, 9.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 3.0], vec![2, 2]).unwrap();
        let result = eig_builtin(Value::Tensor(a), vec![Value::Tensor(b)]).expect("eig(A,B)");
        let values = column_vector_from_value(result);
        assert_values_close_unordered(
            values,
            vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_generalized_two_outputs_reconstruct() {
        let a_tensor = Tensor::new(vec![4.0, 0.0, 1.0, 9.0], vec![2, 2]).unwrap();
        let b_tensor = Tensor::new(vec![2.0, 0.0, 0.0, 3.0], vec![2, 2]).unwrap();
        let eval = evaluate(
            Value::Tensor(a_tensor.clone()),
            &[Value::Tensor(b_tensor.clone())],
            false,
        )
        .expect("evaluate");
        let a = matrix_from_value(Value::Tensor(a_tensor));
        let b = matrix_from_value(Value::Tensor(b_tensor));
        let v = matrix_from_value(eval.right());
        let d = matrix_from_value(eval.diagonal_matrix());
        let lhs = &a * &v;
        let rhs = &b * &v * &d;
        assert_matrix_close(&lhs, &rhs, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_generalized_three_outputs_satisfy_left_equation() {
        let a_tensor = Tensor::new(vec![4.0, 0.0, 1.0, 9.0], vec![2, 2]).unwrap();
        let b_tensor = Tensor::new(vec![2.0, 0.0, 0.0, 3.0], vec![2, 2]).unwrap();
        let eval = evaluate(
            Value::Tensor(a_tensor.clone()),
            &[Value::Tensor(b_tensor.clone())],
            true,
        )
        .expect("evaluate");
        let a = matrix_from_value(Value::Tensor(a_tensor));
        let b = matrix_from_value(Value::Tensor(b_tensor));
        let v = matrix_from_value(eval.right());
        let d = matrix_from_value(eval.diagonal_matrix());
        let w = matrix_from_value(eval.left().expect("left eigenvectors"));

        let identity = DMatrix::<Complex64>::identity(v.ncols(), v.ncols());
        let gram = w.adjoint() * &b * &v;
        assert_matrix_close(&gram, &identity, 1e-10);

        let lhs = w.adjoint() * &a;
        let rhs = &d * w.adjoint() * &b;
        assert_matrix_close(&lhs, &rhs, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_generalized_vector_option_returns_column_vector_second_output() {
        let a = Tensor::new(vec![2.0, 0.0, 0.0, 9.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 3.0], vec![2, 2]).unwrap();
        let eval = evaluate(
            Value::Tensor(a),
            &[Value::Tensor(b), Value::from("vector")],
            false,
        )
        .expect("evaluate");
        match eval.diagonal() {
            Value::Tensor(t) => assert_eq!(t.shape(), [2, 1]),
            Value::ComplexTensor(ct) => assert_eq!(ct.shape(), [2, 1]),
            other => panic!("expected vector second output, got {other:?}"),
        }
        let _ = matrix_from_value(eval.diagonal_matrix());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_generalized_singular_b_returns_infinite_eigenvalue() {
        let a = Tensor::new(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let result = eig_builtin(Value::Tensor(a), vec![Value::Tensor(b)]).expect("eig(A,B)");
        let values = column_vector_from_value(result);
        assert!(values.iter().any(|value| (value.re - 1.0).abs() < 1e-10));
        assert!(values.iter().any(|value| value.re.is_infinite()));

        // A small but nonzero homogeneous denominator remains finite.
        let eval = evaluate(Value::Num(1.0), &[Value::Num(1e-16)], false).unwrap();
        let values = column_vector_from_value(eval.eigenvalues());
        assert!(values[0].re.is_finite());
        assert!((values[0].re / 1e16 - 1.0).abs() < 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_generalized_singular_pencil_preserves_class_and_left_equations() {
        for single in [false, true] {
            for complex in [false, true] {
                let finite = Complex64::new(2.0, if complex { 3.0 } else { 0.0 });
                let infinite = Complex64::new(4.0, if complex { -2.0 } else { 0.0 });
                let coefficient = |entries: [Complex64; 3]| {
                    let data = [
                        entries[0],
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        entries[1],
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        entries[2],
                    ];
                    if complex {
                        Value::ComplexTensor(if single {
                            ComplexTensor::from_f32(
                                data.iter().map(|v| (v.re as f32, v.im as f32)).collect(),
                                vec![3, 3],
                            )
                            .unwrap()
                        } else {
                            ComplexTensor::new(
                                data.iter().map(|v| (v.re, v.im)).collect(),
                                vec![3, 3],
                            )
                            .unwrap()
                        })
                    } else {
                        Value::Tensor(if single {
                            Tensor::from_f32(data.iter().map(|v| v.re as f32).collect(), vec![3, 3])
                                .unwrap()
                        } else {
                            Tensor::new(data.iter().map(|v| v.re).collect(), vec![3, 3]).unwrap()
                        })
                    }
                };
                let a = coefficient([finite, infinite, Complex64::new(0.0, 0.0)]);
                let b = coefficient([
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ]);
                let eval = evaluate(a.clone(), std::slice::from_ref(&b), true)
                    .expect("full singular pencil");
                let values = column_vector_from_value(eval.eigenvalues());
                let expected_dtype = if single {
                    NumericDType::F32
                } else {
                    NumericDType::F64
                };
                for output in [
                    eval.eigenvalues(),
                    eval.diagonal_matrix(),
                    eval.right(),
                    eval.left().unwrap(),
                ] {
                    assert_eq!(value_precision(&output), expected_dtype);
                }
                let a = matrix_from_value(a);
                let b = matrix_from_value(b);
                let v = matrix_from_value(eval.right());
                let w = matrix_from_value(eval.left().unwrap());
                let mut counts = [0; 3];
                let tolerance = if single { 1e-5 } else { 1e-10 };
                for (column, value) in values.iter().enumerate() {
                    let right = v.column(column);
                    let left = w.column(column).adjoint();
                    assert!(right.iter().all(|v| v.re.is_finite() && v.im.is_finite()));
                    assert!(left.iter().all(|v| v.re.is_finite() && v.im.is_finite()));
                    if value.re.is_nan() || value.im.is_nan() {
                        // 0/0 is indeterminate: do not invent a finite residual
                        // or a W'*B*V identity for this undefined eigenvalue.
                        counts[2] += 1;
                    } else if value.re.is_infinite() || value.im.is_infinite() {
                        counts[1] += 1;
                        assert!((&b * right).norm() < tolerance);
                        assert!((&left * &b).norm() < tolerance);
                    } else {
                        counts[0] += 1;
                        assert!((*value - finite).norm() < tolerance);
                        assert!((&a * right - (&b * right) * *value).norm() < tolerance);
                        assert!((&left * &a - (&left * &b) * *value).norm() < tolerance);
                        assert!(
                            ((&left * &b * right)[(0, 0)] - Complex64::new(1.0, 0.0)).norm()
                                < tolerance
                        );
                    }
                }
                assert_eq!(counts, [1, 1, 1]);
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_vector_option_gpu_fallback_restores_owner_residency() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 1.0, 0.0, 3.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let eval = evaluate(Value::GpuTensor(handle), &[Value::from("vector")], false)
                .expect("evaluate");
            match eval.diagonal() {
                Value::GpuTensor(handle) => assert_eq!(handle.shape(), &[2, 1]),
                other => panic!("unexpected eigenvalue output: {other:?}"),
            }
            assert!(matches!(eval.right(), Value::GpuTensor(_)));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_gpu_provider_fallback_restores_owner_residency() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 0.0, 0.0, 3.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                eig_builtin(Value::GpuTensor(handle), vec![Value::from("nobalance")]).expect("eig");
            let tensor = test_support::gather(result).expect("gather owner-resident output");
            assert_eq!(tensor.materialize_f64(), vec![2.0, 3.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_generalized_scalar_pair() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let args = vec![Value::Int(IntValue::I32(3))];
        let result = evaluate(Value::Num(4.0), &args, false)
            .expect("scalar generalized eig")
            .eigenvalues();
        let values = column_vector_from_value(result);
        assert_eq!(values.len(), 1);
        assert!((values[0] - Complex64::new(4.0 / 3.0, 0.0)).norm() < 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_rejects_unknown_option_with_stable_identifier() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = evaluate(Value::Tensor(tensor), &[Value::from("invalid")], false).unwrap_err();
        assert_eq!(err.identifier(), EIG_ERROR_INVALID_ARGUMENT.identifier);
        assert!(error_message(err).contains("unknown option"));
    }

    #[test]
    fn eig_invalid_input_identifier_is_stable_for_nd_arrays() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]).expect("tensor");
        let err = evaluate(Value::Tensor(tensor), &[], false).unwrap_err();
        assert_eq!(err.identifier(), EIG_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn eig_wgpu_matches_cpu_for_real_spectrum() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        let tensor = Tensor::new(vec![4.0, 2.0, 1.0, 3.0], vec![2, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");

        let gpu_eval = evaluate(Value::GpuTensor(handle), &[], true).expect("gpu evaluate");

        let eig_gpu_value = gpu_eval.eigenvalues();
        assert!(matches!(eig_gpu_value, Value::GpuTensor(_)));
        let diag_gpu_value = gpu_eval.diagonal_matrix();
        assert!(matches!(diag_gpu_value, Value::GpuTensor(_)));
        let right_gpu_value = gpu_eval.right();
        assert!(matches!(right_gpu_value, Value::GpuTensor(_)));
        let left_gpu_value = gpu_eval.left().expect("left eigenvectors");
        assert!(matches!(left_gpu_value, Value::GpuTensor(_)));

        let eig_gpu = test_support::gather(eig_gpu_value).expect("gather eigenvalues");
        let diag_gpu = test_support::gather(diag_gpu_value).expect("gather diagonal");
        let right_gpu = test_support::gather(right_gpu_value).expect("gather right vectors");
        let left_gpu = test_support::gather(left_gpu_value).expect("gather left vectors");

        let host_eval = evaluate(Value::Tensor(tensor.clone()), &[], true).expect("host evaluate");

        let eig_host = match host_eval.eigenvalues() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor eigenvalues, got {other:?}"),
        };
        let diag_host = match host_eval.diagonal_matrix() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor diagonal, got {other:?}"),
        };
        let right_host = match host_eval.right() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor right eigenvectors, got {other:?}"),
        };
        let left_host = match host_eval.left().expect("host left eigenvectors") {
            Value::Tensor(t) => t,
            other => panic!("expected tensor left eigenvectors, got {other:?}"),
        };

        assert_tensor_close(&eig_gpu, &eig_host, tol);
        assert_tensor_close(&diag_gpu, &diag_host, tol);
        assert_tensor_close(&right_gpu, &right_host, tol);
        assert_tensor_close(&left_gpu, &left_host, tol);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn eig_wgpu_nobalance_fallback_restores_residency() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![1.0, 1.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let eval = evaluate(Value::GpuTensor(handle), &[Value::from("nobalance")], false)
            .expect("evaluate");
        assert!(matches!(eval.eigenvalues(), Value::GpuTensor(_)));

        let complex = ComplexTensor::new(vec![(2.0, 1.0)], vec![1, 1]).unwrap();
        let source = gpu_helpers::upload_complex_tensor(provider, &complex).unwrap();
        let eval = evaluate(Value::GpuTensor(source), &[], false).expect("resident complex eig");
        let Value::GpuTensor(eigenvalues) = eval.eigenvalues() else {
            panic!("complex eig did not restore resident eigenvalues");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&eigenvalues),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        let gathered = block_on(gpu_helpers::download_value_preserving_residency_async(
            provider,
            &eigenvalues,
        ))
        .unwrap();
        let Value::ComplexTensor(gathered) = gathered else {
            panic!("complex eig lost complex eigenvalue storage");
        };
        let (re, im) = gathered.materialize_f64()[0];
        assert!((re - 2.0).abs() < 1e-5);
        assert!((im - 1.0).abs() < 1e-5);
    }

    fn eig_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let rest = crate::execution::RootedValues::new(rest)
            .map_err(|error| error.into_runtime_error())?;
        block_on(super::eig_builtin_context(None, value, rest.into()))
    }

    fn evaluate(value: Value, args: &[Value], require_left: bool) -> BuiltinResult<EigEval> {
        block_on(super::evaluate(
            value,
            crate::call::FunctionArgumentView::from_slice(args),
            require_left,
        ))
    }
}
