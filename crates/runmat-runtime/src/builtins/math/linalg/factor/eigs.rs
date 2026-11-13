//! MATLAB-compatible `eigs` builtin for subset eigenvalue queries.
//!
//! Native sparse and selected callable operands use owned restarted iteration.
//! Dense and explicit GPU/nonfloating compatibility inputs retain the shared
//! exact eig decomposition. No iterative route materializes a dense A.

use std::cmp::Ordering;

use num_complex::Complex64;
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_builtins::{
    EIGS_ERROR_INTERNAL as ERROR_INTERNAL, EIGS_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT,
    EIGS_ERROR_INVALID_INPUT as ERROR_INVALID_INPUT, EIGS_ERROR_UNSUPPORTED as ERROR_UNSUPPORTED,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    ComplexTensor, IntValue, NumericDType, NumericScalar, StructValue, Tensor, Value,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::eig;
mod allocation;
mod controls;
mod dense;
mod iterative;

const NAME: &str = "eigs";
const DEFAULT_K: usize = 6;

#[cfg(test)]
use runmat_builtins::EIGS_EXTENSIONS;
use runmat_builtins::{
    EIGS_GPU_INPUT_EXTENSION, EIGS_INTEGER_SIGMA_EXTENSION, EIGS_INTEGER_START_VECTOR_EXTENSION,
    EIGS_NONFLOATING_MATRIX_EXTENSION,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::factor::eigs")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("eigs-subset"),
    supported_precisions: &[ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Subset eigenvalue selection is currently a host residency sink: gpuArray inputs gather before exact dense fallback and selected outputs are returned on the host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::factor::eigs")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "eigs executes eagerly and terminates fusion planning.",
};

#[runtime_builtin(
    name = "eigs",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::eigs",
    context = true
)]
async fn eigs_builtin_context(
    runtime: Option<crate::context::RuntimeContext>,
    a: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if let Some(runtime) = &runtime {
        runtime
            .cpu_cooperation()
            .checkpoint(0)
            .await
            .map_err(|e| e.into_runtime_error_in_context(Some(runtime)))?;
    }
    let resident_inputs =
        crate::value_contains_gpu(&a) || rest.iter().any(crate::value_contains_gpu);
    if resident_inputs {
        crate::compatibility::ensure_builtin_extension_enabled(&EIGS_GPU_INPUT_EXTENSION, NAME)?;
    }
    ensure_eigs_coefficient_class_enabled(&a)?;
    let selected = allocation::Selected::admit(&a, &rest, resident_inputs, runtime.as_ref())?;
    // Root/carrier admission is part of the selected host execution too. Keep
    // its original allocation/control cause, but never replay it through gather.
    async {
        if matches!(&selected, allocation::Selected::Dense(_)) {
            preflight_eigs_extensions(runtime.as_ref(), rest.view())?;
        }
        crate::builtins::common::validation::reject_typed_complex_integer(&a, "eigs")?;
        for value in &rest {
            crate::builtins::common::validation::reject_typed_complex_integer(value, "eigs")?;
        }
        let a = crate::execution::RootedValue::in_optional_context(a, runtime.as_ref())?;
        let args = crate::builtins::common::mapped_callable::prepend_rooted_argument(
            &rest,
            a,
            runtime.as_ref(),
        )
        .await?;
        let legacy = match selected {
            allocation::Selected::Iterative => {
                return iterative::evaluate(args, runtime.as_ref()).await;
            }
            allocation::Selected::Dense(work) => work,
        };
        legacy.check()?;
        let args = gather_eigs_arguments(args, runtime.as_ref()).await?;
        legacy.check()?;
        let request = Request::parse(args, runtime.as_ref()).await?;
        legacy.check()?;
        let eval = dense::compute(request, runtime.as_ref()).await?;
        legacy.check()?;

        let output = match crate::output_count::current_output_count() {
            Some(0) => eig::work::output_list(runtime.as_ref(), [], 0).await,
            Some(1) => eig::work::output_list(runtime.as_ref(), [Some(eval.values)], 1).await,
            Some(2) => {
                eig::work::output_list(
                    runtime.as_ref(),
                    [Some(eval.vectors), Some(eval.diagonal)],
                    2,
                )
                .await
            }
            Some(3) => {
                eig::work::output_list(
                    runtime.as_ref(),
                    [
                        Some(eval.vectors),
                        Some(eval.diagonal),
                        Some(Value::Num(0.)),
                    ],
                    3,
                )
                .await
            }
            Some(_) => Err(eigs_error(&ERROR_INVALID_ARGUMENT)),
            None => Ok(eval.values),
        };
        legacy.finish(output)
    }
    .await
    .map_err(with_context)
}

async fn gather_eigs_arguments(
    args: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    let mut resident = false;
    for value in &args {
        crate::dispatcher::try_visit_gathered_gpu_handles(value, &mut |_| resident = true)?;
    }
    if !resident {
        return Ok(args);
    }
    args.try_map_preserving_roots_async(args.len(), runtime, async |index| {
        crate::dispatcher::gather_if_needed_in_context_async(&args[index], runtime)
            .await
            .map_err(with_context)
    })
    .await
}

#[cfg(test)]
async fn eigs_builtin(a: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    eigs_builtin_context(Some(runtime), a, rest).await
}

fn ensure_eigs_coefficient_class_enabled(value: &Value) -> BuiltinResult<()> {
    if is_nonfloating_supported_coefficient_matrix(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EIGS_NONFLOATING_MATRIX_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

fn preflight_eigs_extensions(
    runtime: Option<&crate::context::RuntimeContext>,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    let mut cursor = 0usize;
    if let Some(candidate) = rest.first() {
        if is_empty_matrix(candidate) {
            cursor = 1;
        } else if is_generalized_b(candidate) {
            ensure_eigs_coefficient_class_enabled(candidate)?;
            cursor = 1;
        }
    }
    if cursor < rest.len() && parse_integer_scalar(&rest[cursor])?.is_some() {
        cursor += 1;
    }
    if cursor < rest.len() && (is_text(&rest[cursor]) || is_numeric_sigma_value(&rest[cursor])) {
        if is_typed_integer_value(&rest[cursor]) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &EIGS_INTEGER_SIGMA_EXTENSION,
                NAME,
            )?;
        }
        cursor += 1;
    }
    if cursor < rest.len() {
        preflight_start_vector_extension(
            runtime,
            rest.range(cursor..).expect("bounded eigs option range"),
        )?;
    }
    Ok(())
}

fn preflight_start_vector_extension(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    if args.len() == 1 {
        if let Value::Struct(options) = &args[0] {
            for (name, value) in &options.fields {
                if (name.trim().eq_ignore_ascii_case("startvector")
                    || name.trim().eq_ignore_ascii_case("v0"))
                    && is_typed_integer_value(value)
                {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &EIGS_INTEGER_START_VECTOR_EXTENSION,
                        NAME,
                    )?;
                }
            }
            return Ok(());
        }
    }
    let mut idx = 0usize;
    while idx + 1 < args.len() {
        if text_value_in_context(runtime, &args[idx])?.is_some_and(|name| {
            name.trim().eq_ignore_ascii_case("startvector")
                || name.trim().eq_ignore_ascii_case("v0")
        }) && is_typed_integer_value(&args[idx + 1])
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &EIGS_INTEGER_START_VECTOR_EXTENSION,
                NAME,
            )?;
        }
        idx += 2;
    }
    Ok(())
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::SparseTensor(sparse) if sparse.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_numeric_sigma_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Complex(_, _)
            | Value::Tensor(_)
            | Value::ComplexTensor(_)
            | Value::LogicalArray(_)
            | Value::Bool(_)
    )
}

fn is_nonfloating_supported_coefficient_matrix(value: &Value) -> bool {
    match value {
        Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::Tensor(tensor) => !matches!(
            tensor.numeric_dtype(),
            NumericDType::F64 | NumericDType::F32
        ),
        Value::SparseTensor(sparse) => {
            sparse.is_logical()
                || sparse
                    .numeric_dtype()
                    .is_some_and(|dtype| !matches!(dtype, NumericDType::F64 | NumericDType::F32))
        }
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_is_logical(handle)
                || runmat_accelerate_api::handle_integer_type(handle).is_some()
        }
        Value::ComplexTensor(_) => false,
        _ => false,
    }
}

#[derive(Clone, Copy)]
enum Sigma {
    LargestAbs,
    SmallestAbs,
    LargestReal,
    SmallestReal,
    BothEndsReal,
    LargestImag,
    SmallestImag,
    BothEndsImag,
    Near(Complex64),
}

impl Default for Sigma {
    fn default() -> Self {
        Self::LargestAbs
    }
}

struct Request {
    a: Value,
    b: Option<Value>,
    k: usize,
    sigma: Sigma,
    cholesky: Option<dense::Cholesky>,
}

#[derive(Default)]
struct ParsedOptions {
    is_cholesky: bool,
    cholesky_permutation: Option<runmat_value::ExactArrayStorage<usize>>,
    is_function_symmetric: bool,
    is_symmetric_definite: bool,
}

struct SubsetEval {
    values: Value,
    vectors: Value,
    diagonal: Value,
}

impl Request {
    async fn parse(
        mut args: crate::call::BuiltinArguments,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<Self> {
        let a = args.pop_front().ok_or_else(|| {
            error_with_detail("eigs: missing coefficient matrix", &ERROR_INVALID_INPUT)
        })?;
        if matches!(
            a,
            Value::ScopedFunctionHandle(_)
                | Value::FunctionHandle(_)
                | Value::ExternalFunctionHandle(_)
                | Value::MethodFunctionHandle(_)
                | Value::BoundFunctionHandle(_)
                | Value::Closure(_)
        ) {
            return Err(error_with_detail(
                "eigs: function-handle matrix forms require iterative operator support",
                &ERROR_UNSUPPORTED,
            ));
        }

        let mut a = matrix_host_value(a)?;
        ensure_exact_eigs_integer_boundary(&a, "A")?;
        let n = matrix_order(&a)?;
        let mut cursor = 0usize;
        let mut b = None;
        let mut k = DEFAULT_K.min(n);
        let mut sigma = Sigma::LargestAbs;
        let mut imaginary_alias = controls::ImaginaryAlias::Explicit;
        let mut options = ParsedOptions::default();

        if let Some(candidate) = args.first() {
            if is_empty_matrix(candidate) {
                cursor += 1;
            } else if is_generalized_b(candidate) {
                ensure_eigs_coefficient_class_enabled(candidate)?;
                let candidate = matrix_host_value(
                    args.pop_front()
                        .expect("first generalized coefficient exists"),
                )?;
                ensure_exact_eigs_integer_boundary(&candidate, "B")?;
                b = Some(candidate);
                cursor = 0;
            }
        }

        if cursor < args.len() {
            if let Some(parsed_k) = parse_k(&args[cursor])? {
                k = parsed_k.min(n);
                cursor += 1;
            }
        }

        if cursor < args.len() {
            if let Some(parsed_sigma) = parse_sigma_in_context(runtime, &args[cursor])? {
                sigma = parsed_sigma;
                if let Some(text) = controls::Text::of(&args[cursor]) {
                    let text = text.trim(&mut None).await.map_err(|error| match error {
                        crate::builtins::math::linalg::ops::linear_operator::CooperativeOperatorError::Execution(source) => source.into_runtime_error_in_context(None),
                        source => build_runtime_error(format!("eigs: selector parsing failed: {source:?}")).with_gpu_gather_retry(crate::GpuGatherRetry::Never).build(),
                    })?;
                    if let Some((_, alias)) = controls::selector(text) {
                        imaginary_alias = alias;
                    }
                }
                cursor += 1;
            }
        }

        if cursor < args.len() {
            let option_args = args
                .view()
                .range(cursor..)
                .expect("bounded eigs option range");
            parse_options(runtime, option_args, n, k, &mut options).await?;
        }

        sigma = imaginary_alias.resolve(
            sigma,
            matches!(&a, Value::ComplexTensor(_) | Value::Complex(..))
                || b.as_ref()
                    .is_some_and(|b| matches!(b, Value::ComplexTensor(_) | Value::Complex(..))),
        );
        validate_option_combination(sigma, &options)?;

        if let Some(b_value) = b.as_mut() {
            let b_order = matrix_order(b_value)?;
            if b_order != n {
                return Err(error_with_detail(
                    "eigs: A and B must be the same size",
                    &ERROR_INVALID_INPUT,
                ));
            }
        } else if options.is_cholesky || options.cholesky_permutation.is_some() {
            return Err(error_with_detail(
                "eigs: Cholesky options require a generalized B matrix",
                &ERROR_INVALID_ARGUMENT,
            ));
        }

        a = matrix_host_value(a)?;

        let cholesky = options.is_cholesky.then(|| {
            options
                .cholesky_permutation
                .map_or(dense::Cholesky::Identity, dense::Cholesky::Permuted)
        });
        Ok(Self {
            a,
            b,
            k,
            sigma,
            cholesky,
        })
    }
}

fn matrix_host_value(value: Value) -> BuiltinResult<Value> {
    if let Value::SparseTensor(sparse) = &value {
        sparse.rows.checked_mul(sparse.cols).ok_or_else(|| {
            error_with_detail(
                "eigs: SparseTensor dense dimensions overflow usize",
                &ERROR_INTERNAL,
            )
        })?;
        if sparse.is_logical() {
            return Err(error_with_detail(
                "eigs: SparseTensor logical storage requires to_dense_logical",
                &ERROR_INTERNAL,
            ));
        }
        if sparse.is_complex() {
            return Err(error_with_detail(
                "eigs: SparseTensor complex storage requires to_dense_complex",
                &ERROR_INTERNAL,
            ));
        }
    }
    Ok(value)
}

fn matrix_order(value: &Value) -> BuiltinResult<usize> {
    let (rows, cols) = match value {
        Value::Tensor(tensor) => (tensor.rows(), tensor.cols()),
        Value::ComplexTensor(tensor) => (tensor.rows(), tensor.cols()),
        Value::LogicalArray(logical) => matrix_shape_from_slice(&logical.shape),
        Value::SparseTensor(sparse) => (sparse.rows, sparse.cols),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => (1, 1),
        other => {
            return Err(error_with_detail(
                format!("eigs: unsupported matrix input {other:?}"),
                &ERROR_INVALID_INPUT,
            ));
        }
    };
    if rows != cols {
        return Err(error_with_detail(
            ERROR_INVALID_INPUT.message,
            &ERROR_INVALID_INPUT,
        ));
    }
    Ok(cols)
}

fn matrix_shape_from_slice(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (shape[0], 1),
        _ => (shape[0], shape[1]),
    }
}

fn is_empty_matrix(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor_element_len(tensor) == 0,
        Value::ComplexTensor(tensor) => complex_tensor_element_len(tensor) == 0,
        Value::LogicalArray(logical) => logical.data.is_empty(),
        Value::SparseTensor(sparse) => sparse.rows == 0 || sparse.cols == 0,
        _ => false,
    }
}

fn is_generalized_b(value: &Value) -> bool {
    if is_text(value) {
        return false;
    }
    if parse_integer_scalar(value).ok().flatten().is_some() {
        return false;
    }
    matches!(
        value,
        Value::Tensor(_)
            | Value::ComplexTensor(_)
            | Value::LogicalArray(_)
            | Value::SparseTensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Complex(_, _)
    )
}

fn parse_k(value: &Value) -> BuiltinResult<Option<usize>> {
    parse_integer_scalar(value)
}

fn parse_integer_scalar(value: &Value) -> BuiltinResult<Option<usize>> {
    if let Some(value) = exact_integer_scalar(value) {
        let Some(raw) = value.try_to_usize() else {
            return Err(error_with_detail(
                "eigs: k is too large for this platform",
                &ERROR_INVALID_ARGUMENT,
            ));
        };
        if raw == 0 {
            return Err(error_with_detail(
                "eigs: k must be a positive integer scalar",
                &ERROR_INVALID_ARGUMENT,
            ));
        }
        return Ok(Some(raw));
    }
    let raw = match value {
        Value::Num(n) => *n,
        Value::Tensor(tensor) if tensor_element_len(tensor) == 1 => scalar_tensor_f64(tensor),
        Value::LogicalArray(logical) if logical.len() == 1 => {
            if logical.data[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        Value::Bool(flag) => {
            if *flag {
                1.0
            } else {
                0.0
            }
        }
        _ => return Ok(None),
    };
    if !raw.is_finite() || raw <= 0.0 || raw.fract() != 0.0 {
        return Err(error_with_detail(
            "eigs: k must be a positive integer scalar",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    if raw > usize::MAX as f64 || (usize::BITS == 64 && raw == usize::MAX as f64) {
        return Err(error_with_detail(
            "eigs: k is too large for this platform",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok(Some(raw as usize))
}

#[cfg(test)]
fn parse_sigma(value: &Value) -> BuiltinResult<Option<Sigma>> {
    parse_sigma_in_context(None, value)
}
fn parse_sigma_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    value: &Value,
) -> BuiltinResult<Option<Sigma>> {
    if let Some(text) = text_value_in_context(runtime, value)? {
        return Ok(Some(parse_sigma_text(&text)?));
    }
    match value {
        Value::Num(n) => Ok(Some(Sigma::Near(Complex64::new(*n, 0.0)))),
        Value::Int(i) => {
            ensure_exact_eigs_integer_boundary(value, "sigma")?;
            Ok(Some(Sigma::Near(Complex64::new(i.to_f64(), 0.0))))
        }
        Value::Complex(re, im) => Ok(Some(Sigma::Near(Complex64::new(*re, *im)))),
        Value::Tensor(tensor) if tensor_element_len(tensor) == 1 => {
            ensure_exact_eigs_integer_boundary(value, "sigma")?;
            Ok(Some(Sigma::Near(Complex64::new(
                scalar_tensor_f64(tensor),
                0.0,
            ))))
        }
        Value::ComplexTensor(tensor) if tensor::is_scalar_complex_tensor(tensor) => {
            let value = tensor::complex_tensor_value_complex64(tensor, 0);
            Ok(Some(Sigma::Near(value)))
        }
        Value::LogicalArray(logical) if logical.len() == 1 => Ok(Some(Sigma::Near(
            Complex64::new(if logical.data[0] != 0 { 1.0 } else { 0.0 }, 0.0),
        ))),
        Value::Bool(flag) => Ok(Some(Sigma::Near(Complex64::new(
            if *flag { 1.0 } else { 0.0 },
            0.0,
        )))),
        _ => Ok(None),
    }
}

fn parse_sigma_text(text: &str) -> BuiltinResult<Sigma> {
    controls::selector(controls::Text::Utf8(text.trim()))
        .map(|(sigma, _)| sigma)
        .ok_or_else(|| {
            error_with_detail("eigs: unsupported sigma selector", &ERROR_INVALID_ARGUMENT)
        })
}

async fn parse_options(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::FunctionArgumentView<'_>,
    n: usize,
    k: usize,
    options: &mut ParsedOptions,
) -> BuiltinResult<()> {
    if args.len() == 1 {
        if let Value::Struct(opts) = &args[0] {
            return parse_options_struct(runtime, opts, n, k, options).await;
        }
    }
    if args.is_empty() {
        return Ok(());
    }
    if !args.len().is_multiple_of(2) {
        return Err(error_with_detail(
            "eigs: name-value options must appear in pairs",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    let mut idx = 0usize;
    while idx < args.len() {
        let name = text_value_in_context(runtime, &args[idx])?.ok_or_else(|| {
            error_with_detail(
                "eigs: option names must be character vectors or string scalars",
                &ERROR_INVALID_ARGUMENT,
            )
        })?;
        parse_named_option(runtime, &name, &args[idx + 1], n, k, options).await?;
        idx += 2;
    }
    Ok(())
}

async fn parse_options_struct(
    runtime: Option<&crate::context::RuntimeContext>,
    opts: &StructValue,
    n: usize,
    k: usize,
    options: &mut ParsedOptions,
) -> BuiltinResult<()> {
    for (field, value) in &opts.fields {
        parse_named_option(runtime, field, value, n, k, options).await?;
    }
    Ok(())
}

async fn parse_named_option(
    runtime: Option<&crate::context::RuntimeContext>,
    name: &str,
    value: &Value,
    n: usize,
    k: usize,
    options: &mut ParsedOptions,
) -> BuiltinResult<()> {
    match canonical_option_name(name)? {
        "tolerance" => {
            let tol = numeric_scalar(value, "Tolerance")?;
            if !tol.is_finite() || tol <= 0.0 {
                return Err(error_with_detail(
                    "eigs: Tolerance must be a positive real scalar",
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
        }
        "maxiterations" => {
            positive_integer_option(value, "MaxIterations")?;
        }
        "subspacedimension" => {
            let p = positive_integer_option(value, "SubspaceDimension")?;
            let minimum = if options.is_function_symmetric || options.is_symmetric_definite {
                k + 1
            } else {
                k + 2
            };
            if p < minimum {
                return Err(error_with_detail(
                    format!("eigs: SubspaceDimension must be at least {minimum}"),
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
        }
        "startvector" => {
            ensure_exact_eigs_integer_boundary(value, "StartVector")?;
            let len = numeric_vector_len(value, "StartVector")?;
            if len != n {
                return Err(error_with_detail(
                    "eigs: StartVector length must match size(A,1)",
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
        }
        "failuretreatment" => {
            let text = text_value_in_context(runtime, value)?.ok_or_else(|| {
                error_with_detail(
                    "eigs: FailureTreatment must be a text scalar",
                    &ERROR_INVALID_ARGUMENT,
                )
            })?;
            let text = text.trim();
            if !["replacenan", "keep", "drop"]
                .into_iter()
                .any(|name| text.eq_ignore_ascii_case(name))
            {
                return Err(error_with_detail(
                    format!(
                        "eigs: unsupported FailureTreatment '{}'",
                        text.to_ascii_lowercase()
                    ),
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
        }
        "display" => {
            parse_bool(value, name)?;
        }
        "isfunctionsymmetric" => {
            options.is_function_symmetric = parse_bool(value, name)?;
        }
        "issymmetricdefinite" => {
            options.is_symmetric_definite = parse_bool(value, name)?;
        }
        "ischolesky" => {
            options.is_cholesky = parse_bool(value, name)?;
        }
        "choleskypermutation" => {
            options.cholesky_permutation =
                Some(parse_permutation_in_context(runtime, value, n).await?);
        }
        other => {
            return Err(error_with_detail(
                format!("eigs: unsupported option '{other}'"),
                &ERROR_INVALID_ARGUMENT,
            ));
        }
    }
    Ok(())
}

fn canonical_option_name(name: &str) -> BuiltinResult<&'static str> {
    controls::option(controls::Text::Utf8(name.trim()))
        .map(|key| key.name())
        .ok_or_else(|| error_with_detail("eigs: unsupported option", &ERROR_INVALID_ARGUMENT))
}

fn validate_option_combination(sigma: Sigma, options: &ParsedOptions) -> BuiltinResult<()> {
    if (options.is_cholesky || options.cholesky_permutation.is_some())
        && matches!(sigma, Sigma::Near(_) | Sigma::SmallestAbs)
    {
        return Err(error_with_detail(
            "eigs: IsCholesky and CholeskyPermutation cannot be used with numeric sigma or 'smallestabs'",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    if options.is_symmetric_definite && !matches!(sigma, Sigma::Near(_) | Sigma::SmallestAbs) {
        return Err(error_with_detail(
            "eigs: IsSymmetricDefinite requires numeric sigma or 'smallestabs'",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok(())
}

fn total_cmp(lhs: f64, rhs: f64) -> Ordering {
    lhs.total_cmp(&rhs)
}

fn numeric_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(tensor) if tensor_element_len(tensor) == 1 => Ok(scalar_tensor_f64(tensor)),
        Value::LogicalArray(logical) if logical.len() == 1 => {
            Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
        }
        Value::Bool(flag) => Ok(if *flag { 1.0 } else { 0.0 }),
        _ => Err(error_with_detail(
            format!("eigs: {name} must be a real scalar"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn positive_integer_option(value: &Value, name: &str) -> BuiltinResult<usize> {
    if let Some(value) = exact_integer_scalar(value) {
        return value
            .try_to_usize()
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                error_with_detail(
                    format!("eigs: {name} must be a positive integer"),
                    &ERROR_INVALID_ARGUMENT,
                )
            });
    }
    let parsed = numeric_scalar(value, name)?;
    if !parsed.is_finite() || parsed <= 0.0 || parsed.fract() != 0.0 {
        return Err(error_with_detail(
            format!("eigs: {name} must be a positive integer"),
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    if parsed > usize::MAX as f64 || (usize::BITS == 64 && parsed == usize::MAX as f64) {
        return Err(error_with_detail(
            format!("eigs: {name} is too large for this platform"),
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok(parsed as usize)
}

fn parse_bool(value: &Value, name: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::LogicalArray(logical) if logical.len() == 1 => Ok(logical.data[0] != 0),
        Value::Num(n) if *n == 0.0 || *n == 1.0 => Ok(*n != 0.0),
        Value::Int(i) if i.to_i64() == 0 || i.to_i64() == 1 => Ok(i.to_i64() != 0),
        Value::Tensor(tensor) if tensor_element_len(tensor) == 1 => {
            if let Some(value) = exact_integer_scalar(value) {
                let raw = value.try_to_i64();
                if raw == Some(0) || raw == Some(1) {
                    return Ok(raw != Some(0));
                }
            } else {
                let raw = scalar_tensor_f64(tensor);
                if raw == 0.0 || raw == 1.0 {
                    return Ok(raw != 0.0);
                }
            }
            Err(error_with_detail(
                format!("eigs: {name} must be logical scalar true or false"),
                &ERROR_INVALID_ARGUMENT,
            ))
        }
        _ => Err(error_with_detail(
            format!("eigs: {name} must be logical scalar true or false"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn numeric_vector_len(value: &Value, name: &str) -> BuiltinResult<usize> {
    match value {
        Value::Tensor(tensor) if tensor.rows() == 1 || tensor.cols() == 1 => {
            Ok(tensor_element_len(tensor))
        }
        Value::ComplexTensor(tensor) if tensor.rows() == 1 || tensor.cols() == 1 => {
            Ok(complex_tensor_element_len(tensor))
        }
        Value::LogicalArray(logical) => {
            let (rows, cols) = matrix_shape_from_slice(&logical.shape);
            if rows == 1 || cols == 1 {
                Ok(logical.data.len())
            } else {
                Err(error_with_detail(
                    format!("eigs: {name} must be a vector"),
                    &ERROR_INVALID_ARGUMENT,
                ))
            }
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(1),
        _ => Err(error_with_detail(
            format!("eigs: {name} must be a numeric vector"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

#[cfg(test)]
fn parse_permutation(
    value: &Value,
    n: usize,
) -> BuiltinResult<runmat_value::ExactArrayStorage<usize>> {
    futures::executor::block_on(parse_permutation_in_context(None, value, n))
}
async fn parse_permutation_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    value: &Value,
    n: usize,
) -> BuiltinResult<runmat_value::ExactArrayStorage<usize>> {
    let length = match value {
        Value::Tensor(v) if v.integer_storage().is_some() => {
            if v.rows() != 1 && v.cols() != 1 {
                return Err(error_with_detail(
                    "eigs: CholeskyPermutation must be a vector",
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
            v.len()
        }
        Value::ComplexTensor(_) | Value::Complex(_, _) => {
            return Err(error_with_detail(
                "eigs: CholeskyPermutation must be a numeric vector",
                &ERROR_INVALID_ARGUMENT,
            ))
        }
        _ => numeric_vector_len(value, "CholeskyPermutation")?,
    };
    if length != n {
        return Err(error_with_detail(
            "eigs: CholeskyPermutation length must match size(A,1)",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    let mut seen = eig::work::Buffer::<usize>::new(n, runtime)
        .await
        .map_err(with_context)?;
    let mut output =
        crate::context::host_allocation::prepare_host_array_optional::<usize>(runtime, n)
            .map_err(with_context)?;
    let mut budget = runtime.map(crate::context::RuntimeContext::cpu_cooperation);
    for i in 0..n {
        let exact = match value {
            Value::Int(v) => Some(int_value_i128(v)),
            Value::Tensor(v) => v.numeric_value_at(i).and_then(numeric_scalar_i128),
            _ => None,
        };
        let index = if let Some(raw) = exact {
            if raw < 1 || raw > n as i128 {
                return Err(error_with_detail(
                    "eigs: CholeskyPermutation must contain 1-based integer indices",
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
            raw as usize
        } else {
            let raw = eig::host::element(value, i).re;
            if !raw.is_finite() || raw.fract() != 0. || raw < 1. || raw > n as f64 {
                return Err(error_with_detail(
                    "eigs: CholeskyPermutation must contain 1-based integer indices",
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
            raw as usize
        };
        if seen.get()[index - 1] != 0 {
            return Err(error_with_detail(
                "eigs: CholeskyPermutation entries must be unique",
                &ERROR_INVALID_ARGUMENT,
            ));
        }
        seen.get_mut()[index - 1] = 1;
        output.try_push(index).map_err(|_| {
            error_with_detail("eigs: permutation exceeded admitted width", &ERROR_INTERNAL)
        })?;
        if i.is_multiple_of(4096) {
            if let Some(budget) = &mut budget {
                budget
                    .checkpoint(4096)
                    .await
                    .map_err(|e| e.into_runtime_error_in_context(runtime))?;
            }
        }
    }
    seen.check(runtime).map_err(with_context)?;
    output
        .try_finish()
        .map_err(|_| error_with_detail("eigs: incomplete permutation", &ERROR_INTERNAL))
}

fn text_value_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    value: &Value,
) -> BuiltinResult<Option<runmat_value::Utf8Text>> {
    match value {
        Value::String(text) => Ok(text.text_ref().cloned()),
        Value::StringArray(array) if array.data().len() == 1 => {
            Ok(array.data()[0].text_ref().cloned())
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            crate::call::argument_text::copy_chars(chars.data(), runtime)
                .map(Some)
                .map_err(|mut error| {
                    if error
                        .source
                        .as_ref()
                        .is_some_and(|source| source.is::<runmat_value::Utf16DecodeError>())
                    {
                        let source = error.source.take().expect("checked UTF16 cause");
                        let mut error =
                            error_with_detail(source.to_string(), &ERROR_INVALID_ARGUMENT);
                        error.source = Some(source);
                        error
                    } else {
                        with_context(error)
                    }
                })
        }
        _ => Ok(None),
    }
}

fn is_text(value: &Value) -> bool {
    matches!(value, Value::String(_))
        || matches!(value,Value::StringArray(a) if a.data().len()==1)
        || matches!(value,Value::CharArray(a) if a.rows==1)
}

fn error_with_detail(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(NAME)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn eigs_error(descriptor: &'static BuiltinErrorDescriptor) -> RuntimeError {
    error_with_detail(descriptor.message, descriptor)
}

fn with_context(mut error: RuntimeError) -> RuntimeError {
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    if error.context.builtin.is_none() {
        error.context = error.context.with_builtin(NAME);
    }
    error
}

fn exact_integer_scalar(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor_element_len(tensor) == 1 => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    }
}

fn ensure_exact_eigs_integer_boundary(value: &Value, role: &str) -> BuiltinResult<()> {
    const MAX_EXACT_INTEGER: i128 = 1_i128 << 53;
    let check = |exact: i128| {
        if (-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&exact) {
            Ok(())
        } else {
            Err(error_with_detail(
                format!("eigs: integer {role} values must be exactly representable as double"),
                &ERROR_INVALID_ARGUMENT,
            ))
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
        Value::SparseTensor(sparse) => {
            for index in 0..sparse.nnz() {
                if let Some(exact) = sparse.numeric_value_at(index).and_then(numeric_scalar_i128) {
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

fn scalar_tensor_f64(tensor: &Tensor) -> f64 {
    tensor::tensor_value_f64(tensor, 0)
}

fn tensor_element_len(tensor: &Tensor) -> usize {
    tensor.len()
}

fn complex_tensor_element_len(tensor: &ComplexTensor) -> usize {
    tensor.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::builtin_catalog_entry_by_name;
    use runmat_value::{CharArray, IntegerComplexStorage, IntegerStorage, SparseTensor};

    fn real_matrix(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new_2d(data, rows, cols).unwrap())
    }

    #[test]
    fn eigs_rejects_typed_complex_integer_inputs() {
        let tensor = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![u64::MAX]),
                IntegerStorage::U64(vec![1]),
            )
            .expect("storage"),
            vec![1, 1],
        )
        .expect("tensor");
        let err = block_on(eigs_builtin(
            Value::ComplexTensor(tensor),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("typed complex integer input must reject");
        assert!(err.message().contains("complex numbers with integer types"));
    }

    #[test]
    fn eigs_sigma_parser_reads_typed_integer_complex_scalar_without_mirror() {
        let storage =
            IntegerComplexStorage::new(IntegerStorage::I16(vec![3]), IntegerStorage::I16(vec![-2]))
                .expect("storage");
        let tensor = ComplexTensor::new_integer(storage, vec![1, 1]).expect("tensor");

        match parse_sigma(&Value::ComplexTensor(tensor)).expect("sigma") {
            Some(Sigma::Near(value)) => assert_eq!(value, Complex64::new(3.0, -2.0)),
            _ => panic!("expected near sigma"),
        }
    }

    fn tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn outputs(value: Value) -> Vec<Value> {
        match value {
            Value::OutputList(values) => values
                .try_into_owned()
                .expect("unique output list")
                .try_into_unbounded_vec()
                .expect("unbounded output list"),
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn eigs_selected_real_columns_preserve_native_single_storage() {
        let matrix = Value::Tensor(
            Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("single vectors"),
        );
        let selected = tensor(
            block_on(dense::publish_columns(
                None,
                NumericDType::F32,
                2,
                1,
                |row, _| eig::host::element(&matrix, row + 2),
            ))
            .expect("selected column"),
        );
        assert_eq!(selected.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(selected.shape(), [2, 1]);
        assert_eq!(selected.materialize_f64(), vec![3.0, 4.0]);
    }

    fn call(a: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
        block_on(eigs_builtin(a, rest))
    }

    #[test]
    fn eigs_nonfloating_coefficients_follow_compatibility_mode() {
        let integer_matrix = || {
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![1, 0, 0, 8]), vec![2, 2])
                    .expect("integer coefficient matrix"),
            )
        };
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error =
                call(integer_matrix(), Vec::new()).expect_err("MATLAB mode rejects integer A");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:EigsNonfloatingMatrixExtension")
            );

            let a = real_matrix(vec![1.0, 0.0, 0.0, 8.0], 2, 2);
            let error = call(a, vec![integer_matrix()])
                .expect_err("MATLAB mode rejects integer generalized B");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:EigsNonfloatingMatrixExtension")
            );

            let a = real_matrix(vec![1.0, 0.0, 0.0, 8.0], 2, 2);
            let k = Value::Int(IntValue::I16(1));
            assert!(call(a, vec![k]).is_ok(), "integer k is not a coefficient");
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            assert!(call(integer_matrix(), Vec::new()).is_ok());
        }

        let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![2, 2],
                runmat_accelerate_api::NumericElementType::I16,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle;
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = call(Value::GpuTensor(handle.clone()), Vec::new())
                .expect_err("MATLAB mode rejects resident integer A before gather");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:EigsGpuInputExtension")
            );
        }
    }

    #[test]
    fn eigs_integer_controls_have_role_specific_strict_mode_errors() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let a = real_matrix(vec![1.0, 0.0, 0.0, 8.0], 2, 2);
        let sigma =
            Value::Tensor(Tensor::new_integer(IntegerStorage::I16(vec![7]), vec![1, 1]).unwrap());
        let error = call(a.clone(), vec![Value::Num(1.0), sigma])
            .expect_err("typed-integer sigma must gate");
        assert_eq!(
            error.identifier(),
            EIGS_INTEGER_SIGMA_EXTENSION.error_identifier
        );

        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.insert(
            "StartVector",
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![2, 1]).unwrap(),
            ),
        );
        let error = call(a, vec![Value::Num(1.0), Value::Struct(options.finish())])
            .expect_err("typed-integer StartVector must gate");
        assert_eq!(
            error.identifier(),
            EIGS_INTEGER_START_VECTOR_EXTENSION.error_identifier
        );
    }

    #[test]
    fn eigs_nested_resident_start_vector_gates_before_recursive_gather() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let a = real_matrix(vec![1.0, 0.0, 0.0, 8.0], 2, 2);
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.insert(
            "StartVector",
            Value::GpuTensor(
                runmat_accelerate_api::test_support::TestAllocationOwner::new()
                    .allocate_zeroed(
                        vec![2, 1],
                        runmat_accelerate_api::NumericElementType::F64,
                        runmat_accelerate_api::GpuTensorStorage::Real,
                    )
                    .unwrap()
                    .handle,
            ),
        );
        let error = call(a, vec![Value::Num(1.0), Value::Struct(options.finish())])
            .expect_err("nested resident StartVector must gate before gather");
        assert_eq!(
            error.identifier(),
            EIGS_GPU_INPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn eigs_admits_all_integer_coefficient_classes_and_rejects_wide_boundaries() {
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
            let a = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).unwrap());
            assert!(call(a, Vec::new()).is_ok());
        }

        let wide = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]).unwrap(),
        );
        let error = call(wide, Vec::new()).expect_err("wide coefficient must reject");
        assert!(error.message().contains("exactly representable as double"));

        let a = real_matrix(vec![1.0, 0.0, 0.0, 8.0], 2, 2);
        let wide_sigma = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]).unwrap(),
        );
        let error = call(a, vec![Value::Num(1.0), wide_sigma]).expect_err("wide sigma must reject");
        assert!(error.message().contains("exactly representable as double"));
    }

    #[test]
    fn eigs_preserves_single_outputs_and_registers_integer_policy() {
        let a = Value::Tensor(Tensor::from_f32(vec![1.0, 0.0, 0.0, 8.0], vec![2, 2]).unwrap());
        let output = tensor(call(a, vec![Value::Num(1.0)]).unwrap());
        assert_eq!(output.numeric_dtype(), NumericDType::F32);

        let builtin = builtin_catalog_entry_by_name(NAME).expect("registered eigs catalog entry");
        assert_eq!(builtin.extensions, &EIGS_EXTENSIONS);
        assert_eq!(builtin.integer_capabilities.len(), 4);

        use crate::context::{
            host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeContext,
            RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
        };
        struct Managed(ManagedHostEffectLedger);
        impl RuntimeParallelService for Managed {
            fn supports(&self, _: ParallelCapability) -> bool {
                false
            }
            fn placement_resources(&self) -> RuntimeParallelResources {
                let snapshot = self.0.snapshot();
                RuntimeParallelResources {
                    memory_available_bytes: Some(snapshot.free_bytes()),
                    epoch: snapshot.epoch,
                    ..Default::default()
                }
            }
            fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
                Some(self.0.clone())
            }
        }
        for single in [false, true] {
            for n in [1, 2] {
                for limit in [0, 1 << 20] {
                    let ledger = ManagedHostEffectLedger::new(limit);
                    let runtime = RuntimeContext::new(std::rc::Rc::new(
                        crate::execution::RuntimeExecutionService::new(),
                    ))
                    .with_service_ports(
                        RuntimeServicePorts::default()
                            .with_parallel(std::rc::Rc::new(Managed(ledger.clone()))),
                    );
                    let input = if single {
                        Value::Tensor(
                            Tensor::from_f32(
                                if n == 1 {
                                    vec![8.]
                                } else {
                                    vec![1., 0., 0., 8.]
                                },
                                vec![n, n],
                            )
                            .unwrap(),
                        )
                    } else {
                        Value::Tensor(
                            Tensor::new(
                                if n == 1 {
                                    vec![8.]
                                } else {
                                    vec![1., 0., 0., 8.]
                                },
                                vec![n, n],
                            )
                            .unwrap(),
                        )
                    };
                    let result = block_on(eigs_builtin_context(
                        Some(runtime),
                        input,
                        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(
                            1.,
                        )]),
                    ));
                    if limit == 0 {
                        let error = result.expect_err("zero budget must deny before dense work");
                        let mut source: &(dyn std::error::Error + 'static) = &error;
                        while !source.is::<crate::context::host_allocation::HostAllocationError>() {
                            source = source.source().expect("original admission cause");
                        }
                    } else {
                        let output = result.expect("funded full EVD");
                        let Value::Tensor(value) = &output else {
                            panic!("eigs subset retains matrix storage")
                        };
                        assert_eq!(
                            value.numeric_dtype(),
                            if single {
                                NumericDType::F32
                            } else {
                                NumericDType::F64
                            }
                        );
                        assert_eq!(value.shape(), [1, 1]);
                        assert!(
                            (value.numeric_value_at(0).unwrap().materialize_f64() - 8.).abs()
                                < 1e-5
                        );
                        assert!(ledger.snapshot().live_claimed_bytes > 0);
                        drop(output);
                    }
                    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
                    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
                }
            }
        }
    }

    #[test]
    fn eigs_defaults_to_largest_magnitude_six_clamped() {
        let a = real_matrix(vec![1.0, 0.0, 0.0, 4.0], 2, 2);
        let out = tensor(call(a, Vec::new()).unwrap());
        assert_eq!(out.shape(), [2, 1]);
        assert_eq!(out.materialize_f64(), vec![4.0, 1.0]);
    }

    #[test]
    fn eigs_selects_smallestabs_from_sparse_input() {
        let sparse =
            SparseTensor::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![10.0, 2.0, -4.0])
                .unwrap();
        let out = tensor(
            call(
                Value::SparseTensor(sparse),
                vec![
                    Value::Num(2.0),
                    Value::CharArray(CharArray::new_row("smallestabs")),
                ],
            )
            .unwrap(),
        );
        for (actual, expected) in out.materialize_f64().iter().zip([2.0, -4.0]) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn eigs_supports_two_and_three_output_forms() {
        let guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let a = real_matrix(vec![2.0, 0.0, 0.0, 5.0], 2, 2);
        let out = outputs(call(a, vec![Value::Num(1.0)]).unwrap());
        drop(guard);
        assert_eq!(out.len(), 3);
        assert_eq!(tensor(out[0].clone()).shape(), [2, 1]);
        let d = tensor(out[1].clone());
        assert_eq!(d.shape(), [1, 1]);
        assert_eq!(d.materialize_f64(), vec![5.0]);
        assert_eq!(out[2], Value::Num(0.0));
    }

    #[test]
    fn eigs_numeric_sigma_selects_nearest() {
        let a = real_matrix(vec![1.0, 0.0, 0.0, 8.0], 2, 2);
        let out = tensor(call(a, vec![Value::Num(1.0), Value::Num(7.25)]).unwrap());
        assert_eq!(out.materialize_f64(), vec![8.0]);
    }

    #[test]
    fn eigs_reads_integer_tensor_k_and_sigma_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let a = real_matrix(vec![1.0, 0.0, 0.0, 8.0], 2, 2);
        let k = Tensor::new_integer(IntegerStorage::U64(vec![1]), vec![1, 1]).unwrap();
        let sigma = Tensor::new_integer(IntegerStorage::I16(vec![7]), vec![1, 1]).unwrap();
        let out = tensor(call(a, vec![Value::Tensor(k), Value::Tensor(sigma)]).unwrap());
        assert_eq!(out.materialize_f64(), vec![8.0]);
    }

    #[test]
    fn eigs_reads_integer_tensor_vector_options_exactly() {
        let permutation =
            Tensor::new_integer(IntegerStorage::U16(vec![2, 1, 3]), vec![1, 3]).unwrap();
        assert_eq!(
            parse_permutation(&Value::Tensor(permutation), 3)
                .unwrap()
                .as_ref(),
            &[2, 1, 3]
        );

        let initial = Tensor::new_integer(IntegerStorage::I16(vec![3, 2, 1]), vec![3, 1]).unwrap();
        assert_eq!(
            numeric_vector_len(&Value::Tensor(initial.clone()), "StartVector").unwrap(),
            3
        );
        block_on(parse_named_option(
            None,
            "StartVector",
            &Value::Tensor(initial),
            3,
            1,
            &mut ParsedOptions::default(),
        ))
        .expect("actual dense start-vector admission");
        let wide = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![1, (1u64 << 53) + 1, 3]),
                vec![3, 1],
            )
            .unwrap(),
        );
        let error = block_on(parse_named_option(
            None,
            "StartVector",
            &wide,
            3,
            1,
            &mut ParsedOptions::default(),
        ))
        .expect_err("actual role rejects wide native integers");
        assert!(error.message().contains("exactly representable as double"));

        let fail = Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).unwrap();
        assert!(parse_bool(&Value::Tensor(fail), "Fail").unwrap());
    }

    #[test]
    fn eigs_generalized_reuses_eig_path() {
        let a = real_matrix(vec![2.0, 0.0, 0.0, 9.0], 2, 2);
        let b = real_matrix(vec![1.0, 0.0, 0.0, 3.0], 2, 2);
        let out = tensor(
            call(
                a,
                vec![
                    b,
                    Value::Num(1.0),
                    Value::CharArray(CharArray::new_row("smallestreal")),
                ],
            )
            .unwrap(),
        );
        assert_eq!(out.materialize_f64(), vec![2.0]);
    }

    #[test]
    fn eigs_validates_options_without_affecting_exact_fallback() {
        let mut opts = runmat_value::UnboundedStructValueBuilder::new();
        opts.insert("tol", Value::Num(1e-9));
        let maxit = Tensor::new_integer(IntegerStorage::U16(vec![100]), vec![1, 1]).unwrap();
        opts.insert("maxit", Value::Tensor(maxit));
        let p = Tensor::new_integer(IntegerStorage::U16(vec![4]), vec![1, 1]).unwrap();
        opts.insert("p", Value::Tensor(p));
        opts.insert(
            "v0",
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap()),
        );
        opts.insert("fail", Value::CharArray(CharArray::new_row("keep")));
        let disp = Tensor::new_integer(IntegerStorage::U8(vec![0]), vec![1, 1]).unwrap();
        opts.insert("disp", Value::Tensor(disp));
        let a = real_matrix(vec![1.0, 0.0, 0.0, 3.0], 2, 2);
        let out = tensor(
            call(
                a,
                vec![
                    Value::Num(1.0),
                    Value::String("lm".into()),
                    Value::Struct(opts.finish()),
                ],
            )
            .unwrap(),
        );
        assert_eq!(out.materialize_f64(), vec![3.0]);
    }

    #[test]
    fn eigs_validates_generalized_option_sigma_constraints() {
        let a = real_matrix(vec![2.0, 0.0, 0.0, 9.0], 2, 2);
        let b = real_matrix(vec![1.0, 0.0, 0.0, 3.0], 2, 2);
        let mut chol_opts = runmat_value::UnboundedStructValueBuilder::new();
        chol_opts.insert("IsCholesky", Value::Bool(true));
        assert!(call(
            a.clone(),
            vec![
                b.clone(),
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Struct(chol_opts.finish())
            ],
        )
        .is_err());

        let mut spd_opts = runmat_value::UnboundedStructValueBuilder::new();
        spd_opts.insert("IsSymmetricDefinite", Value::Bool(true));
        assert!(call(
            a,
            vec![
                b,
                Value::Num(1.0),
                Value::CharArray(CharArray::new_row("largestabs")),
                Value::Struct(spd_opts.finish()),
            ],
        )
        .is_err());
    }

    #[test]
    fn eigs_rejects_too_small_subspace_dimension() {
        let a = real_matrix(vec![1.0, 0.0, 0.0, 3.0], 2, 2);
        let mut opts = runmat_value::UnboundedStructValueBuilder::new();
        opts.insert("p", Value::Num(2.0));
        assert!(call(
            a,
            vec![
                Value::Num(1.0),
                Value::String("lm".into()),
                Value::Struct(opts.finish())
            ]
        )
        .is_err());
    }

    #[test]
    fn eigs_positive_integer_options_read_typed_storage_and_reject_float_boundary() {
        let wide = if usize::BITS == 64 {
            9_007_199_254_740_993
        } else {
            u32::MAX as u64
        };
        let typed =
            Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("typed option");

        assert_eq!(
            positive_integer_option(&Value::Tensor(typed), "MaxIterations").expect("typed option"),
            wide as usize
        );

        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        assert!(positive_integer_option(&Value::Num(boundary), "MaxIterations").is_err());
        assert!(parse_k(&Value::Num(usize::MAX as f64 + 1.0)).is_err());
    }

    #[test]
    fn eigs_accepts_cholesky_factor_for_generalized_b() {
        let a = real_matrix(vec![2.0, 0.0, 0.0, 9.0], 2, 2);
        let r = real_matrix(vec![1.0, 0.0, 0.0, 2.0], 2, 2);
        let mut opts = runmat_value::UnboundedStructValueBuilder::new();
        opts.insert("IsCholesky", Value::Bool(true));
        let out = tensor(
            call(
                a,
                vec![
                    r,
                    Value::Num(1.0),
                    Value::CharArray(CharArray::new_row("smallestreal")),
                    Value::Struct(opts.finish()),
                ],
            )
            .unwrap(),
        );
        assert_eq!(out.materialize_f64(), vec![2.0]);
    }

    #[test]
    fn eigs_cholesky_factor_reads_typed_integer_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let a = real_matrix(vec![2.0, 0.0, 0.0, 9.0], 2, 2);
        let r = Tensor::new_integer(IntegerStorage::U16(vec![1, 0, 0, 2]), vec![2, 2]).unwrap();
        let mut opts = runmat_value::UnboundedStructValueBuilder::new();
        opts.insert("IsCholesky", Value::Bool(true));

        let out = tensor(
            call(
                a,
                vec![
                    Value::Tensor(r),
                    Value::Num(1.0),
                    Value::CharArray(CharArray::new_row("smallestreal")),
                    Value::Struct(opts.finish()),
                ],
            )
            .unwrap(),
        );

        assert_eq!(out.materialize_f64(), vec![2.0]);
    }

    #[test]
    fn eigs_start_vector_length_reads_typed_integer_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let a = real_matrix(vec![1.0, 0.0, 0.0, 3.0], 2, 2);
        let start = Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![2, 1]).unwrap();
        let mut opts = runmat_value::UnboundedStructValueBuilder::new();
        opts.insert("StartVector", Value::Tensor(start));

        let out = tensor(call(a, vec![Value::Num(1.0), Value::Struct(opts.finish())]).unwrap());

        assert_eq!(out.shape(), [1, 1]);
        assert_eq!(out.materialize_f64(), vec![3.0]);
    }

    #[test]
    fn eigs_rejects_bad_k_sigma_and_function_handle() {
        let a = real_matrix(vec![1.0, 0.0, 0.0, 3.0], 2, 2);
        assert!(call(a.clone(), vec![Value::Num(0.0)]).is_err());
        assert!(call(
            a.clone(),
            vec![Value::Num(1.0), Value::CharArray(CharArray::new_row("bad"))],
        )
        .is_err());
        assert!(call(Value::FunctionHandle("afun".into()), vec![Value::Num(2.0)]).is_err());
    }
}
