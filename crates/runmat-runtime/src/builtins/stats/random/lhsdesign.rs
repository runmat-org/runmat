//! Latin hypercube experimental designs.
use runmat_builtins::{
    LHSDESIGN_ERROR_INTERNAL as ERROR_INTERNAL,
    LHSDESIGN_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT,
    LHSDESIGN_INTEGER_DIMENSION_EXTENSION as INTEGER_DIMENSION_EXTENSION,
    LHSDESIGN_INTEGER_ITERATIONS_EXTENSION as INTEGER_ITERATIONS_EXTENSION,
    LHSDESIGN_INTEGER_SMOOTH_EXTENSION as INTEGER_SMOOTH_EXTENSION,
    LHSDESIGN_RESIDENT_INPUT_EXTENSION as RESIDENT_INPUT_EXTENSION,
};

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::NumericScalar;
use runmat_value::{Tensor, Value};

use crate::builtins::common::random;
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "lhsdesign";
const DEFAULT_ITERATIONS: usize = 5;
const MAX_SCORE_TERMS: usize = 200_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Criterion {
    None,
    Maximin,
    Correlation,
}

#[derive(Clone, Copy, Debug)]
struct LhsOptions {
    n: usize,
    p: usize,
    smooth: bool,
    criterion: Criterion,
    iterations: usize,
}

fn lhs_error(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    lhs_error(&ERROR_INVALID_ARGUMENT, message)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    lhs_error(&ERROR_INTERNAL, message)
}

#[runtime_builtin(
    name = "lhsdesign",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::random::lhsdesign"
)]
pub(crate) async fn lhsdesign_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    ensure_extensions(args.view())?;
    let args = gather_args(args).await?;
    let options = parse_args(args)?;
    let mut turn = random::acquire_turn(None, NAME).await?;
    let design = compute_lhsdesign(&mut turn, options)?;
    Tensor::new(design, vec![options.n, options.p])
        .map(tensor::tensor_into_value)
        .map_err(|err| internal(format!("lhsdesign: {err}")))
}

fn is_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn contains_explicit_gpu(value: &Value) -> bool {
    match value {
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_explicit(handle),
        Value::Cell(cell) => cell.data().iter().any(contains_explicit_gpu),
        Value::Struct(value) => value.fields.values().any(contains_explicit_gpu),
        Value::Object(value) => value.properties().values().any(contains_explicit_gpu),
        Value::Closure(value) => value.captures.iter().any(contains_explicit_gpu),
        Value::OutputList(values) => values.iter().any(contains_explicit_gpu),
        _ => false,
    }
}

fn ensure_extensions(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    if args.iter().take(2).any(is_typed_integer) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_DIMENSION_EXTENSION, NAME)?;
    }
    for pair in args.range(args.len().min(2)..)?.chunks_exact(2) {
        if !is_typed_integer(&pair[1]) {
            continue;
        }
        let extension = match keyword_of(&pair[0])?.as_deref() {
            Some("iterations") => &INTEGER_ITERATIONS_EXTENSION,
            Some("smooth") => &INTEGER_SMOOTH_EXTENSION,
            _ => continue,
        };
        crate::compatibility::ensure_builtin_extension_enabled(extension, NAME)?;
    }
    if args.iter().any(contains_explicit_gpu) {
        crate::compatibility::ensure_builtin_extension_enabled(&RESIDENT_INPUT_EXTENSION, NAME)?;
    }
    Ok(())
}

async fn gather_args(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    let runtime = crate::context::legacy::active();
    crate::dispatcher::gather_builtin_arguments_async(args, runtime.as_ref())
        .await
        .map_err(|source| {
            let mut error = invalid(format!("lhsdesign: {source}"));
            error.gpu_gather_retry = source.gpu_gather_retry;
            error.source = Some(Box::new(source));
            error
        })
}

fn parse_args(args: crate::call::BuiltinArguments) -> BuiltinResult<LhsOptions> {
    if args.len() < 2 {
        return Err(invalid("lhsdesign: n and p are required"));
    }
    let n = positive_usize(&args[0], "n")?;
    let p = positive_usize(&args[1], "p")?;
    let mut options = LhsOptions {
        n,
        p,
        smooth: true,
        criterion: Criterion::Maximin,
        iterations: DEFAULT_ITERATIONS,
    };
    let mut idx = 2usize;
    while idx < args.len() {
        let Some(name) = keyword_of(&args[idx])? else {
            return Err(invalid("lhsdesign: options must be name-value pairs"));
        };
        idx += 1;
        if idx >= args.len() {
            return Err(invalid(format!("lhsdesign: {name} requires a value")));
        }
        match name.as_str() {
            "smooth" => options.smooth = parse_on_off_bool(&args[idx], "Smooth")?,
            "criterion" => options.criterion = parse_criterion(&args[idx])?,
            "iterations" => options.iterations = positive_usize(&args[idx], "Iterations")?,
            other => return Err(invalid(format!("lhsdesign: unsupported option '{other}'"))),
        }
        idx += 1;
    }
    Ok(options)
}

fn positive_usize(value: &Value, label: &str) -> BuiltinResult<usize> {
    if let Value::Int(value) = value {
        return value
            .try_to_usize()
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                invalid(format!(
                    "lhsdesign: {label} must be a positive integer scalar"
                ))
            });
    }
    if let Value::Tensor(tensor) = value {
        if !tensor::is_scalar_tensor(tensor) {
            return Err(invalid(format!(
                "lhsdesign: {label} must be a positive integer scalar"
            )));
        }
        return positive_usize_scalar(
            tensor
                .numeric_value_at(0)
                .expect("validated scalar tensor has one numeric value"),
            label,
        );
    }
    let number = match value {
        Value::Num(number) => *number,
        _ => {
            return Err(invalid(format!(
                "lhsdesign: {label} must be a positive integer scalar"
            )))
        }
    };
    positive_floating_usize(number, label)
}

fn positive_usize_scalar(value: NumericScalar, label: &str) -> BuiltinResult<usize> {
    let parsed = match value {
        NumericScalar::I8(value) => usize::try_from(value).ok(),
        NumericScalar::I16(value) => usize::try_from(value).ok(),
        NumericScalar::I32(value) => usize::try_from(value).ok(),
        NumericScalar::I64(value) => usize::try_from(value).ok(),
        NumericScalar::U8(value) => Some(usize::from(value)),
        NumericScalar::U16(value) => Some(usize::from(value)),
        NumericScalar::U32(value) => usize::try_from(value).ok(),
        NumericScalar::U64(value) => usize::try_from(value).ok(),
        NumericScalar::F32(value) => return positive_floating_usize(f64::from(value), label),
        NumericScalar::F64(value) => return positive_floating_usize(value, label),
    };
    parsed.filter(|value| *value > 0).ok_or_else(|| {
        invalid(format!(
            "lhsdesign: {label} must be a positive integer scalar"
        ))
    })
}

fn positive_floating_usize(number: f64, label: &str) -> BuiltinResult<usize> {
    if !(number.is_finite() && number >= 1.0 && number.fract() == 0.0) {
        return Err(invalid(format!(
            "lhsdesign: {label} must be a positive integer scalar"
        )));
    }
    if number > usize::MAX as f64 || (usize::BITS == 64 && number == usize::MAX as f64) {
        return Err(invalid(format!("lhsdesign: {label} is too large")));
    }
    Ok(number as usize)
}

fn scalar_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Int(value) => Some(value.to_f64()),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Some(tensor::tensor_value_f64(tensor, 0))
        }
        _ => None,
    }
}

fn parse_on_off_bool(value: &Value, label: &str) -> BuiltinResult<bool> {
    if let Some(keyword) = keyword_of(value)? {
        return match keyword.as_str() {
            "on" | "true" => Ok(true),
            "off" | "false" => Ok(false),
            _ => Err(invalid(format!("lhsdesign: {label} must be 'on' or 'off'"))),
        };
    }
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return match integer.try_to_usize() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(invalid(format!(
                "lhsdesign: {label} must be 'on', 'off', or logical"
            ))),
        };
    }
    let number = scalar_f64(value).ok_or_else(|| {
        invalid(format!(
            "lhsdesign: {label} must be 'on', 'off', or logical"
        ))
    })?;
    if number == 0.0 {
        Ok(false)
    } else if number == 1.0 {
        Ok(true)
    } else {
        Err(invalid(format!(
            "lhsdesign: {label} must be 'on', 'off', or logical"
        )))
    }
}

fn parse_criterion(value: &Value) -> BuiltinResult<Criterion> {
    let Some(keyword) = keyword_of(value)? else {
        return Err(invalid(
            "lhsdesign: Criterion must be 'maximin', 'correlation', or 'none'",
        ));
    };
    match keyword.as_str() {
        "none" => Ok(Criterion::None),
        "maximin" => Ok(Criterion::Maximin),
        "correlation" => Ok(Criterion::Correlation),
        _ => Err(invalid(
            "lhsdesign: Criterion must be 'maximin', 'correlation', or 'none'",
        )),
    }
}

fn compute_lhsdesign(
    turn: &mut random::RandomTurn,
    options: LhsOptions,
) -> BuiltinResult<Vec<f64>> {
    let len = options
        .n
        .checked_mul(options.p)
        .ok_or_else(|| invalid("lhsdesign: requested design is too large"))?;
    if len == 0 {
        return Ok(Vec::new());
    }
    let candidates = if matches!(options.criterion, Criterion::None) {
        1
    } else {
        options.iterations
    };
    validate_score_budget(options.n, options.p, candidates, options.criterion)?;
    let smooth = options.smooth && options.criterion != Criterion::Correlation;
    let mut best = Vec::new();
    let mut best_score = f64::NEG_INFINITY;
    for _ in 0..candidates {
        let candidate = candidate_design(turn, options.n, options.p, len, smooth)?;
        let score = design_score(&candidate, options.n, options.p, options.criterion);
        if best.is_empty() || score > best_score {
            best_score = score;
            best = candidate;
        }
    }
    Ok(best)
}

fn validate_score_budget(
    n: usize,
    p: usize,
    candidates: usize,
    criterion: Criterion,
) -> BuiltinResult<()> {
    let terms = match criterion {
        Criterion::None => Some(0),
        Criterion::Maximin => n
            .checked_mul(n.saturating_sub(1))
            .and_then(|value| value.checked_div(2))
            .and_then(|value| value.checked_mul(p))
            .and_then(|value| value.checked_mul(candidates)),
        Criterion::Correlation => p
            .checked_mul(p.saturating_sub(1))
            .and_then(|value| value.checked_div(2))
            .and_then(|value| value.checked_mul(n))
            .and_then(|value| value.checked_mul(candidates)),
    }
    .ok_or_else(|| invalid("lhsdesign: requested design scoring work is too large"))?;
    if terms > MAX_SCORE_TERMS {
        return Err(invalid(
            "lhsdesign: requested design scoring work is too large",
        ));
    }
    Ok(())
}

fn candidate_design(
    turn: &mut random::RandomTurn,
    n: usize,
    p: usize,
    len: usize,
    smooth: bool,
) -> BuiltinResult<Vec<f64>> {
    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(|_| invalid("lhsdesign: requested design is too large"))?;
    data.resize(len, 0.0);
    for col in 0..p {
        let permutation = random_permutation(turn, n)?;
        let jitter = if smooth {
            Some(random::generate_uniform(turn, n, NAME)?)
        } else {
            None
        };
        for row in 0..n {
            let offset = jitter.as_ref().map(|values| values[row]).unwrap_or(0.5);
            data[row + col * n] = (permutation[row] as f64 + offset) / n as f64;
        }
    }
    Ok(data)
}

fn random_permutation(turn: &mut random::RandomTurn, n: usize) -> BuiltinResult<Vec<usize>> {
    let uniforms = random::generate_uniform(turn, n.saturating_sub(1), NAME)?;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(n)
        .map_err(|_| invalid("lhsdesign: requested design is too large"))?;
    indices.extend(0..n);
    for i in (1..n).rev() {
        let u = uniforms[n - 1 - i];
        let j = ((u * (i + 1) as f64).floor() as usize).min(i);
        indices.swap(i, j);
    }
    Ok(indices)
}

fn design_score(data: &[f64], n: usize, p: usize, criterion: Criterion) -> f64 {
    match criterion {
        Criterion::None => 0.0,
        Criterion::Maximin => minimum_pairwise_distance_sq(data, n, p),
        Criterion::Correlation => -sum_squared_column_correlations(data, n, p),
    }
}

fn minimum_pairwise_distance_sq(data: &[f64], n: usize, p: usize) -> f64 {
    if n < 2 {
        return f64::INFINITY;
    }
    let mut best = f64::INFINITY;
    for a in 0..n - 1 {
        for b in a + 1..n {
            let mut dist = 0.0;
            for col in 0..p {
                let delta = data[a + col * n] - data[b + col * n];
                dist += delta * delta;
            }
            best = best.min(dist);
        }
    }
    best
}

fn sum_squared_column_correlations(data: &[f64], n: usize, p: usize) -> f64 {
    if p < 2 || n < 2 {
        return 0.0;
    }
    let mut means = vec![0.0; p];
    for col in 0..p {
        means[col] = (0..n).map(|row| data[row + col * n]).sum::<f64>() / n as f64;
    }
    let mut centered_norms = vec![0.0; p];
    for col in 0..p {
        centered_norms[col] = (0..n)
            .map(|row| {
                let centered = data[row + col * n] - means[col];
                centered * centered
            })
            .sum::<f64>()
            .sqrt();
    }
    let mut sum = 0.0;
    for a in 0..p - 1 {
        for b in a + 1..p {
            let denom = centered_norms[a] * centered_norms[b];
            if denom == 0.0 {
                continue;
            }
            let dot = (0..n)
                .map(|row| (data[row + a * n] - means[a]) * (data[row + b * n] - means[b]))
                .sum::<f64>();
            let corr = dot / denom;
            sum += corr * corr;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::IntegerStorage;

    fn tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn reset_rng() -> impl Drop {
        let guard = random::test_guard();
        random::reset_rng();
        guard
    }

    fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        Value::Tensor(tensor)
    }

    #[test]
    fn basic_design_has_latin_bins_per_column() {
        let _guard = reset_rng();
        let out = block_on(lhsdesign_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(6.0),
                Value::Num(3.0),
            ]),
        ))
        .unwrap();
        let tensor = tensor(out);
        assert_eq!(tensor.shape(), &[6, 3]);
        assert!(tensor
            .materialize_f64()
            .iter()
            .all(|value| *value > 0.0 && *value < 1.0));
        for col in 0..3 {
            let mut bins = (0..6)
                .map(|row| (tensor.materialize_f64()[row + col * 6] * 6.0).floor() as usize)
                .collect::<Vec<_>>();
            bins.sort_unstable();
            assert_eq!(bins, vec![0, 1, 2, 3, 4, 5]);
        }
    }

    #[test]
    fn smooth_off_uses_interval_midpoints() {
        let _guard = reset_rng();
        let out = block_on(lhsdesign_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(4.0),
                Value::Num(2.0),
                Value::from("Smooth"),
                Value::from("off"),
                Value::from("Criterion"),
                Value::from("none"),
            ]),
        ))
        .unwrap();
        let tensor = tensor(out);
        for value in tensor.materialize_f64() {
            let scaled = value * 4.0;
            assert!((scaled.fract() - 0.5).abs() < 1.0e-12);
        }
    }

    #[test]
    fn criterion_correlation_and_iterations_are_accepted() {
        let _guard = reset_rng();
        let out = block_on(lhsdesign_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(8.0),
                Value::Num(3.0),
                Value::from("Criterion"),
                Value::from("correlation"),
                Value::from("Iterations"),
                Value::Num(4.0),
            ]),
        ))
        .unwrap();
        let tensor = tensor(out);
        assert_eq!(tensor.shape(), &[8, 3]);
        for value in tensor.materialize_f64() {
            let scaled = value * 8.0;
            assert!((scaled.fract() - 0.5).abs() < 1.0e-12);
        }
    }

    #[test]
    fn typed_integer_scalar_arguments_are_exact() {
        let _guard = reset_rng();
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let out = block_on(lhsdesign_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                poisoned_int_tensor(IntegerStorage::U16(vec![4]), vec![1, 1]),
                poisoned_int_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]),
                Value::from("Smooth"),
                poisoned_int_tensor(IntegerStorage::U8(vec![0]), vec![1, 1]),
                Value::from("Criterion"),
                Value::from("correlation"),
                Value::from("Iterations"),
                poisoned_int_tensor(IntegerStorage::U8(vec![2]), vec![1, 1]),
            ]),
        ))
        .expect("lhsdesign");
        let tensor = tensor(out);
        assert_eq!(tensor.shape(), &[4, 2]);
        for value in tensor.materialize_f64() {
            let scaled = value * 4.0;
            assert!((scaled.fract() - 0.5).abs() < 1.0e-12);
        }
    }

    #[test]
    fn smooth_reads_every_integer_storage_variant_not_the_float_mirror() {
        for storage in [
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ] {
            assert!(
                parse_on_off_bool(&poisoned_int_tensor(storage, vec![1, 1]), "Smooth").unwrap()
            );
        }
    }

    #[test]
    fn invalid_options_are_rejected() {
        let err = block_on(lhsdesign_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(4.0),
                Value::Num(2.0),
                Value::from("Criterion"),
                Value::from("bad"),
            ]),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:lhsdesign:InvalidArgument"));
    }

    #[test]
    fn invalid_name_value_arity_is_rejected() {
        let err = block_on(lhsdesign_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(4.0),
                Value::Num(2.0),
                Value::from("Smooth"),
            ]),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:lhsdesign:InvalidArgument"));
    }

    #[test]
    fn excessive_scoring_work_is_rejected() {
        let err = block_on(lhsdesign_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(20_000.0),
                Value::Num(8.0),
                Value::from("Iterations"),
                Value::Num(5.0),
            ]),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:lhsdesign:InvalidArgument"));
    }

    #[test]
    fn typed_integer_counts_are_exact_and_lossy_f64_is_rejected() {
        assert_eq!(
            positive_usize(&Value::Int(runmat_value::IntValue::U16(3)), "n").unwrap(),
            3
        );
        assert!(positive_usize(&Value::Int(runmat_value::IntValue::I8(-1)), "n").is_err());
        assert!(positive_usize(&Value::Num(1.5), "n").is_err());
        assert!(positive_usize(&Value::Num(usize::MAX as f64 + 1.0), "n").is_err());
    }

    #[test]
    fn typed_integer_roles_are_independently_gated() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let cases = [
            (
                vec![Value::Int(runmat_value::IntValue::U16(4)), Value::Num(2.0)],
                "RunMat:compatibility:LhsdesignIntegerDimensionExtension",
            ),
            (
                vec![
                    Value::Num(4.0),
                    Value::Num(2.0),
                    Value::from("Iterations"),
                    Value::Int(runmat_value::IntValue::U8(2)),
                ],
                "RunMat:compatibility:LhsdesignIntegerIterationsExtension",
            ),
            (
                vec![
                    Value::Num(4.0),
                    Value::Num(2.0),
                    Value::from("Smooth"),
                    Value::Int(runmat_value::IntValue::U8(1)),
                ],
                "RunMat:compatibility:LhsdesignIntegerSmoothExtension",
            ),
        ];
        for (args, identifier) in cases {
            let error = block_on(lhsdesign_builtin(
                crate::builtins::common::test_support::builtin_arguments(args),
            ))
            .unwrap_err();
            assert_eq!(error.identifier(), Some(identifier));
        }
    }

    #[test]
    fn typed_integer_counts_use_authoritative_storage_and_platform_bounds() {
        assert_eq!(
            positive_usize(
                &poisoned_int_tensor(IntegerStorage::U64(vec![7]), vec![1, 1]),
                "n",
            )
            .unwrap(),
            7
        );
        let platform_limit = positive_usize(
            &poisoned_int_tensor(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]),
            "n",
        );
        if usize::BITS == 64 {
            assert_eq!(platform_limit.unwrap(), usize::MAX);
        } else {
            assert!(platform_limit.is_err());
        }
    }

    #[test]
    fn explicit_gpu_fallback_is_gated_but_automatic_residency_is_transparent() {
        use crate::builtins::common::{gpu_helpers, test_support};

        let _guard = reset_rng();
        test_support::with_test_provider(|provider| {
            let count = Tensor::new(vec![4.0], vec![1, 1]).unwrap();
            let explicit = gpu_helpers::upload_tensor(provider, &count).expect("upload");
            let explicit = explicit
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("validated provenance");
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(lhsdesign_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::GpuTensor(explicit),
                    Value::Num(2.0),
                ]),
            ))
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:LhsdesignExplicitGpuInputExtension")
            );

            let automatic = gpu_helpers::upload_tensor(provider, &count).expect("upload");
            let automatic = automatic
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .expect("validated provenance");
            let output = block_on(lhsdesign_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::GpuTensor(automatic),
                    Value::Num(2.0),
                ]),
            ))
            .expect("automatic residency may gather transparently");
            let Value::Tensor(output) = output else {
                panic!("lhsdesign remains a host implementation");
            };
            assert_eq!(output.shape(), &[4, 2]);
        });
    }
}
