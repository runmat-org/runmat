//! Latin hypercube experimental designs.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random;
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "lhsdesign";
const DEFAULT_ITERATIONS: usize = 5;
const MAX_SCORE_TERMS: usize = 200_000_000;

const OUTPUT_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "N-by-P Latin hypercube design.",
}];

const PARAM_N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of sample points.",
};

const PARAM_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of variables.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options such as Smooth, Criterion, and Iterations.",
};

const INPUTS_N_P: [BuiltinParamDescriptor; 2] = [PARAM_N, PARAM_P];
const INPUTS_N_P_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_N, PARAM_P, PARAM_OPTIONS];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "X = lhsdesign(n, p)",
        inputs: &INPUTS_N_P,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "X = lhsdesign(n, p, Name, Value)",
        inputs: &INPUTS_N_P_OPTIONS,
        outputs: &OUTPUT_X,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LHSDESIGN.INVALID_ARGUMENT",
    identifier: Some("RunMat:lhsdesign:InvalidArgument"),
    when: "Sample counts, dimensions, criteria, or name-value options are malformed.",
    message: "lhsdesign: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LHSDESIGN.INTERNAL",
    identifier: Some("RunMat:lhsdesign:Internal"),
    when: "Internal random generation or tensor allocation fails.",
    message: "lhsdesign: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const LHSDESIGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

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

fn lhs_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    if args.len() >= 2 {
        Type::Unknown
    } else {
        Type::Num
    }
}

#[runtime_builtin(
    name = "lhsdesign",
    category = "stats/random",
    summary = "Generate a Latin hypercube sample design.",
    keywords = "lhsdesign,latin hypercube,design of experiments,random,statistics",
    type_resolver(lhs_type),
    descriptor(crate::builtins::stats::random::lhsdesign::LHSDESIGN_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::random::lhsdesign"
)]
pub(crate) async fn lhsdesign_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let options = parse_args(args)?;
    let design = compute_lhsdesign(options)?;
    Tensor::new(design, vec![options.n, options.p])
        .map(tensor::tensor_into_value)
        .map_err(|err| internal(format!("lhsdesign: {err}")))
}

fn parse_args(args: Vec<Value>) -> BuiltinResult<LhsOptions> {
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
        let Some(name) = keyword_of(&args[idx]) else {
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
    let number = scalar_f64(value).ok_or_else(|| {
        invalid(format!(
            "lhsdesign: {label} must be a positive integer scalar"
        ))
    })?;
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
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data.first().copied(),
        _ => None,
    }
}

fn parse_on_off_bool(value: &Value, label: &str) -> BuiltinResult<bool> {
    if let Some(keyword) = keyword_of(value) {
        return match keyword.as_str() {
            "on" | "true" => Ok(true),
            "off" | "false" => Ok(false),
            _ => Err(invalid(format!("lhsdesign: {label} must be 'on' or 'off'"))),
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
    let Some(keyword) = keyword_of(value) else {
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

fn compute_lhsdesign(options: LhsOptions) -> BuiltinResult<Vec<f64>> {
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
        let candidate = candidate_design(options.n, options.p, len, smooth)?;
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

fn candidate_design(n: usize, p: usize, len: usize, smooth: bool) -> BuiltinResult<Vec<f64>> {
    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(|_| invalid("lhsdesign: requested design is too large"))?;
    data.resize(len, 0.0);
    for col in 0..p {
        let permutation = random_permutation(n)?;
        let jitter = if smooth {
            Some(random::generate_uniform(n, NAME)?)
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

fn random_permutation(n: usize) -> BuiltinResult<Vec<usize>> {
    let uniforms = random::generate_uniform(n.saturating_sub(1), NAME)?;
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

    fn tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn reset_rng() -> std::sync::MutexGuard<'static, ()> {
        let guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        guard
    }

    #[test]
    fn basic_design_has_latin_bins_per_column() {
        let _guard = reset_rng();
        let out = block_on(lhsdesign_builtin(vec![Value::Num(6.0), Value::Num(3.0)])).unwrap();
        let tensor = tensor(out);
        assert_eq!(tensor.shape, vec![6, 3]);
        assert!(tensor.data.iter().all(|value| *value > 0.0 && *value < 1.0));
        for col in 0..3 {
            let mut bins = (0..6)
                .map(|row| (tensor.data[row + col * 6] * 6.0).floor() as usize)
                .collect::<Vec<_>>();
            bins.sort_unstable();
            assert_eq!(bins, vec![0, 1, 2, 3, 4, 5]);
        }
    }

    #[test]
    fn smooth_off_uses_interval_midpoints() {
        let _guard = reset_rng();
        let out = block_on(lhsdesign_builtin(vec![
            Value::Num(4.0),
            Value::Num(2.0),
            Value::from("Smooth"),
            Value::from("off"),
            Value::from("Criterion"),
            Value::from("none"),
        ]))
        .unwrap();
        let tensor = tensor(out);
        for value in tensor.data {
            let scaled = value * 4.0;
            assert!((scaled.fract() - 0.5).abs() < 1.0e-12);
        }
    }

    #[test]
    fn criterion_correlation_and_iterations_are_accepted() {
        let _guard = reset_rng();
        let out = block_on(lhsdesign_builtin(vec![
            Value::Num(8.0),
            Value::Num(3.0),
            Value::from("Criterion"),
            Value::from("correlation"),
            Value::from("Iterations"),
            Value::Num(4.0),
        ]))
        .unwrap();
        let tensor = tensor(out);
        assert_eq!(tensor.shape, vec![8, 3]);
        for value in tensor.data {
            let scaled = value * 8.0;
            assert!((scaled.fract() - 0.5).abs() < 1.0e-12);
        }
    }

    #[test]
    fn invalid_options_are_rejected() {
        let err = block_on(lhsdesign_builtin(vec![
            Value::Num(4.0),
            Value::Num(2.0),
            Value::from("Criterion"),
            Value::from("bad"),
        ]))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:lhsdesign:InvalidArgument"));
    }

    #[test]
    fn invalid_name_value_arity_is_rejected() {
        let err = block_on(lhsdesign_builtin(vec![
            Value::Num(4.0),
            Value::Num(2.0),
            Value::from("Smooth"),
        ]))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:lhsdesign:InvalidArgument"));
    }

    #[test]
    fn excessive_scoring_work_is_rejected() {
        let err = block_on(lhsdesign_builtin(vec![
            Value::Num(20_000.0),
            Value::Num(8.0),
            Value::from("Iterations"),
            Value::Num(5.0),
        ]))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:lhsdesign:InvalidArgument"));
    }

    #[test]
    fn typed_integer_counts_are_exact_and_lossy_f64_is_rejected() {
        assert_eq!(
            positive_usize(&Value::Int(runmat_builtins::IntValue::U16(3)), "n").unwrap(),
            3
        );
        assert!(positive_usize(&Value::Int(runmat_builtins::IntValue::I8(-1)), "n").is_err());
        assert!(positive_usize(&Value::Num(1.5), "n").is_err());
        assert!(positive_usize(&Value::Num(usize::MAX as f64 + 1.0), "n").is_err());
    }
}
