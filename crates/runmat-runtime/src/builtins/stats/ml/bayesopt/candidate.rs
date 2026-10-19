use super::*;

#[derive(Clone)]
pub(super) struct Point {
    pub(super) values: Vec<VariableValue>,
    pub(super) encoded: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct PointKey(Vec<PointCoordinate>);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum PointCoordinate {
    Numeric(u64),
    Categorical(String),
}

impl Point {
    pub(super) fn key(&self) -> PointKey {
        PointKey(self.values.iter().map(VariableValue::key).collect())
    }

    pub(super) fn to_table(&self, variables: &[Variable]) -> BuiltinResult<Value> {
        let mut names = Vec::with_capacity(variables.len());
        let mut columns = Vec::with_capacity(variables.len());
        for (variable, value) in variables.iter().zip(&self.values) {
            names.push(variable.name.clone());
            columns.push(value.column_value()?);
        }
        crate::builtins::table::table_from_columns(names, columns).map_err(|err| {
            bayesopt_error_with_source(
                format!(
                    "bayesopt: candidate table construction failed: {}",
                    err.message()
                ),
                &ERROR_INVALID_ARGUMENT,
                err,
            )
        })
    }
}

#[derive(Clone)]
pub(super) enum VariableValue {
    Numeric(f64),
    Categorical(String),
}

impl VariableValue {
    fn key(&self) -> PointCoordinate {
        match self {
            Self::Numeric(value) => {
                PointCoordinate::Numeric(if *value == 0.0 { 0.0f64 } else { *value }.to_bits())
            }
            Self::Categorical(value) => PointCoordinate::Categorical(value.clone()),
        }
    }

    fn column_value(&self) -> BuiltinResult<Value> {
        match self {
            Self::Numeric(value) => Tensor::new(vec![*value], vec![1, 1])
                .map(Value::Tensor)
                .map_err(|err| bayesopt_error(format!("bayesopt: {err}"), &ERROR_INVALID_ARGUMENT)),
            Self::Categorical(value) => StringArray::new(vec![value.clone()], vec![1, 1])
                .map(Value::StringArray)
                .map_err(|err| bayesopt_error(format!("bayesopt: {err}"), &ERROR_INVALID_ARGUMENT)),
        }
    }
}

pub(super) fn candidate_set(
    runtime: Option<&crate::context::RuntimeContext>,
    variables: &[Variable],
    max_evals: usize,
) -> BuiltinResult<Vec<Point>> {
    let active_dims = variables.iter().filter(|var| var.optimize).count().max(1);
    let mut count = (max_evals * 24).clamp(128, MAX_CANDIDATES);
    count = count.max(max_evals);
    let mut candidates = Vec::with_capacity(count + 2);
    candidates.push(point_from_unit(variables, &vec![0.5; active_dims]));
    candidates.push(point_from_unit(variables, &vec![0.0; active_dims]));
    candidates.push(point_from_unit(variables, &vec![1.0; active_dims]));
    for idx in 0..count {
        check_cancellation(runtime)?;
        let units = (0..active_dims)
            .map(|dim| halton(idx + 1, prime(dim)))
            .collect::<Vec<_>>();
        candidates.push(point_from_unit(variables, &units));
    }
    let mut seen = HashSet::new();
    candidates.retain(|point| seen.insert(point.key()));
    if candidates.is_empty() {
        return Err(bayesopt_error(
            "bayesopt: candidate generation produced no points",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok(candidates)
}

fn point_from_unit(variables: &[Variable], units: &[f64]) -> Point {
    let mut active = 0usize;
    let mut values = Vec::with_capacity(variables.len());
    let mut encoded = Vec::new();
    for variable in variables {
        let unit = if variable.optimize {
            let value = units.get(active).copied().unwrap_or(0.5).clamp(0.0, 1.0);
            active += 1;
            value
        } else {
            0.0
        };
        match &variable.kind {
            VariableKind::Real { lower, upper } => {
                let value = decode_real(*lower, *upper, variable.transform, unit);
                values.push(VariableValue::Numeric(value));
                if variable.optimize {
                    encoded.push(encode_real(*lower, *upper, variable.transform, value));
                }
            }
            VariableKind::Integer { lower, upper } => {
                let value = decode_integer(*lower, *upper, variable.transform, unit);
                values.push(VariableValue::Numeric(value as f64));
                if variable.optimize {
                    let denom = (i128::from(*upper) - i128::from(*lower)).max(1) as f64;
                    encoded.push((i128::from(value) - i128::from(*lower)) as f64 / denom);
                }
            }
            VariableKind::Categorical { categories } => {
                let len = categories.len();
                let idx = if variable.optimize {
                    ((unit * len as f64).floor() as usize).min(len.saturating_sub(1))
                } else {
                    0
                };
                values.push(VariableValue::Categorical(categories[idx].clone()));
                if variable.optimize {
                    if len <= 1 {
                        encoded.push(0.0);
                    } else {
                        encoded.push(idx as f64 / (len - 1) as f64);
                    }
                }
            }
        }
    }
    Point { values, encoded }
}

fn decode_real(lower: f64, upper: f64, transform: Transform, unit: f64) -> f64 {
    match transform {
        Transform::None if unit <= 0.0 => lower,
        Transform::None if unit >= 1.0 => upper,
        Transform::None => lower * (1.0 - unit) + upper * unit,
        Transform::Log => (lower.ln() + (upper.ln() - lower.ln()) * unit).exp(),
    }
}

fn encode_real(lower: f64, upper: f64, transform: Transform, value: f64) -> f64 {
    match transform {
        Transform::None if value <= lower => 0.0,
        Transform::None if value >= upper => 1.0,
        Transform::None if (upper - lower).is_finite() => (value - lower) / (upper - lower),
        Transform::None => (value * 0.5 - lower * 0.5) / (upper * 0.5 - lower * 0.5),
        Transform::Log => (value.ln() - lower.ln()) / (upper.ln() - lower.ln()),
    }
    .clamp(0.0, 1.0)
}

fn decode_integer(lower: i64, upper: i64, transform: Transform, unit: f64) -> i64 {
    if unit <= 0.0 {
        return lower;
    }
    if unit >= 1.0 {
        return upper;
    }
    let value = match transform {
        Transform::None => {
            let span = i128::from(upper) - i128::from(lower);
            i128::from(lower) + (span as f64 * unit).round() as i128
        }
        Transform::Log => {
            let lo = (lower.max(1) as f64).ln();
            let hi = (upper.max(1) as f64).ln();
            (lo + (hi - lo) * unit).exp().round() as i128
        }
    };
    value.clamp(i128::from(lower), i128::from(upper)) as i64
}

fn halton(mut index: usize, base: usize) -> f64 {
    let mut f = 1.0;
    let mut r = 0.0;
    while index > 0 {
        f /= base as f64;
        r += f * (index % base) as f64;
        index /= base;
    }
    r
}

fn prime(index: usize) -> usize {
    const PRIMES: [usize; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
    PRIMES[index % PRIMES.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_identity_keeps_text_categories_distinct_from_numeric_values() {
        let numeric = VariableValue::Numeric(1.0);
        let text = VariableValue::Categorical("1.00000000000000000".into());
        assert_ne!(numeric.key(), text.key());
        assert_eq!(
            VariableValue::Numeric(-0.0).key(),
            VariableValue::Numeric(0.0).key()
        );
    }

    #[test]
    fn wide_finite_real_range_does_not_make_nan_candidates() {
        let lower = -f64::MAX;
        let upper = f64::MAX;
        assert_eq!(decode_real(lower, upper, Transform::None, 0.0), lower);
        assert_eq!(decode_real(lower, upper, Transform::None, 1.0), upper);
        let middle = decode_real(lower, upper, Transform::None, 0.5);
        assert!(middle.is_finite());
        assert_eq!(encode_real(lower, upper, Transform::None, middle), 0.5);
    }

    #[test]
    fn full_signed_integer_span_never_overflows() {
        assert_eq!(
            decode_integer(i64::MIN, i64::MAX, Transform::None, 0.0),
            i64::MIN
        );
        assert_eq!(
            decode_integer(i64::MIN, i64::MAX, Transform::None, 1.0),
            i64::MAX
        );
        let middle = decode_integer(i64::MIN, i64::MAX, Transform::None, 0.5);
        assert_eq!(middle, 0);
    }
}
