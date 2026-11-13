//! Decode public reduction syntax exactly once, before materializing data.
use crate::builtins::common::{random_args::keyword_of, tensor};
use crate::{BuiltinResult, RuntimeError};
pub(super) use runmat_builtins::StatisticalMeanKind as ReductionKind;
use runmat_value::Value;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Axes {
    Default,
    All,
    Dims(Vec<usize>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MissingMode {
    Include,
    Omit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Keyword {
    All,
    Include,
    Omit,
    Weights,
}

impl Keyword {
    fn parse(value: &Value, kind: ReductionKind) -> BuiltinResult<Option<Self>> {
        let keyword = keyword_of(value)?;
        Ok(match keyword.as_deref() {
            Some("all") => Some(Self::All),
            Some("includenan") => Some(Self::Include),
            Some("omitnan") => Some(Self::Omit),
            Some("includemissing") if kind.allows_missing_synonyms() => Some(Self::Include),
            Some("omitmissing") if kind.allows_missing_synonyms() => Some(Self::Omit),
            Some("weights") if kind.allows_weights() => Some(Self::Weights),
            _ => None,
        })
    }
}

pub(super) struct Options {
    pub(super) axes: Axes,
    pub(super) missing: MissingMode,
    pub(super) weights: Option<Value>,
    pub(super) typed_integer_control: bool,
}

pub(super) async fn parse(
    kind: ReductionKind,
    args: &crate::call::BuiltinArguments,
) -> BuiltinResult<Options> {
    let mut axes = None;
    let mut missing = None;
    let mut weights = None;
    let mut typed_integer_control = false;
    let mut index = 0;
    while index < args.len() {
        let value = &args[index];
        match Keyword::parse(value, kind)? {
            Some(Keyword::All) if axes.is_none() => axes = Some(Axes::All),
            Some(Keyword::All) => {
                return Err(invalid(kind, "only one operating dimension is allowed"))
            }
            Some(Keyword::Include) if missing.is_none() => missing = Some(MissingMode::Include),
            Some(Keyword::Omit) if missing.is_none() => missing = Some(MissingMode::Omit),
            Some(Keyword::Include | Keyword::Omit) => {
                return Err(invalid(kind, "only one missing-value flag is allowed"));
            }
            Some(Keyword::Weights) => {
                if weights.is_some() {
                    return Err(invalid(kind, "Weights may be specified only once"));
                }
                let Some(weight) = args.get(index + 1) else {
                    return Err(invalid(kind, "Weights requires a value"));
                };
                weights = Some(weight.clone());
                index += 1;
            }
            None if keyword_of(value)?.is_some() => {
                return Err(invalid(kind, "unsupported reduction option"));
            }
            None if axes.is_none() => {
                typed_integer_control = super::data::integer_control(value);
                let dimensions = tensor::dims_from_value_async(value)
                    .await
                    .map_err(|source| invalid_source(kind, "invalid dimension control", source))?
                    .ok_or_else(|| invalid(kind, "expected a dimension or dimension vector"))?;
                if dimensions.is_empty() || dimensions.contains(&0) {
                    return Err(invalid(kind, "dimensions must be positive integers"));
                }
                let mut deduplicated = dimensions.clone();
                deduplicated.sort_unstable();
                deduplicated.dedup();
                if deduplicated.len() != dimensions.len() {
                    return Err(invalid(kind, "dimension vector must contain unique axes"));
                }
                axes = Some(Axes::Dims(dimensions));
            }
            None => return Err(invalid(kind, "only one operating dimension is allowed")),
        }
        index += 1;
    }
    let axes = axes.unwrap_or(Axes::Default);
    if weights.is_some() {
        match &axes {
            Axes::All => return Err(invalid(kind, "weights are not supported with \"all\"")),
            Axes::Dims(dimensions) if dimensions.len() > 1 => {
                return Err(invalid(
                    kind,
                    "weights are not supported with vector dimensions",
                ));
            }
            Axes::Default | Axes::Dims(_) => {}
        }
    }
    Ok(Options {
        axes,
        missing: missing.unwrap_or(MissingMode::Include),
        weights,
        typed_integer_control,
    })
}

pub(super) fn invalid(kind: ReductionKind, message: &str) -> RuntimeError {
    crate::build_runtime_error(format!("{}: {message}", kind.name()))
        .with_builtin(kind.name())
        .build()
}

fn invalid_source(kind: ReductionKind, message: &str, source: String) -> RuntimeError {
    let mut error = invalid(kind, message);
    error.source = Some(Box::new(std::io::Error::other(source)));
    error
}

#[cfg(test)]
mod tests {
    use super::*;
    async fn parse(kind: ReductionKind, args: Vec<Value>) -> BuiltinResult<Options> {
        let retained = crate::call::BuiltinArguments::from_borrowed(&args, None)?;
        super::parse(kind, &retained).await
    }
    use futures::executor::block_on;

    #[test]
    fn duplicate_selectors_and_flags_are_rejected_before_data_access() {
        let duplicate = block_on(parse(
            ReductionKind::Geometric,
            vec![Value::Num(1.0), Value::Num(2.0)],
        ));
        assert!(duplicate.is_err());
        let flags = block_on(parse(
            ReductionKind::RootMeanSquare,
            vec![Value::from("omitnan"), Value::from("includemissing")],
        ));
        assert!(flags.is_err());
    }

    #[test]
    fn weighted_multi_axis_is_rejected_before_materialization() {
        let axes = Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let outcome = block_on(parse(
            ReductionKind::RootMeanSquaredError,
            vec![axes, Value::from("Weights"), Value::Num(1.0)],
        ));
        assert!(outcome.is_err());
    }

    #[test]
    fn missing_aliases_follow_the_shared_typed_policy() {
        for kind in [ReductionKind::Geometric, ReductionKind::Harmonic] {
            assert!(block_on(parse(kind, vec![Value::from("omitmissing")])).is_err());
            assert!(block_on(parse(kind, vec![Value::from("includemissing")])).is_err());
            assert!(block_on(parse(kind, vec![Value::from("omitnan")])).is_ok());
        }
        for kind in [
            ReductionKind::RootMeanSquare,
            ReductionKind::RootMeanSquaredError,
        ] {
            assert!(block_on(parse(kind, vec![Value::from("omitmissing")])).is_ok());
            assert!(block_on(parse(kind, vec![Value::from("includemissing")])).is_ok());
        }
    }
}
