//! One selected search mechanism, with concrete host count controls.
use super::super::{controls::unsupported, invalid, option_metadata::Policy};
use crate::builtins::math::optim::common;
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use runmat_builtins::{OptimPatternPoll, OptimPatternSearch};
use runmat_value::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::builtins::math::optim::patternsearch) enum Search {
    Poll(OptimPatternPoll),
    Custom,
    Rbf,
    Nelder { iterations: usize },
    Genetic { iterations: usize },
    Latin { iterations: usize, points: usize },
}
impl Search {
    pub fn poll(self) -> Option<OptimPatternPoll> {
        match self {
            Self::Poll(method) => Some(method),
            Self::Latin { .. }
            | Self::Custom
            | Self::Nelder { .. }
            | Self::Genetic { .. }
            | Self::Rbf => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Poll(method) => method.name(),
            Self::Latin { .. } => "searchlhs",
            Self::Custom => "custom",
            Self::Rbf => "rbfsurrogate",
            Self::Nelder { .. } => "searchneldermead",
            Self::Genetic { .. } => "searchga",
        }
    }
    pub fn redundant(self, poll: OptimPatternPoll) -> bool {
        self.poll() == Some(poll)
    }
}
struct Parts<'a> {
    target: &'a Value,
    iterations: Option<&'a Value>,
    points: Option<&'a Value>,
}
fn parts(value: &Value) -> BuiltinResult<Parts<'_>> {
    if let Value::Cell(cell) = value {
        cell.validate_layout()
            .map_err(|_| invalid("SearchFcn cell layout is invalid"))?;
        let data = cell.data();
        if data.is_empty() || data.len() > 3 {
            return Err(invalid(
                "SearchFcn cell must contain a search and at most two parameters",
            ));
        }
        Ok(Parts {
            target: &data[0],
            iterations: data.get(1),
            points: data.get(2),
        })
    } else {
        Ok(Parts {
            target: value,
            iterations: None,
            points: None,
        })
    }
}
fn kind(target: &Value) -> BuiltinResult<Option<OptimPatternSearch>> {
    let name = match target {
        Value::FunctionHandle(handle) => return Ok(OptimPatternSearch::parse(handle.name())),
        Value::BoundFunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::Closure(_)
        | Value::ScopedFunctionHandle(_) => return Ok(None),
        _ => crate::builtins::common::validation::value_text_scalar_string(target)
            .ok_or_else(|| unsupported("SearchFcn must select a supported search method"))?,
    };
    OptimPatternSearch::parse(&name)
        .map(Some)
        .ok_or_else(|| invalid("unrecognized SearchFcn"))
}
pub(in crate::builtins::math::optim::patternsearch) fn metadata(
    value: &Value,
    policy: Policy,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    syntax(value).map_err(|error| policy.invalid(error.message()))?;
    let parts = parts(value)?;
    let selected = kind(parts.target)?;
    if selected == Some(OptimPatternSearch::NelderMead) {
        crate::builtins::math::optim::fminsearch::options::admit_in(parts.points, runtime)?;
    }
    if selected == Some(OptimPatternSearch::Genetic) {
        if let Some(options) = parts.points {
            crate::builtins::math::optim::ga::auxiliary::metadata(options, runtime)?;
        }
    }
    for value in [
        parts.iterations,
        if selected == Some(OptimPatternSearch::LatinHypercube) {
            parts.points
        } else {
            None
        },
    ]
    .into_iter()
    .flatten()
    {
        if common::is_typed_integer_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
                runtime,
                policy.integer,
                policy.name,
            )?;
        }
    }
    Ok(())
}
pub(in crate::builtins::math::optim::patternsearch) fn syntax(value: &Value) -> BuiltinResult<()> {
    let parts = parts(value)?;
    let selected = kind(parts.target)?;
    if !matches!(
        selected,
        Some(
            OptimPatternSearch::LatinHypercube
                | OptimPatternSearch::NelderMead
                | OptimPatternSearch::Genetic
        )
    ) && (parts.iterations.is_some() || parts.points.is_some())
    {
        return Err(unsupported(
            "parameters are supported only for searchlhs, searchneldermead and searchga",
        ));
    }
    if selected == Some(OptimPatternSearch::NelderMead)
        && parts.points.is_some_and(|value| {
            !matches!(value, Value::Struct(_)) && !super::super::input::empty(value)
        })
    {
        return Err(invalid(
            "searchneldermead optionsNM must be a scalar structure or []",
        ));
    }
    if selected == Some(OptimPatternSearch::Genetic) {
        if let Some(options) = parts.points {
            crate::builtins::math::optim::ga::auxiliary::syntax(options)?;
        }
    }
    for value in [
        parts.iterations,
        if selected == Some(OptimPatternSearch::LatinHypercube) {
            parts.points
        } else {
            None
        },
    ]
    .into_iter()
    .flatten()
    {
        // An opaque cell child is never inspected via rich value facts or
        // numeric/capture conversion. Parameters are explicit host controls.
        if !matches!(value, Value::Num(_) | Value::Int(_) | Value::Tensor(_)) {
            return Err(invalid(
                "searchlhs parameters must be host real numeric scalar counts",
            ));
        }
        count(Some(value), "searchlhs parameter", 1)?;
    }
    Ok(())
}
fn count(value: Option<&Value>, name: &str, default: usize) -> BuiltinResult<usize> {
    let value = common::option_usize_value("patternsearch", value, name, default)?;
    if value == 0 {
        Err(invalid(format!("{name} must be positive")))
    } else {
        Ok(value)
    }
}
pub(in crate::builtins::math::optim::patternsearch) fn parse(
    value: &Value,
    n: usize,
) -> BuiltinResult<Search> {
    let parts = parts(value)?;
    match kind(parts.target)? {
        Some(OptimPatternSearch::Poll(method)) => {
            if parts.iterations.is_some() || parts.points.is_some() {
                return Err(invalid("poll-method searches do not accept parameters"));
            }
            Ok(Search::Poll(method))
        }
        Some(OptimPatternSearch::LatinHypercube) => Ok(Search::Latin {
            iterations: count(parts.iterations, "searchlhs iterlim", 1)?,
            points: count(
                parts.points,
                "searchlhs level",
                n.checked_mul(15)
                    .ok_or_else(|| invalid("searchlhs default point count overflow"))?,
            )?,
        }),
        Some(OptimPatternSearch::NelderMead) => Ok(Search::Nelder {
            iterations: count(parts.iterations, "searchneldermead iterlim", 1)?,
        }),
        Some(OptimPatternSearch::Genetic) => Ok(Search::Genetic {
            iterations: count(parts.iterations, "searchga iterlim", 1)?,
        }),
        Some(OptimPatternSearch::RbfSurrogate) => Ok(Search::Rbf),
        None => Ok(Search::Custom),
    }
}

pub(in crate::builtins::math::optim::patternsearch) fn custom_target(
    value: &Value,
) -> BuiltinResult<&Value> {
    let parts = parts(value)?;
    if kind(parts.target)?.is_some() {
        return Err(invalid("custom search target changed mechanism"));
    }
    Ok(parts.target)
}
/// Resolve the real supplied handle under the original owner, before selector
/// parsing. A same-named user definition is a custom callback, never a built-in
/// search selected by its display spelling.
pub(in crate::builtins::math::optim::patternsearch) fn prepare(
    value: &Value,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Value> {
    let parts = parts(value)?;
    if !parts.target.is_function_handle() {
        return Ok(value.clone());
    }
    let target = {
        let _scope = runtime.map(RuntimeContext::enter);
        crate::canonicalize_callback_handle_for_semantic_resolution(parts.target.clone())?
    };
    if let Value::Cell(cell) = value {
        let mut values = super::super::buffer(cell.data().len())?;
        values.push(target);
        values.extend(cell.data().iter().skip(1).cloned());
        Ok(Value::Cell(
            runmat_value::CellArray::new_with_shape(values, cell.shape.to_vec())
                .map_err(|_| invalid("SearchFcn cell layout changed"))?,
        ))
    } else {
        Ok(target)
    }
}

pub(in crate::builtins::math::optim::patternsearch) fn nelder_options(
    value: &Value,
) -> BuiltinResult<Option<&Value>> {
    let parts = parts(value)?;
    if kind(parts.target)? != Some(OptimPatternSearch::NelderMead) {
        return Err(invalid("Nelder-Mead search target changed mechanism"));
    }
    Ok(parts.points)
}

pub(in crate::builtins::math::optim::patternsearch) fn genetic_options(
    value: &Value,
) -> BuiltinResult<Option<&Value>> {
    let parts = parts(value)?;
    if kind(parts.target)? != Some(OptimPatternSearch::Genetic) {
        return Err(invalid("genetic search target changed mechanism"));
    }
    Ok(parts.points)
}
