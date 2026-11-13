//! Effective classic controls. Public selector decoding is canonical; rejected
//! selectors never become aliases for a different algorithm or serial mode.
use super::{invalid, mesh, poll::Acceptance, timing::TimeLimit};
use crate::builtins::math::optim::common::{self, ScalarOptionFields};
use crate::{BuiltinResult, RuntimeError};
use runmat_builtins::{
    OptimDisplay, OptimOptionField as Field, OptimPatternAlgorithm,
    OptimPatternDefault as DefaultValue, OptimPatternOrder, OptimPatternPoll,
    OptimPatternScalarRule as Rule,
};

pub(super) struct Controls {
    pub equalities: bool,
    pub iterations: usize,
    pub evaluations: usize,
    pub mesh: mesh::Parameters,
    pub constraint_tolerance: f64,
    pub step_tolerance: f64,
    pub function_tolerance: f64,
    pub binding: f64,
    pub initial_penalty: f64,
    pub penalty_factor: f64,
    pub scale_mesh: bool,
    pub polling: Polling,
    pub search: Option<super::search::Search>,
    pub complete_poll: bool,
    pub complete_search: bool,
    pub display: OptimDisplay,
    pub plot_interval: usize,
    pub maximum_time: TimeLimit,
    pub cache: Option<super::cache::Parameters>,
    pub vectorized: bool,
    pub parallel: crate::execution::CallbackParallelMode,
    pub remember_direction: bool,
}

pub(super) fn unsupported(detail: impl std::fmt::Display) -> RuntimeError {
    crate::build_runtime_error(format!("patternsearch: {detail}"))
        .with_builtin("patternsearch")
        .with_identifier("RunMat:patternsearch:UnsupportedOption")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

fn text(fields: &ScalarOptionFields<'_>, field: Field) -> BuiltinResult<String> {
    let Some(DefaultValue::Text(default)) = DefaultValue::for_field(field) else {
        return Err(invalid("internal text-control default mismatch"));
    };
    fields
        .exact(field)
        .map(|value| {
            crate::builtins::common::validation::value_text_scalar_string(value)
                .ok_or_else(|| invalid(format!("{} must be scalar text", field.name())))
        })
        .unwrap_or_else(|| Ok(default.to_owned()))
}

fn default_real(field: Field) -> BuiltinResult<f64> {
    let Some(DefaultValue::Number(default)) = DefaultValue::for_field(field) else {
        return Err(invalid("internal scalar-control default mismatch"));
    };
    Ok(default)
}

fn real(fields: &ScalarOptionFields<'_>, field: Field) -> BuiltinResult<f64> {
    let default = default_real(field)?;
    let value = fields
        .exact(field)
        .map(|value| super::option_metadata::number(field, value, super::OPTIONS))
        .transpose()?
        .unwrap_or(default);
    if !Rule::for_field(field).is_some_and(|rule| rule.accepts(value)) {
        return Err(invalid(format!(
            "{} is outside its supported scalar range",
            field.name()
        )));
    }
    Ok(value)
}

fn count(fields: &ScalarOptionFields<'_>, field: Field, n: usize) -> BuiltinResult<usize> {
    let default = match DefaultValue::for_field(field) {
        Some(DefaultValue::DimensionCount(factor)) => factor
            .checked_mul(n)
            .ok_or_else(|| invalid(format!("default {} count overflow", field.name())))?,
        Some(DefaultValue::Number(number)) => number as usize,
        _ => return Err(invalid("internal count-control default mismatch")),
    };
    let raw = fields.exact(field);
    if raw
        .and_then(crate::builtins::common::validation::value_text_scalar_string)
        .is_some_and(|value| Some(value.as_str()) == DefaultValue::dimension_expression(field))
    {
        return Ok(default);
    }
    // Native integer structural counts retain usize exactness; do not route
    // them through a floating range proof or round at the 2^53 boundary.
    let value = common::option_usize_value("patternsearch", raw, field.name(), default)?;
    if matches!(Rule::for_field(field), Some(Rule::Count { zero: false })) && value == 0 {
        return Err(invalid(format!("{} must be positive", field.name())));
    }
    Ok(value)
}

pub(super) fn switch(fields: &ScalarOptionFields<'_>, field: Field) -> BuiltinResult<bool> {
    if fields.exact(field).is_some() {
        return fields.switch("patternsearch", field);
    }
    match DefaultValue::for_field(field) {
        Some(DefaultValue::Logical(value)) => Ok(value),
        Some(DefaultValue::Text(value)) => runmat_builtins::OptimSwitch::parse(value)
            .map(|value| value.enabled())
            .ok_or_else(|| invalid("internal switch default mismatch")),
        _ => Err(invalid("internal switch default mismatch")),
    }
}

mod selection;
pub(super) use selection::{algorithm, select, Polling, Selection};

pub(super) fn parallel_requested(fields: &ScalarOptionFields<'_>) -> BuiltinResult<bool> {
    if switch(fields, Field::UseVectorized)? {
        return Ok(false);
    }
    fields
        .exact(Field::UseParallel)
        .map(|value| {
            common::quasi_newton_controls::parallel_mode("patternsearch", value)
                .map(|mode| mode != crate::execution::CallbackParallelMode::Off)
        })
        .transpose()
        .map(|value| value.unwrap_or(false))
}

pub(super) fn execution_switch(field: Field, fields: &ScalarOptionFields<'_>) -> BuiltinResult<()> {
    match field {
        Field::UseVectorized => switch(fields, field).map(|_| ()),
        Field::UseParallel => {
            if let Some(value) = fields.exact(field) {
                common::quasi_newton_controls::parallel_mode("patternsearch", value)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

impl Controls {
    pub fn basis(poll: OptimPatternPoll) -> mesh::Basis {
        if poll.minimal() {
            mesh::Basis::Minimal
        } else {
            mesh::Basis::Maximal
        }
    }
    pub fn acceptance(poll: OptimPatternPoll, nonlinear: bool) -> Acceptance {
        if poll.mesh_adaptive() {
            return if nonlinear {
                Acceptance::MadsDecrease
            } else {
                Acceptance::Strict
            };
        }
        if nonlinear
            || matches!(
                poll,
                OptimPatternPoll::GssMaximal | OptimPatternPoll::GssMinimal
            )
        {
            Acceptance::SufficientDecrease
        } else {
            Acceptance::Strict
        }
    }

    pub fn read(
        fields: &ScalarOptionFields<'_>,
        n: usize,
        equalities: bool,
        linear: bool,
        nonlinear: bool,
    ) -> BuiltinResult<Self> {
        let Selection {
            polling,
            search,
            display,
        } = select(fields, n)?;
        let classic = polling.classic();
        let poll = polling.method();
        let nonuniform = classic.is_none();
        if equalities
            && (poll.is_some_and(OptimPatternPoll::mesh_adaptive)
                || search.is_some_and(|search| {
                    search.poll().is_some_and(OptimPatternPoll::mesh_adaptive)
                }))
        {
            return Err(invalid(
                "MADS polling/search does not support linear equality constraints",
            ));
        }
        if equalities && polling.algorithm() == OptimPatternAlgorithm::NupsMads {
            return Err(unsupported(
                "nups-mads does not support linear equality constraints",
            ));
        }
        let adaptive = polling.adaptive();
        let fixed_controls = classic.is_some() && !adaptive
            || search
                .is_some_and(|search| search.poll().is_some_and(|method| !method.mesh_adaptive()));
        let vectorized = switch(fields, Field::UseVectorized)?;
        let parallel = if vectorized {
            crate::execution::CallbackParallelMode::Off
        } else {
            fields
                .exact(Field::UseParallel)
                .map(|value| common::quasi_newton_controls::parallel_mode("patternsearch", value))
                .transpose()?
                .unwrap_or_default()
        };
        let parallel_requested = parallel != crate::execution::CallbackParallelMode::Off;
        let maximum_time = fields
            .exact(Field::MaxTime)
            .map(|value| {
                if crate::builtins::duration::is_duration_object(value) {
                    let tensor =
                        crate::builtins::duration::duration_tensor_from_duration_value(value)?;
                    if tensor.len() != 1 || tensor.as_f64_slice().is_none() {
                        return Err(invalid("MaxTime duration must be a real double scalar"));
                    }
                    TimeLimit::Days(tensor.as_f64_slice().expect("double duration")[0]).validate()
                } else {
                    TimeLimit::Seconds(real(fields, Field::MaxTime)?).validate()
                }
            })
            .transpose()?
            .unwrap_or(TimeLimit::Unbounded);
        Ok(Self {
            equalities,
            iterations: count(fields, Field::MaxIter, n)?,
            evaluations: count(fields, Field::MaxFunEvals, n)?,
            mesh: mesh::Parameters {
                initial: real(fields, Field::InitialMeshSize)?,
                maximum: if fixed_controls || nonuniform {
                    real(fields, Field::MaxMeshSize)?
                } else {
                    1.0
                },
                expansion: if fixed_controls {
                    real(fields, Field::MeshExpansionFactor)?
                } else if nonuniform {
                    2.0
                } else {
                    4.0
                },
                contraction: if fixed_controls {
                    real(fields, Field::MeshContractionFactor)?
                } else if nonuniform {
                    0.5
                } else {
                    0.25
                },
                tolerance: real(fields, Field::MeshTolerance)?,
                rotate: classic.is_some()
                    && !adaptive
                    && poll.is_some_and(OptimPatternPoll::minimal)
                    && !equalities
                    && switch(fields, Field::MeshRotate)?,
                accelerate: classic.is_some()
                    && !adaptive
                    && switch(fields, Field::AccelerateMesh)?,
            },
            constraint_tolerance: real(fields, Field::TolCon)?,
            step_tolerance: if adaptive {
                default_real(Field::TolX)?
            } else {
                real(fields, Field::TolX)?
            },
            function_tolerance: if adaptive {
                default_real(Field::FunctionTolerance)?
            } else {
                real(fields, Field::FunctionTolerance)?
            },
            binding: if linear {
                real(fields, Field::TolBind)?
            } else {
                default_real(Field::TolBind)?
            },
            initial_penalty: if nonlinear {
                real(fields, Field::InitialPenalty)?
            } else {
                default_real(Field::InitialPenalty)?
            },
            penalty_factor: if nonlinear {
                real(fields, Field::PenaltyFactor)?
            } else {
                default_real(Field::PenaltyFactor)?
            },
            scale_mesh: (!equalities || nonuniform) && switch(fields, Field::ScaleMesh)?,
            polling,
            search,
            complete_poll: parallel_requested
                || (classic.is_some() && switch(fields, Field::UseCompletePoll)?),
            complete_search: search.is_some_and(|search| {
                !matches!(
                    search,
                    super::search::Search::Custom
                        | super::search::Search::Nelder { .. }
                        | super::search::Search::Rbf
                ) && !poll.is_some_and(|poll| search.redundant(poll))
            }) && (parallel_requested
                || switch(fields, Field::UseCompleteSearch)?),
            display,
            plot_interval: if common::output_callback::callback_count(
                fields.exact(Field::PlotFcn),
                common::output_callback::CallbackForm::HandlesOrNames,
            )? > 0
            {
                count(fields, Field::PlotInterval, n)?
            } else {
                1
            },
            maximum_time,
            vectorized,
            parallel,
            remember_direction: classic
                .is_some_and(|classic| classic.order == OptimPatternOrder::Success)
                || classic.is_some()
                    && common::output_callback::callback_count(
                        fields.exact(Field::OutputFcn),
                        common::output_callback::CallbackForm::Handles,
                    )? > 0,
            cache: if !parallel_requested && switch(fields, Field::Cache)? {
                Some(super::cache::Parameters {
                    capacity: count(fields, Field::CacheSize, n)?,
                    tolerance: real(fields, Field::CacheTol)?,
                })
            } else {
                None
            },
        })
    }
}
