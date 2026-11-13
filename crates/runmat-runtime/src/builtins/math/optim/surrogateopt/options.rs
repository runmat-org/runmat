//! One prepared, rooted public option generation and typed numerical controls.
use super::{invalid, parameters::Selected, IDENTITY, INVALID_ARGUMENT};
use crate::{
    builtins::math::optim::common::{self, surrogate_options as controls, ScalarCooperation},
    execution::RootedValue,
    BuiltinResult,
};
use runmat_builtins::{OptimDisplay, OptimOptionField as Field, OptimSolver};
use runmat_value::Value;
use std::num::NonZeroUsize;

pub(super) struct Options {
    pub controls: Selected,
    pub display: OptimDisplay,
    pub output: Option<Value>,
    pub plot: Option<Value>,
    pub initial: Option<Value>,
    pub checkpoint: Option<String>,
    pub public: RootedValue,
    pub declared: RootedValue,
}
impl Options {
    pub(super) async fn parse(value: Value, c: &mut ScalarCooperation) -> BuiltinResult<Self> {
        let policy = controls::Policy {
            identity: IDENTITY,
            error: &INVALID_ARGUMENT,
            integer: Some(&runmat_builtins::SURROGATEOPT_INTEGER_OPTIONS_EXTENSION),
            resident: Some(&runmat_builtins::SURROGATEOPT_RESIDENT_OPTIONS_EXTENSION),
        };
        let declared = RootedValue::in_optional_context(value, c.runtime())?;
        let prepared = controls::prepare(policy, declared.value().clone(), c).await?;
        let Value::Struct(fields) = prepared.value() else {
            unreachable!("prepared options structure")
        };
        let solver = OptimSolver::Surrogateopt;
        let mut complete = runmat_value::UnboundedStructValueBuilder::new();
        complete.insert("Solver", Value::from(solver.name()));
        for field in Field::ALL {
            if !solver.accepts(field) {
                continue;
            }
            let name = field.name_for_solver(solver);
            if let Some(value) = fields.fields.get(name).filter(|value| {
                !controls::empty(value) || matches!(field, Field::OutputFcn | Field::PlotFcn)
            }) {
                complete.insert(name, value.clone());
            } else if let Some(value) = controls::default_value(field, c.runtime())? {
                complete.insert(name, value);
            }
            c.completed(1).await?;
        }
        let public =
            RootedValue::in_optional_context(Value::Struct(complete.finish()), c.runtime())?;
        let Value::Struct(fields) = public.value() else {
            unreachable!("public option structure")
        };
        let get = |field: Field| {
            fields
                .fields
                .get(field.name_for_solver(solver))
                .filter(|v| !controls::empty(v))
        };
        let count = |field: Field| {
            get(field)
                .map(|v| common::option_usize_value(super::NAME, Some(v), field.name(), 0))
                .transpose()
        };
        let number = |field: Field, default| {
            get(field)
                .map(|v| common::option_f64_value(super::NAME, Some(v), field.name(), default))
                .transpose()
                .map(|v| v.unwrap_or(default))
        };
        let evaluations = count(Field::MaxFunEvals)?;
        let minimum_surrogate = count(Field::MinSurrogatePoints)?;
        let batch = NonZeroUsize::new(count(Field::BatchUpdateInterval)?.unwrap_or(1))
            .ok_or_else(|| invalid("BatchUpdateInterval must be positive"))?;
        let sample_distance = number(Field::MinSampleDistance, 1e-6)?;
        let nonlinear_tolerance = number(Field::TolCon, 1e-3)?;
        let objective_limit = get(Field::ObjectiveLimit)
            .map(|v| common::quasi_newton_controls::objective_limit(super::NAME, v))
            .transpose()?
            .unwrap_or(f64::NEG_INFINITY);
        let time = get(Field::MaxTime)
            .map(|v| {
                common::global_time::parse(
                    common::global_time::Policy {
                        identity: IDENTITY,
                        error: &INVALID_ARGUMENT,
                    },
                    v,
                )
            })
            .transpose()?
            .unwrap_or(common::global_time::TimeLimit::Unbounded);
        let parallel = get(Field::UseParallel)
            .map(|v| common::quasi_newton_controls::parallel_mode(super::NAME, v))
            .transpose()?
            .unwrap_or_default();
        let vectorized = get(Field::UseVectorized)
            .map(|v| common::switch_value(super::NAME, Field::UseVectorized.name(), v))
            .transpose()?
            .unwrap_or(false);
        let display = get(Field::Display)
            .and_then(crate::builtins::common::validation::value_text_scalar_string)
            .and_then(|v| OptimDisplay::parse(&v))
            .filter(|v| solver.accepts_display(*v))
            .ok_or_else(|| invalid("invalid Display option"))?;
        let checkpoint = get(Field::CheckpointFile)
            .map(|v| {
                crate::builtins::common::validation::value_text_scalar_string(v)
                    .ok_or_else(|| invalid("CheckpointFile must be scalar text"))
            })
            .transpose()?
            .filter(|v| !v.is_empty());
        let output = get(Field::OutputFcn).cloned();
        let plot = get(Field::PlotFcn).cloned();
        let initial = get(Field::InitialPoints).cloned();
        Ok(Self {
            controls: Selected {
                evaluations,
                minimum_surrogate,
                batch,
                sample_distance,
                nonlinear_tolerance,
                objective_limit,
                time,
                parallel,
                vectorized,
            },
            display,
            output,
            plot,
            initial,
            checkpoint,
            public,
            declared,
        })
    }
}
