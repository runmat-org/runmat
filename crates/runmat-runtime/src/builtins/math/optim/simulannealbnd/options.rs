//! Dimension-resolved options retain the actual public rooted generation.
use super::{common, inputs, invalid, ScalarCooperation};
use crate::{execution::RootedValue, BuiltinResult};
use common::anneal_options::{self as controls, Owner};
use runmat_builtins::{OptimAnnealData, OptimDisplay, OptimOptionField as Field, OptimSolver};
use runmat_value::Value;
pub(super) enum Source<T> {
    Predefined(T),
    Custom(RootedValue),
}
pub(super) enum HybridInterval {
    Never,
    End,
    Iterations(usize),
}
pub(super) struct Options {
    pub public: RootedValue,
    pub datatype: OptimAnnealData,
    pub annealing: Source<runmat_builtins::OptimAnnealing>,
    pub acceptance: Source<runmat_builtins::OptimAcceptance>,
    pub schedule: Source<runmat_builtins::OptimTemperature>,
    pub iterations: Option<usize>,
    pub evaluations: usize,
    pub stall: usize,
    pub tolerance: f64,
    pub maximum: controls::TimeLimit,
    pub objective: f64,
    pub display: OptimDisplay,
    pub display_interval: usize,
    pub plot_interval: usize,
    pub reanneal: usize,
    pub initial_temperature: Vec<f64>,
    pub hybrid: Option<super::hybrid::Selected>,
    pub hybrid_interval: HybridInterval,
    pub output: Option<Value>,
    pub plot: Option<Value>,
}
impl Options {
    pub async fn parse(value: Value, c: &mut ScalarCooperation) -> BuiltinResult<Self> {
        let prepared = controls::prepare(value, c).await?;
        let Value::Struct(fields) = prepared.value() else {
            unreachable!("admitted structure")
        };
        let mut complete = runmat_value::UnboundedStructValueBuilder::new();
        complete.insert("Solver", Value::from("simulannealbnd"));
        for field in Field::ALL {
            if !OptimSolver::Simulannealbnd.accepts(field) {
                continue;
            }
            let name = field.name_for_solver(OptimSolver::Simulannealbnd);
            if let Some(value) = fields.fields.get(name).filter(|v| !inputs::is_empty(v)) {
                complete.insert(name, value.clone());
            } else if let Some(value) = controls::default_value(field, c.runtime())? {
                complete.insert(name, value);
            }
            c.completed(1).await?;
        }
        let public =
            RootedValue::in_optional_context(Value::Struct(complete.finish()), c.runtime())?;
        let Value::Struct(fields) = public.value() else {
            unreachable!("public options")
        };
        let get = |field: Field| {
            fields
                .fields
                .get(field.name_for_solver(OptimSolver::Simulannealbnd))
                .filter(|v| !inputs::is_empty(v))
        };
        let number = |field, default| -> BuiltinResult<f64> {
            get(field)
                .map(|v| controls::number(Owner::Solver, v))
                .transpose()
                .map(|v| v.unwrap_or(default))
        };
        let count = |field, default| -> BuiltinResult<usize> {
            get(field)
                .map(|v| {
                    common::option_usize_value("simulannealbnd", Some(v), field.name(), default)
                })
                .transpose()
                .map(|v| v.unwrap_or(default))
        };
        let text = |field: Field, default: &str| -> BuiltinResult<String> {
            get(field)
                .map(|v| controls::text(Owner::Solver, v))
                .transpose()
                .map(|v| v.unwrap_or_else(|| default.to_owned()))
        };
        let datatype =
            OptimAnnealData::parse(&text(Field::DataType, "double")?).expect("validated datatype");
        let annealing = source(
            get(Field::AnnealingFcn),
            runmat_builtins::OptimAnnealing::Fast,
            runmat_builtins::OptimAnnealing::parse,
            true,
            c,
        )?;
        let acceptance = source(
            get(Field::AcceptanceFcn),
            runmat_builtins::OptimAcceptance::Annealing,
            runmat_builtins::OptimAcceptance::parse,
            false,
            c,
        )?;
        let schedule = source(
            get(Field::TemperatureFcn),
            runmat_builtins::OptimTemperature::Exponential,
            runmat_builtins::OptimTemperature::parse,
            false,
            c,
        )?;
        let iterations = if get(Field::MaxIter).is_none_or(controls::infinite) {
            None
        } else {
            Some(count(Field::MaxIter, 0)?)
        };
        // Dynamic defaults resolve after the decision domain is admitted.
        let evaluations = count(Field::MaxFunEvals, 0)?;
        let stall = count(Field::MaxStallIterations, 0)?;
        let tolerance = number(Field::FunctionTolerance, 1e-6)?;
        let objective = number(Field::ObjectiveLimit, f64::NEG_INFINITY)?;
        let maximum = get(Field::MaxTime)
            .map(|v| controls::time(Owner::Solver, v))
            .transpose()?
            .unwrap_or(controls::TimeLimit::Unbounded);
        let display = OptimDisplay::parse(&text(Field::Display, "final")?)
            .filter(|mode| OptimSolver::Simulannealbnd.accepts_display(*mode))
            .expect("validated display");
        let display_interval = count(Field::DisplayInterval, 10)?;
        let plot_interval = count(Field::PlotInterval, 1)?;
        let reanneal = count(Field::ReannealInterval, 100)?;
        let initial_temperature = match get(Field::InitialTemperature) {
            Some(Value::Tensor(v)) => {
                let mut values = Vec::with_capacity(v.len());
                for i in 0..v.len() {
                    values.push(crate::builtins::common::tensor::tensor_value_f64(v, i));
                    c.completed(1).await?;
                }
                values
            }
            Some(v) => vec![controls::number(Owner::Solver, v)?],
            None => vec![100.0],
        };
        let hybrid = get(Field::HybridFcn)
            .map(|v| {
                common::swarm_options::hybrid(common::swarm_options::Owner::Constructor, v)
                    .map_err(|e| invalid(e.message))
            })
            .transpose()?
            .flatten();
        let hybrid_interval = match get(Field::HybridInterval) {
            Some(v) => match crate::builtins::common::validation::value_text_scalar_string(v)
                .map(|v| v.to_ascii_lowercase())
                .as_deref()
            {
                Some("never") => HybridInterval::Never,
                Some("end") => HybridInterval::End,
                Some(_) => return Err(invalid("invalid HybridInterval")),
                None => HybridInterval::Iterations(count(Field::HybridInterval, 1)?),
            },
            None => HybridInterval::End,
        };
        let output = get(Field::OutputFcn).cloned();
        let plot = get(Field::PlotFcn).cloned();
        if datatype == OptimAnnealData::Custom {
            if !matches!(annealing, Source::Custom(_)) {
                return Err(invalid("DataType custom requires a custom AnnealingFcn"));
            }
            if hybrid.is_some() {
                return Err(invalid("DataType custom forbids HybridFcn"));
            }
        }
        let hybrid = if matches!(hybrid_interval, HybridInterval::Never) {
            None
        } else {
            hybrid
                .map(|hybrid| super::hybrid::Selected::new(hybrid, c))
                .transpose()?
        };
        Ok(Self {
            public,
            datatype,
            annealing,
            acceptance,
            schedule,
            iterations,
            evaluations,
            stall,
            tolerance,
            maximum,
            objective,
            display,
            display_interval,
            plot_interval,
            reanneal,
            initial_temperature,
            hybrid,
            hybrid_interval,
            output,
            plot,
        })
    }
    pub async fn resolve(&mut self, n: usize, c: &mut ScalarCooperation) -> BuiltinResult<()> {
        let (default_evaluations, default_stall) = {
            let Value::Struct(fields) = self.public.value() else {
                unreachable!("canonical options")
            };
            (
                fields
                    .fields
                    .get("MaxFunctionEvaluations")
                    .is_none_or(inputs::is_empty),
                fields
                    .fields
                    .get("MaxStallIterations")
                    .is_none_or(inputs::is_empty),
            )
        };
        if default_evaluations {
            self.evaluations = n
                .checked_mul(3000)
                .ok_or_else(|| invalid("default evaluation limit overflow"))?;
        }
        if default_stall {
            self.stall = n
                .checked_mul(500)
                .ok_or_else(|| invalid("default stall limit overflow"))?;
        }
        if self.initial_temperature.len() == 1 {
            let scalar = self.initial_temperature[0];
            self.initial_temperature.reserve(n.saturating_sub(1));
            while self.initial_temperature.len() < n {
                self.initial_temperature.push(scalar);
                c.completed(1).await?;
            }
        }
        if self.initial_temperature.len() != n {
            return Err(invalid(
                "InitialTemperature extent must be scalar or numel(x0)",
            ));
        }
        let Value::Struct(original) = self.public.value() else {
            unreachable!("options")
        };
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        for (name, value) in &original.fields {
            fields.insert(name, value.clone());
            c.completed(1).await?;
        }
        if original
            .fields
            .get("MaxFunctionEvaluations")
            .is_none_or(inputs::is_empty)
        {
            fields.insert(
                "MaxFunctionEvaluations",
                Value::Num(self.evaluations as f64),
            );
        }
        if original
            .fields
            .get("MaxStallIterations")
            .is_none_or(inputs::is_empty)
        {
            fields.insert("MaxStallIterations", Value::Num(self.stall as f64));
        }
        fields.insert(
            "InitialTemperature",
            common::output_callback::vector(&self.initial_temperature, &[n, 1], false, c).await?,
        );
        self.public =
            RootedValue::in_optional_context(Value::Struct(fields.finish()), c.runtime())?;
        Ok(())
    }
}
fn source<T>(
    value: Option<&Value>,
    default: T,
    parse: fn(&str) -> Option<T>,
    named_custom: bool,
    c: &ScalarCooperation,
) -> BuiltinResult<Source<T>> {
    let selected = |value: &Value| -> BuiltinResult<Source<T>> {
        let target = {
            let _entry = c.runtime().map(crate::context::RuntimeContext::enter);
            crate::canonicalize_callback_handle_for_semantic_resolution(value.clone())?
        };
        Ok(Source::Custom(RootedValue::in_optional_context(
            target,
            c.runtime(),
        )?))
    };
    match value {
        None => Ok(Source::Predefined(default)),
        Some(v) if controls::handle(v) => selected(v),
        Some(v) => match parse(&controls::text(Owner::Solver, v)?) {
            Some(kind) => Ok(Source::Predefined(kind)),
            None if named_custom => selected(v),
            None => Err(invalid("unknown predefined operator")),
        },
    }
}
