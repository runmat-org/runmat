//! Display reduction over supplied state; never a solver stop certificate.
use super::{
    super::{common, invalid, options, stall::Window, ScalarCooperation, POLICY},
    plot_state::State,
};
use crate::BuiltinResult;
use runmat_builtins::{
    OptimGeneticStall, OptimGeneticStateField as StateField, OptimOptionField as Field,
};
use runmat_value::Value;

pub(super) async fn values(
    controls: &Value,
    state: &State<'_>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<(Vec<String>, Vec<f64>)> {
    let fields = common::ScalarOptionFields::from_struct_for_solver(
        if let Value::Struct(value) = controls {
            Some(value)
        } else {
            None
        },
        runmat_builtins::OptimSolver::Ga,
    );
    let mut selected = runmat_value::UnboundedStructValueBuilder::new();
    for field in [
        Field::MaxIter,
        Field::MaxStallIterations,
        Field::MaxTime,
        Field::FunctionTolerance,
        Field::StallTest,
    ] {
        if let Some(value) = fields.exact(field) {
            selected.insert(field.name(), value.clone());
        }
    }
    let prepared = options::Prepared::new(
        Value::Struct(selected.finish()),
        crate::builtins::math::optim::ga::options::Activity::Linear,
        cooperation,
    )
    .await?;
    let fields = common::ScalarOptionFields::from_struct_for_solver(
        Some(prepared.structure()),
        runmat_builtins::OptimSolver::Ga,
    );
    let generation = state.count(StateField::Generation, cooperation).await?;
    let last = state
        .count(StateField::LastImprovement, cooperation)
        .await?;
    if last > generation {
        return Err(invalid("LastImprovement cannot exceed Generation"));
    }
    let maximum = match fields.exact(Field::MaxIter) {
        Some(value) => common::option_usize_value("ga", Some(value), "MaxGenerations", 0)?,
        None => {
            let population = state.field(StateField::Population)?;
            let fact = crate::value_fact::value_fact(population);
            let dimensions = fact.shape.known_dims().ok_or_else(|| {
                invalid("stopping plot needs resolved MaxGenerations for opaque populations")
            })?;
            let variables = dimensions.get(1).copied().flatten().ok_or_else(|| {
                invalid("stopping plot cannot derive the default generation limit")
            })?;
            variables
                .checked_mul(100)
                .ok_or_else(|| invalid("default generation limit overflow"))?
        }
    };
    let stall = fields.count("ga", Field::MaxStallIterations, 50)?;
    let tolerance = fields
        .exact(Field::FunctionTolerance)
        .map(|value| options::real(options::Owner::Solver, value))
        .transpose()?
        .unwrap_or(1e-6);
    let test = fields
        .exact(Field::StallTest)
        .map(|value| {
            options::text(value).and_then(|text| {
                OptimGeneticStall::parse(&text).ok_or_else(|| invalid("unrecognized StallTest"))
            })
        })
        .transpose()?
        .unwrap_or(OptimGeneticStall::AverageChange);
    let history = state.vector(StateField::Best, cooperation).await?;
    let mut window = Window::new(stall);
    for value in history {
        window
            .observe(value.is_finite().then_some(value), cooperation)
            .await?;
    }
    let function = if tolerance == 0. {
        if window.unchanged() {
            1.
        } else {
            0.
        }
    } else {
        match window.average(test) {
            Some(0.) => 1.,
            Some(value) => (tolerance / value).clamp(0., 1.),
            None => 0.,
        }
    };
    let maximum_time = fields
        .exact(Field::MaxTime)
        .map(|value| common::global_time::parse(options::Owner::Solver.policy(), value))
        .transpose()?
        .unwrap_or(common::global_time::TimeLimit::Unbounded);
    let time = if matches!(maximum_time, common::global_time::TimeLimit::Unbounded) {
        0.
    } else {
        let runtime = cooperation.runtime();
        let token = state.field(StateField::StartTime)?;
        let arguments =
            crate::call::BuiltinArguments::from_borrowed(std::slice::from_ref(token), runtime)?;
        let elapsed =
            crate::builtins::timing::toc::toc_builtin(runtime.cloned(), Some(1), arguments).await?;
        let elapsed = std::time::Duration::try_from_secs_f64(elapsed).map_err(|source| {
            crate::build_runtime_error("genetic stopping elapsed time is not representable")
                .with_source(source)
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
        })?;
        maximum_time
            .fraction(elapsed)
            .expect("bounded stopping-plot time limit")
            .clamp(0., 1.)
    };
    let mut labels = POLICY.capacity(4, cooperation.runtime())?;
    for label in [
        "FunctionTolerance",
        "MaxGenerations",
        "MaxStallGenerations",
        "MaxTime",
    ] {
        labels.push(label.to_owned());
    }
    let fraction = |value: usize, limit: usize| {
        if limit == 0 {
            1.
        } else {
            (value as f64 / limit as f64).clamp(0., 1.)
        }
    };
    let mut values = POLICY.capacity(4, cooperation.runtime())?;
    values.extend([
        function,
        fraction(generation, maximum),
        fraction(generation - last, stall),
        time,
    ]);
    Ok((labels, values))
}
