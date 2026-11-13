//! Retire seed/options assembly before option materialization; retire that
//! phase before constructing the original geometry/control generation.
use super::*;
use runmat_value::UnboundedStructValueBuilder;
pub(super) struct Input<'a> {
    pub fitness: Value,
    pub seed: &'a [f64],
    pub explicit: Option<&'a Value>,
    pub inherited: (bool, CallbackParallelMode),
    pub public: PublicLinear<'a>,
    pub bounds: (&'a [f64], &'a [f64]),
    pub has_linear: bool,
    pub cooperation: &'a mut ScalarCooperation,
}
struct Remaining<'a> {
    fitness: Value,
    variables: usize,
    public: PublicLinear<'a>,
    bounds: (&'a [f64], &'a [f64]),
    has_linear: bool,
    cooperation: &'a mut ScalarCooperation,
}
struct Selected<'a> {
    value: Value,
    remaining: Remaining<'a>,
}
struct Materialized<'a> {
    prepared: Prepared,
    remaining: Remaining<'a>,
}
pub(super) fn prepare<'a>(
    input: Input<'a>,
) -> impl std::future::Future<Output = BuiltinResult<Controls>> + 'a {
    crate::execution::after_result(select(input), |selected| {
        crate::execution::after_result(materialize(selected), finish)
    })
}
async fn select<'a>(input: Input<'a>) -> BuiltinResult<Selected<'a>> {
    let Input {
        fitness,
        seed,
        explicit,
        inherited,
        public,
        bounds,
        has_linear,
        cooperation,
    } = input;
    cooperation.completed(0).await?;
    crate::context::host_allocation::require_unbounded_workspace(
            cooperation.runtime(), "ga",
            "finite host-memory execution is unsupported; genetic population, callback and feasibility workspaces have no complete finite bound",
        )?;
    let mut fields = match explicit {
        Some(Value::Struct(value)) => UnboundedStructValueBuilder::from_structure(value.clone()),
        Some(value) if !super::super::inputs::raw::unspecified(value) => {
            return Err(super::super::invalid(
                "optionsGA must be a scalar structure or []",
            ))
        }
        _ => UnboundedStructValueBuilder::new(),
    };
    let population = explicit
        .map(Prepared::population)
        .transpose()?
        .unwrap_or(runmat_builtins::OptimGeneticPopulation::DoubleVector);
    let mut seed_compatible = true;
    if population == runmat_builtins::OptimGeneticPopulation::BitString {
        for chunk in seed.chunks(1024) {
            seed_compatible &= chunk.iter().all(|&value| value == 0.0 || value == 1.0);
            cooperation.completed(chunk.len() as u64).await?;
        }
    }
    if explicit.is_none() {
        fields.insert("UseVectorized", Value::Bool(inherited.0));
        fields.insert(
            "UseParallel",
            match inherited.1 {
                CallbackParallelMode::Off => Value::Bool(false),
                CallbackParallelMode::On => Value::Bool(true),
                CallbackParallelMode::Auto => Value::from("auto"),
            },
        );
    }
    let mut has_initial = false;
    let mut selected = UnboundedStructValueBuilder::new();
    for (key, value) in fields.fields.into_entries() {
        if runmat_builtins::OptimOptionField::parse_for_solver(
            key.as_str(),
            runmat_builtins::OptimSolver::Ga,
        ) == Some(runmat_builtins::OptimOptionField::InitialPopulationMatrix)
        {
            if super::super::inputs::raw::unspecified(&value) {
                continue;
            }
            has_initial = true;
        }
        selected.insert_text(key, value);
        cooperation.completed(1).await?;
    }
    if !has_initial && seed_compatible {
        let mut coordinates = POLICY.capacity(seed.len(), cooperation.runtime())?;
        for chunk in seed.chunks(1024) {
            coordinates.extend_from_slice(chunk);
            cooperation.completed(chunk.len() as u64).await?;
        }
        selected.insert(
            "InitialPopulationMatrix",
            Value::Tensor(
                runmat_value::Tensor::new(coordinates, vec![1, seed.len()])
                    .map_err(super::super::invalid)?,
            ),
        );
    }

    Ok(Selected {
        value: Value::Struct(selected.finish()),
        remaining: Remaining {
            fitness,
            variables: seed.len(),
            public,
            bounds,
            has_linear,
            cooperation,
        },
    })
}
async fn materialize<'a>(selected: Selected<'a>) -> BuiltinResult<Materialized<'a>> {
    let Selected { value, remaining } = selected;
    let prepared = Prepared::new(value, Activity::Linear, remaining.cooperation).await?;
    Ok(Materialized {
        remaining,
        prepared,
    })
}
async fn finish(materialized: Materialized<'_>) -> BuiltinResult<Controls> {
    let Materialized {
        remaining,
        prepared,
    } = materialized;
    let Remaining {
        fitness,
        variables,
        public,
        bounds,
        has_linear,
        cooperation,
    } = remaining;
    let mut public_fields = UnboundedStructValueBuilder::new();
    for name in ["fitnessfcn", "FitnessFcn"] {
        public_fields.insert(name, fitness.clone());
    }
    public_fields.insert("nvars", Value::Num(variables as f64));
    for (name, value) in [
        ("Aineq", public.inequalities.map(|pair| pair.0)),
        ("Bineq", public.inequalities.map(|pair| pair.1)),
        ("Aeq", public.equalities.map(|pair| pair.0)),
        ("Beq", public.equalities.map(|pair| pair.1)),
        ("lb", public.lower),
        ("ub", public.upper),
    ] {
        public_fields.insert(
            name,
            value
                .cloned()
                .unwrap_or_else(super::super::inputs::raw::empty),
        );
    }
    for name in ["nonlcon", "intcon", "options"] {
        public_fields.insert(name, super::super::inputs::raw::empty());
    }
    public_fields.insert("solver", Value::from("ga"));
    let problem = Problem {
        objective: fitness,
        nonlinear: None,
        variables,
        linear: None,
        rng: None,
        public: Value::Struct(public_fields.finish()),
        has_linear,
        has_bounds: bounds.0.iter().chain(bounds.1).any(|x| x.is_finite()),
        mixed_integer: false,
    };
    let options = Options::with_bounds(prepared, &problem, bounds, cooperation).await?;
    let sources = Sources::select(&problem, cooperation)?;
    Ok(Controls {
        ready: super::super::initialization::Ready::new(problem, options, cooperation)?,
        sources,
    })
}
