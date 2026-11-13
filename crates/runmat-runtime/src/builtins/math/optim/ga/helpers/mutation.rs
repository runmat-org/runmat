//! Standalone mutation proposals use the controller's schedules and geometry.
use super::{
    super::{common, inputs, invalid, linear, numeric, options, population, reproduction, POLICY},
    admission, mutation_controls as controls,
};
use crate::{context::RuntimeContext, BuiltinResult};
use linear::Geometry;
use runmat_builtins::{OptimGeneticMutation as Kind, OptimGeneticPopulation};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

async fn execute(
    kind: Kind,
    runtime: Option<&RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let max = match kind {
        Kind::Gaussian => 9,
        Kind::Uniform => 8,
        _ => 7,
    };
    if !(7..=max).contains(&args.len()) {
        return Err(invalid("mutation requires parents,options,nvars,problem,state,scores,population and only supported numerical parameters"));
    }
    crate::output_count::validate_bounded_outputs_in(
        &runmat_builtins::GA_MUTATION_HELPER_DESCRIPTOR,
        &runmat_builtins::GA_ERROR_INVALID_ARGUMENT,
        kind.name(),
        runtime,
    )?;
    let _original = args.fork();
    let mut cooperation = admission::begin(kind.name(), runtime)?;
    cooperation.completed(0).await?;
    let timing = common::solver_timing::SolverTiming::start(runtime)?;
    admission::vector_metadata(&args[0], runtime)?;
    numeric::preflight(
        &args[6],
        "mutation population",
        common::constrained_numeric::Geometry::MatrixAny,
        runtime,
    )?;
    let members = controls::matrix_rows(&args[6])?;
    let state = controls::State::new(kind, &args[4], &cooperation)?;
    let mut parameters = POLICY.capacity(args.len() - 7, runtime)?;
    for parameter in args.iter().skip(7) {
        parameters.push(parameter.clone());
        cooperation.completed(1).await?;
    }
    let options = controls::options(kind, &args[1], members, &parameters, &cooperation)?;
    let population_kind = options::Prepared::population(&options)?;
    if population_kind == OptimGeneticPopulation::Custom
        || population_kind == OptimGeneticPopulation::BitString && kind != Kind::Uniform
    {
        return Err(invalid("this predefined mutation requires doubleVector, except uniform also supports bitstring"));
    }
    let mut raw = inputs::raw::creation_problem(args[2].clone(), &args[3], options)?;
    raw.nonlinear = inputs::raw::empty();
    if matches!(kind, Kind::Gaussian | Kind::Uniform) {
        raw.inequalities = inputs::raw::empty();
        raw.inequality_rhs = inputs::raw::empty();
        raw.equalities = inputs::raw::empty();
        raw.equality_rhs = inputs::raw::empty();
        raw.integers = inputs::raw::empty();
    }
    inputs::preflight(&raw, population_kind, &cooperation)?;
    let prepared = options::Prepared::new(
        std::mem::replace(&mut raw.options, inputs::raw::empty()),
        options::Activity::from_raw(&raw, population_kind),
        &mut cooperation,
    )
    .await?;
    let mut problem = inputs::prepare(raw, population_kind, &mut cooperation).await?;
    let options = options::Options::resolve(prepared, &problem, &mut cooperation).await?;
    let parents = admission::vector(args[0].clone(), &mut cooperation).await?;
    let common::constrained_numeric::Numeric::Real { shape, values } =
        common::constrained_numeric::read(
            args[6].clone(),
            "mutation population",
            common::constrained_numeric::Geometry::MatrixAny,
            &numeric::INPUT_POLICY,
            &mut cooperation,
        )
        .await?
    else {
        return Err(invalid("mutation requires a real numeric population"));
    };
    if shape != [members, problem.variables] {
        return Err(invalid("mutation population must have nvars columns"));
    }
    let source = common::population::Population::new(
        POLICY,
        Tensor::new(values, shape).map_err(invalid)?,
        members,
        problem.variables,
    )?;
    let mut indices = POLICY.capacity(parents.len(), runtime)?;
    for parent in parents {
        if !parent.is_finite() || parent < 1. || parent.fract() != 0. || parent >= usize::MAX as f64
        {
            return Err(invalid("parents must be exact one-based member indices"));
        }
        let parent = parent as usize - 1;
        if parent >= members {
            return Err(invalid("mutation parent index exceeds population"));
        }
        indices.push(parent);
        cooperation.completed(1).await?;
    }
    let mut mutation = reproduction::Mutation::new(&options, &mut cooperation).await?;
    controls::restore(&mut mutation, state, &options, &mut cooperation).await?;
    let mut domain = if matches!(
        kind,
        Kind::Power | Kind::AdaptiveFeasible | Kind::PositiveBasis
    ) {
        let input = problem
            .linear
            .take()
            .ok_or_else(|| invalid("mutation original geometry is unavailable"))?
            .into_repair(&mut cooperation)
            .await?;
        let model = super::super::super::intlinprog::feasibility::Domain::new(
            input,
            timing.clone(),
            options.maximum,
            numeric::linear_tolerance(options.constraint_tolerance),
            &mut cooperation,
        )
        .await?;
        Some(linear::Domain::new(model, &mut cooperation).await?)
    } else {
        None
    };
    let children = indices.len();
    let size = children
        .checked_mul(problem.variables)
        .ok_or_else(|| invalid("mutation output extent overflow"))?;
    let mut output = POLICY.capacity(size, runtime)?;
    output.resize(size, 0.);
    for (row, parent) in indices.into_iter().enumerate() {
        let original = population::coordinates(&source, parent, &mut cooperation).await?;
        if let Some(domain) = &mut domain {
            if !domain
                .classify(&original, options.constraint_tolerance, &mut cooperation)
                .await?
                .feasible
            {
                return Err(invalid(
                    "feasible mutation requires a parent admitted to the original domain",
                ));
            }
        }
        let proposal = mutation
            .propose(
                &original,
                domain.as_mut(),
                &options,
                &timing,
                &mut cooperation,
            )
            .await?;
        let point = match proposal {
            reproduction::mutation::Proposal::Point(point) => point,
            reproduction::mutation::Proposal::Retain => original,
            reproduction::mutation::Proposal::TimeLimit => {
                return Err(invalid("mutation stopped at its time limit"))
            }
            reproduction::mutation::Proposal::Unverified => {
                return Err(invalid("mutation could not verify its numerical proposal"))
            }
        };
        for (column, value) in point.into_iter().enumerate() {
            output[row + children * column] = value;
            cooperation.completed(1).await?;
        }
    }
    if crate::output_count::current_output_count_in(runtime) == Some(0) {
        return Ok(Value::output_list(Vec::new()));
    }
    Tensor::new(output, vec![children, problem.variables])
        .map(Value::Tensor)
        .map_err(invalid)
}

#[runtime_builtin(
    name = "mutationgaussian",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::mutation"
)]
async fn gaussian(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Gaussian, runtime.as_ref(), args)
        .await
        .map_err(|mut e| {
            e.gpu_gather_retry = crate::GpuGatherRetry::Never;
            e
        })
}
#[runtime_builtin(
    name = "mutationuniform",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::mutation"
)]
async fn uniform(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Uniform, runtime.as_ref(), args)
        .await
        .map_err(|mut e| {
            e.gpu_gather_retry = crate::GpuGatherRetry::Never;
            e
        })
}
#[runtime_builtin(
    name = "mutationpower",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::mutation"
)]
async fn power(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Power, runtime.as_ref(), args)
        .await
        .map_err(|mut e| {
            e.gpu_gather_retry = crate::GpuGatherRetry::Never;
            e
        })
}
#[runtime_builtin(
    name = "mutationadaptfeasible",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::mutation"
)]
async fn adaptive(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::AdaptiveFeasible, runtime.as_ref(), args)
        .await
        .map_err(|mut e| {
            e.gpu_gather_retry = crate::GpuGatherRetry::Never;
            e
        })
}
#[runtime_builtin(
    name = "mutationpositivebasis",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::mutation"
)]
async fn positive(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::PositiveBasis, runtime.as_ref(), args)
        .await
        .map_err(|mut e| {
            e.gpu_gather_retry = crate::GpuGatherRetry::Never;
            e
        })
}
