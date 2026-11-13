//! Standalone creators share population storage and admitted numerical work.
//! A selected creator never invokes the supplied fitness function.
use super::{
    super::{common, creation, inputs, invalid, options, ScalarCooperation, POLICY},
    admission,
};
use crate::{context::RuntimeContext, BuiltinResult};
use runmat_builtins::{
    OptimGeneticCreation as Kind, OptimGeneticPopulation, OptimOptionField as Field,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, Value};

fn controls(
    kind: Kind,
    value: &Value,
    parameters: &[Value],
    cooperation: &ScalarCooperation,
) -> BuiltinResult<Value> {
    admission::options(value)?;
    let source = if let Value::Struct(value) = value {
        Some(value)
    } else {
        None
    };
    let fields = common::ScalarOptionFields::from_struct_for_solver(
        source,
        runmat_builtins::OptimSolver::Ga,
    );
    let mut result = runmat_value::UnboundedStructValueBuilder::new();
    for field in [
        Field::PopulationType,
        Field::PopulationSize,
        Field::InitialPopulationMatrix,
        Field::TolCon,
    ] {
        if let Some(value) = fields.exact(field) {
            result.insert(field.name(), value.clone());
        }
    }
    if kind != Kind::LinearFeasible {
        if let Some(value) = fields.exact(Field::InitialPopulationRange) {
            result.insert(Field::InitialPopulationRange.name(), value.clone());
        }
    }
    let selector = if parameters.is_empty() {
        Value::from(kind.name())
    } else {
        let mut values = POLICY.capacity(
            parameters
                .len()
                .checked_add(1)
                .ok_or_else(|| invalid("creation parameter extent overflow"))?,
            cooperation.runtime(),
        )?;
        values.push(Value::from(kind.name()));
        values.extend(parameters.iter().cloned());
        let n = values.len();
        Value::Cell(CellArray::new(values, 1, n).map_err(invalid)?)
    };
    result.insert(Field::CreationFcn.name(), selector);
    Ok(Value::Struct(result.finish()))
}
async fn execute(
    kind: Kind,
    runtime: Option<&RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if args.len() < 3 || kind != Kind::NonlinearFeasible && args.len() != 3 {
        return Err(invalid("creation requires nvars,problem,options and only the nonlinear creator accepts parameter pairs"));
    }
    crate::output_count::validate_bounded_outputs_in(
        &runmat_builtins::GA_CREATION_HELPER_DESCRIPTOR,
        &runmat_builtins::GA_ERROR_INVALID_ARGUMENT,
        kind.name(),
        runtime,
    )?;
    let _original = args.fork();
    let mut cooperation = admission::begin(kind.name(), runtime)?;
    cooperation.completed(0).await?;
    let timing = if kind != Kind::Uniform {
        Some(common::solver_timing::SolverTiming::start(runtime)?)
    } else {
        None
    };
    let mut parameters = POLICY.capacity(args.len() - 3, runtime)?;
    for value in args.iter().skip(3) {
        parameters.push(value.clone());
        cooperation.completed(1).await?;
    }
    let controls = controls(kind, &args[2], &parameters, &cooperation)?;
    let population = options::Prepared::population(&controls)?;
    if population == OptimGeneticPopulation::Custom
        || population == OptimGeneticPopulation::BitString && kind != Kind::Uniform
    {
        return Err(invalid(
            "this predefined creator requires doubleVector, except uniform also supports bitstring",
        ));
    }
    let mut raw = inputs::raw::creation_problem(args[0].clone(), &args[1], controls)?;
    // Uniform samples its range/bounds; enforcing linear/integer membership
    // remains the invoking solver's subsequent admission transaction.
    if kind == Kind::Uniform {
        raw.inequalities = inputs::raw::empty();
        raw.inequality_rhs = inputs::raw::empty();
        raw.equalities = inputs::raw::empty();
        raw.equality_rhs = inputs::raw::empty();
    }
    if kind != Kind::NonlinearFeasible {
        raw.nonlinear = inputs::raw::empty();
    }
    inputs::preflight(&raw, population, &cooperation)?;
    let prepared = options::Prepared::new(
        std::mem::replace(&mut raw.options, inputs::raw::empty()),
        options::Activity::from_raw(&raw, population),
        &mut cooperation,
    )
    .await?;
    let mut problem = inputs::prepare(raw, population, &mut cooperation).await?;
    let mut options = options::Options::resolve(prepared, &problem, &mut cooperation).await?;
    let value = if kind == Kind::Uniform {
        let mut draft = creation::Draft::new(
            options.field(Field::InitialPopulationMatrix).cloned(),
            options.layout.total(),
            problem.variables,
            population == OptimGeneticPopulation::BitString,
            &mut cooperation,
        )
        .await?;
        draft
            .fill_uniform(
                &options.ranges,
                population == OptimGeneticPopulation::BitString,
                &mut cooperation,
            )
            .await?;
        draft.finish().value()
    } else {
        let initial = creation::create(
            &mut problem,
            &mut options,
            timing
                .as_ref()
                .expect("feasible creator captured its selected clock"),
            None,
            &mut cooperation,
        )
        .await?;
        match initial.admission {
            creation::Admission::Complete => initial.generation.value(),
            creation::Admission::Infeasible => {
                return Err(invalid(
                    "no linearly/integer feasible population could be produced",
                ))
            }
            creation::Admission::TimeLimit => {
                return Err(invalid("creation stopped before completing its population"))
            }
            creation::Admission::Unverified => {
                return Err(invalid(
                    "creation could not verify its numerical feasibility proposal",
                ))
            }
        }
    };
    if crate::output_count::current_output_count_in(runtime) == Some(0) {
        Ok(Value::output_list(Vec::new()))
    } else {
        Ok(value)
    }
}

#[runtime_builtin(
    name = "gacreationuniform",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::creation"
)]
async fn uniform_registered(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Uniform, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "gacreationlinearfeasible",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::creation"
)]
async fn linear_feasible_registered(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::LinearFeasible, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "gacreationnonlinearfeasible",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::creation"
)]
async fn nonlinear_feasible_registered(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::NonlinearFeasible, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "gacreationuniformint",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::creation"
)]
async fn uniform_integer_registered(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::UniformInteger, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "gacreationsobol",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::creation"
)]
async fn sobol_registered(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Sobol, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}
