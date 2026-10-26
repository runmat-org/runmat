//! One registered worker call performs one constrained feasible start. It is
//! not a replacement Runtime or a serial fallback for selected parallel work.
use super::super::super::fmincon::feasibility;
use super::super::{
    common::{constrained_numeric as numeric, global_time::TimeLimit, solver_timing::SolverTiming},
    invalid,
    numeric::INPUT_POLICY,
    ScalarCooperation, POLICY,
};
use crate::{execution::RootedValueSequence, BuiltinResult};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

pub(super) const NAME: &str = "__runmat_ga_feasible_start";
#[runtime_builtin(
    name = "__runmat_ga_feasible_start",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::creation::worker"
)]
async fn registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(runtime.as_ref(), args).await.map_err(|mut error| {
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })
}
async fn execute(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs_in(
        &runmat_builtins::GA_FEASIBLE_START_DESCRIPTOR,
        &runmat_builtins::GA_ERROR_INVALID_ARGUMENT,
        NAME,
        runtime,
    )?;
    if args.len() != 4 {
        return Err(invalid(
            "feasible-start worker requires x,geometry,nonlcon,SolverOpts",
        ));
    }
    let _original = args.fork();
    crate::context::host_allocation::require_unbounded_workspace(
        runtime,
        NAME,
        "finite host-memory execution is unsupported for constrained feasible-start workspaces",
    )?;
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: NAME },
        runtime,
    );
    cooperation.completed(0).await?;
    let timing = SolverTiming::start(runtime)?;
    super::super::numeric::preflight(&args[0], "x", numeric::Geometry::Vector, runtime)?;
    let geometry = super::geometry_wire::Original::preflight(&args[1], &cooperation)?;
    super::geometry_wire::callback(&args[2])?;
    let selected = super::super::callbacks::select(&args[2], &cooperation)?;
    let prepared = feasibility::prepare_options(args[3].clone(), &mut cooperation).await?;
    let numeric::Numeric::Real { values: x, .. } = numeric::read(
        args[0].clone(),
        "x",
        numeric::Geometry::Vector,
        &INPUT_POLICY,
        &mut cooperation,
    )
    .await?
    else {
        return Err(invalid("feasible-start worker requires a real start"));
    };
    if x.is_empty() {
        return Err(invalid("feasible-start worker requires a nonempty start"));
    }
    let geometry = geometry.materialize(x.len(), &mut cooperation).await?;
    let mut domain = feasibility::Domain::new(
        geometry,
        prepared,
        feasibility::Limit {
            timing,
            maximum: TimeLimit::Unbounded,
        },
    )?;
    // The parent checks its selected logical/physical deadline at wave edges.
    // A worker does not fabricate an equivalent physical timeout from it.
    let candidate = domain
        .run(x, &args[2], &selected, INPUT_POLICY, &mut cooperation)
        .await?;
    if crate::output_count::current_output_count_in(runtime) == Some(0) {
        Ok(Value::output_list(Vec::new()))
    } else {
        encode(candidate)
    }
}

fn count(value: usize) -> BuiltinResult<Value> {
    u64::try_from(value)
        .map(|value| Value::Int(runmat_value::IntValue::U64(value)))
        .map_err(|_| invalid("worker count cannot be represented in its result"))
}
fn encode(candidate: feasibility::Candidate) -> BuiltinResult<Value> {
    let n = candidate.x.len();
    let mut result = runmat_value::UnboundedStructValueBuilder::new();
    result.insert(
        "x",
        Value::Tensor(Tensor::new(candidate.x, vec![1, n]).map_err(invalid)?),
    );
    result.insert(
        "violation",
        Value::Num(candidate.violation.unwrap_or(f64::NAN)),
    );
    result.insert("calls", count(candidate.constraint_calls)?);
    result.insert("points", count(candidate.points)?);
    result.insert("timedOut", Value::Bool(candidate.timed_out));
    result.insert("hasExtents", Value::Bool(candidate.extents.is_some()));
    let (inequalities, equalities) = candidate.extents.unwrap_or((0, 0));
    result.insert("inequalities", count(inequalities)?);
    result.insert("equalities", count(equalities)?);
    Ok(Value::Struct(result.finish()))
}

pub(super) struct Result {
    pub x: Vec<f64>,
    pub violation: Option<f64>,
    pub calls: usize,
    pub points: usize,
    pub timed_out: bool,
    pub extents: Option<(usize, usize)>,
}
pub(super) async fn decode(
    returned: RootedValueSequence,
    variables: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Result> {
    let returned = {
        let _entry = cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        returned.try_into_single()?
    };
    returned
        .with_owned_async(|value| async {
            let Value::Struct(value) = value else {
                return Err(invalid(
                    "feasible-start worker result must be a scalar structure",
                ));
            };
            if value.fields.len() != 8 {
                return Err(invalid("feasible-start result schema changed"));
            }
            let field = |name: &str| {
                value
                    .fields
                    .get(name)
                    .ok_or_else(|| invalid("feasible-start result is missing a field"))
            };
            let exact = |name: &str| match field(name)? {
                Value::Int(runmat_value::IntValue::U64(value)) => usize::try_from(*value)
                    .map_err(|_| invalid("worker count exceeds host capacity")),
                _ => Err(invalid("worker counters must be exact uint64 scalars")),
            };
            let boolean = |name: &str| match field(name)? {
                Value::Bool(value) => Ok(*value),
                _ => Err(invalid("worker flags must be logical scalar")),
            };
            let calls = exact("calls")?;
            let points = exact("points")?;
            let timed_out = boolean("timedOut")?;
            let dimensions = (exact("inequalities")?, exact("equalities")?);
            let extents = if boolean("hasExtents")? {
                Some(dimensions)
            } else {
                if dimensions != (0, 0) {
                    return Err(invalid("worker reported counts without constraint extents"));
                }
                None
            };
            let violation = match field("violation")? {
                Value::Num(value) if value.is_nan() => None,
                Value::Num(value) if value.is_finite() && *value >= 0. => Some(*value),
                _ => {
                    return Err(invalid(
                        "worker violation must be nonnegative finite or undefined",
                    ))
                }
            };
            let Value::Tensor(point) = field("x")? else {
                return Err(invalid("worker point must be a double row"));
            };
            if point.shape() != [1, variables]
                || point.numeric_dtype() != runmat_value::NumericDType::F64
            {
                return Err(invalid("worker point geometry or class changed"));
            }
            let mut x = POLICY.capacity(variables, cooperation.runtime())?;
            for &value in point.as_f64_slice().expect("double worker point") {
                if !value.is_finite() {
                    return Err(invalid("worker point must be finite"));
                }
                x.push(value);
                cooperation.completed(1).await?;
            }
            Ok(Result {
                x,
                violation,
                calls,
                points,
                timed_out,
                extents,
            })
        })
        .await
}
