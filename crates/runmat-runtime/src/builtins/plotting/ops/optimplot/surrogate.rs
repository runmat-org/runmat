//! Public surrogate progress callback: original measured metadata selects the
//! real chart series, while the shared receiver owner stores every payload.
use super::{values::Owner, *};
use crate::builtins::plotting::state::{SurrogatePlotSample, SurrogatePlotStep};
use runmat_builtins::SurrogateSampleOrigin;
use runmat_value::StructValue;
const NAME: &str = "surrogateoptplot";
const OWNER: Owner = Owner::new(NAME);
#[runtime_builtin(
    name = "surrogateoptplot",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::optimplot::surrogate"
)]
pub(crate) async fn surrogateoptplot_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    evaluate(runtime.as_ref(), args).await.map_err(|mut error| {
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })
}
async fn evaluate(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::context::host_allocation::require_unbounded_workspace(
        runtime,
        NAME,
        "finite host-memory execution has no complete progress chart and scene allocation bound",
    )?;
    if args.len() != 3
        || crate::output_count::current_output_count_in(runtime).is_some_and(|n| n > 1)
    {
        return Err(OWNER.invalid("expected (x,optimValues,state) and at most one stop output"));
    }
    let phase = crate::builtins::common::validation::value_text_scalar_string(&args[2])
        .and_then(|v| OutputState::parse(&v))
        .ok_or_else(|| OWNER.invalid("state must be init, iter, or done"))?;
    let Value::Struct(fields) = &args[1] else {
        return Err(OWNER.invalid("optimValues must be a scalar struct"));
    };
    let count = required(fields, "funccount")?;
    let resets = required(fields, "surrogateResetCount")?;
    let reset = logical(required(fields, "surrogateReset")?)?;
    let resume = fields
        .fields
        .get("checkpointResume")
        .map(logical)
        .transpose()?
        .unwrap_or(false);
    let origin = required(fields, "currentFlag")?;
    let origin = if super::empty(origin) {
        None
    } else {
        Some(
            crate::builtins::common::validation::value_text_scalar_string(origin)
                .and_then(|text| SurrogateSampleOrigin::parse(&text))
                .ok_or_else(|| {
                    OWNER.invalid("currentFlag must be initial, random, adaptive, or []")
                })?,
        )
    };
    let best = [
        required(fields, "fval")?,
        required(fields, "constrviolation")?,
    ];
    let incumbent = [
        required(fields, "incumbentFval")?,
        required(fields, "incumbentConstrviolation")?,
    ];
    let current = [
        required(fields, "currentFval")?,
        required(fields, "currentConstrviolation")?,
    ];
    let tolerance = fields.fields.get("constraintTolerance");
    if origin.is_none() && current.iter().any(|value| !super::empty(value)) {
        return Err(OWNER.invalid("currentFlag is required for a current measured row"));
    }
    // All consumed outer classes/shapes/intents precede the first transfer.
    // x, flags for earlier bests, Ineq arrays and unrelated fields are unused.
    preflight(count, false)?;
    preflight(resets, false)?;
    if let Some(value) = tolerance {
        preflight(value, false)?;
    }
    for value in best.into_iter().chain(incumbent).chain(current) {
        preflight(value, true)?;
    }
    let mut c = ScalarCooperation::for_builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: NAME },
        runtime,
    );
    c.completed(0).await?;
    let count = counter(count, runtime).await?;
    let reset_count = counter(resets, runtime).await?;
    if reset && reset_count == 0 {
        return Err(OWNER.invalid("a surrogate reset requires a positive reset count"));
    }
    let tolerance = match tolerance {
        Some(value) => OWNER.scalar(&super::gather(value, runtime).await?)?,
        None => 1e-3,
    };
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(OWNER.invalid("constraintTolerance must be finite and nonnegative"));
    }
    let best = sample(best, None, runtime).await?;
    let incumbent = sample(incumbent, None, runtime).await?;
    let current = sample(current, origin, runtime).await?;
    c.completed(0).await?;
    crate::builtins::plotting::state::update_surrogate_plot(
        phase,
        SurrogatePlotStep {
            count,
            best,
            incumbent,
            current,
            reset,
            resume,
            reset_count,
            tolerance,
        },
        runtime,
    )?;
    c.completed(0).await?;
    Ok(
        if crate::output_count::current_output_count_in(runtime) == Some(0) {
            Value::output_list(Vec::new())
        } else {
            Value::Bool(false)
        },
    )
}
fn required<'a>(fields: &'a StructValue, name: &str) -> BuiltinResult<&'a Value> {
    fields
        .fields
        .get(name)
        .ok_or_else(|| OWNER.invalid(&format!("optimValues.{name} is required")))
}
fn logical(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::LogicalArray(value) if value.data.len() == 1 => Ok(value.data[0] != 0),
        _ => Err(OWNER.invalid("surrogateReset must be a logical scalar")),
    }
}
fn preflight(value: &Value, optional: bool) -> BuiltinResult<()> {
    OWNER.real(value)?;
    let count = match value {
        Value::Num(_) => Some(1),
        Value::Tensor(value) => Some(value.len()),
        Value::GpuTensor(value) => value
            .shape()
            .iter()
            .try_fold(1usize, |n, &m| n.checked_mul(m)),
        _ => None,
    };
    if count != Some(1) && !(optional && count == Some(0)) {
        return Err(
            OWNER.invalid("measured fields must be floating scalars or permitted empty arrays")
        );
    }
    if crate::builtins::common::residency::host_gather_intent(value)?
        == crate::builtins::common::residency::HostGatherIntent::ExplicitOrUnproven
    {
        return Err(OWNER.invalid("gather explicit or unproven device metadata before plotting"));
    }
    Ok(())
}
async fn counter(
    value: &Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<usize> {
    runmat_types::decode_shape_dimension(OWNER.scalar(&super::gather(value, runtime).await?)?)
        .map_err(|_| OWNER.invalid("counts must be nonnegative platform-representable integers"))
}
async fn sample(
    values: [&Value; 2],
    origin: Option<SurrogateSampleOrigin>,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<SurrogatePlotSample> {
    let mut prepared = [None; 2];
    for (index, value) in values.into_iter().enumerate() {
        let value = super::gather(value, runtime).await?;
        if !super::empty(&value) {
            let value = OWNER.scalar(&value)?;
            if value.is_infinite() {
                return Err(OWNER.invalid("measured values must be finite, NaN, or []"));
            }
            prepared[index] = Some(value);
        }
    }
    Ok(SurrogatePlotSample {
        objective: prepared[0],
        maximum: prepared[1],
        origin,
    })
}
