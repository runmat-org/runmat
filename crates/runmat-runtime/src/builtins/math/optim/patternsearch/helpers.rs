//! Predefined Pattern charts use the existing selected axes/history owner.
//! They inspect only their concrete public metadata roles, never the objective.

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn nonlinear_chart_metric_uses_original_rows_without_invoking_callbacks() {
        let mut cooperation = ScalarCooperation::for_builtin(
            runmat_builtins::BuiltinCatalogIdentity {
                name: "psplotmaxconstr",
            },
            None,
        );
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert(
            "nonlinineq",
            Value::Tensor(runmat_value::Tensor::new(vec![-1.0, 0.25], vec![2, 1]).unwrap()),
        );
        fields.insert("nonlineq", Value::Num(-0.5));
        fields.insert("unrelated", Value::FunctionHandle("must-not-run".into()));
        let metadata = Value::Struct(fields.finish());
        assert_eq!(
            violation(
                &metadata,
                OptimPlotKind::PatternMaxConstraint,
                &mut cooperation
            )
            .await
            .unwrap(),
            0.5
        );
        let empty = Value::Struct(runmat_value::UnboundedStructValueBuilder::new().finish());
        assert_eq!(
            violation(
                &empty,
                OptimPlotKind::PatternMaxConstraint,
                &mut cooperation
            )
            .await
            .unwrap(),
            0.0
        );
        let mut invalid = runmat_value::UnboundedStructValueBuilder::new();
        invalid.insert("nonlinineq", Value::from("private-payload-must-not-leak"));
        let invalid = Value::Struct(invalid.finish());
        let failure = violation(
            &invalid,
            OptimPlotKind::PatternMaxConstraint,
            &mut cooperation,
        )
        .await
        .unwrap_err();
        assert!(!failure.message().contains("private-payload"));
    }
}
use super::{buffer, invalid};
use crate::builtins::math::optim::common::{output_callback::OutputState, ScalarCooperation};
use crate::{BuiltinResult, RuntimeError};
use runmat_builtins::OptimPlotKind;
use runmat_value::{NumericDType, Value};
pub(crate) mod bindings;

fn error(kind: OptimPlotKind, message: impl std::fmt::Display) -> RuntimeError {
    crate::build_runtime_error(format!("{}: {message}", kind.name()))
        .with_builtin(kind.name())
        .with_identifier("RunMat:patternsearch:PlotArgument")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn field<'a>(metadata: &'a Value, name: &str, kind: OptimPlotKind) -> BuiltinResult<&'a Value> {
    let Value::Struct(metadata) = metadata else {
        return Err(error(kind, "optimvalues must be a scalar structure"));
    };
    metadata
        .fields
        .get(name)
        .ok_or_else(|| error(kind, format!("optimvalues requires {name}")))
}
fn values(value: &Value, kind: OptimPlotKind) -> BuiltinResult<&[f64]> {
    match value {
        Value::Num(value) => Ok(std::slice::from_ref(value)),
        Value::Tensor(value) if value.numeric_dtype() == NumericDType::F64 => {
            Ok(value.as_f64_slice().expect("double metadata"))
        }
        _ => Err(error(
            kind,
            "plot metadata must be concrete full real host double data",
        )),
    }
}
fn scalar(value: &Value, kind: OptimPlotKind) -> BuiltinResult<f64> {
    match values(value, kind)? {
        [value] if value.is_finite() => Ok(*value),
        _ => Err(error(
            kind,
            "plot metric must be a finite real double scalar",
        )),
    }
}
async fn violation(
    metadata: &Value,
    kind: OptimPlotKind,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<f64> {
    let Value::Struct(metadata) = metadata else {
        return Err(error(kind, "optimvalues must be a scalar structure"));
    };
    let mut violation = 0.0f64;
    for (name, equality) in [("nonlinineq", false), ("nonlineq", true)] {
        if let Some(value) = metadata.fields.get(name) {
            for &value in values(value, kind)? {
                if !value.is_finite() {
                    return Err(error(kind, "constraint plot data must be finite"));
                }
                violation = violation.max(if equality {
                    value.abs()
                } else {
                    value.max(0.0)
                });
                cooperation.completed(1).await?;
            }
        }
    }
    Ok(violation)
}

async fn run(
    kind: OptimPlotKind,
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::context::host_allocation::require_unbounded_workspace(runtime,kind.name(),
        "finite host-memory execution is unsupported; progress-scene workspaces lack a complete finite bound")?;
    if args.len() != 2 {
        return Err(error(kind, "expected optimvalues,flag"));
    }
    if crate::output_count::current_output_count_in(runtime).is_some_and(|n| n > 1) {
        return Err(error(kind, "plot helper returns at most one stop flag"));
    }
    let _original = args.fork();
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: kind.name() },
        runtime,
    );
    cooperation.completed(0).await?;
    let phase = crate::builtins::common::validation::value_text_scalar_string(&args[1])
        .and_then(|phase| OutputState::parse(&phase))
        .ok_or_else(|| error(kind, "flag must be init, iter, or done"))?;
    let iteration = scalar(field(&args[0], "iteration", kind)?, kind)?;
    if iteration < 0.0 || iteration.fract() != 0.0 {
        return Err(error(kind, "iteration must be nonnegative integral"));
    }
    let mut point = Vec::new();
    let metric = match kind {
        OptimPlotKind::PatternBestPoint => {
            let coordinates = values(field(&args[0], "x", kind)?, kind)?;
            point = buffer(coordinates.len())?;
            for &coordinate in coordinates {
                if !coordinate.is_finite() {
                    return Err(error(kind, "point coordinates must be finite"));
                }
                point.push(coordinate);
                cooperation.completed(1).await?;
            }
            0.0
        }
        OptimPlotKind::PatternMaxConstraint => violation(&args[0], kind, &mut cooperation).await?,
        OptimPlotKind::PatternBestFunction => scalar(field(&args[0], "fval", kind)?, kind)?,
        OptimPlotKind::PatternFunctionCount => {
            let count = scalar(field(&args[0], "funccount", kind)?, kind)?;
            if count < 0.0 || count.fract() != 0.0 {
                return Err(error(kind, "funccount must be nonnegative integral"));
            }
            count
        }
        OptimPlotKind::PatternMeshSize => {
            let size = scalar(field(&args[0], "meshsize", kind)?, kind)?;
            if size < 0.0 {
                return Err(error(kind, "meshsize must be nonnegative"));
            }
            size
        }
        _ => return Err(invalid("invalid Pattern chart role")),
    };
    draw(kind, phase, point, iteration, metric, runtime)?;
    Ok(
        if crate::output_count::current_output_count_in(runtime) == Some(0) {
            Value::output_list(Vec::new())
        } else {
            Value::Bool(false)
        },
    )
}
fn draw(
    kind: OptimPlotKind,
    phase: OutputState,
    point: Vec<f64>,
    iteration: f64,
    metric: f64,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    #[cfg(feature = "plot-core")]
    {
        crate::builtins::plotting::state::update_optimization_plot(
            kind, phase, point, iteration, metric, runtime,
        )
    }
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = (phase, point, iteration, metric, runtime);
        Err(error(kind, "predefined plots require plot-core/UI"))
    }
}
