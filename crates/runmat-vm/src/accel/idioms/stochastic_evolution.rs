use crate::accel::idioms::{AccelStmtIdiom, StmtIdiomContext};
use crate::compiler::core::Compiler;
use crate::compiler::end_expr::expr_is_one;
use crate::compiler::CompileError;
use crate::instr::Instr;
use once_cell::sync::OnceCell;
#[cfg(feature = "native-accel")]
use runmat_accelerate::fusion_residency;
use runmat_builtins::Value;
use runmat_hir::{HirExpr, HirStmt, VarId};
use runmat_runtime::RuntimeError;

pub(crate) struct Plan<'a> {
    pub state: VarId,
    pub drift: &'a HirExpr,
    pub scale: &'a HirExpr,
    pub steps: &'a HirExpr,
}

pub(crate) struct Idiom;

fn stochastic_evolution_disabled() -> bool {
    static DISABLE: OnceCell<bool> = OnceCell::new();
    *DISABLE.get_or_init(|| {
        std::env::var("RUNMAT_DISABLE_STOCHASTIC_EVOLUTION")
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

fn is_randn_call(expr: &HirExpr) -> bool {
    match &expr.kind {
        runmat_hir::HirExprKind::FuncCall(name, _) => name.eq_ignore_ascii_case("randn"),
        _ => false,
    }
}

fn matches_var(expr: &HirExpr, var: VarId) -> bool {
    matches!(expr.kind, runmat_hir::HirExprKind::Var(id) if id == var)
}

fn extract_drift_and_scale(
    expr: &HirExpr,
    state_var: VarId,
    z_var: VarId,
) -> Option<(&HirExpr, &HirExpr)> {
    use runmat_hir::HirExprKind as EK;
    use runmat_parser::BinOp;

    let (maybe_state_side, maybe_exp_side) = match &expr.kind {
        EK::Binary(lhs, BinOp::ElemMul, rhs) => (lhs.as_ref(), rhs.as_ref()),
        _ => return None,
    };

    let exp_side = if matches_var(maybe_state_side, state_var) && is_exp_call(maybe_exp_side) {
        maybe_exp_side
    } else if matches_var(maybe_exp_side, state_var) && is_exp_call(maybe_state_side) {
        maybe_state_side
    } else {
        return None;
    };

    let exp_arg = match &exp_side.kind {
        EK::FuncCall(name, args) if name.eq_ignore_ascii_case("exp") && args.len() == 1 => &args[0],
        _ => return None,
    };

    match &exp_arg.kind {
        EK::Binary(lhs, BinOp::Add, rhs) => {
            if let Some(scale_expr) = extract_scale_term(lhs, z_var) {
                Some((rhs.as_ref(), scale_expr))
            } else if let Some(scale_expr) = extract_scale_term(rhs, z_var) {
                Some((lhs.as_ref(), scale_expr))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn extract_scale_term(expr: &HirExpr, z_var: VarId) -> Option<&HirExpr> {
    use runmat_hir::HirExprKind as EK;
    use runmat_parser::BinOp;

    match &expr.kind {
        EK::Binary(lhs, BinOp::ElemMul, rhs) => {
            if matches_var(lhs, z_var) {
                Some(rhs.as_ref())
            } else if matches_var(rhs, z_var) {
                Some(lhs.as_ref())
            } else {
                None
            }
        }
        _ => None,
    }
}

fn is_exp_call(expr: &HirExpr) -> bool {
    matches!(&expr.kind, runmat_hir::HirExprKind::FuncCall(name, _) if name.eq_ignore_ascii_case("exp"))
}

impl<'a> AccelStmtIdiom<'a> for Idiom {
    type Plan = Plan<'a>;

    fn detect(ctx: &StmtIdiomContext<'a>) -> Option<Self::Plan> {
        if stochastic_evolution_disabled() {
            return None;
        }
        use runmat_hir::HirExprKind as EK;

        match &ctx.expr.kind {
            EK::Range(start, step, end) => {
                if !expr_is_one(start) {
                    return None;
                }
                if let Some(step_expr) = step {
                    if !expr_is_one(step_expr) {
                        return None;
                    }
                }
                if ctx.body.len() != 2 {
                    return None;
                }
                let (z_var, randn_expr) = match &ctx.body[0] {
                    HirStmt::Assign(var, expr, _, _) => (*var, expr),
                    _ => return None,
                };
                if !is_randn_call(randn_expr) {
                    return None;
                }
                let (state_var, update_expr) = match &ctx.body[1] {
                    HirStmt::Assign(var, expr, _, _) => (*var, expr),
                    _ => return None,
                };
                let (drift, scale) = extract_drift_and_scale(update_expr, state_var, z_var)?;
                Some(Plan {
                    state: state_var,
                    drift,
                    scale,
                    steps: end,
                })
            }
            _ => None,
        }
    }

    fn lower(compiler: &mut Compiler, plan: Self::Plan) -> Result<(), CompileError> {
        compiler.emit(Instr::LoadVar(plan.state.0));
        compiler.compile_expr(plan.drift)?;
        compiler.compile_expr(plan.scale)?;
        compiler.compile_expr(plan.steps)?;
        compiler.emit(Instr::StochasticEvolution);
        compiler.emit(Instr::StoreVar(plan.state.0));
        Ok(())
    }
}

pub async fn execute_stochastic_evolution(
    state: Value,
    drift: Value,
    scale: Value,
    steps: Value,
) -> Result<Value, RuntimeError> {
    let steps_u32 = parse_steps_value(&steps).await?;
    if steps_u32 == 0 {
        return Ok(state);
    }

    #[cfg(feature = "native-accel")]
    {
        if let Some(provider) = runmat_accelerate_api::provider() {
            let (state_handle, state_owned) =
                ensure_gpu_tensor_for_stochastic(provider, &state).await?;
            let drift_scalar =
                scalar_from_value_scalar(&drift, "stochastic_evolution drift").await?;
            let scale_scalar =
                scalar_from_value_scalar(&scale, "stochastic_evolution scale").await?;
            match provider.stochastic_evolution(
                &state_handle,
                drift_scalar,
                scale_scalar,
                steps_u32,
            ) {
                Ok(output) => {
                    if let Some(temp) = state_owned {
                        let _ = provider.free(&temp);
                    }
                    fusion_residency::mark(&output);
                    return Ok(Value::GpuTensor(output));
                }
                Err(err) => {
                    log::debug!("stochastic_evolution provider fallback to host: {}", err);
                    if let Some(temp) = state_owned {
                        let _ = provider.free(&temp);
                    }
                }
            }
        }
    }

    let gathered_state = runmat_runtime::dispatcher::gather_if_needed_async(&state)
        .await
        .map_err(|e| format!("stochastic_evolution: {e}"))?;
    let mut tensor_value = match gathered_state {
        Value::Tensor(t) => t,
        other => runmat_runtime::builtins::common::tensor::value_into_tensor_for(
            "stochastic_evolution",
            other,
        )?,
    };
    let drift_scalar = scalar_from_value_scalar(&drift, "stochastic_evolution drift").await?;
    let scale_scalar = scalar_from_value_scalar(&scale, "stochastic_evolution scale").await?;
    runmat_runtime::builtins::stats::random::stochastic_evolution::stochastic_evolution_host(
        &mut tensor_value,
        drift_scalar,
        scale_scalar,
        steps_u32,
    )?;
    Ok(Value::Tensor(tensor_value))
}

async fn scalar_from_value_scalar(value: &Value, label: &str) -> Result<f64, RuntimeError> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(t) if t.data.len() == 1 => Ok(t.data[0]),
        Value::Tensor(t) => Err(format!(
            "{label}: expected scalar tensor, got {} elements",
            t.data.len()
        )
        .into()),
        Value::GpuTensor(_) => {
            let gathered = runmat_runtime::dispatcher::gather_if_needed_async(value)
                .await
                .map_err(|e| format!("{label}: {e}"))?;
            match gathered {
                Value::Num(n) => Ok(n),
                Value::Int(i) => Ok(i.to_f64()),
                Value::Tensor(t) if t.data.len() == 1 => Ok(t.data[0]),
                Value::Tensor(t) => Err(format!(
                    "{label}: expected scalar tensor, got {} elements",
                    t.data.len()
                )
                .into()),
                other => Err(format!("{label}: expected numeric scalar, got {:?}", other).into()),
            }
        }
        other => Err(format!("{label}: expected numeric scalar, got {:?}", other).into()),
    }
}

async fn parse_steps_value(value: &Value) -> Result<u32, RuntimeError> {
    let raw = scalar_from_value_scalar(value, "stochastic_evolution steps").await?;
    if !raw.is_finite() || raw < 0.0 {
        return Err(crate::interpreter::errors::mex(
            "InvalidSteps",
            "stochastic_evolution: steps must be a non-negative scalar",
        ));
    }
    Ok(raw.round() as u32)
}

#[cfg(feature = "native-accel")]
async fn ensure_gpu_tensor_for_stochastic(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    value: &Value,
) -> Result<
    (
        runmat_accelerate_api::GpuTensorHandle,
        Option<runmat_accelerate_api::GpuTensorHandle>,
    ),
    RuntimeError,
> {
    match value {
        Value::GpuTensor(handle) => Ok((handle.clone(), None)),
        Value::Tensor(tensor) => {
            let handle = upload_tensor_view(provider, tensor)?;
            Ok((handle.clone(), Some(handle)))
        }
        _ => {
            let gathered = runmat_runtime::dispatcher::gather_if_needed_async(value)
                .await
                .map_err(|e| format!("stochastic_evolution: {e}"))?;
            match gathered {
                Value::Tensor(t) => {
                    let handle = upload_tensor_view(provider, &t)?;
                    Ok((handle.clone(), Some(handle)))
                }
                other => {
                    let tensor = runmat_runtime::builtins::common::tensor::value_into_tensor_for(
                        "stochastic_evolution",
                        other,
                    )?;
                    let handle = upload_tensor_view(provider, &tensor)?;
                    Ok((handle.clone(), Some(handle)))
                }
            }
        }
    }
}

#[cfg(feature = "native-accel")]
fn upload_tensor_view(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    tensor: &runmat_builtins::Tensor,
) -> Result<runmat_accelerate_api::GpuTensorHandle, RuntimeError> {
    let view = runmat_accelerate_api::HostTensorView {
        data: &tensor.data,
        shape: &tensor.shape,
    };
    provider
        .upload(&view)
        .map_err(|e| crate::interpreter::errors::mex("UploadFailed", &e.to_string()))
}
