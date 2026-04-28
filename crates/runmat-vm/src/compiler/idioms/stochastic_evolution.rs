use crate::compiler::core::Compiler;
use crate::compiler::end_expr::expr_is_one;
use crate::compiler::CompileError;
use crate::instr::Instr;
use once_cell::sync::OnceCell;
use runmat_hir::{HirExpr, HirExprKind, HirStmt, VarId};

pub(crate) struct Plan<'a> {
    pub state: VarId,
    pub drift: &'a HirExpr,
    pub scale: &'a HirExpr,
    pub steps: &'a HirExpr,
}

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
        HirExprKind::FuncCall(name, _) => name.eq_ignore_ascii_case("randn"),
        _ => false,
    }
}

fn matches_var(expr: &HirExpr, var: VarId) -> bool {
    matches!(expr.kind, HirExprKind::Var(id) if id == var)
}

fn is_exp_call(expr: &HirExpr) -> bool {
    matches!(&expr.kind, HirExprKind::FuncCall(name, _) if name.eq_ignore_ascii_case("exp"))
}

fn extract_scale_term(expr: &HirExpr, z_var: VarId) -> Option<&HirExpr> {
    use runmat_parser::BinOp;

    match &expr.kind {
        HirExprKind::Binary(lhs, BinOp::ElemMul, rhs) => {
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

fn extract_drift_and_scale(
    expr: &HirExpr,
    state_var: VarId,
    z_var: VarId,
) -> Option<(&HirExpr, &HirExpr)> {
    use runmat_parser::BinOp;

    let (maybe_state_side, maybe_exp_side) = match &expr.kind {
        HirExprKind::Binary(lhs, BinOp::ElemMul, rhs) => (lhs.as_ref(), rhs.as_ref()),
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
        HirExprKind::FuncCall(name, args)
            if name.eq_ignore_ascii_case("exp") && args.len() == 1 =>
        {
            &args[0]
        }
        _ => return None,
    };

    match &exp_arg.kind {
        HirExprKind::Binary(lhs, runmat_parser::BinOp::Add, rhs) => {
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

pub(crate) fn detect(stmt: &HirStmt) -> Option<Plan<'_>> {
    if stochastic_evolution_disabled() {
        return None;
    }

    let HirStmt::For { expr, body, .. } = stmt else {
        return None;
    };

    let HirExprKind::Range(start, step, end) = &expr.kind else {
        return None;
    };
    if !expr_is_one(start) {
        return None;
    }
    if let Some(step_expr) = step {
        if !expr_is_one(step_expr) {
            return None;
        }
    }
    if body.len() != 2 {
        return None;
    }

    let (z_var, randn_expr) = match &body[0] {
        HirStmt::Assign(var, expr, _, _) => (*var, expr),
        _ => return None,
    };
    if !is_randn_call(randn_expr) {
        return None;
    }
    let (state_var, update_expr) = match &body[1] {
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

pub(crate) fn lower(compiler: &mut Compiler, plan: Plan<'_>) -> Result<(), CompileError> {
    compiler.emit(Instr::LoadVar(plan.state.0));
    compiler.compile_expr(plan.drift)?;
    compiler.compile_expr(plan.scale)?;
    compiler.compile_expr(plan.steps)?;
    compiler.emit(Instr::StochasticEvolution);
    compiler.emit(Instr::StoreVar(plan.state.0));
    Ok(())
}
