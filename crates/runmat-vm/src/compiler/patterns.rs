//! Lowering-time pattern detection.

use crate::compiler::core::Compiler;
use crate::compiler::end_expr::expr_is_one;
use crate::compiler::CompileError;
use crate::instr::Instr;
use once_cell::sync::OnceCell;
use runmat_hir::{HirExpr, HirStmt};

pub(crate) struct StochasticEvolutionPlan<'a> {
    pub state: runmat_hir::VarId,
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
        runmat_hir::HirExprKind::FuncCall(name, _) => name.eq_ignore_ascii_case("randn"),
        _ => false,
    }
}

fn matches_var(expr: &HirExpr, var: runmat_hir::VarId) -> bool {
    matches!(expr.kind, runmat_hir::HirExprKind::Var(id) if id == var)
}

fn extract_drift_and_scale(
    expr: &HirExpr,
    state_var: runmat_hir::VarId,
    z_var: runmat_hir::VarId,
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

fn extract_scale_term(expr: &HirExpr, z_var: runmat_hir::VarId) -> Option<&HirExpr> {
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

impl Compiler {
    pub(crate) fn compile_stochastic_evolution(
        &mut self,
        plan: StochasticEvolutionPlan<'_>,
    ) -> Result<(), CompileError> {
        self.emit(Instr::LoadVar(plan.state.0));
        self.compile_expr(plan.drift)?;
        self.compile_expr(plan.scale)?;
        self.compile_expr(plan.steps)?;
        self.emit(Instr::StochasticEvolution);
        self.emit(Instr::StoreVar(plan.state.0));
        Ok(())
    }

    pub(crate) fn detect_stochastic_evolution<'a>(
        &self,
        expr: &'a HirExpr,
        body: &'a [HirStmt],
    ) -> Option<StochasticEvolutionPlan<'a>> {
        if stochastic_evolution_disabled() {
            return None;
        }
        use runmat_hir::HirExprKind as EK;

        match &expr.kind {
            EK::Range(start, step, end) => {
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
                Some(StochasticEvolutionPlan {
                    state: state_var,
                    drift,
                    scale,
                    steps: end,
                })
            }
            _ => None,
        }
    }
}
