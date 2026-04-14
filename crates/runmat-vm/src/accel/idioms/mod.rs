use crate::compiler::core::Compiler;
use crate::compiler::CompileError;
use runmat_hir::{HirExpr, HirStmt};

pub mod stochastic_evolution;

pub(crate) enum StmtIdiomPlan<'a> {
    StochasticEvolution(stochastic_evolution::Plan<'a>),
}

pub(crate) fn detect_stmt_idiom<'a>(
    expr: &'a HirExpr,
    body: &'a [HirStmt],
) -> Option<StmtIdiomPlan<'a>> {
    stochastic_evolution::detect(expr, body).map(StmtIdiomPlan::StochasticEvolution)
}

pub(crate) fn lower_stmt_idiom(
    compiler: &mut Compiler,
    plan: StmtIdiomPlan<'_>,
) -> Result<(), CompileError> {
    match plan {
        StmtIdiomPlan::StochasticEvolution(plan) => stochastic_evolution::lower(compiler, plan),
    }
}
