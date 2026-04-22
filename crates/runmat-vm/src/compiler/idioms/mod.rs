use crate::compiler::core::Compiler;
use crate::compiler::CompileError;
use runmat_hir::HirStmt;

mod stochastic_evolution;

pub(crate) enum StmtIdiomPlan<'a> {
    StochasticEvolution(stochastic_evolution::Plan<'a>),
}

pub(crate) fn try_lower_stmt_idiom(
    compiler: &mut Compiler,
    stmt: &HirStmt,
) -> Result<bool, CompileError> {
    let Some(plan) = detect_stmt_idiom(stmt) else {
        return Ok(false);
    };
    lower_stmt_idiom(compiler, plan)?;
    Ok(true)
}

fn detect_stmt_idiom(stmt: &HirStmt) -> Option<StmtIdiomPlan<'_>> {
    stochastic_evolution::detect(stmt).map(StmtIdiomPlan::StochasticEvolution)
}

fn lower_stmt_idiom(compiler: &mut Compiler, plan: StmtIdiomPlan<'_>) -> Result<(), CompileError> {
    match plan {
        StmtIdiomPlan::StochasticEvolution(plan) => stochastic_evolution::lower(compiler, plan),
    }
}
