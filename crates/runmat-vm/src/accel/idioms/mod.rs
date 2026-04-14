use crate::compiler::core::Compiler;
use crate::compiler::CompileError;
use runmat_hir::{HirExpr, HirStmt};

pub mod stochastic_evolution;

pub(crate) struct StmtIdiomContext<'a> {
    pub expr: &'a HirExpr,
    pub body: &'a [HirStmt],
}

pub(crate) trait AccelStmtIdiom<'a> {
    type Plan;

    fn detect(ctx: &StmtIdiomContext<'a>) -> Option<Self::Plan>;
    fn lower(compiler: &mut Compiler, plan: Self::Plan) -> Result<(), CompileError>;
}

pub(crate) enum StmtIdiomPlan<'a> {
    StochasticEvolution(stochastic_evolution::Plan<'a>),
}

pub(crate) fn detect_stmt_idiom<'a>(
    expr: &'a HirExpr,
    body: &'a [HirStmt],
) -> Option<StmtIdiomPlan<'a>> {
    let ctx = StmtIdiomContext { expr, body };
    <stochastic_evolution::Idiom as AccelStmtIdiom<'a>>::detect(&ctx)
        .map(StmtIdiomPlan::StochasticEvolution)
}

pub(crate) fn lower_stmt_idiom(
    compiler: &mut Compiler,
    plan: StmtIdiomPlan<'_>,
) -> Result<(), CompileError> {
    match plan {
        StmtIdiomPlan::StochasticEvolution(plan) => {
            <stochastic_evolution::Idiom as AccelStmtIdiom<'_>>::lower(compiler, plan)
        }
    }
}
