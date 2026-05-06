use crate::{MirAssembly, MirBody, MirOperand, MirSourceMap, MirTerminator, MirTerminatorKind};
use runmat_hir::{HirAssembly, HirFunction, SemanticError};

use super::{control_flow::ControlFlowBuilder, expr::lower_operand, MirLoweringContext};

pub fn lower_assembly(hir: &HirAssembly) -> Result<MirAssembly, SemanticError> {
    let mut assembly = MirAssembly::default();
    for function in &hir.functions {
        assembly
            .bodies
            .insert(function.id, lower_function(function)?);
    }
    Ok(assembly)
}

pub fn lower_function(function: &HirFunction) -> Result<MirBody, SemanticError> {
    let mut ctx = MirLoweringContext::new();
    let (locals, local_sources) = ctx.locals_for_function(function);
    let returns: Vec<MirOperand> = function
        .outputs
        .iter()
        .map(|binding| {
            let expr = runmat_hir::HirExpr {
                id: runmat_hir::ExprId(usize::MAX),
                kind: runmat_hir::HirExprKind::Binding(*binding),
                span: function.span,
            };
            lower_operand(&ctx, &expr)
        })
        .collect::<Result<_, _>>()?;
    let return_terminator = MirTerminator {
        kind: MirTerminatorKind::Return(returns),
        span: function.span,
    };
    let (blocks, statement_sources) =
        ControlFlowBuilder::new().lower_function_body(&ctx, &function.body, return_terminator)?;
    Ok(MirBody {
        function: function.id,
        locals,
        blocks,
        source_map: MirSourceMap {
            function: Some(function.id),
            statements: statement_sources,
            locals: local_sources,
        },
    })
}
