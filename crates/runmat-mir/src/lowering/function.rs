use crate::{MirAssembly, MirBody, MirOperand, MirSourceMap, MirTerminator, MirTerminatorKind};
use runmat_hir::{HirAssembly, HirFunction, SemanticError};

use super::{control_flow::ControlFlowBuilder, expr::lower_simple_operand, MirLoweringContext};

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
            lower_simple_operand(&ctx, &expr)?.ok_or_else(|| {
                SemanticError::new("function return binding did not lower to a simple MIR operand")
            })
        })
        .collect::<Result<_, _>>()?;
    let return_terminator = MirTerminator {
        kind: MirTerminatorKind::Return(returns),
        span: function.span,
    };
    let (blocks, statement_sources) =
        ControlFlowBuilder::new().lower_function_body(&ctx, &function.body, return_terminator)?;
    let (temp_locals, temp_sources) = ctx.take_temp_locals();
    let mut locals = locals;
    let mut local_sources = local_sources;
    locals.extend(temp_locals);
    local_sources.extend(temp_sources);

    Ok(MirBody {
        function: function.id,
        abi: function.abi.clone(),
        locals,
        blocks,
        source_map: MirSourceMap {
            function: Some(function.id),
            statements: statement_sources,
            locals: local_sources,
        },
    })
}
