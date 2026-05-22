use crate::{MirAssembly, MirBody, MirOperand, MirTerminator, MirTerminatorKind};
use runmat_hir::{HirAssembly, HirError, HirFunction};
use std::collections::HashSet;

use super::{control_flow::ControlFlowBuilder, expr::lower_simple_operand, MirLoweringContext};

pub fn lower_assembly(hir: &HirAssembly) -> Result<MirAssembly, HirError> {
    let mut assembly = MirAssembly::default();
    let async_functions: HashSet<_> = hir
        .functions
        .iter()
        .filter(|function| function.modifiers.is_async)
        .map(|function| function.id)
        .collect();
    for function in &hir.functions {
        assembly.bodies.insert(
            function.id,
            lower_function_with_context(
                function,
                MirLoweringContext::with_async_functions(async_functions.clone()),
            )?,
        );
    }
    Ok(assembly)
}

fn lower_function_with_context(
    function: &HirFunction,
    mut ctx: MirLoweringContext,
) -> Result<MirBody, HirError> {
    let mut locals = ctx.locals_for_function(function);
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
                HirError::new("function return binding did not lower to a simple MIR operand")
            })
        })
        .collect::<Result<_, _>>()?;
    let return_terminator = MirTerminator {
        kind: MirTerminatorKind::Return(returns),
        span: function.span,
    };
    let blocks =
        ControlFlowBuilder::new().lower_function_body(&ctx, &function.body, return_terminator)?;
    let temp_locals = ctx.take_temp_locals();
    locals.extend(temp_locals);

    Ok(MirBody {
        function: function.id,
        abi: function.abi.clone(),
        locals,
        blocks,
    })
}
