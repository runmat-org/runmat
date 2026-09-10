use crate::{
    MirAssembly, MirBody, MirFunctionMetadata, MirOperand, MirTerminator, MirTerminatorKind,
};
use runmat_hir::{HirAssembly, HirError, HirFunction};
use std::collections::HashSet;

use super::{control_flow::ControlFlowBuilder, expr::lower_simple_operand, MirLoweringContext};

pub fn lower_assembly(hir: &HirAssembly) -> Result<MirAssembly, HirError> {
    let mut assembly = MirAssembly {
        classes: hir
            .classes
            .iter()
            .map(|class| class.declaration.clone())
            .collect(),
        ..MirAssembly::default()
    };
    assembly.classes.sort_by_key(|class| class.id);
    assembly.entrypoints = hir
        .entrypoints
        .iter()
        .map(|entrypoint| entrypoint.target)
        .collect();
    assembly.entrypoints.sort();
    assembly.entrypoints.dedup();
    let async_functions: HashSet<_> = hir
        .functions
        .iter()
        .filter(|function| function.modifiers.is_async)
        .map(|function| function.id)
        .collect();
    for function in &hir.functions {
        let module = hir
            .modules
            .iter()
            .find(|module| module.id == function.module)
            .ok_or_else(|| HirError::new("function references a missing source module"))?;
        let source = u32::try_from(module.source_id.0)
            .map(runmat_types::ProgramSourceId)
            .map_err(|_| HirError::new("function source identity exceeds the portable schema"))?;
        let class_method_owner = class_method_owner(hir, function)?;
        assembly.functions.insert(
            function.id,
            MirFunctionMetadata {
                source,
                name: function.name.clone(),
                parent: function.parent,
                enclosing_class: function.enclosing_class,
                class_method_owner,
                kind: function.kind.clone(),
                argument_validations: function.argument_validations.clone(),
                captures: function.captures.clone(),
                modifiers: function.modifiers.clone(),
                span: function.span,
            },
        );
        assembly.bodies.insert(
            function.id,
            lower_function_with_context(
                function,
                MirLoweringContext::with_async_functions(async_functions.clone(), function.id),
            )?,
        );
    }
    Ok(assembly)
}

fn class_method_owner(
    hir: &HirAssembly,
    function: &HirFunction,
) -> Result<Option<runmat_types::ClassMethodOwner>, HirError> {
    let runmat_hir::FunctionKind::ClassMethod { is_static } = &function.kind else {
        return Ok(None);
    };
    let is_static = *is_static;
    let class_id = function
        .enclosing_class
        .ok_or_else(|| HirError::new("class method is missing its enclosing class identity"))?;
    let class = hir
        .classes
        .iter()
        .find(|class| class.declaration.id == class_id)
        .ok_or_else(|| HirError::new("class method references a missing class declaration"))?;
    let method = class
        .declaration
        .methods
        .iter()
        .find(|method| method.function == function.id)
        .ok_or_else(|| HirError::new("class method is absent from its class declaration"))?;
    if method.is_static != is_static {
        return Err(HirError::new(
            "class method static ownership disagrees with its function kind",
        ));
    }
    Ok(Some(runmat_types::ClassMethodOwner {
        declaring_class: runmat_types::ClassIdentity::from_qualified_name(&class.declaration.name)
            .map_err(|error| HirError::new(error.to_string()))?,
        method: method.name.clone(),
        is_static,
    }))
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
