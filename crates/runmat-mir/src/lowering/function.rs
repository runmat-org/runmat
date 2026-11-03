use crate::{
    MirAssembly, MirBody, MirFunctionMetadata, MirOperand, MirTerminator, MirTerminatorKind,
};
use runmat_hir::{BindingRole, HirAssembly, HirError, HirFunction};
use std::collections::HashSet;

use super::{control_flow::ControlFlowBuilder, expr::lower_simple_operand, MirLoweringContext};

pub fn lower_assembly(hir: &HirAssembly) -> Result<MirAssembly, HirError> {
    for (owner, binding) in &hir.statement_result_bindings {
        let function = hir
            .functions
            .iter()
            .find(|function| function.id == *owner)
            .ok_or_else(|| HirError::new("statement result references a missing function"))?;
        if !function.locals.contains(binding)
            || !hir.bindings.iter().any(|candidate| {
                candidate.id == *binding
                    && candidate.owner == runmat_hir::BindingOwner::Function(*owner)
            })
        {
            return Err(HirError::new(
                "statement result binding differs from its exact function owner",
            ));
        }
    }
    let mut assembly = MirAssembly {
        classes: hir
            .classes
            .iter()
            .map(|class| {
                let module = hir
                    .modules
                    .get(class.module.0)
                    .filter(|module| module.id == class.module)
                    .ok_or_else(|| HirError::new("class references a missing source module"))?;
                if class.declaration.source != module.source_id {
                    return Err(HirError::new(
                        "class declaration source differs from its HIR module",
                    ));
                }
                Ok(class.declaration.clone())
            })
            .collect::<Result<Vec<_>, HirError>>()?,
        class_property_defaults: super::class_defaults::project(hir),
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
    let external_async_functions = hir
        .bound_function_modifiers
        .iter()
        .filter_map(|(id, modifiers)| modifiers.is_async.then_some(*id))
        .collect::<HashSet<_>>();
    let incoming_workspace: HashSet<_> = hir
        .bindings
        .iter()
        .filter(|binding| binding.role == BindingRole::IncomingWorkspace)
        .map(|binding| binding.id)
        .collect();
    let external_workspace: HashSet<_> = hir
        .bindings
        .iter()
        .filter(|binding| binding.role == BindingRole::ExternalWorkspace)
        .map(|binding| binding.id)
        .collect();
    for function in &hir.functions {
        let module = hir
            .modules
            .iter()
            .find(|module| module.id == function.module)
            .ok_or_else(|| HirError::new("function references a missing source module"))?;
        let source = module.source_id;
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
                source_role: hir.function_source_role(function),
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
                MirLoweringContext::with_async_functions(
                    async_functions.clone(),
                    external_async_functions.clone(),
                    function.id,
                )
                .with_statement_ans(hir.statement_result_bindings.get(&function.id).copied()),
                &incoming_workspace,
                &external_workspace,
            )?,
        );
    }
    super::lexical_captures::propagate(hir, &mut assembly)?;
    Ok(assembly)
}

fn class_method_owner(
    hir: &HirAssembly,
    function: &HirFunction,
) -> Result<Option<runmat_types::ClassMethodOwner>, HirError> {
    let runmat_hir::FunctionKind::ClassMethod { is_static, role } = &function.kind else {
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
    if function.module != class.module {
        return Err(HirError::new(
            "class method source module differs from its declaration",
        ));
    }
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
        role: *role,
    }))
}

fn lower_function_with_context(
    function: &HirFunction,
    mut ctx: MirLoweringContext,
    incoming_workspace: &HashSet<runmat_hir::BindingId>,
    external_workspace: &HashSet<runmat_hir::BindingId>,
) -> Result<MirBody, HirError> {
    let mut locals = ctx.locals_for_function(function, incoming_workspace, external_workspace);
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
    let lowered =
        ControlFlowBuilder::new().lower_function_body(&ctx, &function.body, return_terminator)?;
    let temp_locals = ctx.take_temp_locals();
    locals.extend(temp_locals);

    let body = MirBody {
        function: function.id,
        abi: function.abi.clone(),
        locals,
        blocks: lowered.blocks,
        catch_scopes: lowered.catch_scopes,
    };
    body.validate_statement_outputs().map_err(HirError::new)?;
    Ok(body)
}
