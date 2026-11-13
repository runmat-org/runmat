//! Isolated numeric-text evaluation through the existing dynamic compiler and
//! interpreter. The parent source service owns provenance and selected imports.

use super::*;
use crate::interpreter::dispatch::MatrixConstructionAtSource;
use runmat_parser::{BinOp, Expr, Program, Stmt, UnOp};
use runmat_runtime::call::descriptor::MissingCallableTarget;
use runmat_runtime::context::NumericTextEvaluation;

fn empty_result() -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    runmat_runtime::execution::RootedValueSequence::new(runmat_value::ValueSequence::empty())
        .map_err(|error| error.into_runtime_error())
}

/// The restricted mode is a structural admission of the parsed expression,
/// before lowering, provider preparation, or any callable effect. A catalog
/// call must have an explicitly pure, capability-free numeric expression
/// contract; an unresolved user function is not accepted here.
fn restricted_expression(expr: &Expr, functions: &FunctionRegistry) -> bool {
    match expr {
        Expr::Number(..) | Expr::ImaginaryNumber(..) | Expr::IntegerLiteral(..) => true,
        Expr::Ident(name, _) => {
            !functions.names.contains_key(name)
                && runmat_builtins::builtin_constant_catalog_entry_by_name(name).is_some_and(
                    |entry| {
                        matches!(
                            entry.kind,
                            runmat_builtins::BuiltinConstantKind::RealDouble
                                | runmat_builtins::BuiltinConstantKind::ComplexDouble
                                | runmat_builtins::BuiltinConstantKind::Logical
                        )
                    },
                )
        }
        Expr::Unary(op, value, _) => {
            matches!(
                op,
                UnOp::Plus
                    | UnOp::Minus
                    | UnOp::Not
                    | UnOp::Transpose
                    | UnOp::NonConjugateTranspose
            ) && restricted_expression(value, functions)
        }
        Expr::Binary(left, op, right, _) => {
            matches!(
                op,
                BinOp::Add
                    | BinOp::Sub
                    | BinOp::Mul
                    | BinOp::RightDiv
                    | BinOp::LeftDiv
                    | BinOp::Pow
                    | BinOp::ElemMul
                    | BinOp::ElemDiv
                    | BinOp::ElemPow
                    | BinOp::ElemLeftDiv
                    | BinOp::Colon
                    | BinOp::AndAnd
                    | BinOp::OrOr
                    | BinOp::BitAnd
                    | BinOp::BitOr
                    | BinOp::Equal
                    | BinOp::NotEqual
                    | BinOp::Less
                    | BinOp::LessEqual
                    | BinOp::Greater
                    | BinOp::GreaterEqual
            ) && restricted_expression(left, functions)
                && restricted_expression(right, functions)
        }
        Expr::Tensor(rows, _) => rows
            .iter()
            .flatten()
            .all(|value| restricted_expression(value, functions)),
        Expr::Range(start, step, end, _) => {
            restricted_expression(start, functions)
                && step
                    .as_deref()
                    .is_none_or(|value| restricted_expression(value, functions))
                && restricted_expression(end, functions)
        }
        Expr::FuncCall(name, arguments, _) => runmat_builtins::builtin_catalog_entry_by_name(name)
            .is_some_and(|entry| {
                !functions.names.contains_key(name)
                    && entry.contract.purity == runmat_builtins::BuiltinPurity::Pure
                    && entry.contract.workspace_effect.is_none()
                    && entry.contract.environment_effect.is_none()
                    && entry.contract.capabilities.is_empty()
                    && entry.contract.effects.iter().all(|effect| {
                        matches!(
                            effect,
                            runmat_types::EffectKind::MayThrow
                                | runmat_types::EffectKind::MaySuspend
                        )
                    })
                    && arguments
                        .iter()
                        .all(|value| restricted_expression(value, functions))
            }),
        _ => false,
    }
}

fn restricted_program(program: &Program, functions: &FunctionRegistry) -> bool {
    matches!(program.body.as_slice(), [Stmt::ExprStmt(expr, _, _)] if restricted_expression(expr, functions))
}

pub async fn evaluate_numeric_text_source_in_context(
    runtime: runmat_runtime::context::RuntimeContext,
    source: String,
    source_label: String,
    source_ref: runmat_runtime::source_context::RuntimeSourceRef,
    method: NumericTextEvaluation,
    function_registry: &FunctionRegistry,
    selected: DynamicSessionEvalAdmission,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    if selected.source_lease.source_ref() != source_ref {
        return Err(runmat_runtime::build_runtime_error(
            "numeric text source differs from its captured source lease",
        )
        .with_identifier("RunMat:SourceEvaluation:SourceMismatch")
        .build());
    }
    // The source lease contains the exact matrix-expression text. Syntax
    // failure is a failed conversion; source/capability failures are not.
    let ast = match parse_dynamic_eval("str2num", &source) {
        Ok(ast) => ast,
        Err(_) => return empty_result(),
    };
    if method == NumericTextEvaluation::Restricted && !restricted_program(&ast, function_registry) {
        return empty_result();
    }
    let names = HashMap::new();
    let assigned = HashSet::new();
    let requires_product = |canonical: &crate::bytecode::Bytecode, inherited: &FunctionRegistry| {
        selected
            .port
            .host()
            .requires_selected_product(canonical, inherited)
    };
    let compiled = compile_dynamic_workspace_eval_inner_typed(DynamicEvalCompileRequest {
        builtin: "str2num",
        ast: &ast,
        workspace_bindings: &names,
        workspace_value_count: 0,
        lowering_bindings: &names,
        function_registry,
        source_ref: Some(source_ref),
        requested_outputs: 1,
        result_capture: DynamicEvalResultCapture::ExpressionStatement,
        select_product: Some(&requires_product),
        selected_sources: super::super::eval_static_selection::EvalSelectedSources::default(),
    });
    let Some(mut compiled) = (match compiled {
        Ok(compiled) => compiled,
        Err(failure)
            if failure.source_kind == Some(runmat_hir::HirErrorKind::UndefinedVariable) =>
        {
            return empty_result();
        }
        Err(failure) => return Err(failure.into()),
    }) else {
        return empty_result();
    };
    // The first compile supplies compiler-owned HIR site IDs. Only an exact
    // direct entry-frontier call can be observed before this expression runs;
    // a wider expression keeps ordinary dynamic resolution because earlier
    // elements or arguments may change path/source state. Selection itself is
    // the shared live observer and Core still seals the resulting import.
    if method == NumericTextEvaluation::All {
        if let DynamicEvalCompiled::Selected(provisional) = &compiled {
            let sites = provisional
                .hir
                .entrypoints
                .first()
                .and_then(|entry| {
                    provisional
                        .hir
                        .functions
                        .iter()
                        .find(|function| function.id == entry.target)
                })
                .map(super::super::eval_hir_sites::numeric_text_entry_sites)
                .unwrap_or_default();
            if !sites.is_empty() {
                let alias_floor = provisional
                    .canonical_bytecode
                    .function_registry
                    .functions
                    .keys()
                    .chain(
                        provisional
                            .canonical_bytecode
                            .layout
                            .iter()
                            .flat_map(|layout| layout.functions.keys()),
                    )
                    .map(|function| function.0)
                    .max()
                    .unwrap_or(0);
                let selected_sources = super::super::eval_static_selection::select_sites(
                    &sites,
                    alias_floor,
                    &runtime,
                    function_registry,
                    &names,
                    &selected.port,
                )
                .await?;
                if !selected_sources.sources.is_empty()
                    || !selected_sources.external_sources.is_empty()
                    || !selected_sources.natives.is_empty()
                {
                    compiled =
                        compile_dynamic_workspace_eval_inner_typed(DynamicEvalCompileRequest {
                            builtin: "str2num",
                            ast: &ast,
                            workspace_bindings: &names,
                            workspace_value_count: 0,
                            lowering_bindings: &names,
                            function_registry,
                            source_ref: Some(source_ref),
                            requested_outputs: 1,
                            result_capture: DynamicEvalResultCapture::ExpressionStatement,
                            select_product: Some(&requires_product),
                            selected_sources,
                        })?
                        .ok_or_else(|| {
                            runmat_runtime::build_runtime_error(
                                "selected numeric text has no compiler product",
                            )
                            .with_identifier("RunMat:SourceEvaluation:MissingNumericResult")
                            .build()
                        })?;
                }
            }
        }
    }
    let (bytecode, result, runtime, _publication) = match compiled {
        DynamicEvalCompiled::Ordinary(bytecode, result) => (
            *bytecode,
            result,
            runtime.without_selected_graph_program(),
            None,
        ),
        DynamicEvalCompiled::Selected(product) => {
            let admitted = selected.port.host().admit(
                &runtime,
                &selected.source_lease,
                &source,
                None,
                *product,
                function_registry,
            )?;
            (
                admitted.execution_bytecode,
                admitted.result,
                admitted.runtime,
                Some(admitted.publication),
            )
        }
    };
    let result = result.ok_or_else(|| {
        runmat_runtime::build_runtime_error(
            "numeric-text expression has no compiler-owned result slot",
        )
        .with_identifier("RunMat:SourceEvaluation:MissingNumericResult")
        .build()
    })?;
    // No base-workspace snapshot or projection is created here. Ordinary
    // local expression slots live only for this interpreter invocation.
    let admission =
        crate::runtime::workspace::detached_transactional_workspace_admission_with_global_bindings(
            names,
            assigned,
            runmat_runtime::workspace::bindings::FrameGlobalBindings::default(),
        );
    let execution = interpret_workspace(
        &bytecode,
        crate::InterpreterVars::from_owned(Vec::new()),
        Some(&source_label),
        runtime.with_invocation_arity(runmat_runtime::context::RuntimeInvocationArity::Script),
        Some(admission),
        Some(selected.port.clone()),
    )
    .await;
    let InterpreterOutcome::Completed(completion) = match execution.outcome {
        Err(error)
            if error
                .source
                .as_deref()
                .and_then(|source| source.downcast_ref::<MatrixConstructionAtSource>())
                .is_some_and(|cause| cause.in_source_script(source_ref)) =>
        {
            return empty_result();
        }
        Err(error)
            if error
                .source
                .as_deref()
                .and_then(|source| source.downcast_ref::<MissingCallableTarget>())
                .is_some_and(|cause| cause.in_source_script(source_ref)) =>
        {
            return empty_result();
        }
        outcome => outcome?,
    };
    completion.try_map_owned(
        |completion| {
            let value = completion
                .values
                .get(result.slot)
                .map(|value| value.clone())
                .ok_or_else(|| {
                    runmat_runtime::build_runtime_error(
                        "numeric-text result slot was not populated by its compiled expression",
                    )
                    .with_identifier("RunMat:SourceEvaluation:MissingNumericResult")
                    .build()
                })?;
            if method == NumericTextEvaluation::Restricted
                && !runmat_runtime::builtins::common::validation::value_is_numeric_or_logical(
                    &value,
                )
            {
                return Ok(runmat_value::ValueSequence::empty());
            }
            runmat_value::ValueSequence::single(value)
                .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
        },
        |error| error.into_runtime_error(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restricted_constants_follow_catalog_identity_and_captured_shadowing() {
        let program = parse_dynamic_eval("str2num", "[pi true 2j]").unwrap();
        let mut functions = FunctionRegistry::default();
        assert!(restricted_program(&program, &functions));
        functions
            .names
            .insert("pi".into(), runmat_hir::FunctionId(0));
        assert!(!restricted_program(&program, &functions));

        let expression = parse_dynamic_eval("str2num", "[1+2]").unwrap();
        let bindings = HashMap::new();
        let registry = FunctionRegistry::default();
        let compiled = compile_dynamic_workspace_eval_inner_typed(DynamicEvalCompileRequest {
            builtin: "str2num",
            ast: &expression,
            workspace_bindings: &bindings,
            workspace_value_count: 0,
            lowering_bindings: &bindings,
            function_registry: &registry,
            source_ref: None,
            requested_outputs: 1,
            result_capture: DynamicEvalResultCapture::ExpressionStatement,
            select_product: None,
            selected_sources:
                crate::call::builtins::eval_static_selection::EvalSelectedSources::default(),
        })
        .map_err(RuntimeError::from)
        .unwrap()
        .unwrap();
        let DynamicEvalCompiled::Ordinary(bytecode, Some(result)) = compiled else {
            panic!("numeric expression has no compiler-owned output slot");
        };
        assert!(bytecode.instructions.windows(2).any(|pair| matches!(pair,
            [crate::bytecode::Instr::EmitStackTop { label: crate::bytecode::EmitLabel::InternalCapture }, crate::bytecode::Instr::StoreVar(output)]
                if *output == result.slot
        )));
    }

    #[test]
    fn selected_numeric_entry_is_the_exact_first_hir_call() {
        let bindings = HashMap::new();
        let sites = |source| {
            let ast = parse_dynamic_eval("str2num", source).expect("numeric text syntax");
            let lowered = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::new(&bindings))
                .expect("numeric text HIR");
            let entry = lowered.assembly.entrypoints.first().expect("script entry");
            let function = lowered
                .assembly
                .functions
                .iter()
                .find(|function| function.id == entry.target)
                .expect("entry body");
            super::super::super::eval_hir_sites::numeric_text_entry_sites(function)
        };
        let direct = sites("[selected_numeric(2)]");
        assert_eq!(direct.len(), 1);
        assert_eq!(direct[0].name, "selected_numeric");
        assert!(sites("[1, selected_numeric(2)]").is_empty());
        assert!(sites("[selected_numeric(other_call())]").is_empty());
    }
}
