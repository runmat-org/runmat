use crate::schema::{
    normalize_literal_string, DatasetSchema, DatasetSchemaProvider, FsDatasetSchemaProvider,
};
use runmat_hir::{
    CallSyntax, HirCallableRef, HirDiagnostic, HirDiagnosticSeverity, LoweringResult, QualifiedName,
};
use std::collections::{HashMap, HashSet};

#[derive(Clone)]
struct DatasetBinding {
    arrays: HashMap<String, usize>,
}

#[derive(Clone)]
struct ArrayBinding {
    rank: Option<usize>,
}

pub fn lint_data_api(result: &LoweringResult) -> Vec<HirDiagnostic> {
    lint_data_api_with_provider(result, &FsDatasetSchemaProvider)
}

pub fn lint_data_api_with_provider(
    result: &LoweringResult,
    provider: &dyn DatasetSchemaProvider,
) -> Vec<HirDiagnostic> {
    let mir = match runmat_mir::lowering::lower_assembly(&result.assembly) {
        Ok(mir) => mir,
        Err(err) => return vec![mir_lowering_diagnostic(err)],
    };
    let store = runmat_mir::analysis::analyze_assembly(&mir);
    lint_data_api_from_mir(&mir, &store, provider)
}

fn mir_lowering_diagnostic(err: runmat_hir::SemanticError) -> HirDiagnostic {
    HirDiagnostic::new(
        "lint.mir.lowering_failed",
        HirDiagnosticSeverity::Error,
        format!("MIR lowering failed: {}", err.message),
        err.span.unwrap_or(runmat_hir::Span { start: 0, end: 0 }),
    )
    .with_category("mir-lowering")
}

fn lint_data_api_from_mir(
    mir: &runmat_mir::MirAssembly,
    _store: &runmat_mir::analysis::AnalysisStore,
    provider: &dyn DatasetSchemaProvider,
) -> Vec<HirDiagnostic> {
    let mut diagnostics = Vec::new();
    let mut datasets = HashMap::<runmat_mir::analysis::MirLocalKey, DatasetBinding>::new();
    let mut arrays = HashMap::<runmat_mir::analysis::MirLocalKey, ArrayBinding>::new();
    let mut tx_locals = HashSet::<runmat_mir::analysis::MirLocalKey>::new();
    let mut data_namespaces = HashSet::<runmat_mir::analysis::MirLocalKey>::new();
    let mut slice_ranks = HashMap::<runmat_mir::analysis::MirLocalKey, usize>::new();

    for body in mir.bodies.values() {
        let mut non_tx_write_count = 0usize;
        for block in &body.blocks {
            for stmt in &block.statements {
                match &stmt.kind {
                    runmat_mir::MirStmtKind::Assign { place, value } => {
                        let Some(local) = assigned_local(place) else {
                            continue;
                        };
                        let local_key = mir_local_key(body, local);
                        if let runmat_mir::MirRvalue::Aggregate {
                            kind: runmat_mir::MirAggregateKind::Cell,
                            elements,
                            ..
                        } = value
                        {
                            slice_ranks.insert(local_key, elements.len());
                        }
                        let runmat_mir::MirRvalue::Call(call) = value else {
                            continue;
                        };
                        let name = mir_call_name(&call.callee);
                        if call.syntax == runmat_hir::CallSyntax::Plain
                            && name.as_deref() == Some("data")
                            && call.args.is_empty()
                        {
                            data_namespaces.insert(local_key);
                        }
                        if is_mir_data_open(
                            body,
                            call.syntax.clone(),
                            name.as_deref(),
                            &call.args,
                            &data_namespaces,
                        ) {
                            if let Some(path) = mir_data_open_path(name.as_deref(), &call.args)
                                .and_then(mir_literal_string)
                            {
                                if let Some(dataset) = infer_dataset_binding(provider, &path) {
                                    datasets.insert(local_key, dataset);
                                }
                            } else {
                                diagnostics.push(diagnostic(
                                    "lint.data.no_untyped_open",
                                    "data.open should use a literal dataset path for static checking",
                                    stmt.span,
                                ));
                            }
                        } else if call.syntax == runmat_hir::CallSyntax::Method
                            && name.as_deref() == Some("array")
                        {
                            if let (Some(dataset_local), Some(array_name)) = (
                                call.args.first().and_then(mir_arg_local),
                                call.args.get(1).and_then(mir_literal_string),
                            ) {
                                if let Some(dataset) =
                                    datasets.get(&mir_local_key(body, dataset_local))
                                {
                                    let rank = dataset.arrays.get(&array_name).copied();
                                    if rank.is_none() {
                                        diagnostics.push(diagnostic(
                                            "lint.data.unknown_array_name",
                                            format!(
                                                "dataset schema does not contain array '{array_name}'"
                                            ),
                                            stmt.span,
                                        ));
                                    }
                                    arrays.insert(local_key, ArrayBinding { rank });
                                }
                            }
                        } else if call.syntax == runmat_hir::CallSyntax::Method
                            && name.as_deref() == Some("begin")
                        {
                            tx_locals.insert(local_key);
                        }
                        if call.syntax == runmat_hir::CallSyntax::Method
                            && name.as_deref() == Some("read")
                        {
                            check_mir_array_read(
                                body,
                                call,
                                &arrays,
                                &slice_ranks,
                                stmt.span,
                                &mut diagnostics,
                            );
                        }
                    }
                    runmat_mir::MirStmtKind::Expr(runmat_mir::MirRvalue::Call(call)) => {
                        let name = mir_call_name(&call.callee);
                        if call.syntax == runmat_hir::CallSyntax::Method
                            && name.as_deref() == Some("read")
                        {
                            check_mir_array_read(
                                body,
                                call,
                                &arrays,
                                &slice_ranks,
                                stmt.span,
                                &mut diagnostics,
                            );
                        } else if call.syntax == runmat_hir::CallSyntax::Method
                            && name.as_deref() == Some("write")
                        {
                            let in_tx =
                                call.args
                                    .first()
                                    .and_then(mir_arg_local)
                                    .is_some_and(|local| {
                                        tx_locals.contains(&mir_local_key(body, local))
                                    });
                            if !in_tx {
                                non_tx_write_count += 1;
                                if non_tx_write_count > 1 {
                                    diagnostics.push(diagnostic(
                                        "lint.data.no_multiwrite_outside_tx",
                                        "multiple dataset writes should be grouped in a transaction",
                                        stmt.span,
                                    ));
                                }
                            }
                        } else if call.syntax == runmat_hir::CallSyntax::Method
                            && name.as_deref() == Some("commit")
                        {
                            diagnostics.push(diagnostic(
                                "lint.data.ignore_commit_result",
                                "transaction commit result should be checked",
                                stmt.span,
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    diagnostics
}

fn mir_local_key(
    body: &runmat_mir::MirBody,
    local: runmat_mir::MirLocalId,
) -> runmat_mir::analysis::MirLocalKey {
    runmat_mir::analysis::MirLocalKey {
        function: body.function,
        local,
    }
}

fn check_mir_array_read(
    body: &runmat_mir::MirBody,
    call: &runmat_mir::MirCall,
    arrays: &HashMap<runmat_mir::analysis::MirLocalKey, ArrayBinding>,
    slice_ranks: &HashMap<runmat_mir::analysis::MirLocalKey, usize>,
    span: runmat_hir::Span,
    diagnostics: &mut Vec<HirDiagnostic>,
) {
    if let (Some(array_local), Some(slice)) = (
        call.args.first().and_then(mir_arg_local),
        call.args.get(1).map(|arg| arg.operand()),
    ) {
        if let Some(array) = arrays.get(&mir_local_key(body, array_local)) {
            if let (Some(expected), Some(actual)) =
                (array.rank, mir_slice_rank(body, slice, slice_ranks))
            {
                if expected != actual {
                    diagnostics.push(diagnostic(
                        "lint.data.invalid_slice_rank",
                        format!("slice rank {actual} does not match array rank {expected}"),
                        span,
                    ));
                }
            }
        }
    }
}

fn infer_dataset_binding(
    provider: &dyn DatasetSchemaProvider,
    path: &str,
) -> Option<DatasetBinding> {
    let schema: DatasetSchema = provider.load_schema(path)?;
    Some(DatasetBinding {
        arrays: schema.arrays,
    })
}

fn assigned_local(place: &runmat_mir::MirPlace) -> Option<runmat_mir::MirLocalId> {
    match place {
        runmat_mir::MirPlace::Local(local) => Some(*local),
        _ => None,
    }
}

fn mir_call_name(callee: &HirCallableRef) -> Option<String> {
    match callee {
        HirCallableRef::Builtin(id) => Some(id.0.clone()),
        HirCallableRef::Unresolved(name) => Some(qualified_name(name)),
        _ => None,
    }
}

fn mir_arg_local(arg: &runmat_mir::MirCallArg) -> Option<runmat_mir::MirLocalId> {
    match arg.operand() {
        runmat_mir::MirOperand::Local(local) => Some(*local),
        _ => None,
    }
}

fn mir_literal_string(arg: &runmat_mir::MirCallArg) -> Option<String> {
    match arg.operand() {
        runmat_mir::MirOperand::Constant(runmat_mir::MirConstant::String(value)) => {
            Some(normalize_literal_string(&value.0))
        }
        _ => None,
    }
}

fn mir_slice_rank(
    body: &runmat_mir::MirBody,
    operand: &runmat_mir::MirOperand,
    slice_ranks: &HashMap<runmat_mir::analysis::MirLocalKey, usize>,
) -> Option<usize> {
    match operand {
        runmat_mir::MirOperand::Constant(runmat_mir::MirConstant::EmptyArray) => Some(0),
        runmat_mir::MirOperand::Local(local) => {
            slice_ranks.get(&mir_local_key(body, *local)).copied()
        }
        _ => None,
    }
}

fn is_mir_data_open(
    body: &runmat_mir::MirBody,
    syntax: CallSyntax,
    name: Option<&str>,
    args: &[runmat_mir::MirCallArg],
    data_namespaces: &HashSet<runmat_mir::analysis::MirLocalKey>,
) -> bool {
    name == Some("data.open")
        || (syntax == CallSyntax::Method
            && name == Some("open")
            && args.first().is_some_and(|arg| {
                matches!(
                    arg.operand(),
                    runmat_mir::MirOperand::Constant(runmat_mir::MirConstant::Symbol(symbol))
                        if symbol.0 == "data"
                ) || match arg.operand() {
                    runmat_mir::MirOperand::Local(local) => {
                        data_namespaces.contains(&mir_local_key(body, *local))
                    }
                    _ => false,
                }
            }))
}

fn mir_data_open_path<'a>(
    name: Option<&str>,
    args: &'a [runmat_mir::MirCallArg],
) -> Option<&'a runmat_mir::MirCallArg> {
    if name == Some("data.open") {
        args.first()
    } else {
        args.get(1)
    }
}

fn qualified_name(name: &QualifiedName) -> String {
    name.0
        .iter()
        .map(|part| part.0.as_str())
        .collect::<Vec<_>>()
        .join(".")
}

fn diagnostic(
    code: &'static str,
    message: impl Into<String>,
    span: runmat_hir::Span,
) -> HirDiagnostic {
    HirDiagnostic::new(code, HirDiagnosticSeverity::Warning, message, span)
        .with_category("data-api")
}
