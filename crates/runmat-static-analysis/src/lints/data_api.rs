use crate::schema::{
    normalize_literal_string, DatasetSchema, DatasetSchemaProvider, FsDatasetSchemaProvider,
};
use runmat_builtins::{BuiltinSemanticKind, DataApiOp};
use runmat_hir::{CallSyntax, HirDiagnostic, HirDiagnosticSeverity, LoweringResult};
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
                        let op = data_api_op(call);
                        let method_op = data_api_method_op(call);
                        if call.syntax == runmat_hir::CallSyntax::Plain
                            && op == Some(DataApiOp::Namespace)
                            && call.args.is_empty()
                        {
                            data_namespaces.insert(local_key);
                        }
                        if is_mir_data_open(body, call, &call.args, &data_namespaces) {
                            if let Some(path) =
                                mir_data_open_path(call, &call.args).and_then(mir_literal_string)
                            {
                                datasets.insert(local_key, infer_dataset_binding(provider, &path));
                            } else {
                                diagnostics.push(diagnostic(
                                    "lint.data.no_untyped_open",
                                    "data.open should use a literal dataset path for static checking",
                                    stmt.span,
                                ));
                            }
                        } else if call.syntax == runmat_hir::CallSyntax::Method
                            && method_op == Some(DataApiOp::Array)
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
                            && method_op == Some(DataApiOp::BeginTransaction)
                            && call
                                .args
                                .first()
                                .and_then(mir_arg_local)
                                .is_some_and(|local| {
                                    datasets.contains_key(&mir_local_key(body, local))
                                })
                        {
                            tx_locals.insert(local_key);
                        }
                        if call.syntax == runmat_hir::CallSyntax::Method
                            && method_op == Some(DataApiOp::Read)
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
                        let method_op = data_api_method_op(call);
                        if call.syntax == runmat_hir::CallSyntax::Method
                            && method_op == Some(DataApiOp::Read)
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
                            && method_op == Some(DataApiOp::Write)
                        {
                            let receiver = call.args.first().and_then(mir_arg_local);
                            let in_tx = receiver.is_some_and(|local| {
                                tx_locals.contains(&mir_local_key(body, local))
                            });
                            let is_data_write = receiver.is_some_and(|local| {
                                let key = mir_local_key(body, local);
                                datasets.contains_key(&key)
                                    || arrays.contains_key(&key)
                                    || tx_locals.contains(&key)
                            });
                            if !is_data_write {
                                continue;
                            }
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
                            && method_op == Some(DataApiOp::Commit)
                            && call
                                .args
                                .first()
                                .and_then(mir_arg_local)
                                .is_some_and(|local| {
                                    let key = mir_local_key(body, local);
                                    tx_locals.contains(&key) || datasets.contains_key(&key)
                                })
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

fn infer_dataset_binding(provider: &dyn DatasetSchemaProvider, path: &str) -> DatasetBinding {
    let arrays = provider
        .load_schema(path)
        .map(|schema: DatasetSchema| schema.arrays)
        .unwrap_or_default();
    DatasetBinding { arrays }
}

fn assigned_local(place: &runmat_mir::MirPlace) -> Option<runmat_mir::MirLocalId> {
    match place {
        runmat_mir::MirPlace::Local(local) => Some(*local),
        _ => None,
    }
}

fn data_api_op(call: &runmat_mir::MirCall) -> Option<DataApiOp> {
    match call.semantic_kind {
        BuiltinSemanticKind::DataApi(op) => Some(op),
        _ => None,
    }
}

fn data_api_method_op(call: &runmat_mir::MirCall) -> Option<DataApiOp> {
    if call.syntax != CallSyntax::Method {
        return None;
    }
    call_callee_name(call).and_then(|name| runmat_builtins::data_api_method_op_for_name(&name))
}

fn call_callee_name(call: &runmat_mir::MirCall) -> Option<String> {
    match &call.callee {
        runmat_hir::HirCallableRef::Builtin(id) => Some(id.0.clone()),
        runmat_hir::HirCallableRef::Unresolved(name) => Some(
            name.0
                .iter()
                .map(|part| part.0.as_str())
                .collect::<Vec<_>>()
                .join("."),
        ),
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
    call: &runmat_mir::MirCall,
    args: &[runmat_mir::MirCallArg],
    data_namespaces: &HashSet<runmat_mir::analysis::MirLocalKey>,
) -> bool {
    (call.syntax == CallSyntax::Plain
        && data_api_op(call) == Some(DataApiOp::Open)
        && is_data_open_callee(call))
        || (call.syntax == CallSyntax::Method
            && data_api_method_op(call) == Some(DataApiOp::Open)
            && args
                .first()
                .is_some_and(|arg| is_data_namespace(body, arg, data_namespaces)))
}

fn mir_data_open_path<'a>(
    call: &runmat_mir::MirCall,
    args: &'a [runmat_mir::MirCallArg],
) -> Option<&'a runmat_mir::MirCallArg> {
    if call.syntax == CallSyntax::Plain {
        args.first()
    } else {
        args.get(1)
    }
}

fn is_data_namespace(
    body: &runmat_mir::MirBody,
    arg: &runmat_mir::MirCallArg,
    data_namespaces: &HashSet<runmat_mir::analysis::MirLocalKey>,
) -> bool {
    matches!(
        arg.operand(),
        runmat_mir::MirOperand::Constant(runmat_mir::MirConstant::Symbol(symbol))
            if runmat_builtins::is_data_namespace_symbol(&symbol.0)
    ) || match arg.operand() {
        runmat_mir::MirOperand::Local(local) => {
            data_namespaces.contains(&mir_local_key(body, *local))
        }
        _ => false,
    }
}

fn is_data_open_callee(call: &runmat_mir::MirCall) -> bool {
    call_callee_name(call).is_some_and(|name| runmat_builtins::is_data_open_name(&name))
}

fn diagnostic(
    code: &'static str,
    message: impl Into<String>,
    span: runmat_hir::Span,
) -> HirDiagnostic {
    HirDiagnostic::new(code, HirDiagnosticSeverity::Warning, message, span)
        .with_category("data-api")
}
