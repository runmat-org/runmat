use crate::schema::{
    normalize_literal_string, DatasetSchema, DatasetSchemaProvider, FsDatasetSchemaProvider,
};
use runmat_hir::{
    HirClassMember, HirDiagnostic, HirDiagnosticSeverity, HirExpr, HirExprKind, HirLValue, HirStmt,
    LoweringResult, VarId,
};
use std::collections::{HashMap, HashSet};

#[derive(Clone)]
struct DatasetBinding {
    arrays: HashMap<String, usize>,
}

#[derive(Clone)]
struct ArrayBinding {
    array_name: String,
    rank: Option<usize>,
}

pub fn lint_data_api(result: &LoweringResult) -> Vec<HirDiagnostic> {
    let provider = FsDatasetSchemaProvider;
    lint_data_api_with_provider(result, &provider)
}

pub fn lint_data_api_with_provider(
    result: &LoweringResult,
    provider: &dyn DatasetSchemaProvider,
) -> Vec<HirDiagnostic> {
    let mut diags = Vec::new();
    let mut non_tx_write_count = 0usize;
    let mut tx_vars = HashSet::<VarId>::new();
    collect_tx_bindings_from_stmts(&result.hir.body, &mut tx_vars);
    for stmt in &result.hir.body {
        walk_stmt_general(stmt, &mut diags, &mut non_tx_write_count, &tx_vars);
    }

    let mut datasets = HashMap::<VarId, DatasetBinding>::new();
    let mut arrays = HashMap::<VarId, ArrayBinding>::new();
    for stmt in &result.hir.body {
        analyze_stmt_bindings(
            stmt,
            result,
            provider,
            &mut datasets,
            &mut arrays,
            &mut diags,
        );
    }

    diags
}

fn collect_tx_bindings_from_stmts(stmts: &[HirStmt], tx_vars: &mut HashSet<VarId>) {
    for stmt in stmts {
        match stmt {
            HirStmt::Assign(var_id, expr, _, _)
            | HirStmt::AssignLValue(HirLValue::Var(var_id), expr, _, _) => {
                if expr_is_begin_call(expr) {
                    tx_vars.insert(*var_id);
                }
            }
            HirStmt::MultiAssign(var_ids, expr, _, _) => {
                if expr_is_begin_call(expr) {
                    for var_id in var_ids.iter().flatten() {
                        tx_vars.insert(*var_id);
                    }
                }
            }
            HirStmt::If {
                then_body,
                elseif_blocks,
                else_body,
                ..
            } => {
                collect_tx_bindings_from_stmts(then_body, tx_vars);
                for (_, body) in elseif_blocks {
                    collect_tx_bindings_from_stmts(body, tx_vars);
                }
                if let Some(body) = else_body {
                    collect_tx_bindings_from_stmts(body, tx_vars);
                }
            }
            HirStmt::While { body, .. }
            | HirStmt::For { body, .. }
            | HirStmt::Function { body, .. } => {
                collect_tx_bindings_from_stmts(body, tx_vars);
            }
            HirStmt::Switch {
                cases, otherwise, ..
            } => {
                for (_, body) in cases {
                    collect_tx_bindings_from_stmts(body, tx_vars);
                }
                if let Some(body) = otherwise {
                    collect_tx_bindings_from_stmts(body, tx_vars);
                }
            }
            HirStmt::TryCatch {
                try_body,
                catch_body,
                ..
            } => {
                collect_tx_bindings_from_stmts(try_body, tx_vars);
                collect_tx_bindings_from_stmts(catch_body, tx_vars);
            }
            HirStmt::ClassDef { members, .. } => {
                for member in members {
                    if let HirClassMember::Methods { body, .. } = member {
                        collect_tx_bindings_from_stmts(body, tx_vars);
                    }
                }
            }
            HirStmt::ExprStmt(_, _, _)
            | HirStmt::Break(_)
            | HirStmt::Continue(_)
            | HirStmt::Return(_)
            | HirStmt::Global(_, _)
            | HirStmt::Persistent(_, _)
            | HirStmt::Import { .. }
            | HirStmt::AssignLValue(_, _, _, _) => {}
        }
    }
}

fn expr_is_begin_call(expr: &HirExpr) -> bool {
    match &expr.kind {
        HirExprKind::MethodCall(_, method, _) => method == "begin",
        HirExprKind::FuncCall(name, _) => name == "Dataset.begin",
        _ => false,
    }
}

fn literal_string(expr: &HirExpr) -> Option<String> {
    match &expr.kind {
        HirExprKind::String(s) => Some(normalize_literal_string(s)),
        _ => None,
    }
}

fn slice_rank(expr: &HirExpr) -> Option<usize> {
    match &expr.kind {
        HirExprKind::Cell(rows) => Some(rows.iter().map(|row| row.len()).sum()),
        HirExprKind::Colon => Some(1),
        _ => None,
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

fn infer_binding_from_expr(
    expr: &HirExpr,
    datasets: &HashMap<VarId, DatasetBinding>,
    provider: &dyn DatasetSchemaProvider,
) -> (Option<DatasetBinding>, Option<ArrayBinding>) {
    match &expr.kind {
        HirExprKind::FuncCall(name, args) if name == "data.open" => {
            if let Some(path_expr) = args.first() {
                if let Some(path) = literal_string(path_expr) {
                    return (infer_dataset_binding(provider, &path), None);
                }
            }
            (None, None)
        }
        HirExprKind::MethodCall(base, method, args) if method == "array" => {
            if let HirExprKind::Var(ds_var) = base.kind {
                if let Some(dataset) = datasets.get(&ds_var) {
                    if let Some(name_expr) = args.first() {
                        if let Some(name) = literal_string(name_expr) {
                            let rank = dataset.arrays.get(&name).copied();
                            return (
                                None,
                                Some(ArrayBinding {
                                    array_name: name,
                                    rank,
                                }),
                            );
                        }
                    }
                }
            }
            (None, None)
        }
        HirExprKind::FuncCall(name, args) if name == "Dataset.array" => {
            if let Some(base) = args.first() {
                if let HirExprKind::Var(ds_var) = base.kind {
                    if let Some(dataset) = datasets.get(&ds_var) {
                        if let Some(name_expr) = args.get(1) {
                            if let Some(name) = literal_string(name_expr) {
                                let rank = dataset.arrays.get(&name).copied();
                                return (
                                    None,
                                    Some(ArrayBinding {
                                        array_name: name,
                                        rank,
                                    }),
                                );
                            }
                        }
                    }
                }
            }
            (None, None)
        }
        _ => (None, None),
    }
}

fn analyze_data_expr(
    expr: &HirExpr,
    result: &LoweringResult,
    datasets: &HashMap<VarId, DatasetBinding>,
    arrays: &HashMap<VarId, ArrayBinding>,
    diags: &mut Vec<HirDiagnostic>,
) {
    match &expr.kind {
        HirExprKind::MethodCall(base, method, args) => {
            if method == "array" {
                if let HirExprKind::Var(ds_var) = base.kind {
                    if let Some(dataset) = datasets.get(&ds_var) {
                        if let Some(name_expr) = args.first() {
                            if let Some(name) = literal_string(name_expr) {
                                if !dataset.arrays.contains_key(&name) {
                                    diags.push(HirDiagnostic {
                                        message: format!(
                                            "array '{name}' is not present in inferred dataset schema"
                                        ),
                                        span: name_expr.span,
                                        code: "lint.data.unknown_array_name",
                                        severity: HirDiagnosticSeverity::Warning,
                                    });
                                }
                            }
                        }
                    }
                }
            }

            if method == "read" || method == "write" {
                if let HirExprKind::Var(array_var) = base.kind {
                    if let Some(array_binding) = arrays.get(&array_var) {
                        if let Some(rank) = array_binding.rank {
                            let slice_arg = args.first();
                            if let Some(slice_expr) = slice_arg {
                                if let Some(actual_slice_rank) = slice_rank(slice_expr) {
                                    if actual_slice_rank > rank {
                                        let array_var_name = result
                                            .var_names
                                            .get(&array_var)
                                            .cloned()
                                            .unwrap_or_else(|| format!("v{}", array_var.0));
                                        diags.push(HirDiagnostic {
                                            message: format!(
                                                "slice rank {actual_slice_rank} exceeds array rank {rank} for '{array_var_name}' ({})",
                                                array_binding.array_name
                                            ),
                                            span: slice_expr.span,
                                            code: "lint.data.invalid_slice_rank",
                                            severity: HirDiagnosticSeverity::Warning,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }

            analyze_data_expr(base, result, datasets, arrays, diags);
            for arg in args {
                analyze_data_expr(arg, result, datasets, arrays, diags);
            }
        }
        HirExprKind::FuncCall(name, args) => {
            if name == "Dataset.array" {
                if let Some(base) = args.first() {
                    if let HirExprKind::Var(ds_var) = base.kind {
                        if let Some(dataset) = datasets.get(&ds_var) {
                            if let Some(name_expr) = args.get(1) {
                                if let Some(array_name) = literal_string(name_expr) {
                                    if !dataset.arrays.contains_key(&array_name) {
                                        diags.push(HirDiagnostic {
                                            message: format!(
                                                "array '{array_name}' is not present in inferred dataset schema"
                                            ),
                                            span: name_expr.span,
                                            code: "lint.data.unknown_array_name",
                                            severity: HirDiagnosticSeverity::Warning,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if name == "DataArray.read" || name == "DataArray.write" {
                if let Some(base) = args.first() {
                    if let HirExprKind::Var(array_var) = base.kind {
                        if let Some(array_binding) = arrays.get(&array_var) {
                            if let Some(rank) = array_binding.rank {
                                let slice_arg = args.get(1);
                                if let Some(slice_expr) = slice_arg {
                                    if let Some(actual_slice_rank) = slice_rank(slice_expr) {
                                        if actual_slice_rank > rank {
                                            let array_var_name = result
                                                .var_names
                                                .get(&array_var)
                                                .cloned()
                                                .unwrap_or_else(|| format!("v{}", array_var.0));
                                            diags.push(HirDiagnostic {
                                                message: format!(
                                                    "slice rank {actual_slice_rank} exceeds array rank {rank} for '{array_var_name}' ({})",
                                                    array_binding.array_name
                                                ),
                                                span: slice_expr.span,
                                                code: "lint.data.invalid_slice_rank",
                                                severity: HirDiagnosticSeverity::Warning,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            for arg in args {
                analyze_data_expr(arg, result, datasets, arrays, diags);
            }
        }
        HirExprKind::Unary(_, inner) => analyze_data_expr(inner, result, datasets, arrays, diags),
        HirExprKind::Binary(lhs, _, rhs) => {
            analyze_data_expr(lhs, result, datasets, arrays, diags);
            analyze_data_expr(rhs, result, datasets, arrays, diags);
        }
        HirExprKind::Tensor(rows) | HirExprKind::Cell(rows) => {
            for row in rows {
                for value in row {
                    analyze_data_expr(value, result, datasets, arrays, diags);
                }
            }
        }
        HirExprKind::Index(base, args) | HirExprKind::IndexCell(base, args) => {
            analyze_data_expr(base, result, datasets, arrays, diags);
            for arg in args {
                analyze_data_expr(arg, result, datasets, arrays, diags);
            }
        }
        HirExprKind::Range(start, step, end) => {
            analyze_data_expr(start, result, datasets, arrays, diags);
            if let Some(step) = step {
                analyze_data_expr(step, result, datasets, arrays, diags);
            }
            analyze_data_expr(end, result, datasets, arrays, diags);
        }
        HirExprKind::Member(base, _) | HirExprKind::AnonFunc { body: base, .. } => {
            analyze_data_expr(base, result, datasets, arrays, diags)
        }
        HirExprKind::MemberDynamic(base, field) => {
            analyze_data_expr(base, result, datasets, arrays, diags);
            analyze_data_expr(field, result, datasets, arrays, diags);
        }
        HirExprKind::FuncHandle(_)
        | HirExprKind::MetaClass(_)
        | HirExprKind::Var(_)
        | HirExprKind::String(_)
        | HirExprKind::Number(_)
        | HirExprKind::Constant(_)
        | HirExprKind::Colon
        | HirExprKind::End => {}
    }
}

fn analyze_stmt_bindings(
    stmt: &HirStmt,
    result: &LoweringResult,
    provider: &dyn DatasetSchemaProvider,
    datasets: &mut HashMap<VarId, DatasetBinding>,
    arrays: &mut HashMap<VarId, ArrayBinding>,
    diags: &mut Vec<HirDiagnostic>,
) {
    match stmt {
        HirStmt::ExprStmt(expr, _, _) => analyze_data_expr(expr, result, datasets, arrays, diags),
        HirStmt::Assign(var, expr, _, _) => {
            analyze_data_expr(expr, result, datasets, arrays, diags);
            let (dataset_binding, array_binding) =
                infer_binding_from_expr(expr, datasets, provider);
            if let Some(binding) = dataset_binding {
                datasets.insert(*var, binding);
                arrays.remove(var);
                return;
            }
            if let Some(binding) = array_binding {
                arrays.insert(*var, binding);
                datasets.remove(var);
                return;
            }
            datasets.remove(var);
            arrays.remove(var);
        }
        HirStmt::MultiAssign(_, expr, _, _) => {
            analyze_data_expr(expr, result, datasets, arrays, diags)
        }
        HirStmt::AssignLValue(lv, expr, _, _) => {
            analyze_data_expr(expr, result, datasets, arrays, diags);
            match lv {
                HirLValue::Var(var) => {
                    datasets.remove(var);
                    arrays.remove(var);
                }
                HirLValue::Index(base, args) | HirLValue::IndexCell(base, args) => {
                    analyze_data_expr(base, result, datasets, arrays, diags);
                    for arg in args {
                        analyze_data_expr(arg, result, datasets, arrays, diags);
                    }
                }
                HirLValue::Member(base, _) => {
                    analyze_data_expr(base, result, datasets, arrays, diags)
                }
                HirLValue::MemberDynamic(base, field) => {
                    analyze_data_expr(base, result, datasets, arrays, diags);
                    analyze_data_expr(field, result, datasets, arrays, diags);
                }
            }
        }
        HirStmt::If {
            cond,
            then_body,
            elseif_blocks,
            else_body,
            ..
        } => {
            analyze_data_expr(cond, result, datasets, arrays, diags);
            for nested in then_body {
                analyze_stmt_bindings(nested, result, provider, datasets, arrays, diags);
            }
            for (cond, body) in elseif_blocks {
                analyze_data_expr(cond, result, datasets, arrays, diags);
                for nested in body {
                    analyze_stmt_bindings(nested, result, provider, datasets, arrays, diags);
                }
            }
            if let Some(body) = else_body {
                for nested in body {
                    analyze_stmt_bindings(nested, result, provider, datasets, arrays, diags);
                }
            }
        }
        HirStmt::While { cond, body, .. } => {
            analyze_data_expr(cond, result, datasets, arrays, diags);
            for nested in body {
                analyze_stmt_bindings(nested, result, provider, datasets, arrays, diags);
            }
        }
        HirStmt::For { expr, body, .. } => {
            analyze_data_expr(expr, result, datasets, arrays, diags);
            for nested in body {
                analyze_stmt_bindings(nested, result, provider, datasets, arrays, diags);
            }
        }
        HirStmt::Switch {
            expr,
            cases,
            otherwise,
            ..
        } => {
            analyze_data_expr(expr, result, datasets, arrays, diags);
            for (case_expr, body) in cases {
                analyze_data_expr(case_expr, result, datasets, arrays, diags);
                for nested in body {
                    analyze_stmt_bindings(nested, result, provider, datasets, arrays, diags);
                }
            }
            if let Some(body) = otherwise {
                for nested in body {
                    analyze_stmt_bindings(nested, result, provider, datasets, arrays, diags);
                }
            }
        }
        HirStmt::TryCatch {
            try_body,
            catch_body,
            ..
        } => {
            for nested in try_body {
                analyze_stmt_bindings(nested, result, provider, datasets, arrays, diags);
            }
            for nested in catch_body {
                analyze_stmt_bindings(nested, result, provider, datasets, arrays, diags);
            }
        }
        HirStmt::Function { body, .. } => {
            for nested in body {
                analyze_stmt_bindings(nested, result, provider, datasets, arrays, diags);
            }
        }
        HirStmt::ClassDef { members, .. } => {
            for member in members {
                if let HirClassMember::Methods { body, .. } = member {
                    for nested in body {
                        analyze_stmt_bindings(nested, result, provider, datasets, arrays, diags);
                    }
                }
            }
        }
        HirStmt::Break(_)
        | HirStmt::Continue(_)
        | HirStmt::Return(_)
        | HirStmt::Global(_, _)
        | HirStmt::Persistent(_, _)
        | HirStmt::Import { .. } => {}
    }
}

fn walk_expr_general(
    expr: &HirExpr,
    diags: &mut Vec<HirDiagnostic>,
    non_tx_write_count: &mut usize,
    tx_vars: &HashSet<VarId>,
) {
    match &expr.kind {
        HirExprKind::FuncCall(name, args) => {
            if name == "data.open" {
                let first = args.first();
                let is_typed_open = args.len() > 1;
                let first_is_literal =
                    matches!(first.map(|a| &a.kind), Some(HirExprKind::String(_)));
                if !first_is_literal && !is_typed_open {
                    diags.push(HirDiagnostic {
                        message: "data.open with dynamic path should include explicit schema for type safety"
                            .to_string(),
                        span: expr.span,
                        code: "lint.data.no_untyped_open",
                        severity: HirDiagnosticSeverity::Warning,
                    });
                }
            }
            for arg in args {
                walk_expr_general(arg, diags, non_tx_write_count, tx_vars);
            }
        }
        HirExprKind::MethodCall(base, method, args) => {
            if method == "write" {
                let in_tx =
                    matches!(base.kind, HirExprKind::Var(var_id) if tx_vars.contains(&var_id));
                if !in_tx {
                    *non_tx_write_count += 1;
                    if *non_tx_write_count > 1 {
                        diags.push(HirDiagnostic {
                            message:
                                "multiple data writes detected outside explicit transaction; consider ds.begin() + tx.commit()"
                                    .to_string(),
                            span: expr.span,
                            code: "lint.data.no_multiwrite_outside_tx",
                            severity: HirDiagnosticSeverity::Warning,
                        });
                    }
                }
            }
            walk_expr_general(base, diags, non_tx_write_count, tx_vars);
            for arg in args {
                walk_expr_general(arg, diags, non_tx_write_count, tx_vars);
            }
        }
        HirExprKind::Unary(_, inner) => {
            walk_expr_general(inner, diags, non_tx_write_count, tx_vars)
        }
        HirExprKind::Binary(lhs, _, rhs) => {
            walk_expr_general(lhs, diags, non_tx_write_count, tx_vars);
            walk_expr_general(rhs, diags, non_tx_write_count, tx_vars);
        }
        HirExprKind::Tensor(rows) | HirExprKind::Cell(rows) => {
            for row in rows {
                for value in row {
                    walk_expr_general(value, diags, non_tx_write_count, tx_vars);
                }
            }
        }
        HirExprKind::Index(base, args) | HirExprKind::IndexCell(base, args) => {
            walk_expr_general(base, diags, non_tx_write_count, tx_vars);
            for arg in args {
                walk_expr_general(arg, diags, non_tx_write_count, tx_vars);
            }
        }
        HirExprKind::Range(start, step, end) => {
            walk_expr_general(start, diags, non_tx_write_count, tx_vars);
            if let Some(step) = step {
                walk_expr_general(step, diags, non_tx_write_count, tx_vars);
            }
            walk_expr_general(end, diags, non_tx_write_count, tx_vars);
        }
        HirExprKind::Member(base, _) | HirExprKind::AnonFunc { body: base, .. } => {
            walk_expr_general(base, diags, non_tx_write_count, tx_vars)
        }
        HirExprKind::MemberDynamic(base, field) => {
            walk_expr_general(base, diags, non_tx_write_count, tx_vars);
            walk_expr_general(field, diags, non_tx_write_count, tx_vars);
        }
        HirExprKind::FuncHandle(_)
        | HirExprKind::MetaClass(_)
        | HirExprKind::Var(_)
        | HirExprKind::String(_)
        | HirExprKind::Number(_)
        | HirExprKind::Constant(_)
        | HirExprKind::Colon
        | HirExprKind::End => {}
    }
}

fn walk_stmt_general(
    stmt: &HirStmt,
    diags: &mut Vec<HirDiagnostic>,
    non_tx_write_count: &mut usize,
    tx_vars: &HashSet<VarId>,
) {
    match stmt {
        HirStmt::ExprStmt(expr, _, _) => {
            if matches!(&expr.kind, HirExprKind::MethodCall(_, method, _) if method == "commit") {
                diags.push(HirDiagnostic {
                    message: "consider checking transaction commit outcomes in shared workflows"
                        .to_string(),
                    span: expr.span,
                    code: "lint.data.ignore_commit_result",
                    severity: HirDiagnosticSeverity::Information,
                });
            }
            walk_expr_general(expr, diags, non_tx_write_count, tx_vars)
        }
        HirStmt::Assign(_, expr, _, _) => {
            walk_expr_general(expr, diags, non_tx_write_count, tx_vars)
        }
        HirStmt::MultiAssign(_, expr, _, _) => {
            walk_expr_general(expr, diags, non_tx_write_count, tx_vars)
        }
        HirStmt::AssignLValue(lv, expr, _, _) => {
            match lv {
                HirLValue::Var(_) => {}
                HirLValue::Index(base, args) | HirLValue::IndexCell(base, args) => {
                    walk_expr_general(base, diags, non_tx_write_count, tx_vars);
                    for arg in args {
                        walk_expr_general(arg, diags, non_tx_write_count, tx_vars);
                    }
                }
                HirLValue::Member(base, _) => {
                    walk_expr_general(base, diags, non_tx_write_count, tx_vars)
                }
                HirLValue::MemberDynamic(base, field) => {
                    walk_expr_general(base, diags, non_tx_write_count, tx_vars);
                    walk_expr_general(field, diags, non_tx_write_count, tx_vars);
                }
            }
            walk_expr_general(expr, diags, non_tx_write_count, tx_vars);
        }
        HirStmt::If {
            cond,
            then_body,
            elseif_blocks,
            else_body,
            ..
        } => {
            walk_expr_general(cond, diags, non_tx_write_count, tx_vars);
            for stmt in then_body {
                walk_stmt_general(stmt, diags, non_tx_write_count, tx_vars);
            }
            for (cond, body) in elseif_blocks {
                walk_expr_general(cond, diags, non_tx_write_count, tx_vars);
                for stmt in body {
                    walk_stmt_general(stmt, diags, non_tx_write_count, tx_vars);
                }
            }
            if let Some(body) = else_body {
                for stmt in body {
                    walk_stmt_general(stmt, diags, non_tx_write_count, tx_vars);
                }
            }
        }
        HirStmt::While { cond, body, .. } => {
            walk_expr_general(cond, diags, non_tx_write_count, tx_vars);
            for stmt in body {
                walk_stmt_general(stmt, diags, non_tx_write_count, tx_vars);
            }
        }
        HirStmt::For { expr, body, .. } => {
            walk_expr_general(expr, diags, non_tx_write_count, tx_vars);
            for stmt in body {
                walk_stmt_general(stmt, diags, non_tx_write_count, tx_vars);
            }
        }
        HirStmt::Switch {
            expr,
            cases,
            otherwise,
            ..
        } => {
            walk_expr_general(expr, diags, non_tx_write_count, tx_vars);
            for (case_expr, body) in cases {
                walk_expr_general(case_expr, diags, non_tx_write_count, tx_vars);
                for stmt in body {
                    walk_stmt_general(stmt, diags, non_tx_write_count, tx_vars);
                }
            }
            if let Some(body) = otherwise {
                for stmt in body {
                    walk_stmt_general(stmt, diags, non_tx_write_count, tx_vars);
                }
            }
        }
        HirStmt::TryCatch {
            try_body,
            catch_body,
            ..
        } => {
            for stmt in try_body {
                walk_stmt_general(stmt, diags, non_tx_write_count, tx_vars);
            }
            for stmt in catch_body {
                walk_stmt_general(stmt, diags, non_tx_write_count, tx_vars);
            }
        }
        HirStmt::Function { body, .. } => {
            for stmt in body {
                walk_stmt_general(stmt, diags, non_tx_write_count, tx_vars);
            }
        }
        HirStmt::ClassDef { members, .. } => {
            for member in members {
                if let HirClassMember::Methods { body, .. } = member {
                    for stmt in body {
                        walk_stmt_general(stmt, diags, non_tx_write_count, tx_vars);
                    }
                }
            }
        }
        HirStmt::Break(_)
        | HirStmt::Continue(_)
        | HirStmt::Return(_)
        | HirStmt::Global(_, _)
        | HirStmt::Persistent(_, _)
        | HirStmt::Import { .. } => {}
    }
}
