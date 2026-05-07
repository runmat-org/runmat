use crate::schema::{
    normalize_literal_string, DatasetSchema, DatasetSchemaProvider, FsDatasetSchemaProvider,
};
use runmat_hir::{
    BindingId, CallSyntax, HirBlock, HirCallableRef, HirDiagnostic, HirDiagnosticSeverity, HirExpr,
    HirExprKind, HirPlace, HirStmt, HirStmtKind, LoweringResult, QualifiedName,
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
    let _analysis_store = runmat_mir::lowering::lower_assembly(&result.assembly)
        .ok()
        .map(|mir| runmat_mir::analysis::analyze_assembly(&mir));
    let mut ctx = DataLintContext {
        provider,
        datasets: HashMap::new(),
        arrays: HashMap::new(),
        tx_bindings: HashSet::new(),
        non_tx_write_count: 0,
        diagnostics: Vec::new(),
    };

    for function in &result.assembly.functions {
        ctx.collect_tx_bindings(&function.body);
    }
    for function in &result.assembly.functions {
        ctx.walk_block(&function.body);
    }

    ctx.diagnostics
}

struct DataLintContext<'a> {
    provider: &'a dyn DatasetSchemaProvider,
    datasets: HashMap<BindingId, DatasetBinding>,
    arrays: HashMap<BindingId, ArrayBinding>,
    tx_bindings: HashSet<BindingId>,
    non_tx_write_count: usize,
    diagnostics: Vec<HirDiagnostic>,
}

impl DataLintContext<'_> {
    fn collect_tx_bindings(&mut self, block: &HirBlock) {
        for stmt in &block.statements {
            match &stmt.kind {
                HirStmtKind::Assign(place, expr, _) if is_begin_call(expr) => {
                    if let HirPlace::Binding(binding) = place {
                        self.tx_bindings.insert(*binding);
                    }
                }
                HirStmtKind::MultiAssign(targets, expr, _) if is_begin_call(expr) => {
                    for target in &targets.targets {
                        if let runmat_hir::OutputTarget::Place(HirPlace::Binding(binding)) = target
                        {
                            self.tx_bindings.insert(*binding);
                        }
                    }
                }
                _ => self.walk_nested_blocks(stmt, |this, block| this.collect_tx_bindings(block)),
            }
        }
    }

    fn walk_block(&mut self, block: &HirBlock) {
        for stmt in &block.statements {
            self.walk_stmt(stmt);
        }
    }

    fn walk_stmt(&mut self, stmt: &HirStmt) {
        match &stmt.kind {
            HirStmtKind::Assign(place, expr, _) => {
                self.walk_expr(expr, false);
                if let HirPlace::Binding(binding) = place {
                    let (dataset, array) = self.infer_binding(expr);
                    if let Some(dataset) = dataset {
                        self.datasets.insert(*binding, dataset);
                    }
                    if let Some(array) = array {
                        self.arrays.insert(*binding, array);
                    }
                }
            }
            HirStmtKind::MultiAssign(_, expr, _) | HirStmtKind::ExprStmt(expr, _) => {
                self.walk_expr(expr, matches!(stmt.kind, HirStmtKind::ExprStmt(_, _)));
            }
            _ => self.walk_nested_blocks(stmt, |this, block| this.walk_block(block)),
        }
    }

    fn walk_nested_blocks(&mut self, stmt: &HirStmt, mut f: impl FnMut(&mut Self, &HirBlock)) {
        match &stmt.kind {
            HirStmtKind::If {
                then_body,
                elseif_blocks,
                else_body,
                ..
            } => {
                f(self, then_body);
                for (_, block) in elseif_blocks {
                    f(self, block);
                }
                if let Some(block) = else_body {
                    f(self, block);
                }
            }
            HirStmtKind::While { body, .. } | HirStmtKind::For { body, .. } => f(self, body),
            HirStmtKind::Switch {
                cases, otherwise, ..
            } => {
                for (_, block) in cases {
                    f(self, block);
                }
                if let Some(block) = otherwise {
                    f(self, block);
                }
            }
            HirStmtKind::TryCatch {
                try_body,
                catch_body,
                ..
            } => {
                f(self, try_body);
                f(self, catch_body);
            }
            _ => {}
        }
    }

    fn walk_expr(&mut self, expr: &HirExpr, statement_position: bool) {
        if let HirExprKind::Call(call) = &expr.kind {
            let name = call_name(&call.callee);
            if is_data_open_call(call.syntax.clone(), name.as_deref(), &call.args)
                && !data_open_path_arg(name.as_deref(), &call.args)
                    .is_some_and(|arg| literal_string(arg).is_some())
            {
                self.diagnostics.push(diagnostic(
                    "lint.data.no_untyped_open",
                    "data.open should use a literal dataset path for static checking",
                    expr.span,
                ));
            }
            if call.syntax == CallSyntax::Method && name.as_deref() == Some("write") {
                let in_tx = call
                    .args
                    .first()
                    .and_then(binding_expr)
                    .is_some_and(|binding| self.tx_bindings.contains(&binding));
                if !in_tx {
                    self.non_tx_write_count += 1;
                    if self.non_tx_write_count > 1 {
                        self.diagnostics.push(diagnostic(
                            "lint.data.no_multiwrite_outside_tx",
                            "multiple dataset writes should be grouped in a transaction",
                            expr.span,
                        ));
                    }
                }
            }
            if statement_position
                && call.syntax == CallSyntax::Method
                && name.as_deref() == Some("commit")
            {
                self.diagnostics.push(diagnostic(
                    "lint.data.ignore_commit_result",
                    "transaction commit result should be checked",
                    expr.span,
                ));
            }
            self.check_array_read(expr, call.args.as_slice(), name.as_deref());
            for arg in &call.args {
                self.walk_expr(arg, false);
            }
            return;
        }

        walk_children(expr, |child| self.walk_expr(child, false));
    }

    fn check_array_read(&mut self, expr: &HirExpr, args: &[HirExpr], name: Option<&str>) {
        if name != Some("read") {
            return;
        }
        let Some(array_binding) = args.first().and_then(binding_expr) else {
            return;
        };
        let Some(array) = self.arrays.get(&array_binding) else {
            return;
        };
        let Some(slice) = args.get(1) else {
            return;
        };
        if let (Some(expected), Some(actual)) = (array.rank, slice_rank(slice)) {
            if expected != actual {
                self.diagnostics.push(diagnostic(
                    "lint.data.invalid_slice_rank",
                    format!("slice rank {actual} does not match array rank {expected}"),
                    expr.span,
                ));
            }
        }
    }

    fn infer_binding(&mut self, expr: &HirExpr) -> (Option<DatasetBinding>, Option<ArrayBinding>) {
        let HirExprKind::Call(call) = &expr.kind else {
            return (None, None);
        };
        let name = call_name(&call.callee);
        if is_data_open_call(call.syntax.clone(), name.as_deref(), &call.args) {
            if let Some(path) =
                data_open_path_arg(name.as_deref(), &call.args).and_then(literal_string)
            {
                return (infer_dataset_binding(self.provider, &path), None);
            }
            return (None, None);
        }
        if call.syntax == CallSyntax::Method && name.as_deref() == Some("array") {
            let dataset = call.args.first().and_then(binding_expr);
            let array_name = call.args.get(1).and_then(literal_string);
            if let (Some(dataset), Some(array_name)) = (dataset, array_name) {
                if let Some(dataset) = self.datasets.get(&dataset) {
                    let rank = dataset.arrays.get(&array_name).copied();
                    if rank.is_none() {
                        self.diagnostics.push(diagnostic(
                            "lint.data.unknown_array_name",
                            format!("dataset schema does not contain array '{array_name}'"),
                            expr.span,
                        ));
                    }
                    return (None, Some(ArrayBinding { rank }));
                }
            }
        }
        (None, None)
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

fn is_begin_call(expr: &HirExpr) -> bool {
    matches!(
        &expr.kind,
        HirExprKind::Call(call)
            if call_name(&call.callee).as_deref() == Some("Dataset.begin")
                || (call.syntax == CallSyntax::Method
                    && call_name(&call.callee).as_deref() == Some("begin"))
    )
}

fn is_data_open_call(syntax: CallSyntax, name: Option<&str>, args: &[HirExpr]) -> bool {
    name == Some("data.open")
        || (syntax == CallSyntax::Method
            && name == Some("open")
            && args.first().is_some_and(|arg| expr_has_name(arg, "data")))
}

fn data_open_path_arg<'a>(name: Option<&str>, args: &'a [HirExpr]) -> Option<&'a HirExpr> {
    if name == Some("data.open") {
        args.first()
    } else {
        args.get(1)
    }
}

fn expr_has_name(expr: &HirExpr, name: &str) -> bool {
    match &expr.kind {
        HirExprKind::Constant(symbol) => symbol.0 == name,
        HirExprKind::Call(call) => call_name(&call.callee).as_deref() == Some(name),
        _ => false,
    }
}

fn call_name(callee: &HirCallableRef) -> Option<String> {
    match callee {
        HirCallableRef::Builtin(id) => Some(id.0.clone()),
        HirCallableRef::Unresolved(name) => Some(qualified_name(name)),
        _ => None,
    }
}

fn qualified_name(name: &QualifiedName) -> String {
    name.0
        .iter()
        .map(|part| part.0.as_str())
        .collect::<Vec<_>>()
        .join(".")
}

fn literal_string(expr: &HirExpr) -> Option<String> {
    match &expr.kind {
        HirExprKind::String(value) => Some(normalize_literal_string(&value.0)),
        _ => None,
    }
}

fn binding_expr(expr: &HirExpr) -> Option<BindingId> {
    match expr.kind {
        HirExprKind::Binding(binding) => Some(binding),
        _ => None,
    }
}

fn slice_rank(expr: &HirExpr) -> Option<usize> {
    match &expr.kind {
        HirExprKind::Cell(rows) => Some(rows.iter().map(Vec::len).sum()),
        HirExprKind::Colon => Some(1),
        _ => None,
    }
}

fn walk_children(expr: &HirExpr, mut f: impl FnMut(&HirExpr)) {
    match &expr.kind {
        HirExprKind::Unary(_, inner) | HirExprKind::Await(inner) | HirExprKind::Spawn(inner) => {
            f(inner)
        }
        HirExprKind::Binary(left, _, right) => {
            f(left);
            f(right);
        }
        HirExprKind::Tensor(rows) | HirExprKind::Cell(rows) => {
            for row in rows {
                for element in row {
                    f(element);
                }
            }
        }
        HirExprKind::Range(start, step, end) => {
            f(start);
            if let Some(step) = step {
                f(step);
            }
            f(end);
        }
        HirExprKind::Index(base, indexing) => {
            f(base);
            for component in &indexing.components {
                match component {
                    runmat_hir::IndexComponent::Expr(expr)
                    | runmat_hir::IndexComponent::Logical(expr) => f(expr),
                    runmat_hir::IndexComponent::Colon | runmat_hir::IndexComponent::End { .. } => {}
                }
            }
        }
        HirExprKind::Member(base, _) => f(base),
        HirExprKind::MemberDynamic(base, member) => {
            f(base);
            f(member);
        }
        HirExprKind::Call(call) => {
            for arg in &call.args {
                f(arg);
            }
        }
        HirExprKind::Number(_)
        | HirExprKind::String(_)
        | HirExprKind::Constant(_)
        | HirExprKind::Binding(_)
        | HirExprKind::Colon
        | HirExprKind::End
        | HirExprKind::CommandCall(_)
        | HirExprKind::FunctionHandle(_)
        | HirExprKind::AnonymousFunction(_)
        | HirExprKind::MetaClass(_) => {}
    }
}

fn diagnostic(
    code: &'static str,
    message: impl Into<String>,
    span: runmat_hir::Span,
) -> HirDiagnostic {
    HirDiagnostic::new(code, HirDiagnosticSeverity::Warning, message, span)
        .with_category("data-api")
}
