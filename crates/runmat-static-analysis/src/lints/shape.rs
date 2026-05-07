use runmat_hir::{
    BindingId, HirBlock, HirCallableRef, HirDiagnostic, HirDiagnosticSeverity, HirExpr,
    HirExprKind, HirPlace, HirStmt, HirStmtKind, IndexComponent, OperatorKind, QualifiedName, Span,
};
use std::collections::HashMap;

pub fn lint_shapes(result: &runmat_hir::LoweringResult) -> Vec<HirDiagnostic> {
    let _analysis_store = runmat_mir::lowering::lower_assembly(&result.assembly)
        .ok()
        .map(|mir| runmat_mir::analysis::analyze_assembly(&mir));
    let mut ctx = ShapeLintContext::default();
    for function in &result.assembly.functions {
        ctx.walk_block(&function.body);
    }
    ctx.diagnostics
}

#[derive(Debug, Clone, PartialEq)]
struct Shape(Vec<Option<usize>>);

#[derive(Default)]
struct ShapeLintContext {
    env: HashMap<BindingId, Shape>,
    diagnostics: Vec<HirDiagnostic>,
}

impl ShapeLintContext {
    fn walk_block(&mut self, block: &HirBlock) {
        for stmt in &block.statements {
            self.walk_stmt(stmt);
        }
    }

    fn walk_stmt(&mut self, stmt: &HirStmt) {
        match &stmt.kind {
            HirStmtKind::Assign(place, expr, _) => {
                let shape = self.infer_expr(expr);
                if let (HirPlace::Binding(binding), Some(shape)) = (place, shape) {
                    self.env.insert(*binding, shape);
                }
            }
            HirStmtKind::MultiAssign(_, expr, _) | HirStmtKind::ExprStmt(expr, _) => {
                self.infer_expr(expr);
            }
            HirStmtKind::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
            } => {
                self.infer_expr(cond);
                self.walk_block(then_body);
                for (cond, block) in elseif_blocks {
                    self.infer_expr(cond);
                    self.walk_block(block);
                }
                if let Some(block) = else_body {
                    self.walk_block(block);
                }
            }
            HirStmtKind::While { cond, body } => {
                self.infer_expr(cond);
                self.walk_block(body);
            }
            HirStmtKind::For { range, body, .. } => {
                self.infer_expr(range);
                self.walk_block(body);
            }
            HirStmtKind::Switch {
                expr,
                cases,
                otherwise,
            } => {
                self.infer_expr(expr);
                for (case, block) in cases {
                    self.infer_expr(case);
                    self.walk_block(block);
                }
                if let Some(block) = otherwise {
                    self.walk_block(block);
                }
            }
            HirStmtKind::TryCatch {
                try_body,
                catch_body,
                ..
            } => {
                self.walk_block(try_body);
                self.walk_block(catch_body);
            }
            HirStmtKind::Global(_)
            | HirStmtKind::Persistent(_)
            | HirStmtKind::Break
            | HirStmtKind::Continue
            | HirStmtKind::Return
            | HirStmtKind::Import(_) => {}
        }
    }

    fn infer_expr(&mut self, expr: &HirExpr) -> Option<Shape> {
        match &expr.kind {
            HirExprKind::Number(_) => Some(Shape(vec![Some(1), Some(1)])),
            HirExprKind::Binding(binding) => self.env.get(binding).cloned(),
            HirExprKind::Unary(_, inner) => self.infer_expr(inner),
            HirExprKind::Binary(left, op, right) => self.infer_binary(expr.span, left, op, right),
            HirExprKind::Tensor(rows) => self.infer_tensor(expr.span, rows),
            HirExprKind::Cell(rows) => {
                Some(Shape(vec![Some(1), Some(rows.iter().map(Vec::len).sum())]))
            }
            HirExprKind::Range(start, step, end) => {
                self.infer_expr(start);
                if let Some(step) = step {
                    self.infer_expr(step);
                }
                self.infer_expr(end);
                Some(Shape(vec![Some(1), None]))
            }
            HirExprKind::Index(base, indexing) => {
                let base_shape = self.infer_expr(base);
                for component in &indexing.components {
                    if let IndexComponent::Expr(idx) | IndexComponent::Logical(idx) = component {
                        let idx_shape = self.infer_expr(idx);
                        if indexing.components.len() == 1 {
                            self.check_logical_index(
                                expr.span,
                                base_shape.as_ref(),
                                idx_shape.as_ref(),
                            );
                        }
                    }
                }
                None
            }
            HirExprKind::Member(base, _) => {
                self.infer_expr(base);
                None
            }
            HirExprKind::MemberDynamic(base, member) => {
                self.infer_expr(base);
                self.infer_expr(member);
                None
            }
            HirExprKind::Call(call) => self.infer_call(expr.span, &call.callee, &call.args),
            HirExprKind::Await(inner) | HirExprKind::Spawn(inner) => self.infer_expr(inner),
            HirExprKind::String(_)
            | HirExprKind::Constant(_)
            | HirExprKind::Colon
            | HirExprKind::End
            | HirExprKind::CommandCall(_)
            | HirExprKind::FunctionHandle(_)
            | HirExprKind::AnonymousFunction(_)
            | HirExprKind::MetaClass(_) => None,
        }
    }

    fn infer_binary(
        &mut self,
        span: Span,
        left: &HirExpr,
        op: &OperatorKind,
        right: &HirExpr,
    ) -> Option<Shape> {
        let lhs = self.infer_expr(left);
        let rhs = self.infer_expr(right);
        match op {
            OperatorKind::MatrixMultiply => {
                if let (Some(lhs), Some(rhs)) = (&lhs, &rhs) {
                    if matrix_dims(lhs)
                        .zip(matrix_dims(rhs))
                        .is_some_and(|((_, lc), (rr, _))| lc.is_some() && rr.is_some() && lc != rr)
                    {
                        self.warn(
                            "lint.shape.matmul",
                            "matrix multiply dimensions do not agree",
                            span,
                        );
                    }
                }
                match (
                    lhs.as_ref().and_then(matrix_dims),
                    rhs.as_ref().and_then(matrix_dims),
                ) {
                    (Some((rows, _)), Some((_, cols))) => Some(Shape(vec![rows, cols])),
                    _ => None,
                }
            }
            OperatorKind::Add
            | OperatorKind::Subtract
            | OperatorKind::ElementwiseMultiply
            | OperatorKind::ElementwiseDivide
            | OperatorKind::ElementwiseLeftDivide
            | OperatorKind::ElementwisePower
            | OperatorKind::Greater
            | OperatorKind::GreaterEqual
            | OperatorKind::Less
            | OperatorKind::LessEqual
            | OperatorKind::Equal
            | OperatorKind::NotEqual => {
                if let (Some(lhs), Some(rhs)) = (&lhs, &rhs) {
                    if !broadcast_compatible(lhs, rhs) {
                        self.warn(
                            "lint.shape.broadcast",
                            "array dimensions are not broadcast compatible",
                            span,
                        );
                    }
                }
                lhs.or(rhs)
            }
            _ => lhs.or(rhs),
        }
    }

    fn infer_tensor(&mut self, span: Span, rows: &[Vec<HirExpr>]) -> Option<Shape> {
        let mut row_dims = Vec::new();
        for row in rows {
            let mut total_cols = 0usize;
            let mut expected_rows = None;
            for element in row {
                let shape = self.infer_expr(element);
                if let Some((rows, cols)) = shape.as_ref().and_then(matrix_dims) {
                    if let (Some(expected), Some(rows)) = (expected_rows, rows) {
                        if expected != rows {
                            self.warn(
                                "lint.shape.horzcat",
                                "horizontal concatenation row dimensions do not agree",
                                span,
                            );
                        }
                    }
                    expected_rows = expected_rows.or(rows);
                    total_cols += cols.unwrap_or(1);
                } else {
                    total_cols += 1;
                }
            }
            row_dims.push((expected_rows.unwrap_or(1), total_cols));
        }
        if let Some((_, first_cols)) = row_dims.first().copied() {
            for (_, cols) in &row_dims {
                if *cols != first_cols {
                    self.warn(
                        "lint.shape.vertcat",
                        "vertical concatenation column dimensions do not agree",
                        span,
                    );
                }
            }
            Some(Shape(vec![
                Some(row_dims.iter().map(|(rows, _)| rows).sum()),
                Some(first_cols),
            ]))
        } else {
            Some(Shape(vec![Some(0), Some(0)]))
        }
    }

    fn infer_call(
        &mut self,
        span: Span,
        callee: &HirCallableRef,
        args: &[HirExpr],
    ) -> Option<Shape> {
        let name = call_name(callee);
        let name = name.as_deref();
        match name {
            Some("ones") | Some("zeros") | Some("rand") => Some(Shape(
                args.iter()
                    .filter_map(const_int)
                    .map(Some)
                    .collect::<Vec<_>>(),
            )),
            Some("dot") => {
                let lhs = args.get(0).and_then(|arg| self.infer_expr(arg));
                let rhs = args.get(1).and_then(|arg| self.infer_expr(arg));
                if let (Some(lhs), Some(rhs)) = (&lhs, &rhs) {
                    if vector_len(lhs)
                        .zip(vector_len(rhs))
                        .is_some_and(|(l, r)| l != r)
                    {
                        self.warn(
                            "lint.shape.dot",
                            "dot product vector lengths do not agree",
                            span,
                        );
                    }
                }
                Some(Shape(vec![Some(1), Some(1)]))
            }
            Some("reshape") => {
                let input = args.get(0).and_then(|arg| self.infer_expr(arg));
                let dims = parse_dims(&args[1..]);
                if dims.iter().filter(|dim| matches!(dim, Dim::Infer)).count() > 1
                    || incompatible_element_count(input.as_ref(), &dims)
                {
                    self.warn(
                        "lint.shape.reshape",
                        "reshape dimensions are not compatible",
                        span,
                    );
                }
                Some(Shape(dims.iter().map(|dim| dim.as_shape_dim()).collect()))
            }
            Some("repmat") => {
                let base = args.get(0).and_then(|arg| self.infer_expr(arg));
                for arg in &args[1..] {
                    if !matches!(parse_dim(arg), Dim::Known(_)) {
                        self.warn(
                            "lint.shape.repmat",
                            "repmat dimensions must be non-negative integers",
                            arg.span,
                        );
                    }
                }
                base
            }
            Some("permute") => {
                let base = args.get(0).and_then(|arg| self.infer_expr(arg));
                let order = args.get(1).and_then(vector_literal_ints);
                if let Some(order) = &order {
                    let mut sorted = order.clone();
                    sorted.sort_unstable();
                    if sorted.windows(2).any(|pair| pair[0] == pair[1])
                        || base
                            .as_ref()
                            .is_some_and(|shape| order.len() != shape.0.len())
                    {
                        self.warn(
                            "lint.shape.permute",
                            "permute order is invalid for input rank",
                            span,
                        );
                    }
                }
                base
            }
            Some("sum") | Some("mean") | Some("max") | Some("min") => {
                let base = args.get(0).and_then(|arg| self.infer_expr(arg));
                if let (Some(base), Some(dim)) = (base.as_ref(), args.get(1).and_then(const_int)) {
                    if dim == 0 || dim > base.0.len() {
                        self.warn(
                            "lint.shape.reduction",
                            "reduction dimension is out of range",
                            span,
                        );
                    }
                }
                base
            }
            _ => {
                for arg in args {
                    self.infer_expr(arg);
                }
                None
            }
        }
    }

    fn check_logical_index(&mut self, span: Span, base: Option<&Shape>, idx: Option<&Shape>) {
        if let (Some(base), Some(idx)) = (base, idx) {
            if element_count(base)
                .zip(element_count(idx))
                .is_some_and(|(base, idx)| base != idx)
            {
                self.warn(
                    "lint.shape.logical_index",
                    "logical index shape does not match indexed value",
                    span,
                );
            }
        }
    }

    fn warn(&mut self, code: &'static str, message: &'static str, span: Span) {
        self.diagnostics.push(
            HirDiagnostic::new(code, HirDiagnosticSeverity::Warning, message, span)
                .with_category("shape"),
        );
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Dim {
    Known(usize),
    Infer,
    Unknown,
}

impl Dim {
    fn as_shape_dim(self) -> Option<usize> {
        match self {
            Dim::Known(value) => Some(value),
            Dim::Infer | Dim::Unknown => None,
        }
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

fn const_num(expr: &HirExpr) -> Option<f64> {
    match &expr.kind {
        HirExprKind::Number(value) => value.parse().ok(),
        HirExprKind::Unary(OperatorKind::UnaryMinus, inner) => const_num(inner).map(|value| -value),
        _ => None,
    }
}

fn const_int(expr: &HirExpr) -> Option<usize> {
    let value = const_num(expr)?;
    if value.is_finite() && value >= 0.0 && (value.fract().abs() <= 1e-9) {
        Some(value as usize)
    } else {
        None
    }
}

fn parse_dim(expr: &HirExpr) -> Dim {
    match const_num(expr) {
        Some(value) if value == -1.0 => Dim::Infer,
        Some(value) if value.is_finite() && value >= 0.0 && (value.fract().abs() <= 1e-9) => {
            Dim::Known(value as usize)
        }
        _ => Dim::Unknown,
    }
}

fn parse_dims(args: &[HirExpr]) -> Vec<Dim> {
    if args.len() == 1 {
        if let Some(values) = vector_literal_ints(&args[0]) {
            return values.into_iter().map(Dim::Known).collect();
        }
    }
    args.iter().map(parse_dim).collect()
}

fn vector_literal_ints(expr: &HirExpr) -> Option<Vec<usize>> {
    let HirExprKind::Tensor(rows) = &expr.kind else {
        return None;
    };
    let mut values = Vec::new();
    for row in rows {
        for element in row {
            values.push(const_int(element)?);
        }
    }
    Some(values)
}

fn matrix_dims(shape: &Shape) -> Option<(Option<usize>, Option<usize>)> {
    Some((*shape.0.first()?, *shape.0.get(1)?))
}

fn element_count(shape: &Shape) -> Option<usize> {
    shape
        .0
        .iter()
        .try_fold(1usize, |acc, dim| dim.map(|dim| acc * dim))
}

fn vector_len(shape: &Shape) -> Option<usize> {
    let count = element_count(shape)?;
    if shape.0.len() == 1
        || (shape.0.len() == 2 && (shape.0[0] == Some(1) || shape.0[1] == Some(1)))
    {
        Some(count)
    } else {
        None
    }
}

fn broadcast_compatible(left: &Shape, right: &Shape) -> bool {
    let len = left.0.len().max(right.0.len());
    (0..len).all(|idx| {
        let l = left.0.iter().rev().nth(idx).copied().flatten().unwrap_or(1);
        let r = right
            .0
            .iter()
            .rev()
            .nth(idx)
            .copied()
            .flatten()
            .unwrap_or(1);
        l == r || l == 1 || r == 1
    })
}

fn incompatible_element_count(input: Option<&Shape>, dims: &[Dim]) -> bool {
    let Some(input_count) = input.and_then(element_count) else {
        return false;
    };
    if dims.iter().any(|dim| matches!(dim, Dim::Unknown)) {
        return true;
    }
    let known_product = dims.iter().fold(1usize, |acc, dim| match dim {
        Dim::Known(value) => acc * value,
        Dim::Infer | Dim::Unknown => acc,
    });
    if dims.iter().any(|dim| matches!(dim, Dim::Infer)) {
        known_product == 0 || input_count % known_product != 0
    } else {
        known_product != input_count
    }
}
