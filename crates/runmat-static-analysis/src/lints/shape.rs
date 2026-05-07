use runmat_builtins::{BuiltinSemanticKind, ConcatKind, ShapeTransformKind};
use runmat_hir::{BindingId, HirDiagnostic, HirDiagnosticSeverity, OperatorKind, Span};
use runmat_mir::analysis::{AnalysisStore, MirLocalKey};
use std::collections::HashMap;

pub fn lint_shapes(result: &runmat_hir::LoweringResult) -> Vec<HirDiagnostic> {
    let mir = match runmat_mir::lowering::lower_assembly(&result.assembly) {
        Ok(mir) => mir,
        Err(err) => return vec![mir_lowering_diagnostic(err)],
    };
    let store = runmat_mir::analysis::analyze_assembly(&mir);
    let mut ctx = ShapeLintContext::default();
    ctx.seed_from_analysis(&mir, &store);
    ctx.walk_mir_assembly(&mir);
    ctx.diagnostics
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

#[derive(Debug, Clone, PartialEq)]
struct Shape(Vec<Option<usize>>);

#[derive(Default)]
struct ShapeLintContext {
    env: HashMap<BindingId, Shape>,
    local_env: HashMap<MirLocalKey, Shape>,
    number_env: HashMap<MirLocalKey, f64>,
    int_vector_env: HashMap<MirLocalKey, Vec<usize>>,
    diagnostics: Vec<HirDiagnostic>,
}

#[derive(Default)]
struct MirShapeValue {
    shape: Option<Shape>,
    number: Option<f64>,
    int_vector: Option<Vec<usize>>,
}

impl ShapeLintContext {
    fn seed_from_analysis(&mut self, mir: &runmat_mir::MirAssembly, store: &AnalysisStore) {
        for body in mir.bodies.values() {
            for local in &body.locals {
                let Some(binding) = local.binding else {
                    continue;
                };
                let Some(fact) = store.mir_locals.get(&MirLocalKey {
                    function: body.function,
                    local: local.id,
                }) else {
                    continue;
                };
                if let Some(shape) = shape_from_fact(&fact.shape) {
                    self.env.insert(binding, shape);
                }
            }
        }
    }

    fn walk_mir_assembly(&mut self, mir: &runmat_mir::MirAssembly) {
        for body in mir.bodies.values() {
            for block in &body.blocks {
                for stmt in &block.statements {
                    match &stmt.kind {
                        runmat_mir::MirStmtKind::Assign { place, value } => {
                            let value = self.infer_mir_rvalue(body, value, stmt.span);
                            if let runmat_mir::MirPlace::Local(local) = place {
                                self.record_mir_value(body, *local, value);
                            }
                        }
                        runmat_mir::MirStmtKind::MultiAssign { value, .. }
                        | runmat_mir::MirStmtKind::Expr(value) => {
                            self.infer_mir_rvalue(body, value, stmt.span);
                        }
                        runmat_mir::MirStmtKind::PlaceMutation(_)
                        | runmat_mir::MirStmtKind::WorkspaceEffect { .. }
                        | runmat_mir::MirStmtKind::EnvironmentEffect(_) => {}
                    }
                }
            }
        }
    }

    fn record_mir_value(
        &mut self,
        body: &runmat_mir::MirBody,
        local: runmat_mir::MirLocalId,
        value: MirShapeValue,
    ) {
        let key = MirLocalKey {
            function: body.function,
            local,
        };
        if let Some(shape) = value.shape {
            self.local_env.insert(key, shape);
        }
        if let Some(number) = value.number {
            self.number_env.insert(key, number);
        }
        if let Some(vector) = value.int_vector {
            self.int_vector_env.insert(key, vector);
        }
    }

    fn infer_mir_rvalue(
        &mut self,
        body: &runmat_mir::MirBody,
        value: &runmat_mir::MirRvalue,
        span: Span,
    ) -> MirShapeValue {
        match value {
            runmat_mir::MirRvalue::Use(operand) => self.infer_mir_operand(body, operand),
            runmat_mir::MirRvalue::Unary(op, operand) => {
                let inner = self.infer_mir_operand(body, operand);
                let number = match (op, inner.number) {
                    (OperatorKind::UnaryMinus, Some(value)) => Some(-value),
                    _ => None,
                };
                MirShapeValue {
                    shape: inner.shape,
                    number,
                    int_vector: None,
                }
            }
            runmat_mir::MirRvalue::Binary(left, op, right) => {
                let lhs = self.infer_mir_operand(body, left);
                let rhs = self.infer_mir_operand(body, right);
                MirShapeValue {
                    shape: self.infer_mir_binary(span, lhs.shape.as_ref(), op, rhs.shape.as_ref()),
                    number: None,
                    int_vector: None,
                }
            }
            runmat_mir::MirRvalue::Range { start, step, end } => {
                self.infer_mir_operand(body, start);
                if let Some(step) = step {
                    self.infer_mir_operand(body, step);
                }
                self.infer_mir_operand(body, end);
                MirShapeValue {
                    shape: Some(Shape(vec![Some(1), None])),
                    number: None,
                    int_vector: None,
                }
            }
            runmat_mir::MirRvalue::Call(call) => self.infer_mir_call(body, span, call),
            runmat_mir::MirRvalue::Aggregate {
                kind,
                rows,
                elements,
                ..
            } => self.infer_mir_aggregate(body, span, kind, *rows, elements),
            runmat_mir::MirRvalue::Index { base, indexing } => {
                let base_shape = self.infer_mir_operand(body, base).shape;
                for component in &indexing.components {
                    match component {
                        runmat_mir::MirIndexComponent::Expr(operand)
                        | runmat_mir::MirIndexComponent::Logical(operand) => {
                            let idx_shape = self.infer_mir_operand(body, operand).shape;
                            if indexing.components.len() == 1 {
                                self.check_logical_index(
                                    span,
                                    base_shape.as_ref(),
                                    idx_shape.as_ref(),
                                );
                            }
                        }
                        runmat_mir::MirIndexComponent::Colon
                        | runmat_mir::MirIndexComponent::End { .. } => {}
                    }
                }
                MirShapeValue::default()
            }
            runmat_mir::MirRvalue::Member { base, .. } => self.infer_mir_operand(body, base),
            runmat_mir::MirRvalue::DynamicMember { base, member } => {
                self.infer_mir_operand(body, member);
                self.infer_mir_operand(body, base)
            }
            runmat_mir::MirRvalue::Future { args, .. } => {
                for arg in args {
                    self.infer_mir_operand(body, arg.operand());
                }
                MirShapeValue::default()
            }
            runmat_mir::MirRvalue::Spawn(operand) => self.infer_mir_operand(body, operand),
            runmat_mir::MirRvalue::MetaClass(_)
            | runmat_mir::MirRvalue::Colon
            | runmat_mir::MirRvalue::End => MirShapeValue::default(),
        }
    }

    fn infer_mir_operand(
        &self,
        body: &runmat_mir::MirBody,
        operand: &runmat_mir::MirOperand,
    ) -> MirShapeValue {
        match operand {
            runmat_mir::MirOperand::Constant(runmat_mir::MirConstant::Number(value)) => {
                MirShapeValue {
                    shape: Some(Shape(vec![Some(1), Some(1)])),
                    number: value.parse().ok(),
                    int_vector: None,
                }
            }
            runmat_mir::MirOperand::Local(local) => {
                let key = MirLocalKey {
                    function: body.function,
                    local: *local,
                };
                MirShapeValue {
                    shape: self.local_env.get(&key).cloned(),
                    number: self.number_env.get(&key).copied(),
                    int_vector: self.int_vector_env.get(&key).cloned(),
                }
            }
            runmat_mir::MirOperand::Constant(_)
            | runmat_mir::MirOperand::Temp(_)
            | runmat_mir::MirOperand::FunctionHandle(_) => MirShapeValue::default(),
        }
    }

    fn infer_mir_binary(
        &mut self,
        span: Span,
        lhs: Option<&Shape>,
        op: &OperatorKind,
        rhs: Option<&Shape>,
    ) -> Option<Shape> {
        match op {
            OperatorKind::MatrixMultiply => {
                if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
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
                match (lhs.and_then(matrix_dims), rhs.and_then(matrix_dims)) {
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
                if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
                    if !broadcast_compatible(lhs, rhs) {
                        self.warn(
                            "lint.shape.broadcast",
                            "array dimensions are not broadcast compatible",
                            span,
                        );
                    }
                }
                lhs.cloned().or_else(|| rhs.cloned())
            }
            _ => lhs.cloned().or_else(|| rhs.cloned()),
        }
    }

    fn infer_mir_aggregate(
        &mut self,
        body: &runmat_mir::MirBody,
        span: Span,
        kind: &runmat_mir::MirAggregateKind,
        rows: usize,
        elements: &[runmat_mir::MirOperand],
    ) -> MirShapeValue {
        let values: Vec<_> = elements
            .iter()
            .map(|element| self.infer_mir_operand(body, element))
            .collect();
        let int_vector = values
            .iter()
            .map(|value| value.number.and_then(number_to_int))
            .collect::<Option<Vec<_>>>();
        let shape = match kind {
            runmat_mir::MirAggregateKind::Tensor => {
                let row_count = rows.max(1);
                let cols_per_row = if row_count == 0 {
                    0
                } else {
                    elements.len() / row_count
                };
                let mut row_dims = Vec::new();
                for row_idx in 0..row_count {
                    let start = row_idx * cols_per_row;
                    let end = start + cols_per_row;
                    let mut total_cols = 0usize;
                    let mut expected_rows = None;
                    for value in &values[start..end] {
                        if let Some((rows, cols)) = value.shape.as_ref().and_then(matrix_dims) {
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
            runmat_mir::MirAggregateKind::Cell => Some(Shape(vec![Some(1), Some(elements.len())])),
            runmat_mir::MirAggregateKind::Struct | runmat_mir::MirAggregateKind::ObjectArray(_) => {
                None
            }
        };
        MirShapeValue {
            shape,
            number: None,
            int_vector,
        }
    }

    fn infer_mir_call(
        &mut self,
        body: &runmat_mir::MirBody,
        span: Span,
        call: &runmat_mir::MirCall,
    ) -> MirShapeValue {
        let arg_values: Vec<_> = call
            .args
            .iter()
            .map(|arg| self.infer_mir_operand(body, arg.operand()))
            .collect();
        let shape = match call.semantic_kind {
            BuiltinSemanticKind::ArrayConstructor => sized_constructor_shape(&arg_values),
            BuiltinSemanticKind::ParameterizedArrayConstructor => {
                sized_constructor_shape(arg_values.get(1..).unwrap_or(&[]))
            }
            BuiltinSemanticKind::PermutationConstructor => Some(Shape(vec![
                Some(1),
                arg_values
                    .first()
                    .and_then(|value| value.number.and_then(number_to_int)),
            ])),
            BuiltinSemanticKind::RangeConstructor => Some(Shape(vec![Some(1), None])),
            BuiltinSemanticKind::EmptyConstructor => Some(Shape(vec![Some(0), Some(0)])),
            BuiltinSemanticKind::ShapeTransform(ShapeTransformKind::Dot) => {
                let lhs = arg_values.first().and_then(|value| value.shape.as_ref());
                let rhs = arg_values.get(1).and_then(|value| value.shape.as_ref());
                if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
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
            BuiltinSemanticKind::ShapeTransform(ShapeTransformKind::Reshape) => {
                let input = arg_values.first().and_then(|value| value.shape.as_ref());
                let dims = mir_parse_dims(&arg_values[1..]);
                if dims.iter().filter(|dim| matches!(dim, Dim::Infer)).count() > 1
                    || incompatible_element_count(input, &dims)
                {
                    self.warn(
                        "lint.shape.reshape",
                        "reshape dimensions are not compatible",
                        span,
                    );
                }
                Some(Shape(dims.iter().map(|dim| dim.as_shape_dim()).collect()))
            }
            BuiltinSemanticKind::ShapeTransform(ShapeTransformKind::Repmat) => {
                for arg in &arg_values[1..] {
                    if !matches!(mir_parse_dim(arg), Dim::Known(_)) {
                        self.warn(
                            "lint.shape.repmat",
                            "repmat dimensions must be non-negative integers",
                            span,
                        );
                    }
                }
                arg_values.first().and_then(|value| value.shape.clone())
            }
            BuiltinSemanticKind::ShapeTransform(ShapeTransformKind::Permute) => {
                let base = arg_values.first().and_then(|value| value.shape.clone());
                let order = arg_values.get(1).and_then(|value| value.int_vector.clone());
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
            BuiltinSemanticKind::ShapeTransform(ShapeTransformKind::Transpose) => {
                let base = arg_values.first().and_then(|value| value.shape.clone());
                base.map(|shape| {
                    if shape.0.len() >= 2 {
                        Shape(vec![shape.0[1], shape.0[0]])
                    } else {
                        shape
                    }
                })
            }
            BuiltinSemanticKind::ShapeTransform(ShapeTransformKind::Concatenate(kind)) => {
                self.infer_mir_concat(span, kind, &arg_values)
            }
            BuiltinSemanticKind::ShapeTransform(ShapeTransformKind::General) => {
                arg_values.first().and_then(|value| value.shape.clone())
            }
            BuiltinSemanticKind::Reduction => {
                let base = arg_values.first().and_then(|value| value.shape.clone());
                if let (Some(base_shape), Some(dim)) = (
                    base.as_ref(),
                    arg_values
                        .get(1)
                        .and_then(|value| value.number.and_then(number_to_int)),
                ) {
                    if dim == 0 || dim > base_shape.0.len() {
                        self.warn(
                            "lint.shape.reduction",
                            "reduction dimension is out of range",
                            span,
                        );
                    }
                }
                base
            }
            _ => None,
        };
        MirShapeValue {
            shape,
            number: None,
            int_vector: None,
        }
    }

    fn infer_mir_concat(
        &mut self,
        span: Span,
        kind: ConcatKind,
        arg_values: &[MirShapeValue],
    ) -> Option<Shape> {
        let (dim, values) = match kind {
            ConcatKind::Dimension => {
                let dim = arg_values
                    .first()
                    .and_then(|value| value.number.and_then(number_to_int))?;
                (dim, &arg_values[1..])
            }
            ConcatKind::Horizontal => (2, arg_values),
            ConcatKind::Vertical => (1, arg_values),
        };
        let shapes: Vec<_> = values
            .iter()
            .filter_map(|value| value.shape.as_ref())
            .collect();
        if shapes.is_empty() || dim == 0 {
            return None;
        }
        let rank = shapes
            .iter()
            .map(|shape| shape.0.len())
            .max()
            .unwrap_or(dim);
        let axis = dim - 1;
        if axis >= rank {
            return None;
        }
        let mut out = vec![Some(1); rank];
        for idx in 0..rank {
            if idx == axis {
                out[idx] = shapes
                    .iter()
                    .map(|shape| shape.0.get(idx).copied().flatten())
                    .try_fold(0usize, |sum, dim| dim.map(|dim| sum + dim));
                continue;
            }
            let mut expected = None;
            for shape in &shapes {
                let dim = shape.0.get(idx).copied().flatten().or(Some(1));
                if let (Some(expected), Some(dim)) = (expected, dim) {
                    if expected != dim {
                        self.warn(
                            "lint.shape.concat",
                            "concatenation dimensions do not agree",
                            span,
                        );
                    }
                }
                expected = expected.or(dim);
            }
            out[idx] = expected;
        }
        Some(Shape(out))
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

fn number_to_int(value: f64) -> Option<usize> {
    if value.is_finite() && value >= 0.0 && (value.fract().abs() <= 1e-9) {
        Some(value as usize)
    } else {
        None
    }
}

fn mir_parse_dim(value: &MirShapeValue) -> Dim {
    match value.number {
        Some(value) if value == -1.0 => Dim::Infer,
        Some(value) => number_to_int(value).map(Dim::Known).unwrap_or(Dim::Unknown),
        None => Dim::Unknown,
    }
}

fn mir_parse_dims(args: &[MirShapeValue]) -> Vec<Dim> {
    if args.len() == 1 {
        if let Some(values) = &args[0].int_vector {
            return values.iter().copied().map(Dim::Known).collect();
        }
    }
    args.iter().map(mir_parse_dim).collect()
}

fn sized_constructor_shape(args: &[MirShapeValue]) -> Option<Shape> {
    let dims: Vec<_> = args
        .iter()
        .filter_map(|value| value.number.and_then(number_to_int))
        .map(Some)
        .collect();
    match dims.as_slice() {
        [] => None,
        [dim] => Some(Shape(vec![*dim, *dim])),
        _ => Some(Shape(dims)),
    }
}

fn shape_from_fact(shape: &runmat_hir::ShapeFact) -> Option<Shape> {
    match shape {
        runmat_hir::ShapeFact::Scalar => Some(Shape(vec![Some(1), Some(1)])),
        runmat_hir::ShapeFact::Shaped { dims } => Some(Shape(
            dims.iter()
                .map(|dim| match dim {
                    runmat_hir::DimFact::Known(value) => Some(*value),
                    runmat_hir::DimFact::Symbolic(_) | runmat_hir::DimFact::Unknown => None,
                })
                .collect(),
        )),
        runmat_hir::ShapeFact::Ranked { .. }
        | runmat_hir::ShapeFact::Unknown
        | runmat_hir::ShapeFact::Unreachable => None,
    }
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
