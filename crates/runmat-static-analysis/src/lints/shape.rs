use runmat_builtins::Type;
use runmat_hir::{
    eval_const_num, infer_expr_type_with_env, merge_span, HirClassMember, HirDiagnostic,
    HirDiagnosticSeverity, HirExpr, HirExprKind, HirStmt, LoweringResult, Span, VarId,
};
use runmat_parser as parser;

pub fn lint_shapes(result: &LoweringResult) -> Vec<HirDiagnostic> {
    fn vector_literal_length(expr: &HirExpr) -> Option<usize> {
        let shape = tensor_literal_shape(expr)?;
        match (
            shape.first().copied().flatten(),
            shape.get(1).copied().flatten(),
        ) {
            (Some(r), Some(c)) => {
                if r == 1 {
                    Some(c)
                } else if c == 1 {
                    Some(r)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn concat_dims(ty: &Type) -> Option<(Option<usize>, Option<usize>)> {
        match ty {
            Type::Num | Type::Int | Type::Bool => Some((Some(1), Some(1))),
            Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
                Some(runmat_builtins::shape_rules::matrix_dims(shape))
            }
            _ => None,
        }
    }

    fn format_dim(dim: Option<usize>) -> String {
        dim.map(|v| v.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    }

    fn format_shape(shape: &[Option<usize>]) -> String {
        if shape.len() == 2 {
            return format!("{} x {}", format_dim(shape[0]), format_dim(shape[1]));
        }
        let dims: Vec<String> = shape.iter().map(|d| format_dim(*d)).collect();
        format!("[{}]", dims.join(", "))
    }

    fn matrix_dims_from_type(ty: &Type) -> Option<(Option<usize>, Option<usize>)> {
        match ty {
            Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
                Some(runmat_builtins::shape_rules::matrix_dims(shape))
            }
            _ => None,
        }
    }

    fn element_count(shape: &[Option<usize>]) -> Option<usize> {
        runmat_builtins::shape_rules::element_count_if_known(shape)
    }

    fn vector_length(shape: &[Option<usize>]) -> Option<usize> {
        let count = element_count(shape)?;
        let is_vector = shape.len() == 1
            || (shape.len() == 2
                && (shape[0] == Some(1) || shape[1] == Some(1))
                && shape.iter().all(|d| d.is_some()));
        if is_vector {
            Some(count)
        } else {
            None
        }
    }

    fn tensor_literal_shape(expr: &HirExpr) -> Option<Vec<Option<usize>>> {
        let HirExprKind::Tensor(rows) = &expr.kind else {
            return None;
        };
        if rows.is_empty() {
            return Some(vec![Some(0), Some(0)]);
        }
        let cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        Some(vec![Some(rows.len()), Some(cols)])
    }

    enum DimSpec {
        Known(usize),
        Unknown,
        Negative,
        NonInteger,
    }

    fn parse_dim(expr: &HirExpr) -> DimSpec {
        if let Some(value) = eval_const_num(expr) {
            if value.is_finite() {
                let rounded = value.round();
                if (value - rounded).abs() <= 1e-9 {
                    if rounded < 0.0 {
                        return DimSpec::Negative;
                    }
                    return DimSpec::Known(rounded as usize);
                }
                return DimSpec::NonInteger;
            }
        }
        DimSpec::Unknown
    }

    fn type_shape_for_broadcast(ty: &Type) -> Option<Vec<Option<usize>>> {
        match ty {
            Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
                Some(shape.clone())
            }
            Type::Num | Type::Int | Type::Bool => Some(vec![Some(1), Some(1)]),
            _ => None,
        }
    }

    fn check_binary(
        op: &parser::BinOp,
        lhs: &HirExpr,
        rhs: &HirExpr,
        env: &std::collections::HashMap<VarId, Type>,
        returns: &std::collections::HashMap<String, Vec<Type>>,
        diags: &mut Vec<HirDiagnostic>,
    ) {
        let lhs_ty = infer_expr_type_with_env(lhs, env, returns);
        let rhs_ty = infer_expr_type_with_env(rhs, env, returns);
        match op {
            parser::BinOp::Mul => {
                if let Some(false) =
                    runmat_builtins::shape_rules::matmul_compatible(&lhs_ty, &rhs_ty)
                {
                    let detail = match (
                        matrix_dims_from_type(&lhs_ty),
                        matrix_dims_from_type(&rhs_ty),
                    ) {
                        (Some((lrows, lcols)), Some((rrows, rcols))) => format!(
                            "left is {} x {}, right is {} x {} (inner dimensions {} and {})",
                            format_dim(lrows),
                            format_dim(lcols),
                            format_dim(rrows),
                            format_dim(rcols),
                            format_dim(lcols),
                            format_dim(rrows)
                        ),
                        _ => "unknown shapes".to_string(),
                    };
                    diags.push(HirDiagnostic {
                        message: format!(
                            "Matrix multiply dimension mismatch: {detail} (inner dimensions must match)"
                        ),
                        span: merge_span(lhs.span, rhs.span),
                        code: "lint.shape.matmul",
                        severity: HirDiagnosticSeverity::Warning,
                    });
                }
            }
            parser::BinOp::LeftDiv => {
                if let Some(false) =
                    runmat_builtins::shape_rules::left_divide_compatible(&lhs_ty, &rhs_ty)
                {
                    let detail = match (
                        matrix_dims_from_type(&lhs_ty),
                        matrix_dims_from_type(&rhs_ty),
                    ) {
                        (Some((lrows, _)), Some((rrows, _))) => format!(
                            "left row dimension {}, right row dimension {}",
                            format_dim(lrows),
                            format_dim(rrows)
                        ),
                        _ => "unknown shapes".to_string(),
                    };
                    diags.push(HirDiagnostic {
                        message: format!(
                            "Left divide dimension mismatch: {detail} (row dimensions must match)"
                        ),
                        span: merge_span(lhs.span, rhs.span),
                        code: "lint.shape.ldivide",
                        severity: HirDiagnosticSeverity::Warning,
                    });
                }
            }
            parser::BinOp::RightDiv => {
                if let Some(false) =
                    runmat_builtins::shape_rules::right_divide_compatible(&lhs_ty, &rhs_ty)
                {
                    let detail = match (
                        matrix_dims_from_type(&lhs_ty),
                        matrix_dims_from_type(&rhs_ty),
                    ) {
                        (Some((_, lcols)), Some((_, rcols))) => format!(
                            "left column dimension {}, right column dimension {}",
                            format_dim(lcols),
                            format_dim(rcols)
                        ),
                        _ => "unknown shapes".to_string(),
                    };
                    diags.push(HirDiagnostic {
                        message: format!(
                            "Right divide dimension mismatch: {detail} (column dimensions must match)"
                        ),
                        span: merge_span(lhs.span, rhs.span),
                        code: "lint.shape.rdivide",
                        severity: HirDiagnosticSeverity::Warning,
                    });
                }
            }
            parser::BinOp::Add
            | parser::BinOp::Sub
            | parser::BinOp::ElemMul
            | parser::BinOp::ElemDiv
            | parser::BinOp::ElemPow
            | parser::BinOp::ElemLeftDiv
            | parser::BinOp::Equal
            | parser::BinOp::NotEqual
            | parser::BinOp::Less
            | parser::BinOp::LessEqual
            | parser::BinOp::Greater
            | parser::BinOp::GreaterEqual => {
                let lhs_shape = type_shape_for_broadcast(&lhs_ty);
                let rhs_shape = type_shape_for_broadcast(&rhs_ty);
                if let (Some(a), Some(b)) = (lhs_shape, rhs_shape) {
                    if let Some(false) = runmat_builtins::shape_rules::broadcast_compatible(&a, &b)
                    {
                        let detail = format!(
                            "left is {}, right is {}",
                            format_shape(&a),
                            format_shape(&b)
                        );
                        diags.push(HirDiagnostic {
                            message: format!(
                                "Elementwise/broadcast dimension mismatch: {detail} (broadcasting failed)"
                            ),
                            span: merge_span(lhs.span, rhs.span),
                            code: "lint.shape.broadcast",
                            severity: HirDiagnosticSeverity::Warning,
                        });
                    }
                }
            }
            _ => {}
        }
    }

    fn walk_expr(
        expr: &HirExpr,
        env: &std::collections::HashMap<VarId, Type>,
        returns: &std::collections::HashMap<String, Vec<Type>>,
        diags: &mut Vec<HirDiagnostic>,
    ) {
        match &expr.kind {
            HirExprKind::Unary(_, inner) => walk_expr(inner, env, returns, diags),
            HirExprKind::Binary(lhs, op, rhs) => {
                check_binary(op, lhs, rhs, env, returns, diags);
                walk_expr(lhs, env, returns, diags);
                walk_expr(rhs, env, returns, diags);
            }
            HirExprKind::Tensor(rows) => {
                let mut col_constraint: Option<usize> = None;
                for row in rows {
                    let mut row_dim: Option<usize> = None;
                    let mut row_cols: Option<usize> = Some(0);
                    let mut first_span: Option<Span> = None;
                    for e in row {
                        if first_span.is_none() {
                            first_span = Some(e.span);
                        }
                        let ty = infer_expr_type_with_env(e, env, returns);
                        if let Some((rows_dim, cols_dim)) = concat_dims(&ty) {
                            if let (Some(prev), Some(curr)) = (row_dim, rows_dim) {
                                if prev != curr {
                                    diags.push(HirDiagnostic {
                                        message: format!(
                                            "Horizontal concatenation dimension mismatch: left row dimension {prev}, right row dimension {curr} (row dimensions must match)"
                                        ),
                                        span: merge_span(first_span.unwrap_or(e.span), e.span),
                                        code: "lint.shape.horzcat",
                                        severity: HirDiagnosticSeverity::Warning,
                                    });
                                }
                            }
                            if row_dim.is_none() {
                                row_dim = rows_dim;
                            }
                            match (row_cols, cols_dim) {
                                (Some(total), Some(value)) => row_cols = Some(total + value),
                                _ => row_cols = None,
                            }
                        } else {
                            row_dim = None;
                            row_cols = None;
                        }
                    }

                    if let (Some(prev_cols), Some(curr_cols)) = (col_constraint, row_cols) {
                        if prev_cols != curr_cols {
                            diags.push(HirDiagnostic {
                                message: format!(
                                    "Vertical concatenation dimension mismatch: upper column dimension {prev_cols}, lower column dimension {curr_cols} (column dimensions must match)"
                                ),
                                span: expr.span,
                                code: "lint.shape.vertcat",
                                severity: HirDiagnosticSeverity::Warning,
                            });
                        }
                    }
                    if col_constraint.is_none() {
                        col_constraint = row_cols;
                    }
                }

                for row in rows {
                    for e in row {
                        walk_expr(e, env, returns, diags);
                    }
                }
            }
            HirExprKind::Cell(rows) => {
                for row in rows {
                    for e in row {
                        walk_expr(e, env, returns, diags);
                    }
                }
            }
            HirExprKind::Index(base, idxs) | HirExprKind::IndexCell(base, idxs) => {
                walk_expr(base, env, returns, diags);
                for idx in idxs {
                    walk_expr(idx, env, returns, diags);
                }
                if matches!(expr.kind, HirExprKind::Index(_, _)) && idxs.len() == 1 {
                    let base_ty = infer_expr_type_with_env(base, env, returns);
                    let idx_ty = infer_expr_type_with_env(&idxs[0], env, returns);
                    let base_shape = match base_ty {
                        Type::Tensor { shape: Some(shape) }
                        | Type::Logical { shape: Some(shape) } => Some(shape),
                        _ => None,
                    };
                    let mask_shape = match idx_ty {
                        Type::Logical { shape: Some(shape) }
                        | Type::Tensor { shape: Some(shape) } => Some(shape),
                        _ => None,
                    };
                    if let (Some(base_shape), Some(mask_shape)) = (base_shape, mask_shape) {
                        if let (Some(base_count), Some(mask_count)) =
                            (element_count(&base_shape), element_count(&mask_shape))
                        {
                            if base_count != mask_count {
                                diags.push(HirDiagnostic {
                                    message: format!(
                                        "Logical index size mismatch: mask has {mask_count}, array has {base_count} (must match)"
                                    ),
                                    span: merge_span(base.span, idxs[0].span),
                                    code: "lint.shape.logical_index",
                                    severity: HirDiagnosticSeverity::Warning,
                                });
                            }
                        }
                    }
                }
            }
            HirExprKind::Range(start, step, end) => {
                walk_expr(start, env, returns, diags);
                if let Some(step) = step.as_ref() {
                    walk_expr(step, env, returns, diags);
                }
                walk_expr(end, env, returns, diags);
            }
            HirExprKind::FuncCall(name, args) => {
                if name.eq_ignore_ascii_case("dot") && args.len() >= 2 {
                    let lhs_ty = infer_expr_type_with_env(&args[0], env, returns);
                    let rhs_ty = infer_expr_type_with_env(&args[1], env, returns);
                    let lhs_len = match lhs_ty {
                        Type::Tensor { shape: Some(shape) }
                        | Type::Logical { shape: Some(shape) } => vector_length(&shape),
                        _ => None,
                    };
                    let rhs_len = match rhs_ty {
                        Type::Tensor { shape: Some(shape) }
                        | Type::Logical { shape: Some(shape) } => vector_length(&shape),
                        _ => None,
                    };
                    if let (Some(a), Some(b)) = (lhs_len, rhs_len) {
                        if a != b {
                            diags.push(HirDiagnostic {
                                message: format!(
                                    "Dot product length mismatch: left length {a}, right length {b} (lengths must match)"
                                ),
                                span: merge_span(args[0].span, args[1].span),
                                code: "lint.shape.dot",
                                severity: HirDiagnosticSeverity::Warning,
                            });
                        }
                    }
                }

                if name.eq_ignore_ascii_case("reshape") && args.len() >= 2 {
                    let input_ty = infer_expr_type_with_env(&args[0], env, returns);
                    let input_shape = match input_ty {
                        Type::Tensor { shape: Some(shape) }
                        | Type::Logical { shape: Some(shape) } => Some(shape),
                        _ => None,
                    };
                    let mut dims: Vec<Option<usize>> = Vec::new();
                    let mut negative_count = 0usize;
                    let mut non_integer = false;
                    for arg in args.iter().skip(1) {
                        match parse_dim(arg) {
                            DimSpec::Known(value) => dims.push(Some(value)),
                            DimSpec::Negative => {
                                negative_count += 1;
                                dims.push(None);
                            }
                            DimSpec::NonInteger => {
                                non_integer = true;
                                dims.push(None);
                            }
                            DimSpec::Unknown => dims.push(None),
                        }
                    }
                    if negative_count > 1 {
                        diags.push(HirDiagnostic {
                            message:
                                "Reshape dimension mismatch: more than one negative dimension (only one allowed)"
                                    .to_string(),
                            span: merge_span(args[0].span, args[1].span),
                            code: "lint.shape.reshape",
                            severity: HirDiagnosticSeverity::Warning,
                        });
                    } else if negative_count == 1 && non_integer {
                        diags.push(HirDiagnostic {
                            message:
                                "Reshape dimension mismatch: negative dimensions require integer sizes"
                                    .to_string(),
                            span: merge_span(args[0].span, args[1].span),
                            code: "lint.shape.reshape",
                            severity: HirDiagnosticSeverity::Warning,
                        });
                    }
                    if non_integer {
                        diags.push(HirDiagnostic {
                            message: "Reshape dimension mismatch: non-integer dimensions"
                                .to_string(),
                            span: merge_span(args[0].span, args[1].span),
                            code: "lint.shape.reshape",
                            severity: HirDiagnosticSeverity::Warning,
                        });
                    }
                    if let Some(shape) =
                        runmat_builtins::shape_rules::constructor_shape_from_dims(&dims)
                    {
                        if let Some(input_shape) = input_shape {
                            if let (Some(in_count), Some(out_count)) =
                                (element_count(&input_shape), element_count(&shape))
                            {
                                if in_count != out_count {
                                    diags.push(HirDiagnostic {
                                        message: format!(
                                            "Reshape element count mismatch: input has {in_count}, output has {out_count} (must match)"
                                        ),
                                        span: merge_span(args[0].span, args[1].span),
                                        code: "lint.shape.reshape",
                                        severity: HirDiagnosticSeverity::Warning,
                                    });
                                }
                            }
                        }
                    }
                }

                if (name.eq_ignore_ascii_case("permute") || name.eq_ignore_ascii_case("ipermute"))
                    && args.len() >= 2
                {
                    let input_ty = infer_expr_type_with_env(&args[0], env, returns);
                    let input_rank = match input_ty {
                        Type::Tensor { shape: Some(shape) }
                        | Type::Logical { shape: Some(shape) } => Some(shape.len()),
                        _ => None,
                    };
                    let order_rank = vector_literal_length(&args[1]);
                    if let (Some(in_rank), Some(ord_rank)) = (input_rank, order_rank) {
                        if in_rank != ord_rank {
                            diags.push(HirDiagnostic {
                                message: format!(
                                    "Permute rank mismatch: input rank {in_rank}, order length {ord_rank} (must match)"
                                ),
                                span: merge_span(args[0].span, args[1].span),
                                code: "lint.shape.permute",
                                severity: HirDiagnosticSeverity::Warning,
                            });
                        }
                    }
                    if let HirExprKind::Tensor(rows) = &args[1].kind {
                        let mut seen: std::collections::BTreeSet<usize> =
                            std::collections::BTreeSet::new();
                        let mut duplicate = false;
                        let mut max_index = 0usize;
                        for row in rows {
                            for entry in row {
                                if let Some(value) = eval_const_num(entry) {
                                    let rounded = value.round();
                                    if (value - rounded).abs() <= 1e-9 && rounded >= 1.0 {
                                        let idx = rounded as usize;
                                        max_index = max_index.max(idx);
                                        if !seen.insert(idx) {
                                            duplicate = true;
                                        }
                                    }
                                }
                            }
                        }
                        if duplicate {
                            diags.push(HirDiagnostic {
                                message:
                                    "Permute order mismatch: duplicate dimensions in order vector"
                                        .to_string(),
                                span: args[1].span,
                                code: "lint.shape.permute",
                                severity: HirDiagnosticSeverity::Warning,
                            });
                        }
                        if let Some(in_rank) = input_rank {
                            if max_index > in_rank {
                                diags.push(HirDiagnostic {
                                    message: "Permute order mismatch: order references a dimension larger than the input rank"
                                        .to_string(),
                                    span: args[1].span,
                                    code: "lint.shape.permute",
                                    severity: HirDiagnosticSeverity::Warning,
                                });
                            }
                        }
                    }
                }

                if name.eq_ignore_ascii_case("repmat") && args.len() >= 2 {
                    let mut non_integer = false;
                    let mut negative = false;
                    for arg in args.iter().skip(1) {
                        match parse_dim(arg) {
                            DimSpec::Known(_) => {}
                            DimSpec::NonInteger => non_integer = true,
                            DimSpec::Negative => negative = true,
                            _ => {}
                        }
                    }
                    if non_integer || negative {
                        let reason = if non_integer {
                            "non-integer"
                        } else {
                            "negative"
                        };
                        diags.push(HirDiagnostic {
                            message: format!(
                                "Repmat dimension mismatch: {reason} replication factors"
                            ),
                            span: merge_span(args[0].span, args[1].span),
                            code: "lint.shape.repmat",
                            severity: HirDiagnosticSeverity::Warning,
                        });
                    }
                }

                if (name.eq_ignore_ascii_case("sum")
                    || name.eq_ignore_ascii_case("mean")
                    || name.eq_ignore_ascii_case("prod")
                    || name.eq_ignore_ascii_case("min")
                    || name.eq_ignore_ascii_case("max"))
                    && args.len() >= 2
                {
                    let input_ty = infer_expr_type_with_env(&args[0], env, returns);
                    let input_rank = match input_ty {
                        Type::Tensor { shape: Some(shape) }
                        | Type::Logical { shape: Some(shape) } => Some(shape.len()),
                        _ => None,
                    };
                    if let Some(rank) = input_rank {
                        if let DimSpec::Known(dim) = parse_dim(&args[1]) {
                            if dim == 0 || dim > rank {
                                diags.push(HirDiagnostic {
                                    message: format!(
                                        "Reduction dimension mismatch: dimension {dim} is out of range for rank {rank}"
                                    ),
                                    span: args[1].span,
                                    code: "lint.shape.reduction",
                                    severity: HirDiagnosticSeverity::Warning,
                                });
                            }
                        }
                    }
                }

                for arg in args {
                    walk_expr(arg, env, returns, diags);
                }
            }
            HirExprKind::MethodCall(_, _, args) => {
                for arg in args {
                    walk_expr(arg, env, returns, diags);
                }
            }
            HirExprKind::Member(base, _) | HirExprKind::MemberDynamic(base, _) => {
                walk_expr(base, env, returns, diags);
            }
            HirExprKind::AnonFunc { body, .. } => {
                walk_expr(body, env, returns, diags);
            }
            _ => {}
        }
    }

    fn walk_stmt(
        stmt: &HirStmt,
        env: &std::collections::HashMap<VarId, Type>,
        returns: &std::collections::HashMap<String, Vec<Type>>,
        func_envs: &std::collections::HashMap<String, std::collections::HashMap<VarId, Type>>,
        diags: &mut Vec<HirDiagnostic>,
    ) {
        match stmt {
            HirStmt::Assign(_, expr, _, _)
            | HirStmt::ExprStmt(expr, _, _)
            | HirStmt::MultiAssign(_, expr, _, _) => walk_expr(expr, env, returns, diags),
            HirStmt::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
                ..
            } => {
                walk_expr(cond, env, returns, diags);
                for s in then_body {
                    walk_stmt(s, env, returns, func_envs, diags);
                }
                for (cond, body) in elseif_blocks {
                    walk_expr(cond, env, returns, diags);
                    for s in body {
                        walk_stmt(s, env, returns, func_envs, diags);
                    }
                }
                if let Some(body) = else_body {
                    for s in body {
                        walk_stmt(s, env, returns, func_envs, diags);
                    }
                }
            }
            HirStmt::While { cond, body, .. } => {
                walk_expr(cond, env, returns, diags);
                for s in body {
                    walk_stmt(s, env, returns, func_envs, diags);
                }
            }
            HirStmt::For { expr, body, .. } => {
                walk_expr(expr, env, returns, diags);
                for s in body {
                    walk_stmt(s, env, returns, func_envs, diags);
                }
            }
            HirStmt::Switch {
                expr,
                cases,
                otherwise,
                ..
            } => {
                walk_expr(expr, env, returns, diags);
                for (case_expr, case_body) in cases {
                    walk_expr(case_expr, env, returns, diags);
                    for s in case_body {
                        walk_stmt(s, env, returns, func_envs, diags);
                    }
                }
                if let Some(body) = otherwise {
                    for s in body {
                        walk_stmt(s, env, returns, func_envs, diags);
                    }
                }
            }
            HirStmt::Function { name, body, .. } => {
                let func_env = func_envs.get(name).cloned().unwrap_or_default();
                for s in body {
                    walk_stmt(s, &func_env, returns, func_envs, diags);
                }
            }
            HirStmt::ClassDef { members, .. } => {
                for member in members {
                    if let HirClassMember::Methods { body, .. } = member {
                        for s in body {
                            walk_stmt(s, env, returns, func_envs, diags);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let mut diags = Vec::new();
    let global_env = result.inferred_globals.clone();
    for stmt in &result.hir.body {
        walk_stmt(
            stmt,
            &global_env,
            &result.inferred_function_returns,
            &result.inferred_function_envs,
            &mut diags,
        );
    }
    diags
}
