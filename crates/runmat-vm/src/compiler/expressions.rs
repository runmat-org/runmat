//! Expression lowering.

use crate::compiler::core::Compiler;
use crate::compiler::end_expr::{end_numeric_expr, range_dynamic_end_spec};
use crate::compiler::imports::CallImportResolution;
use crate::compiler::CompileError;
use crate::instr::{EndExpr, Instr};
use runmat_hir::{HirExpr, HirExprKind};

impl Compiler {
    pub(crate) fn compile_expr_impl(&mut self, expr: &HirExpr) -> Result<(), CompileError> {
        match &expr.kind {
            HirExprKind::Number(n) => {
                let val: f64 = n.parse().map_err(|_| "invalid number")?;
                self.emit(Instr::LoadConst(val));
            }
            HirExprKind::String(s) => {
                if s.starts_with('"') && s.ends_with('"') {
                    let inner = &s[1..s.len() - 1];
                    let clean = inner.replace("\"\"", "\"");
                    self.emit(Instr::LoadString(clean));
                } else if s.starts_with('\'') && s.ends_with('\'') {
                    let inner = &s[1..s.len() - 1];
                    let clean = inner.replace("''", "'");
                    self.emit(Instr::LoadCharRow(clean));
                } else {
                    self.emit(Instr::LoadString(s.clone()));
                }
            }
            HirExprKind::Var(id) => {
                self.emit(Instr::LoadVar(id.0));
            }
            HirExprKind::Constant(name) => {
                let constants = runmat_builtins::constants();
                if let Some(constant) = constants.iter().find(|c| c.name == name) {
                    match &constant.value {
                        runmat_builtins::Value::Num(val) => {
                            self.emit(Instr::LoadConst(*val));
                        }
                        runmat_builtins::Value::Complex(re, im) => {
                            self.emit(Instr::LoadComplex(*re, *im));
                        }
                        runmat_builtins::Value::Bool(b) => {
                            self.emit(Instr::LoadBool(*b));
                        }
                        _ => {
                            return Err(self.compile_error(format!(
                                "Constant {name} is not a number or boolean"
                            )));
                        }
                    }
                } else {
                    if let Some(cls) = self.resolve_unqualified_static_property(name)? {
                        self.emit(Instr::LoadStaticProperty(cls, name.clone()));
                        return Ok(());
                    }
                    return Err(
                        self.compile_error(format!("Unknown constant or static property: {name}"))
                    );
                }
            }
            HirExprKind::Unary(op, e) => {
                self.compile_expr(e)?;
                match op {
                    runmat_parser::UnOp::Plus => {
                        self.emit(Instr::UPlus);
                    }
                    runmat_parser::UnOp::Minus => {
                        self.emit(Instr::Neg);
                    }
                    runmat_parser::UnOp::Transpose => {
                        self.emit(Instr::ConjugateTranspose);
                    }
                    runmat_parser::UnOp::NonConjugateTranspose => {
                        self.emit(Instr::Transpose);
                    }
                    runmat_parser::UnOp::Not => {
                        self.emit(Instr::CallBuiltin("not".to_string(), 1));
                    }
                }
            }
            HirExprKind::Binary(a, op, b) => {
                use runmat_parser::BinOp;
                match op {
                    BinOp::AndAnd => {
                        self.compile_expr(a)?;
                        self.emit(Instr::LoadConst(0.0));
                        self.emit(Instr::NotEqual);
                        let j_false = self.emit(Instr::JumpIfFalse(usize::MAX));
                        self.compile_expr(b)?;
                        self.emit(Instr::LoadConst(0.0));
                        self.emit(Instr::NotEqual);
                        let end = self.emit(Instr::Jump(usize::MAX));
                        let after_cond = self.instructions.len();
                        self.patch(j_false, Instr::JumpIfFalse(after_cond));
                        self.emit(Instr::LoadConst(0.0));
                        let end_pc = self.instructions.len();
                        self.patch(end, Instr::Jump(end_pc));
                    }
                    BinOp::OrOr => {
                        self.compile_expr(a)?;
                        self.emit(Instr::LoadConst(0.0));
                        self.emit(Instr::NotEqual);
                        let j_true = self.emit(Instr::JumpIfFalse(usize::MAX));
                        self.emit(Instr::LoadConst(1.0));
                        let end = self.emit(Instr::Jump(usize::MAX));
                        let after_check = self.instructions.len();
                        self.patch(j_true, Instr::JumpIfFalse(after_check));
                        self.compile_expr(b)?;
                        self.emit(Instr::LoadConst(0.0));
                        self.emit(Instr::NotEqual);
                        let end_pc = self.instructions.len();
                        self.patch(end, Instr::Jump(end_pc));
                    }
                    BinOp::BitAnd => {
                        self.compile_expr(a)?;
                        self.emit(Instr::LoadConst(0.0));
                        self.emit(Instr::CallBuiltin("ne".to_string(), 2));
                        self.compile_expr(b)?;
                        self.emit(Instr::LoadConst(0.0));
                        self.emit(Instr::CallBuiltin("ne".to_string(), 2));
                        self.emit(Instr::ElemMul);
                    }
                    BinOp::BitOr => {
                        self.compile_expr(a)?;
                        self.emit(Instr::LoadConst(0.0));
                        self.emit(Instr::CallBuiltin("ne".to_string(), 2));
                        self.compile_expr(b)?;
                        self.emit(Instr::LoadConst(0.0));
                        self.emit(Instr::CallBuiltin("ne".to_string(), 2));
                        self.emit(Instr::Add);
                        self.emit(Instr::LoadConst(0.0));
                        self.emit(Instr::CallBuiltin("ne".to_string(), 2));
                    }
                    _ => {
                        self.compile_expr(a)?;
                        self.compile_expr(b)?;
                        match op {
                            BinOp::Add => self.emit(Instr::Add),
                            BinOp::Sub => self.emit(Instr::Sub),
                            BinOp::Mul => self.emit(Instr::Mul),
                            BinOp::RightDiv => self.emit(Instr::RightDiv),
                            BinOp::LeftDiv => self.emit(Instr::LeftDiv),
                            BinOp::Pow => self.emit(Instr::Pow),
                            BinOp::ElemMul => self.emit(Instr::ElemMul),
                            BinOp::ElemDiv => self.emit(Instr::ElemDiv),
                            BinOp::ElemLeftDiv => self.emit(Instr::ElemLeftDiv),
                            BinOp::ElemPow => self.emit(Instr::ElemPow),
                            BinOp::Equal => self.emit(Instr::Equal),
                            BinOp::NotEqual => self.emit(Instr::NotEqual),
                            BinOp::Less => self.emit(Instr::Less),
                            BinOp::LessEqual => self.emit(Instr::LessEqual),
                            BinOp::Greater => self.emit(Instr::Greater),
                            BinOp::GreaterEqual => self.emit(Instr::GreaterEqual),
                            BinOp::Colon => {
                                return Err(self.compile_error("colon operator not supported"));
                            }
                            _ => unreachable!(),
                        };
                    }
                }
            }
            HirExprKind::Range(start, step, end) => {
                self.compile_expr(start)?;
                if let Some(step) = step {
                    self.compile_expr(step)?;
                    self.compile_expr(end)?;
                    self.emit(Instr::CreateRange(true));
                } else {
                    self.compile_expr(end)?;
                    self.emit(Instr::CreateRange(false));
                }
            }
            HirExprKind::FuncCall(name, args) => {
                let call_arg_spans: Vec<runmat_hir::Span> = args.iter().map(|a| a.span).collect();
                if name == "feval" {
                    if args.is_empty() {
                        return Err(self.compile_error("feval: missing function argument"));
                    }
                    self.compile_expr(&args[0])?;
                    let rest = &args[1..];
                    let has_expand = rest
                        .iter()
                        .any(|a| matches!(a.kind, HirExprKind::IndexCell(_, _)));
                    if has_expand {
                        let mut specs: Vec<crate::instr::ArgSpec> = Vec::with_capacity(rest.len());
                        for arg in rest {
                            if let HirExprKind::IndexCell(base, indices) = &arg.kind {
                                let is_expand_all = indices.len() == 1
                                    && matches!(indices[0].kind, HirExprKind::Colon);
                                if is_expand_all {
                                    specs.push(crate::instr::ArgSpec {
                                        is_expand: true,
                                        num_indices: 0,
                                        expand_all: true,
                                    });
                                    self.compile_expr(base)?;
                                } else {
                                    specs.push(crate::instr::ArgSpec {
                                        is_expand: true,
                                        num_indices: indices.len(),
                                        expand_all: false,
                                    });
                                    self.compile_expr(base)?;
                                    for i in indices {
                                        self.compile_expr(i)?;
                                    }
                                }
                            } else {
                                specs.push(crate::instr::ArgSpec {
                                    is_expand: false,
                                    num_indices: 0,
                                    expand_all: false,
                                });
                                self.compile_expr(arg)?;
                            }
                        }
                        self.emit_call_with_arg_spans(
                            Instr::CallFevalExpandMulti(specs),
                            &call_arg_spans,
                        );
                    } else {
                        for arg in rest {
                            self.compile_expr(arg)?;
                        }
                        self.emit_call_with_arg_spans(
                            Instr::CallFeval(rest.len()),
                            &call_arg_spans,
                        );
                    }
                    return Ok(());
                }
                let has_any_expand = args
                    .iter()
                    .any(|a| matches!(a.kind, HirExprKind::IndexCell(_, _)));
                if self.functions.contains_key(name) {
                    if has_any_expand {
                        let mut specs: Vec<crate::instr::ArgSpec> = Vec::with_capacity(args.len());
                        for arg in args {
                            if let HirExprKind::IndexCell(base, indices) = &arg.kind {
                                let is_expand_all = indices.len() == 1
                                    && matches!(indices[0].kind, HirExprKind::Colon);
                                if is_expand_all {
                                    specs.push(crate::instr::ArgSpec {
                                        is_expand: true,
                                        num_indices: 0,
                                        expand_all: true,
                                    });
                                    self.compile_expr(base)?;
                                } else {
                                    specs.push(crate::instr::ArgSpec {
                                        is_expand: true,
                                        num_indices: indices.len(),
                                        expand_all: false,
                                    });
                                    self.compile_expr(base)?;
                                    for i in indices {
                                        self.compile_expr(i)?;
                                    }
                                }
                            } else {
                                specs.push(crate::instr::ArgSpec {
                                    is_expand: false,
                                    num_indices: 0,
                                    expand_all: false,
                                });
                                self.compile_expr(arg)?;
                            }
                        }
                        self.emit_call_with_arg_spans(
                            Instr::CallFunctionExpandMulti(name.clone(), specs),
                            &call_arg_spans,
                        );
                    } else {
                        for arg in args {
                            self.compile_expr(arg)?;
                        }
                        self.emit_call_with_arg_spans(
                            Instr::CallFunction(name.clone(), args.len()),
                            &call_arg_spans,
                        );
                    }
                } else {
                    let CallImportResolution {
                        resolved,
                        mut static_candidates,
                    } = self.resolve_call_imports(name)?;
                    if self.functions.contains_key(&resolved) {
                        if has_any_expand {
                            let mut specs: Vec<crate::instr::ArgSpec> =
                                Vec::with_capacity(args.len());
                            for arg in args {
                                if let HirExprKind::IndexCell(base, indices) = &arg.kind {
                                    let is_expand_all = indices.len() == 1
                                        && matches!(indices[0].kind, HirExprKind::Colon);
                                    if is_expand_all {
                                        specs.push(crate::instr::ArgSpec {
                                            is_expand: true,
                                            num_indices: 0,
                                            expand_all: true,
                                        });
                                        self.compile_expr(base)?;
                                    } else {
                                        specs.push(crate::instr::ArgSpec {
                                            is_expand: true,
                                            num_indices: indices.len(),
                                            expand_all: false,
                                        });
                                        self.compile_expr(base)?;
                                        for i in indices {
                                            self.compile_expr(i)?;
                                        }
                                    }
                                } else {
                                    specs.push(crate::instr::ArgSpec {
                                        is_expand: false,
                                        num_indices: 0,
                                        expand_all: false,
                                    });
                                    self.compile_expr(arg)?;
                                }
                            }
                            self.emit_call_with_arg_spans(
                                Instr::CallFunctionExpandMulti(resolved.clone(), specs),
                                &call_arg_spans,
                            );
                            return Ok(());
                        } else {
                            let mut total_argc: usize = 0;
                            for arg in args {
                                if let HirExprKind::FuncCall(inner, inner_args) = &arg.kind {
                                    if self.functions.contains_key(inner) {
                                        for a in inner_args {
                                            self.compile_expr(a)?;
                                        }
                                        let outc = self
                                            .functions
                                            .get(inner)
                                            .map(|f| f.outputs.len().max(1))
                                            .unwrap_or(1);
                                        self.emit(Instr::CallFunctionMulti(
                                            inner.clone(),
                                            inner_args.len(),
                                            outc,
                                        ));
                                        self.emit(Instr::Unpack(outc));
                                        total_argc += outc;
                                        continue;
                                    }
                                }
                                self.compile_expr(arg)?;
                                total_argc += 1;
                            }
                            self.emit_call_with_arg_spans(
                                Instr::CallFunction(resolved.clone(), total_argc),
                                &call_arg_spans,
                            );
                            return Ok(());
                        }
                    }
                    if !runmat_builtins::builtin_functions()
                        .iter()
                        .any(|b| b.name == resolved)
                        && static_candidates.len() == 1
                    {
                        let (cls, method) = static_candidates.remove(0);
                        for arg in args {
                            self.compile_expr(arg)?;
                        }
                        self.emit_call_with_arg_spans(
                            Instr::CallStaticMethod(cls, method, args.len()),
                            &call_arg_spans,
                        );
                        return Ok(());
                    }
                    if !runmat_builtins::builtin_functions()
                        .iter()
                        .any(|b| b.name == resolved)
                        && static_candidates.len() > 1
                    {
                        return Err(self.compile_error(format!(
                            "ambiguous unqualified static method '{}' via Class.* imports: {}",
                            name,
                            static_candidates
                                .iter()
                                .map(|(c, _)| c.clone())
                                .collect::<Vec<_>>()
                                .join(", ")
                        )));
                    }
                    if !has_any_expand {
                        let mut total_argc = 0usize;
                        let mut did_expand_inner = false;
                        let mut pending_simple: Vec<&runmat_hir::HirExpr> = Vec::new();
                        for arg in args {
                            if let HirExprKind::FuncCall(inner, inner_args) = &arg.kind {
                                if self.functions.contains_key(inner) {
                                    for a in inner_args {
                                        self.compile_expr(a)?;
                                    }
                                    let outc = self
                                        .functions
                                        .get(inner)
                                        .map(|f| f.outputs.len().max(1))
                                        .unwrap_or(1);
                                    self.emit(Instr::CallFunctionMulti(
                                        inner.clone(),
                                        inner_args.len(),
                                        outc,
                                    ));
                                    self.emit(Instr::Unpack(outc));
                                    total_argc += outc;
                                    did_expand_inner = true;
                                } else {
                                    pending_simple.push(arg);
                                }
                            } else {
                                pending_simple.push(arg);
                            }
                        }
                        if did_expand_inner {
                            for arg in pending_simple {
                                self.compile_expr(arg)?;
                                total_argc += 1;
                            }
                            self.emit_call_with_arg_spans(
                                Instr::CallBuiltin(resolved, total_argc),
                                &call_arg_spans,
                            );
                            return Ok(());
                        }
                    }
                    if has_any_expand {
                        let mut specs: Vec<crate::instr::ArgSpec> = Vec::with_capacity(args.len());
                        for arg in args {
                            if let HirExprKind::IndexCell(base, indices) = &arg.kind {
                                let is_expand_all = indices.len() == 1
                                    && matches!(indices[0].kind, HirExprKind::Colon);
                                if is_expand_all {
                                    specs.push(crate::instr::ArgSpec {
                                        is_expand: true,
                                        num_indices: 0,
                                        expand_all: true,
                                    });
                                    self.compile_expr(base)?;
                                } else {
                                    specs.push(crate::instr::ArgSpec {
                                        is_expand: true,
                                        num_indices: indices.len(),
                                        expand_all: false,
                                    });
                                    self.compile_expr(base)?;
                                    for i in indices {
                                        self.compile_expr(i)?;
                                    }
                                }
                            } else {
                                specs.push(crate::instr::ArgSpec {
                                    is_expand: false,
                                    num_indices: 0,
                                    expand_all: false,
                                });
                                self.compile_expr(arg)?;
                            }
                        }
                        self.emit_call_with_arg_spans(
                            Instr::CallBuiltinExpandMulti(resolved, specs),
                            &call_arg_spans,
                        );
                    } else {
                        for arg in args {
                            self.compile_expr(arg)?;
                        }
                        self.emit_call_with_arg_spans(
                            Instr::CallBuiltin(resolved, args.len()),
                            &call_arg_spans,
                        );
                    }
                    return Ok(());
                }
                return Ok(());
            }
            HirExprKind::Tensor(matrix_data) | HirExprKind::Cell(matrix_data) => {
                let rows = matrix_data.len();
                if matches!(expr.kind, HirExprKind::Tensor(_))
                    && rows == 1
                    && matrix_data.first().map(|r| r.len()).unwrap_or(0) == 1
                {
                    if let HirExprKind::IndexCell(base, indices) = &matrix_data[0][0].kind {
                        if indices.len() == 1 && matches!(indices[0].kind, HirExprKind::Colon) {
                            let mut specs: Vec<crate::instr::ArgSpec> = Vec::with_capacity(2);
                            specs.push(crate::instr::ArgSpec {
                                is_expand: false,
                                num_indices: 0,
                                expand_all: false,
                            });
                            self.emit(Instr::LoadConst(2.0));
                            specs.push(crate::instr::ArgSpec {
                                is_expand: true,
                                num_indices: 0,
                                expand_all: true,
                            });
                            self.compile_expr(base)?;
                            self.emit(Instr::CallBuiltinExpandMulti("cat".to_string(), specs));
                            return Ok(());
                        }
                    }
                }
                let has_non_literals = matrix_data.iter().any(|row| {
                    row.iter()
                        .any(|expr| !matches!(expr.kind, HirExprKind::Number(_)))
                });
                if has_non_literals {
                    for row in matrix_data {
                        for element in row {
                            self.compile_expr(element)?;
                        }
                    }
                    let row_lengths: Vec<usize> = matrix_data.iter().map(|row| row.len()).collect();
                    if matches!(expr.kind, HirExprKind::Cell(_)) {
                        let rectangular = row_lengths.iter().all(|&c| c == row_lengths[0]);
                        if rectangular {
                            let cols = if rows > 0 { row_lengths[0] } else { 0 };
                            self.emit(Instr::CreateCell2D(rows, cols));
                        } else {
                            let total: usize = row_lengths.iter().sum();
                            self.emit(Instr::CreateCell2D(1, total));
                        }
                    } else {
                        for &row_len in &row_lengths {
                            self.emit(Instr::LoadConst(row_len as f64));
                        }
                        self.emit(Instr::CreateMatrixDynamic(rows));
                    }
                } else {
                    let cols = if rows > 0 { matrix_data[0].len() } else { 0 };
                    for row in matrix_data {
                        for element in row {
                            self.compile_expr(element)?;
                        }
                    }
                    if matches!(expr.kind, HirExprKind::Cell(_)) {
                        self.emit(Instr::CreateCell2D(rows, cols));
                    } else {
                        self.emit(Instr::CreateMatrix(rows, cols));
                    }
                }
            }
            HirExprKind::Index(base, indices) => {
                let has_colon = indices.iter().any(|e| matches!(e.kind, HirExprKind::Colon));
                let has_end = indices.iter().any(Self::expr_contains_end);
                let has_vector = indices.iter().any(|e| {
                    matches!(e.kind, HirExprKind::Range(_, _, _) | HirExprKind::Tensor(_))
                        || matches!(
                            e.ty,
                            runmat_hir::Type::Tensor { .. }
                                | runmat_hir::Type::Bool
                                | runmat_hir::Type::Logical { .. }
                        )
                        || !matches!(
                            e.kind,
                            HirExprKind::Number(_) | HirExprKind::Colon | HirExprKind::End
                        )
                });
                {
                    let mut has_any_range_end = false;
                    let mut range_dims: Vec<usize> = Vec::new();
                    let mut range_has_step: Vec<bool> = Vec::new();
                    let mut range_start_exprs: Vec<Option<EndExpr>> = Vec::new();
                    let mut range_step_exprs: Vec<Option<EndExpr>> = Vec::new();
                    let mut end_offsets: Vec<EndExpr> = Vec::new();
                    for (dim, index) in indices.iter().enumerate() {
                        if let HirExprKind::Range(_start, step, _end) = &index.kind {
                            if let Some((start_end, step_end, end_expr)) =
                                range_dynamic_end_spec(index)
                            {
                                has_any_range_end = true;
                                range_dims.push(dim);
                                range_has_step.push(step.is_some());
                                range_start_exprs.push(start_end);
                                range_step_exprs.push(step_end);
                                end_offsets.push(end_expr);
                            }
                        }
                    }
                    if has_any_range_end {
                        self.compile_expr(base)?;
                        for (ri, &dim) in range_dims.iter().enumerate() {
                            if let HirExprKind::Range(start, step, _end) = &indices[dim].kind {
                                if range_start_exprs[ri].is_some() {
                                    self.emit(Instr::LoadConst(0.0));
                                } else {
                                    self.compile_expr(start)?;
                                }
                                if let Some(st) = step {
                                    if range_step_exprs[ri].is_some() {
                                        self.emit(Instr::LoadConst(0.0));
                                    } else {
                                        self.compile_expr(st)?;
                                    }
                                }
                            }
                        }
                        let mut colon_mask: u32 = 0;
                        let mut end_mask: u32 = 0;
                        let mut numeric_count = 0usize;
                        for (dim, index) in indices.iter().enumerate() {
                            match &index.kind {
                                HirExprKind::Colon => colon_mask |= 1u32 << dim,
                                HirExprKind::End => end_mask |= 1u32 << dim,
                                HirExprKind::Range(_, _, _) => {
                                    if range_dynamic_end_spec(index).is_some() {
                                        continue;
                                    }
                                    self.compile_expr(index)?;
                                    numeric_count += 1;
                                }
                                _ => {
                                    self.compile_expr(index)?;
                                    numeric_count += 1;
                                }
                            }
                        }
                        self.emit(Instr::IndexSliceExpr {
                            dims: indices.len(),
                            numeric_count,
                            colon_mask,
                            end_mask,
                            range_dims,
                            range_has_step,
                            range_start_exprs,
                            range_step_exprs,
                            range_end_exprs: end_offsets,
                            end_numeric_exprs: Vec::new(),
                        });
                        return Ok(());
                    }
                }
                if has_colon
                    || has_vector
                    || has_end
                    || indices.len() > 2
                    || Self::expr_contains_end(base)
                {
                    self.compile_expr(base)?;
                    let mut colon_mask: u32 = 0;
                    let mut end_mask: u32 = 0;
                    let mut numeric_count = 0usize;
                    let mut end_offsets: Vec<(usize, EndExpr)> = Vec::new();
                    for (dim, index) in indices.iter().enumerate() {
                        if matches!(index.kind, HirExprKind::Colon) {
                            colon_mask |= 1u32 << dim;
                        } else if matches!(index.kind, HirExprKind::End) {
                            end_mask |= 1u32 << dim;
                        } else {
                            if let Some(off) = end_numeric_expr(index) {
                                self.emit(Instr::LoadConst(0.0));
                                end_offsets.push((numeric_count, off));
                                numeric_count += 1;
                                continue;
                            }
                            self.compile_expr(index)?;
                            numeric_count += 1;
                        }
                    }
                    if end_offsets.is_empty() {
                        self.emit(Instr::IndexSlice(
                            indices.len(),
                            numeric_count,
                            colon_mask,
                            end_mask,
                        ));
                    } else {
                        self.emit(Instr::IndexSliceExpr {
                            dims: indices.len(),
                            numeric_count,
                            colon_mask,
                            end_mask,
                            range_dims: Vec::new(),
                            range_has_step: Vec::new(),
                            range_start_exprs: Vec::new(),
                            range_step_exprs: Vec::new(),
                            range_end_exprs: Vec::new(),
                            end_numeric_exprs: end_offsets,
                        });
                    }
                } else {
                    self.compile_expr(base)?;
                    for index in indices {
                        self.compile_expr(index)?;
                    }
                    self.emit(Instr::Index(indices.len()));
                }
            }
            HirExprKind::Colon => {
                self.emit(Instr::LoadConst(0.0));
            }
            HirExprKind::End => {
                self.emit(Instr::LoadConst(-0.0));
            }
            HirExprKind::Member(base, field) => match &base.kind {
                HirExprKind::MetaClass(cls_name) => {
                    self.emit(Instr::LoadStaticProperty(cls_name.clone(), field.clone()));
                }
                HirExprKind::FuncCall(name, args) if name == "classref" && args.len() == 1 => {
                    self.compile_expr(base)?;
                    self.emit(Instr::LoadMember(field.clone()));
                }
                _ => {
                    self.compile_expr(base)?;
                    self.emit(Instr::LoadMember(field.clone()));
                }
            },
            HirExprKind::MemberDynamic(base, name_expr) => {
                self.compile_expr(base)?;
                self.compile_expr(name_expr)?;
                self.emit(Instr::LoadMemberDynamic);
            }
            HirExprKind::DottedInvoke(base, member, args) => {
                self.compile_expr(base)?;
                for arg in args {
                    self.compile_expr(arg)?;
                }
                self.emit(Instr::CallMethodOrMemberIndex(member.clone(), args.len()));
            }
            HirExprKind::MethodCall(b, m, a) if m == &"()".to_string() && a.len() == 1 => {
                self.compile_expr(b)?;
                self.compile_expr(&a[0])?;
                self.emit(Instr::LoadMemberDynamic);
            }
            HirExprKind::MethodCall(base, method, args) => match &base.kind {
                HirExprKind::MetaClass(cls_name) => {
                    for arg in args {
                        self.compile_expr(arg)?;
                    }
                    self.emit(Instr::CallStaticMethod(
                        cls_name.clone(),
                        method.clone(),
                        args.len(),
                    ));
                }
                HirExprKind::FuncCall(name, bargs) if name == "classref" && bargs.len() == 1 => {
                    if let HirExprKind::String(cls) = &bargs[0].kind {
                        let cls_name = Self::normalize_class_literal_name(cls);
                        for arg in args {
                            self.compile_expr(arg)?;
                        }
                        self.emit(Instr::CallStaticMethod(
                            cls_name,
                            method.clone(),
                            args.len(),
                        ));
                    } else {
                        self.compile_expr(base)?;
                        for arg in args {
                            self.compile_expr(arg)?;
                        }
                        self.emit(Instr::CallMethod(method.clone(), args.len()));
                    }
                }
                _ => {
                    self.compile_expr(base)?;
                    for arg in args {
                        self.compile_expr(arg)?;
                    }
                    self.emit(Instr::CallMethod(method.clone(), args.len()));
                }
            },
            HirExprKind::AnonFunc { params, body } => self.compile_anon_func(params, body)?,
            HirExprKind::FuncHandle(name) => {
                self.emit(Instr::LoadString(name.clone()));
                self.emit(Instr::CallBuiltin("make_handle".to_string(), 1));
            }
            HirExprKind::MetaClass(name) => {
                self.emit(Instr::LoadString(name.clone()));
            }
            HirExprKind::IndexCell(base, indices) => {
                self.compile_expr(base)?;
                for index in indices {
                    self.compile_expr(index)?;
                }
                self.emit(Instr::IndexCell(indices.len()));
            }
        }
        Ok(())
    }
}
