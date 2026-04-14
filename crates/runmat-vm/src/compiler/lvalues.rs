//! L-value lowering for assignment targets.

use crate::compiler::core::Compiler;
use crate::compiler::end_expr::{end_numeric_expr, range_dynamic_end_spec};
use crate::compiler::CompileError;
use crate::instr::{EndExpr, Instr};
use runmat_builtins::Type;
use runmat_hir::{HirExpr, HirExprKind, HirLValue};

impl Compiler {
    pub(crate) fn expr_contains_end(expr: &HirExpr) -> bool {
        use runmat_hir::HirExprKind as K;

        match &expr.kind {
            K::End => true,
            K::Unary(_, e) => Self::expr_contains_end(e),
            K::Binary(a, _, b) => Self::expr_contains_end(a) || Self::expr_contains_end(b),
            K::Range(a, b, c) => {
                Self::expr_contains_end(a)
                    || b.as_ref()
                        .map(|e| Self::expr_contains_end(e))
                        .unwrap_or(false)
                    || Self::expr_contains_end(c)
            }
            K::FuncCall(_, args) => args.iter().any(Self::expr_contains_end),
            K::Tensor(rows) | K::Cell(rows) => rows
                .iter()
                .flat_map(|r| r.iter())
                .any(Self::expr_contains_end),
            K::Index(base, idxs) | K::IndexCell(base, idxs) => {
                if Self::expr_contains_end(base) {
                    return true;
                }
                idxs.iter().any(Self::expr_contains_end)
            }
            _ => false,
        }
    }

    fn allow_implicit_struct_materialization(ty: &Type) -> bool {
        matches!(ty, Type::Unknown | Type::Struct { .. })
    }

    fn emit_store_back_member_chain(&mut self, base: &HirExpr) -> Result<(), CompileError> {
        match &base.kind {
            HirExprKind::Var(var_id) => {
                self.emit(Instr::StoreVar(var_id.0));
                Ok(())
            }
            HirExprKind::Member(parent, field) => {
                self.compile_member_base_for_assignment(parent)?;
                self.emit(Instr::Swap);
                self.emit(Instr::StoreMemberOrInit(field.clone()));
                self.emit_store_back_member_chain(parent)
            }
            HirExprKind::MemberDynamic(parent, name_expr) => {
                let tmp = self.alloc_temp();
                self.emit(Instr::StoreVar(tmp));
                self.compile_member_base_for_assignment(parent)?;
                self.compile_expr(name_expr)?;
                self.emit(Instr::LoadVar(tmp));
                self.emit(Instr::StoreMemberDynamicOrInit);
                self.emit_store_back_member_chain(parent)
            }
            _ => Ok(()),
        }
    }

    fn compile_member_base_for_assignment(&mut self, base: &HirExpr) -> Result<(), CompileError> {
        match &base.kind {
            HirExprKind::Member(parent, field) => {
                self.compile_member_base_for_assignment(parent)?;
                self.emit(Instr::LoadMemberOrInit(field.clone()));
                Ok(())
            }
            HirExprKind::MemberDynamic(parent, name_expr) => {
                self.compile_member_base_for_assignment(parent)?;
                self.compile_expr(name_expr)?;
                self.emit(Instr::LoadMemberDynamicOrInit);
                Ok(())
            }
            _ => self.compile_expr(base),
        }
    }

    pub(crate) fn compile_assign_lvalue(
        &mut self,
        lv: &HirLValue,
        rhs: &HirExpr,
    ) -> Result<(), CompileError> {
        match lv {
            HirLValue::Index(base, indices) => {
                if let HirExprKind::Var(var_id) = base.kind {
                    self.emit(Instr::LoadVar(var_id.0));
                    let has_colon = indices.iter().any(|e| matches!(e.kind, HirExprKind::Colon));
                    let has_end = indices.iter().any(|e| matches!(e.kind, HirExprKind::End));
                    let has_vector = indices.iter().any(|e| {
                        matches!(e.kind, HirExprKind::Range(_, _, _) | HirExprKind::Tensor(_))
                            || matches!(
                                e.ty,
                                runmat_hir::Type::Tensor { .. }
                                    | runmat_hir::Type::Bool
                                    | runmat_hir::Type::Logical { .. }
                            )
                    });
                    if has_colon || has_end || has_vector || indices.len() > 2 {
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
                                if let Some((_start_end, _step_end, end_expr)) =
                                    range_dynamic_end_spec(index)
                                {
                                    end_offsets.push((numeric_count, end_expr));
                                    continue;
                                }
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
                        {
                            let mut has_any_range_end = false;
                            let mut range_dims: Vec<usize> = Vec::new();
                            let mut range_has_step: Vec<bool> = Vec::new();
                            let mut range_start_exprs: Vec<Option<EndExpr>> = Vec::new();
                            let mut range_step_exprs: Vec<Option<EndExpr>> = Vec::new();
                            let mut end_offs: Vec<EndExpr> = Vec::new();
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
                                        end_offs.push(end_expr);
                                    }
                                }
                            }
                            if has_any_range_end {
                                for &dim in &range_dims {
                                    if let HirExprKind::Range(start, step, _end) =
                                        &indices[dim].kind
                                    {
                                        if range_start_exprs
                                            [range_dims.iter().position(|&rd| rd == dim).unwrap()]
                                        .is_some()
                                        {
                                            self.emit(Instr::LoadConst(0.0));
                                        } else {
                                            self.compile_expr(start)?;
                                        }
                                        if let Some(st) = step {
                                            if range_step_exprs[range_dims
                                                .iter()
                                                .position(|&rd| rd == dim)
                                                .unwrap()]
                                            .is_some()
                                            {
                                                self.emit(Instr::LoadConst(0.0));
                                            } else {
                                                self.compile_expr(st)?;
                                            }
                                        }
                                    }
                                }
                                let range_value_count = range_has_step
                                    .iter()
                                    .map(|has_step| if *has_step { 2 } else { 1 })
                                    .sum::<usize>();
                                let lvalue_count = 1 + numeric_count + range_value_count;
                                let mut lvalue_temps = Vec::with_capacity(lvalue_count);
                                for _ in 0..lvalue_count {
                                    let temp = self.alloc_temp();
                                    self.emit(Instr::StoreVar(temp));
                                    lvalue_temps.push(temp);
                                }
                                lvalue_temps.reverse();
                                self.compile_expr(rhs)?;
                                let rhs_temp = self.alloc_temp();
                                self.emit(Instr::StoreVar(rhs_temp));
                                for temp in &lvalue_temps {
                                    self.emit(Instr::LoadVar(*temp));
                                }
                                self.emit(Instr::LoadVar(rhs_temp));
                                self.emit(Instr::StoreSliceExpr {
                                    dims: indices.len(),
                                    numeric_count,
                                    colon_mask,
                                    end_mask,
                                    range_dims,
                                    range_has_step,
                                    range_start_exprs,
                                    range_step_exprs,
                                    range_end_exprs: end_offs,
                                    end_numeric_exprs: Vec::new(),
                                });
                            } else {
                                let dims_len = indices.len();
                                let idx_is_scalar = |e: &HirExpr| -> bool {
                                    matches!(e.kind, HirExprKind::Number(_) | HirExprKind::End)
                                };
                                let idx_is_vector = |e: &HirExpr| -> bool {
                                    matches!(
                                        e.kind,
                                        HirExprKind::Colon
                                            | HirExprKind::Range(_, _, _)
                                            | HirExprKind::Tensor(_)
                                    )
                                };
                                let (is_row_slice, is_col_slice) = if dims_len == 2 {
                                    (
                                        idx_is_scalar(&indices[0]) && idx_is_vector(&indices[1]),
                                        idx_is_vector(&indices[0]) && idx_is_scalar(&indices[1]),
                                    )
                                } else {
                                    (false, false)
                                };
                                fn const_vec_len(e: &HirExpr) -> Option<usize> {
                                    match &e.kind {
                                        HirExprKind::Number(_) | HirExprKind::End => Some(1),
                                        HirExprKind::Tensor(rows) => {
                                            Some(rows.iter().map(|r| r.len()).sum())
                                        }
                                        HirExprKind::Range(start, step, end) => {
                                            if let (
                                                HirExprKind::Number(sa),
                                                HirExprKind::Number(ea),
                                            ) = (&start.kind, &end.kind)
                                            {
                                                let s: f64 = sa.parse().ok()?;
                                                let en: f64 = ea.parse().ok()?;
                                                let st: f64 = if let Some(st) = step {
                                                    if let HirExprKind::Number(x) = &st.kind {
                                                        x.parse().ok()?
                                                    } else {
                                                        return None;
                                                    }
                                                } else {
                                                    1.0
                                                };
                                                if st == 0.0 {
                                                    return None;
                                                }
                                                let n = ((en - s) / st).floor() as isize + 1;
                                                if n <= 0 {
                                                    Some(0)
                                                } else {
                                                    Some(n as usize)
                                                }
                                            } else {
                                                None
                                            }
                                        }
                                        HirExprKind::Colon => None,
                                        _ => None,
                                    }
                                }
                                let lvalue_count = 1 + numeric_count;
                                let mut lvalue_temps = Vec::with_capacity(lvalue_count);
                                for _ in 0..lvalue_count {
                                    let temp = self.alloc_temp();
                                    self.emit(Instr::StoreVar(temp));
                                    lvalue_temps.push(temp);
                                }
                                lvalue_temps.reverse();
                                let mut packed = false;
                                if let HirExprKind::FuncCall(fname, fargs) = &rhs.kind {
                                    if self.functions.contains_key(fname)
                                        && (dims_len == 1 || is_row_slice || is_col_slice)
                                    {
                                        for a in fargs {
                                            self.compile_expr(a)?;
                                        }
                                        let outc = self
                                            .functions
                                            .get(fname)
                                            .map(|f| f.outputs.len().max(1))
                                            .unwrap_or(1);
                                        self.emit(Instr::CallFunctionMulti(
                                            fname.clone(),
                                            fargs.len(),
                                            outc,
                                        ));
                                        self.emit(Instr::Unpack(outc));
                                        if dims_len == 1 || is_col_slice {
                                            self.emit(Instr::PackToCol(outc));
                                        } else {
                                            self.emit(Instr::PackToRow(outc));
                                        }
                                        packed = true;
                                    }
                                } else if let HirExprKind::IndexCell(cbase, cidx) = &rhs.kind {
                                    let outc = if dims_len == 1 {
                                        const_vec_len(&indices[0])
                                    } else if is_row_slice {
                                        const_vec_len(&indices[1])
                                    } else if is_col_slice {
                                        const_vec_len(&indices[0])
                                    } else {
                                        None
                                    };
                                    if let Some(n) = outc {
                                        self.compile_expr(cbase)?;
                                        let expand_all = cidx.len() == 1
                                            && matches!(cidx[0].kind, HirExprKind::Colon);
                                        if expand_all {
                                            self.emit(Instr::IndexCellExpand(0, n));
                                        } else {
                                            for i in cidx {
                                                self.compile_expr(i)?;
                                            }
                                            self.emit(Instr::IndexCellExpand(cidx.len(), n));
                                        }
                                        if dims_len == 1 || is_col_slice {
                                            self.emit(Instr::PackToCol(n));
                                        } else {
                                            self.emit(Instr::PackToRow(n));
                                        }
                                        packed = true;
                                    }
                                }
                                if !packed {
                                    self.compile_expr(rhs)?;
                                }
                                let rhs_temp = self.alloc_temp();
                                self.emit(Instr::StoreVar(rhs_temp));
                                for temp in &lvalue_temps {
                                    self.emit(Instr::LoadVar(*temp));
                                }
                                self.emit(Instr::LoadVar(rhs_temp));
                                if end_offsets.is_empty() {
                                    self.emit(Instr::StoreSlice(
                                        indices.len(),
                                        numeric_count,
                                        colon_mask,
                                        end_mask,
                                    ));
                                } else {
                                    self.emit(Instr::StoreSliceExpr {
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
                            }
                            self.emit(Instr::StoreVar(var_id.0));
                        }
                    } else {
                        for index in indices {
                            self.compile_expr(index)?;
                        }
                        if let HirExprKind::FuncCall(fname, fargs) = &rhs.kind {
                            if self.functions.contains_key(fname) && indices.len() == 1 {
                                for a in fargs {
                                    self.compile_expr(a)?;
                                }
                                let outc = self
                                    .functions
                                    .get(fname)
                                    .map(|f| f.outputs.len().max(1))
                                    .unwrap_or(1);
                                self.emit(Instr::CallFunctionMulti(
                                    fname.clone(),
                                    fargs.len(),
                                    outc,
                                ));
                                self.emit(Instr::Unpack(outc));
                                self.emit(Instr::PackToCol(outc));
                            } else {
                                self.compile_expr(rhs)?;
                            }
                        } else {
                            self.compile_expr(rhs)?;
                        }
                        self.emit(Instr::StoreIndex(indices.len()));
                        self.emit(Instr::StoreVar(var_id.0));
                    }
                } else if let HirExprKind::Member(member_base, field) = &base.kind {
                    self.compile_member_base_for_assignment(member_base)?;
                    self.emit(Instr::LoadMemberOrInit(field.clone()));
                    let has_colon = indices.iter().any(|e| matches!(e.kind, HirExprKind::Colon));
                    let has_end = indices.iter().any(|e| matches!(e.kind, HirExprKind::End));
                    let has_vector = indices.iter().any(|e| {
                        matches!(e.kind, HirExprKind::Range(_, _, _) | HirExprKind::Tensor(_))
                            || matches!(e.ty, runmat_hir::Type::Tensor { .. })
                    });
                    if has_colon || has_end || has_vector || indices.len() > 2 {
                        let mut colon_mask: u32 = 0;
                        let mut end_mask: u32 = 0;
                        let mut numeric_count = 0usize;
                        for (dim, index) in indices.iter().enumerate() {
                            if matches!(index.kind, HirExprKind::Colon) {
                                colon_mask |= 1u32 << dim;
                            } else if matches!(index.kind, HirExprKind::End) {
                                end_mask |= 1u32 << dim;
                            } else {
                                self.compile_expr(index)?;
                                numeric_count += 1;
                            }
                        }
                        let lvalue_count = 1 + numeric_count;
                        let mut lvalue_temps = Vec::with_capacity(lvalue_count);
                        for _ in 0..lvalue_count {
                            let temp = self.alloc_temp();
                            self.emit(Instr::StoreVar(temp));
                            lvalue_temps.push(temp);
                        }
                        lvalue_temps.reverse();
                        self.compile_expr(rhs)?;
                        let rhs_temp = self.alloc_temp();
                        self.emit(Instr::StoreVar(rhs_temp));
                        for temp in &lvalue_temps {
                            self.emit(Instr::LoadVar(*temp));
                        }
                        self.emit(Instr::LoadVar(rhs_temp));
                        self.emit(Instr::StoreSlice(
                            indices.len(),
                            numeric_count,
                            colon_mask,
                            end_mask,
                        ));
                    } else {
                        for index in indices {
                            self.compile_expr(index)?;
                        }
                        self.compile_expr(rhs)?;
                        self.emit(Instr::StoreIndex(indices.len()));
                    }
                    self.compile_member_base_for_assignment(member_base)?;
                    self.emit(Instr::Swap);
                    self.emit(Instr::StoreMemberOrInit(field.clone()));
                    self.emit_store_back_member_chain(member_base)?;
                } else {
                    return Err(self.compile_error(
                        "unsupported lvalue target (index on non-variable/non-member)",
                    ));
                }
            }
            HirLValue::IndexCell(base, indices) => {
                if let HirExprKind::Var(var_id) = base.kind {
                    self.emit(Instr::LoadVar(var_id.0));
                    for index in indices {
                        self.compile_expr(index)?;
                    }
                    self.compile_expr(rhs)?;
                    self.emit(Instr::StoreIndexCell(indices.len()));
                    self.emit(Instr::StoreVar(var_id.0));
                } else if let HirExprKind::Member(member_base, field) = &base.kind {
                    self.compile_member_base_for_assignment(member_base)?;
                    self.emit(Instr::LoadMemberOrInit(field.clone()));
                    for index in indices {
                        self.compile_expr(index)?;
                    }
                    self.compile_expr(rhs)?;
                    self.emit(Instr::StoreIndexCell(indices.len()));
                    self.compile_member_base_for_assignment(member_base)?;
                    self.emit(Instr::Swap);
                    self.emit(Instr::StoreMemberOrInit(field.clone()));
                    self.emit_store_back_member_chain(member_base)?;
                } else {
                    self.compile_expr(base)?;
                    for index in indices {
                        self.compile_expr(index)?;
                    }
                    self.compile_expr(rhs)?;
                    self.emit(Instr::StoreIndexCell(indices.len()));
                }
            }
            HirLValue::Member(base, field) => {
                self.compile_member_base_for_assignment(base)?;
                self.compile_expr(rhs)?;
                if Self::allow_implicit_struct_materialization(&base.ty) {
                    self.emit(Instr::StoreMemberOrInit(field.clone()));
                } else {
                    self.emit(Instr::StoreMember(field.clone()));
                }
                self.emit_store_back_member_chain(base)?;
            }
            HirLValue::MemberDynamic(base, name_expr) => {
                self.compile_member_base_for_assignment(base)?;
                self.compile_expr(name_expr)?;
                self.compile_expr(rhs)?;
                if Self::allow_implicit_struct_materialization(&base.ty) {
                    self.emit(Instr::StoreMemberDynamicOrInit);
                } else {
                    self.emit(Instr::StoreMemberDynamic);
                }
                self.emit_store_back_member_chain(base)?;
            }
            _ => return Err(self.compile_error("unsupported lvalue target")),
        }

        Ok(())
    }
}
