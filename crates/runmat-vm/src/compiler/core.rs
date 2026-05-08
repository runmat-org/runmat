use crate::compiler::CompileError;
use crate::functions::UserFunction;
use crate::instr::{ArgSpec, EmitLabel, EndExpr, Instr};
use crate::layout::VmAssemblyLayout;
use runmat_builtins::{self, Type};
use runmat_hir::{
    BindingId, EntrypointId, FunctionId, HirAssembly, HirCallableRef, IndexKind,
    IndexResultContext, LegacyHirExpr as HirExpr, LegacyHirExprKind as HirExprKind,
    LegacyHirProgram as HirProgram, LegacyHirStmt as HirStmt, OperatorKind, RequestedOutputCount,
};
use runmat_mir::{
    BasicBlockId, MirAggregateKind, MirAssembly, MirBody, MirCall, MirCallArg, MirCallee,
    MirConstant, MirIndexComponent, MirIndexing, MirOperand, MirOutputTarget, MirPlace, MirRvalue,
    MirStmt, MirStmtKind, MirTerminatorKind,
};
use std::collections::HashMap;

pub struct LoopLabels {
    pub break_jumps: Vec<usize>,
    pub continue_jumps: Vec<usize>,
}

pub struct Compiler {
    pub instructions: Vec<Instr>,
    pub instr_spans: Vec<runmat_hir::Span>,
    pub call_arg_spans: Vec<Option<Vec<runmat_hir::Span>>>,
    pub var_count: usize,
    pub loop_stack: Vec<LoopLabels>,
    pub functions: HashMap<String, UserFunction>,
    pub imports: Vec<(Vec<String>, bool)>,
    pub var_types: Vec<Type>,
    pub layout: Option<VmAssemblyLayout>,
    pub entrypoint: Option<EntrypointId>,
    pub function: Option<FunctionId>,
    pub body: Option<MirBody>,
    current_span: Option<runmat_hir::Span>,
}

struct SpanGuard {
    compiler: *mut Compiler,
    prev: Option<runmat_hir::Span>,
}

fn end_expr_with_offset(offset: isize) -> EndExpr {
    let magnitude = EndExpr::Const(offset.unsigned_abs() as f64);
    if offset.is_negative() {
        EndExpr::Sub(Box::new(EndExpr::End), Box::new(magnitude))
    } else {
        EndExpr::Add(Box::new(EndExpr::End), Box::new(magnitude))
    }
}

fn mir_indexing_context_matches(actual: IndexResultContext, expected: IndexResultContext) -> bool {
    actual == expected
        || (expected == IndexResultContext::AssignmentTarget
            && actual == IndexResultContext::DeletionTarget)
}

impl SpanGuard {
    fn new(compiler: &mut Compiler, span: runmat_hir::Span) -> Self {
        let prev = compiler.current_span;
        compiler.current_span = Some(span);
        Self {
            compiler: compiler as *mut Compiler,
            prev,
        }
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(compiler) = self.compiler.as_mut() {
                compiler.current_span = self.prev;
            }
        }
    }
}

impl Compiler {
    pub(crate) fn normalize_class_literal_name(raw: &str) -> String {
        if raw.len() >= 2 {
            let bytes = raw.as_bytes();
            let first = bytes[0] as char;
            let last = bytes[raw.len() - 1] as char;
            if (first == '\'' || first == '"') && first == last {
                return raw[1..raw.len() - 1].to_string();
            }
        }
        raw.to_string()
    }

    pub(crate) fn emit_multiassign_outputs(&mut self, vars: &[Option<runmat_hir::VarId>]) {
        for v in vars.iter().flatten() {
            self.emit(Instr::EmitVar {
                var_index: v.0,
                label: EmitLabel::Var(v.0),
            });
        }
    }

    pub fn new(
        hir: &HirAssembly,
        mir: &MirAssembly,
        layout: VmAssemblyLayout,
        entrypoint: EntrypointId,
    ) -> Result<Self, CompileError> {
        let entrypoint_layout = layout.entrypoints.get(&entrypoint).ok_or_else(|| {
            CompileError::new(format!("missing VM layout for entrypoint {entrypoint:?}"))
        })?;
        let function_layout = layout
            .functions
            .get(&entrypoint_layout.target)
            .ok_or_else(|| {
                CompileError::new(format!(
                    "missing VM layout for entrypoint target {:?}",
                    entrypoint_layout.target
                ))
            })?;
        if !hir
            .functions
            .iter()
            .any(|f| f.id == entrypoint_layout.target)
        {
            return Err(CompileError::new(format!(
                "missing HIR function {:?}",
                entrypoint_layout.target
            )));
        }
        let body = mir
            .bodies
            .get(&entrypoint_layout.target)
            .ok_or_else(|| {
                CompileError::new(format!(
                    "missing MIR body for function {:?}",
                    entrypoint_layout.target
                ))
            })?
            .clone();
        let function = entrypoint_layout.target;

        let var_count = function_layout.local_count;
        let mut var_types = Vec::new();
        var_types.resize(var_count, Type::Unknown);

        Ok(Self {
            instructions: Vec::new(),
            instr_spans: Vec::new(),
            call_arg_spans: Vec::new(),
            var_count,
            loop_stack: Vec::new(),
            functions: HashMap::new(),
            imports: Vec::new(),
            var_types,
            layout: Some(layout),
            entrypoint: Some(entrypoint),
            function: Some(function),
            body: Some(body),
            current_span: None,
        })
    }

    pub fn new_for_function(
        hir: &HirAssembly,
        mir: &MirAssembly,
        layout: VmAssemblyLayout,
        function: FunctionId,
    ) -> Result<Self, CompileError> {
        let function_layout = layout.functions.get(&function).ok_or_else(|| {
            CompileError::new(format!("missing VM layout for function {function:?}"))
        })?;
        if !hir.functions.iter().any(|f| f.id == function) {
            return Err(CompileError::new(format!(
                "missing HIR function {function:?}"
            )));
        }
        let body = mir
            .bodies
            .get(&function)
            .ok_or_else(|| {
                CompileError::new(format!("missing MIR body for function {function:?}"))
            })?
            .clone();

        let var_count = function_layout.local_count;
        let mut var_types = Vec::new();
        var_types.resize(var_count, Type::Unknown);

        Ok(Self {
            instructions: Vec::new(),
            instr_spans: Vec::new(),
            call_arg_spans: Vec::new(),
            var_count,
            loop_stack: Vec::new(),
            functions: HashMap::new(),
            imports: Vec::new(),
            var_types,
            layout: Some(layout),
            entrypoint: None,
            function: Some(function),
            body: Some(body),
            current_span: None,
        })
    }

    pub fn new_legacy(prog: &HirProgram) -> Self {
        let mut max_var = 0;
        fn visit_expr(expr: &HirExpr, max: &mut usize) {
            match &expr.kind {
                HirExprKind::Var(id) => {
                    if id.0 + 1 > *max {
                        *max = id.0 + 1;
                    }
                }
                HirExprKind::Unary(_, e) => visit_expr(e, max),
                HirExprKind::Binary(left, _, right) => {
                    visit_expr(left, max);
                    visit_expr(right, max);
                }
                HirExprKind::Tensor(rows) | HirExprKind::Cell(rows) => {
                    for row in rows {
                        for expr in row {
                            visit_expr(expr, max);
                        }
                    }
                }
                HirExprKind::Index(expr, indices) | HirExprKind::IndexCell(expr, indices) => {
                    visit_expr(expr, max);
                    for idx in indices {
                        visit_expr(idx, max);
                    }
                }
                HirExprKind::Range(start, step, end) => {
                    visit_expr(start, max);
                    if let Some(step) = step {
                        visit_expr(step, max);
                    }
                    visit_expr(end, max);
                }
                HirExprKind::FuncCall(_, args) => {
                    for arg in args {
                        visit_expr(arg, max);
                    }
                }
                HirExprKind::MethodCall(base, _, args)
                | HirExprKind::DottedInvoke(base, _, args) => {
                    visit_expr(base, max);
                    for arg in args {
                        visit_expr(arg, max);
                    }
                }
                HirExprKind::Member(base, _) => visit_expr(base, max),
                HirExprKind::MemberDynamic(base, name) => {
                    visit_expr(base, max);
                    visit_expr(name, max);
                }
                HirExprKind::AnonFunc { body, .. } => visit_expr(body, max),
                HirExprKind::Number(_)
                | HirExprKind::String(_)
                | HirExprKind::Constant(_)
                | HirExprKind::Colon
                | HirExprKind::End
                | HirExprKind::FuncHandle(_)
                | HirExprKind::MetaClass(_) => {}
            }
        }

        fn visit_stmts(stmts: &[HirStmt], max: &mut usize) {
            for s in stmts {
                match s {
                    HirStmt::Assign(id, expr, _, _) => {
                        if id.0 + 1 > *max {
                            *max = id.0 + 1;
                        }
                        visit_expr(expr, max);
                    }
                    HirStmt::ExprStmt(expr, _, _) => visit_expr(expr, max),
                    HirStmt::Return(_) => {}
                    HirStmt::If {
                        cond,
                        then_body,
                        elseif_blocks,
                        else_body,
                        ..
                    } => {
                        visit_expr(cond, max);
                        visit_stmts(then_body, max);
                        for (cond, body) in elseif_blocks {
                            visit_expr(cond, max);
                            visit_stmts(body, max);
                        }
                        if let Some(body) = else_body {
                            visit_stmts(body, max);
                        }
                    }
                    HirStmt::While { cond, body, .. } => {
                        visit_expr(cond, max);
                        visit_stmts(body, max);
                    }
                    HirStmt::For {
                        var, expr, body, ..
                    } => {
                        if var.0 + 1 > *max {
                            *max = var.0 + 1;
                        }
                        visit_expr(expr, max);
                        visit_stmts(body, max);
                    }
                    HirStmt::Switch {
                        expr,
                        cases,
                        otherwise,
                        ..
                    } => {
                        visit_expr(expr, max);
                        for (c, b) in cases {
                            visit_expr(c, max);
                            visit_stmts(b, max);
                        }
                        if let Some(b) = otherwise {
                            visit_stmts(b, max);
                        }
                    }
                    HirStmt::TryCatch {
                        try_body,
                        catch_var,
                        catch_body,
                        ..
                    } => {
                        if let Some(v) = catch_var {
                            if v.0 + 1 > *max {
                                *max = v.0 + 1;
                            }
                        }
                        visit_stmts(try_body, max);
                        visit_stmts(catch_body, max);
                    }
                    HirStmt::Global(vars, _) | HirStmt::Persistent(vars, _) => {
                        for (v, _name) in vars {
                            if v.0 + 1 > *max {
                                *max = v.0 + 1;
                            }
                        }
                    }
                    HirStmt::AssignLValue(_, expr, _, _) => visit_expr(expr, max),
                    HirStmt::MultiAssign(vars, expr, _, _) => {
                        for v in vars.iter().flatten() {
                            if v.0 + 1 > *max {
                                *max = v.0 + 1;
                            }
                        }
                        visit_expr(expr, max);
                    }
                    HirStmt::Function { .. }
                    | HirStmt::ClassDef { .. }
                    | HirStmt::Import { .. }
                    | HirStmt::Break(_)
                    | HirStmt::Continue(_) => {}
                }
            }
        }

        visit_stmts(&prog.body, &mut max_var);
        let mut var_types = prog.var_types.clone();
        if var_types.len() < max_var {
            var_types.resize(max_var, Type::Unknown);
        }
        Self {
            instructions: Vec::new(),
            instr_spans: Vec::new(),
            call_arg_spans: Vec::new(),
            var_count: max_var,
            loop_stack: Vec::new(),
            functions: HashMap::new(),
            imports: Vec::new(),
            var_types,
            layout: None,
            entrypoint: None,
            function: None,
            body: None,
            current_span: None,
        }
    }

    pub fn compile(&mut self) -> Result<(), CompileError> {
        let Some(function) = self.function else {
            return Err(CompileError::new("compiler missing selected function"));
        };
        if self.layout.is_none() {
            return Err(CompileError::new("compiler missing VM layout"));
        }
        if !self
            .layout
            .as_ref()
            .is_some_and(|layout| layout.functions.contains_key(&function))
        {
            return Err(CompileError::new(format!(
                "missing VM layout for selected function {function:?}"
            )));
        }
        let body = self
            .body
            .clone()
            .ok_or_else(|| CompileError::new("compiler missing MIR body"))?;

        self.compile_mir_body(&body)?;
        Ok(())
    }

    fn compile_mir_body(&mut self, body: &MirBody) -> Result<(), CompileError> {
        let mut blocks = body.blocks.clone();
        blocks.sort_by_key(|block| block.id.0);

        let mut block_starts = HashMap::new();
        let mut pending_jumps: Vec<(usize, BasicBlockId, bool)> = Vec::new();

        for (block_index, block) in blocks.iter().enumerate() {
            block_starts.insert(block.id, self.instructions.len());
            for stmt in &block.statements {
                self.compile_mir_stmt(stmt)?;
            }
            match &block.terminator.kind {
                MirTerminatorKind::Goto(target) => {
                    let pc = self.emit(Instr::Jump(usize::MAX));
                    pending_jumps.push((pc, *target, false));
                }
                MirTerminatorKind::Branch {
                    cond,
                    then_block,
                    else_block,
                } => {
                    self.compile_mir_operand(cond)?;
                    let false_pc = self.emit(Instr::JumpIfFalse(usize::MAX));
                    pending_jumps.push((false_pc, *else_block, true));
                    let true_pc = self.emit(Instr::Jump(usize::MAX));
                    pending_jumps.push((true_pc, *then_block, false));
                }
                MirTerminatorKind::Switch {
                    discr,
                    cases,
                    otherwise,
                } => {
                    let discr_temp = self.alloc_temp();
                    self.compile_mir_operand(discr)?;
                    self.emit(Instr::StoreVar(discr_temp));
                    for (case, target) in cases {
                        self.emit(Instr::LoadVar(discr_temp));
                        self.compile_mir_operand(case)?;
                        self.emit(Instr::Equal);
                        let next_case_pc = self.emit(Instr::JumpIfFalse(usize::MAX));
                        let target_pc = self.emit(Instr::Jump(usize::MAX));
                        pending_jumps.push((target_pc, *target, false));
                        self.patch(next_case_pc, Instr::JumpIfFalse(self.instructions.len()));
                    }
                    let otherwise_pc = self.emit(Instr::Jump(usize::MAX));
                    pending_jumps.push((otherwise_pc, *otherwise, false));
                }
                MirTerminatorKind::Return(values) => {
                    if values.is_empty() && block_index + 1 < blocks.len() {
                        self.emit(Instr::Return);
                    } else {
                        self.compile_mir_return(&block.terminator.kind)?;
                    }
                }
                _ => return Err(CompileError::new(
                    "MIR bytecode lowering for this control-flow terminator is not implemented yet",
                )),
            }
        }

        for (pc, target, is_conditional) in pending_jumps {
            let target_pc = *block_starts
                .get(&target)
                .ok_or_else(|| CompileError::new(format!("missing MIR target block {target:?}")))?;
            if is_conditional {
                self.patch(pc, Instr::JumpIfFalse(target_pc));
            } else {
                self.patch(pc, Instr::Jump(target_pc));
            }
        }

        Ok(())
    }

    fn compile_mir_stmt(&mut self, stmt: &MirStmt) -> Result<(), CompileError> {
        let _span_guard = SpanGuard::new(self, stmt.span);
        match &stmt.kind {
            MirStmtKind::Assign { place, value } => self.compile_mir_assign(place, value),
            MirStmtKind::Expr(value) => {
                self.compile_mir_rvalue(value)?;
                self.emit(Instr::Pop);
                Ok(())
            }
            MirStmtKind::WorkspaceEffect { .. } | MirStmtKind::EnvironmentEffect(_) => Ok(()),
            MirStmtKind::PlaceMutation(_) => Ok(()),
            MirStmtKind::MultiAssign { targets, value } => {
                self.compile_mir_multi_assign(targets, value)
            }
        }
    }

    fn compile_mir_multi_assign(
        &mut self,
        targets: &runmat_mir::MirOutputTargetList,
        value: &MirRvalue,
    ) -> Result<(), CompileError> {
        match value {
            MirRvalue::Call(call) => {
                self.compile_mir_call_for_multi_assign(call, targets.targets.len())?
            }
            MirRvalue::Index { base, indexing }
                if indexing.kind == IndexKind::Brace
                    && matches!(indexing.result_context, IndexResultContext::ReadCommaList) =>
            {
                self.compile_mir_cell_expand_for_multi_assign(
                    base,
                    indexing,
                    targets.targets.len(),
                )?
            }
            _ => self.compile_mir_rvalue(value)?,
        }
        if !matches!(
            value,
            MirRvalue::Index { indexing, .. }
                if indexing.kind == IndexKind::Brace
                    && matches!(indexing.result_context, IndexResultContext::ReadCommaList)
        ) {
            self.emit(Instr::Unpack(targets.targets.len()));
        }
        for target in targets.targets.iter().rev() {
            match target {
                MirOutputTarget::Place(place) => {
                    let slot = self.mir_place_slot(place)?;
                    self.emit(Instr::StoreVar(slot));
                }
                MirOutputTarget::Discard => {
                    self.emit(Instr::Pop);
                }
                MirOutputTarget::VarargoutExpansion => {
                    return Err(self.compile_error(
                        "MIR bytecode lowering for varargout expansion is not implemented yet",
                    ))
                }
            }
        }
        Ok(())
    }

    fn compile_mir_cell_expand_for_multi_assign(
        &mut self,
        base: &MirOperand,
        indexing: &MirIndexing,
        output_count: usize,
    ) -> Result<(), CompileError> {
        self.compile_mir_operand(base)?;
        let mut index_count = 0usize;
        let mut expand_all = false;
        for component in &indexing.components {
            match component {
                MirIndexComponent::Colon => expand_all = true,
                MirIndexComponent::Expr(operand) => {
                    self.compile_mir_operand(operand)?;
                    index_count += 1;
                }
                MirIndexComponent::End { offset, .. } if *offset <= 0 => {
                    self.emit(Instr::LoadConst(if *offset == 0 {
                        -0.0
                    } else {
                        *offset as f64
                    }));
                    index_count += 1;
                }
                _ => {
                    return Err(self.compile_error(
                        "MIR bytecode lowering for this slice index is not implemented yet",
                    ))
                }
            }
        }
        if expand_all {
            self.emit(Instr::IndexCellExpand(0, output_count));
        } else {
            self.emit(Instr::IndexCellExpand(index_count, output_count));
        }
        Ok(())
    }

    fn compile_mir_call_for_multi_assign(
        &mut self,
        call: &MirCall,
        output_count: usize,
    ) -> Result<(), CompileError> {
        match call.requested_outputs {
            RequestedOutputCount::Exactly(count) | RequestedOutputCount::AtLeast(count)
                if count == output_count => {}
            RequestedOutputCount::UnknownDynamic => {}
            _ => {
                return Err(
                    self.compile_error("MIR multi-assign call output count does not match targets")
                )
            }
        }
        let (specs, has_expansion) = self.mir_call_arg_specs(&call.args);
        match &call.callee {
            MirCallee::Static(HirCallableRef::Function(function)) => {
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                if has_expansion {
                    self.emit(Instr::CallSemanticFunctionExpandMultiOutput(
                        *function,
                        specs,
                        output_count,
                    ));
                    return Ok(());
                }
                self.emit(Instr::CallSemanticFunctionMulti(
                    *function,
                    call.args.len(),
                    output_count,
                ));
            }
            MirCallee::Dynamic(_) => {
                return Err(self.compile_error(
                    "MIR bytecode lowering for dynamic multi-output calls is not implemented yet",
                ))
            }
            MirCallee::Static(_) => {
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                let name = self.mir_builtin_call_name(call)?;
                if has_expansion {
                    self.emit(Instr::CallBuiltinExpandMulti(name, specs));
                } else {
                    self.emit(Instr::CallBuiltin(name, call.args.len()));
                }
            }
        }
        Ok(())
    }

    fn compile_mir_assign(
        &mut self,
        place: &MirPlace,
        value: &MirRvalue,
    ) -> Result<(), CompileError> {
        match place {
            MirPlace::Local(_) | MirPlace::Binding(_) => {
                self.compile_mir_rvalue(value)?;
                let slot = self.mir_place_slot(place)?;
                self.emit(Instr::StoreVar(slot));
                Ok(())
            }
            MirPlace::Index(base, indexing) => {
                let base_slot = self.mir_place_slot(base)?;
                self.emit(Instr::LoadVar(base_slot));
                match indexing.kind {
                    IndexKind::Paren => {
                        if self.mir_indexing_is_simple_expr_indices(indexing) {
                            self.compile_mir_index_components(
                                indexing,
                                IndexResultContext::AssignmentTarget,
                            )?;
                            self.compile_mir_rvalue(value)?;
                            self.emit(Instr::StoreIndex(indexing.components.len()));
                        } else {
                            if self.indexing_needs_slice_expr(indexing) {
                                let (numeric_count, colon_mask, end_mask, end_numeric_exprs) =
                                    self.compile_mir_slice_expr_components(indexing)?;
                                self.compile_mir_rvalue(value)?;
                                self.emit(Instr::StoreSliceExpr {
                                    dims: indexing.components.len(),
                                    numeric_count,
                                    colon_mask,
                                    end_mask,
                                    range_dims: Vec::new(),
                                    range_has_step: Vec::new(),
                                    range_start_exprs: Vec::new(),
                                    range_step_exprs: Vec::new(),
                                    range_end_exprs: Vec::new(),
                                    end_numeric_exprs,
                                });
                            } else {
                                let (numeric_count, colon_mask, end_mask) =
                                    self.compile_mir_slice_components(indexing)?;
                                self.compile_mir_rvalue(value)?;
                                self.emit(Instr::StoreSlice(
                                    indexing.components.len(),
                                    numeric_count,
                                    colon_mask,
                                    end_mask,
                                ));
                            }
                        }
                    }
                    IndexKind::Brace => {
                        self.compile_mir_cell_index_components(
                            indexing,
                            IndexResultContext::AssignmentTarget,
                        )?;
                        self.compile_mir_rvalue(value)?;
                        self.emit(Instr::StoreIndexCell(indexing.components.len()));
                    }
                    IndexKind::Dot => {
                        return Err(self.compile_error(
                            "MIR bytecode lowering for dot assignment is not implemented yet",
                        ))
                    }
                };
                self.emit(Instr::StoreVar(base_slot));
                Ok(())
            }
            MirPlace::Member(base, member) => {
                self.compile_mir_member_base_for_assignment(base)?;
                self.compile_mir_rvalue(value)?;
                self.emit(Instr::StoreMemberOrInit(member.0.clone()));
                self.emit_store_back_mir_member_chain(base)
            }
            MirPlace::DynamicMember(base, member) => {
                self.compile_mir_member_base_for_assignment(base)?;
                self.compile_mir_operand(member)?;
                self.compile_mir_rvalue(value)?;
                self.emit(Instr::StoreMemberDynamicOrInit);
                self.emit_store_back_mir_member_chain(base)
            }
        }
    }

    fn compile_mir_member_base_for_assignment(
        &mut self,
        base: &MirPlace,
    ) -> Result<(), CompileError> {
        match base {
            MirPlace::Index(parent, indexing) => {
                self.compile_mir_place_read(parent)?;
                self.compile_mir_index_after_base(indexing)
            }
            MirPlace::Member(parent, field) => {
                self.compile_mir_member_base_for_assignment(parent)?;
                self.emit(Instr::LoadMemberOrInit(field.0.clone()));
                Ok(())
            }
            MirPlace::DynamicMember(parent, name) => {
                self.compile_mir_member_base_for_assignment(parent)?;
                self.compile_mir_operand(name)?;
                self.emit(Instr::LoadMemberDynamicOrInit);
                Ok(())
            }
            _ => {
                let slot = self.mir_place_slot(base)?;
                self.emit(Instr::LoadVar(slot));
                Ok(())
            }
        }
    }

    fn emit_store_back_mir_member_chain(&mut self, base: &MirPlace) -> Result<(), CompileError> {
        match base {
            MirPlace::Local(_) | MirPlace::Binding(_) => {
                let slot = self.mir_place_slot(base)?;
                self.emit(Instr::StoreVar(slot));
                Ok(())
            }
            MirPlace::Member(parent, field) => {
                self.compile_mir_member_base_for_assignment(parent)?;
                self.emit(Instr::Swap);
                self.emit(Instr::StoreMemberOrInit(field.0.clone()));
                self.emit_store_back_mir_member_chain(parent)
            }
            MirPlace::DynamicMember(parent, name) => {
                let tmp = self.alloc_temp();
                self.emit(Instr::StoreVar(tmp));
                self.compile_mir_member_base_for_assignment(parent)?;
                self.compile_mir_operand(name)?;
                self.emit(Instr::LoadVar(tmp));
                self.emit(Instr::StoreMemberDynamicOrInit);
                self.emit_store_back_mir_member_chain(parent)
            }
            MirPlace::Index(parent, indexing) => {
                let tmp = self.alloc_temp();
                self.emit(Instr::StoreVar(tmp));
                self.compile_mir_place_read(parent)?;
                self.compile_mir_store_indexed_value_from_temp(indexing, tmp)?;
                self.emit_store_back_mir_member_chain(parent)
            }
        }
    }

    fn compile_mir_place_read(&mut self, place: &MirPlace) -> Result<(), CompileError> {
        match place {
            MirPlace::Local(_) | MirPlace::Binding(_) => {
                let slot = self.mir_place_slot(place)?;
                self.emit(Instr::LoadVar(slot));
                Ok(())
            }
            MirPlace::Member(base, member) => {
                self.compile_mir_place_read(base)?;
                self.emit(Instr::LoadMember(member.0.clone()));
                Ok(())
            }
            MirPlace::DynamicMember(base, member) => {
                self.compile_mir_place_read(base)?;
                self.compile_mir_operand(member)?;
                self.emit(Instr::LoadMemberDynamic);
                Ok(())
            }
            MirPlace::Index(base, indexing) => {
                self.compile_mir_place_read(base)?;
                self.compile_mir_index_after_base(indexing)
            }
        }
    }

    fn compile_mir_index_after_base(&mut self, indexing: &MirIndexing) -> Result<(), CompileError> {
        match indexing.kind {
            IndexKind::Paren => {
                if self.mir_indexing_is_simple_expr_indices(indexing) {
                    self.compile_mir_index_components_any_context(indexing)?;
                    self.emit(Instr::Index(indexing.components.len()));
                } else {
                    self.compile_mir_slice_index(indexing)?;
                }
            }
            IndexKind::Brace => {
                self.compile_mir_cell_index_components_any_context(indexing)?;
                self.emit(Instr::IndexCell(indexing.components.len()));
            }
            IndexKind::Dot => {
                return Err(self.compile_error(
                    "MIR bytecode lowering for dot indexing is not implemented yet",
                ))
            }
        }
        Ok(())
    }

    fn mir_indexing_is_simple_expr_indices(&self, indexing: &MirIndexing) -> bool {
        indexing
            .components
            .iter()
            .all(|component| matches!(component, MirIndexComponent::Expr(_)))
            && !indexing.components.iter().any(|component| {
                matches!(component, MirIndexComponent::Expr(MirOperand::Local(local)) if self.mir_local_is_colon(*local))
                    || matches!(component, MirIndexComponent::Expr(operand) if self.mir_operand_end_expr(operand).is_some())
                    || matches!(component, MirIndexComponent::Expr(operand) if self.mir_operand_is_non_scalar_index(operand))
            })
    }

    fn mir_operand_is_non_scalar_index(&self, operand: &MirOperand) -> bool {
        match operand {
            MirOperand::Local(local) => self.mir_local_matches_rvalue(*local, |value| {
                self.mir_rvalue_is_non_scalar_index(value)
            }),
            _ => false,
        }
    }

    fn mir_rvalue_is_non_scalar_index(&self, value: &MirRvalue) -> bool {
        match value {
            MirRvalue::Range { .. } => true,
            MirRvalue::Aggregate { rows, cols, .. } => rows.saturating_mul(*cols) != 1,
            MirRvalue::Use(operand) => self.mir_operand_is_non_scalar_index(operand),
            _ => false,
        }
    }

    fn compile_mir_store_indexed_value_from_temp(
        &mut self,
        indexing: &MirIndexing,
        tmp: usize,
    ) -> Result<(), CompileError> {
        match indexing.kind {
            IndexKind::Paren if self.mir_indexing_is_simple_expr_indices(indexing) => {
                self.compile_mir_index_components_any_context(indexing)?;
                self.emit(Instr::LoadVar(tmp));
                self.emit(Instr::StoreIndex(indexing.components.len()));
                Ok(())
            }
            IndexKind::Brace => {
                self.compile_mir_cell_index_components_any_context(indexing)?;
                self.emit(Instr::LoadVar(tmp));
                self.emit(Instr::StoreIndexCell(indexing.components.len()));
                Ok(())
            }
            _ => Err(self.compile_error(
                "MIR bytecode lowering for indexed member store-back is not implemented yet",
            )),
        }
    }

    fn compile_mir_return(&mut self, terminator: &MirTerminatorKind) -> Result<(), CompileError> {
        match terminator {
            MirTerminatorKind::Return(values) => match values.len() {
                0 => Ok(()),
                1 => {
                    self.compile_mir_operand(&values[0])?;
                    self.emit(Instr::ReturnValue);
                    Ok(())
                }
                _ => {
                    self.emit(Instr::Return);
                    Ok(())
                }
            },
            _ => Err(CompileError::new(
                "MIR bytecode lowering for control-flow terminators is not implemented yet",
            )),
        }
    }

    fn compile_mir_rvalue(&mut self, value: &MirRvalue) -> Result<(), CompileError> {
        match value {
            MirRvalue::Use(operand) => self.compile_mir_operand(operand),
            MirRvalue::Unary(op, operand) => {
                self.compile_mir_operand(operand)?;
                match op {
                    OperatorKind::UnaryPlus => self.emit(Instr::UPlus),
                    OperatorKind::UnaryMinus => self.emit(Instr::Neg),
                    OperatorKind::Not => self.emit(Instr::CallBuiltin("not".to_string(), 1)),
                    OperatorKind::Transpose => self.emit(Instr::Transpose),
                    OperatorKind::ConjugateTranspose => self.emit(Instr::ConjugateTranspose),
                    _ => {
                        return Err(self
                            .compile_error(format!("operator {op:?} is not a MIR unary operator")))
                    }
                };
                Ok(())
            }
            MirRvalue::Binary(left, op, right) => {
                match op {
                    OperatorKind::ShortCircuitAnd => {
                        return self.compile_mir_short_circuit_and(left, right);
                    }
                    OperatorKind::ShortCircuitOr => {
                        return self.compile_mir_short_circuit_or(left, right);
                    }
                    _ => {}
                }
                self.compile_mir_operand(left)?;
                self.compile_mir_operand(right)?;
                match op {
                    OperatorKind::Add => self.emit(Instr::Add),
                    OperatorKind::Subtract => self.emit(Instr::Sub),
                    OperatorKind::MatrixMultiply => self.emit(Instr::Mul),
                    OperatorKind::Mrdivide => self.emit(Instr::RightDiv),
                    OperatorKind::Mldivide => self.emit(Instr::LeftDiv),
                    OperatorKind::MatrixPower => self.emit(Instr::Pow),
                    OperatorKind::ElementwiseMultiply => self.emit(Instr::ElemMul),
                    OperatorKind::ElementwiseDivide => self.emit(Instr::ElemDiv),
                    OperatorKind::ElementwiseLeftDivide => self.emit(Instr::ElemLeftDiv),
                    OperatorKind::ElementwisePower => self.emit(Instr::ElemPow),
                    OperatorKind::Equal => self.emit(Instr::Equal),
                    OperatorKind::NotEqual => self.emit(Instr::NotEqual),
                    OperatorKind::Less => self.emit(Instr::Less),
                    OperatorKind::LessEqual => self.emit(Instr::LessEqual),
                    OperatorKind::Greater => self.emit(Instr::Greater),
                    OperatorKind::GreaterEqual => self.emit(Instr::GreaterEqual),
                    OperatorKind::ElementwiseAnd => {
                        self.emit(Instr::CallBuiltin("and".to_string(), 2))
                    }
                    OperatorKind::ElementwiseOr => {
                        self.emit(Instr::CallBuiltin("or".to_string(), 2))
                    }
                    _ => {
                        return Err(self.compile_error(format!(
                            "operator {op:?} is not supported in primary MIR lowering yet"
                        )))
                    }
                };
                Ok(())
            }
            MirRvalue::Range { start, step, end } => {
                self.compile_mir_operand(start)?;
                if let Some(step) = step {
                    self.compile_mir_operand(step)?;
                    self.compile_mir_operand(end)?;
                    self.emit(Instr::CreateRange(true));
                } else {
                    self.compile_mir_operand(end)?;
                    self.emit(Instr::CreateRange(false));
                }
                Ok(())
            }
            MirRvalue::Call(call) => self.compile_mir_call(call),
            MirRvalue::Aggregate {
                kind,
                rows,
                cols,
                elements,
            } => self.compile_mir_aggregate(kind, *rows, *cols, elements),
            MirRvalue::Index { base, indexing } => self.compile_mir_index(base, indexing),
            MirRvalue::Member { base, member } => {
                self.compile_mir_operand(base)?;
                self.emit(Instr::LoadMember(member.0.clone()));
                Ok(())
            }
            MirRvalue::DynamicMember { base, member } => {
                self.compile_mir_operand(base)?;
                self.compile_mir_operand(member)?;
                self.emit(Instr::LoadMemberDynamic);
                Ok(())
            }
            MirRvalue::MetaClass(name) => {
                self.emit(Instr::LoadString(
                    name.0
                        .iter()
                        .map(|segment| segment.0.as_str())
                        .collect::<Vec<_>>()
                        .join("."),
                ));
                Ok(())
            }
            MirRvalue::Colon => {
                self.emit(Instr::LoadConst(0.0));
                Ok(())
            }
            MirRvalue::End => {
                self.emit(Instr::LoadConst(-0.0));
                Ok(())
            }
            _ => {
                Err(self
                    .compile_error("MIR bytecode lowering for this rvalue is not implemented yet"))
            }
        }
    }

    fn compile_mir_truthy_operand(&mut self, operand: &MirOperand) -> Result<(), CompileError> {
        self.compile_mir_operand(operand)?;
        self.emit(Instr::LoadConst(0.0));
        self.emit(Instr::NotEqual);
        Ok(())
    }

    fn compile_mir_short_circuit_and(
        &mut self,
        left: &MirOperand,
        right: &MirOperand,
    ) -> Result<(), CompileError> {
        self.compile_mir_truthy_operand(left)?;
        let jump_false = self.emit(Instr::JumpIfFalse(usize::MAX));
        self.compile_mir_truthy_operand(right)?;
        let end = self.emit(Instr::Jump(usize::MAX));
        let false_pc = self.instructions.len();
        self.patch(jump_false, Instr::JumpIfFalse(false_pc));
        self.emit(Instr::LoadConst(0.0));
        let end_pc = self.instructions.len();
        self.patch(end, Instr::Jump(end_pc));
        Ok(())
    }

    fn compile_mir_short_circuit_or(
        &mut self,
        left: &MirOperand,
        right: &MirOperand,
    ) -> Result<(), CompileError> {
        self.compile_mir_truthy_operand(left)?;
        let jump_false = self.emit(Instr::JumpIfFalse(usize::MAX));
        self.emit(Instr::LoadConst(1.0));
        let end = self.emit(Instr::Jump(usize::MAX));
        let right_pc = self.instructions.len();
        self.patch(jump_false, Instr::JumpIfFalse(right_pc));
        self.compile_mir_truthy_operand(right)?;
        let end_pc = self.instructions.len();
        self.patch(end, Instr::Jump(end_pc));
        Ok(())
    }

    fn compile_mir_call(&mut self, call: &MirCall) -> Result<(), CompileError> {
        match call.requested_outputs {
            RequestedOutputCount::Zero
            | RequestedOutputCount::One
            | RequestedOutputCount::UnknownDynamic
            | RequestedOutputCount::Exactly(1)
            | RequestedOutputCount::AtLeast(1) => {}
            RequestedOutputCount::Exactly(count) | RequestedOutputCount::AtLeast(count) => {
                return Err(self.compile_error(format!(
                    "MIR bytecode lowering for {count} call outputs is not implemented yet"
                )))
            }
        }

        let (specs, has_expansion) = self.mir_call_arg_specs(&call.args);
        match &call.callee {
            MirCallee::Static(HirCallableRef::Function(function)) => {
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                if has_expansion {
                    self.emit(Instr::CallSemanticFunctionExpandMulti(*function, specs));
                } else {
                    self.emit(Instr::CallSemanticFunction(*function, call.args.len()));
                }
            }
            MirCallee::Dynamic(callee) => {
                self.compile_mir_operand(callee)?;
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                if has_expansion {
                    self.emit(Instr::CallFevalExpandMulti(specs));
                } else {
                    self.emit(Instr::CallFeval(call.args.len()));
                }
            }
            MirCallee::Static(_) => {
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                let name = self.mir_builtin_call_name(call)?;
                if has_expansion {
                    self.emit(Instr::CallBuiltinExpandMulti(name, specs));
                } else {
                    self.emit(Instr::CallBuiltin(name, call.args.len()));
                }
            }
        }
        Ok(())
    }

    fn mir_call_arg_specs(&self, args: &[MirCallArg]) -> (Vec<ArgSpec>, bool) {
        let mut has_expansion = false;
        let specs = args
            .iter()
            .map(|arg| match arg {
                MirCallArg::Single(_) => ArgSpec {
                    is_expand: false,
                    num_indices: 0,
                    expand_all: false,
                },
                MirCallArg::Expansion {
                    indices,
                    expand_all,
                    ..
                } => {
                    has_expansion = true;
                    ArgSpec {
                        is_expand: true,
                        num_indices: indices.len(),
                        expand_all: *expand_all,
                    }
                }
            })
            .collect();
        (specs, has_expansion)
    }

    fn compile_mir_call_arg(&mut self, arg: &MirCallArg) -> Result<(), CompileError> {
        match arg {
            MirCallArg::Single(operand) => self.compile_mir_operand(operand),
            MirCallArg::Expansion { base, indices, .. } => {
                self.compile_mir_operand(base)?;
                for index in indices {
                    self.compile_mir_operand(index)?;
                }
                Ok(())
            }
        }
    }

    fn mir_builtin_call_name(&self, call: &MirCall) -> Result<String, CompileError> {
        let candidate = match &call.callee {
            MirCallee::Static(HirCallableRef::Builtin(id)) => id.0.clone(),
            MirCallee::Static(HirCallableRef::Unresolved(name)) if name.0.len() == 1 => {
                name.0[0].0.clone()
            }
            _ => {
                return Err(CompileError::new(
                    "MIR bytecode lowering for this call callee is not implemented yet",
                ))
            }
        };
        runmat_builtins::builtin_function_by_name(&candidate)
            .map(|builtin| builtin.name.to_string())
            .ok_or_else(|| CompileError::new(format!("unknown builtin function {candidate}")))
    }

    fn compile_mir_aggregate(
        &mut self,
        kind: &MirAggregateKind,
        rows: usize,
        cols: usize,
        elements: &[MirOperand],
    ) -> Result<(), CompileError> {
        if rows.checked_mul(cols) != Some(elements.len()) {
            return Err(
                self.compile_error("MIR aggregate shape does not match aggregate element count")
            );
        }

        for element in elements {
            self.compile_mir_operand(element)?;
        }
        match kind {
            MirAggregateKind::Tensor => self.emit(Instr::CreateMatrix(rows, cols)),
            MirAggregateKind::Cell => self.emit(Instr::CreateCell2D(rows, cols)),
            MirAggregateKind::Struct | MirAggregateKind::ObjectArray(_) => {
                return Err(self.compile_error(
                    "MIR bytecode lowering for this aggregate kind is not implemented yet",
                ))
            }
        };
        Ok(())
    }

    fn compile_mir_index(
        &mut self,
        base: &MirOperand,
        indexing: &MirIndexing,
    ) -> Result<(), CompileError> {
        if !matches!(
            indexing.result_context,
            IndexResultContext::ReadSingle | IndexResultContext::ReadCommaList
        ) {
            return Err(self.compile_error(
                "MIR bytecode lowering for this indexing form is not implemented yet",
            ));
        }

        self.compile_mir_operand(base)?;
        match indexing.kind {
            IndexKind::Paren => {
                if self.mir_indexing_is_simple_expr_indices(indexing) {
                    self.compile_mir_index_components(indexing, IndexResultContext::ReadSingle)?;
                    self.emit(Instr::Index(indexing.components.len()));
                } else {
                    self.compile_mir_slice_index(indexing)?;
                }
            }
            IndexKind::Brace => {
                self.compile_mir_cell_index_components(indexing, indexing.result_context.clone())?;
                self.emit(Instr::IndexCell(indexing.components.len()));
            }
            IndexKind::Dot => {
                return Err(self.compile_error(
                    "MIR bytecode lowering for dot indexing is not implemented yet",
                ))
            }
        };
        Ok(())
    }

    fn compile_mir_index_components(
        &mut self,
        indexing: &MirIndexing,
        expected_context: IndexResultContext,
    ) -> Result<(), CompileError> {
        if !mir_indexing_context_matches(indexing.result_context.clone(), expected_context) {
            return Err(self.compile_error(
                "MIR bytecode lowering for this indexing form is not implemented yet",
            ));
        }
        for component in &indexing.components {
            let MirIndexComponent::Expr(operand) = component else {
                return Err(self.compile_error(
                    "MIR bytecode lowering for non-expression indices is not implemented yet",
                ));
            };
            self.compile_mir_operand(operand)?;
        }
        Ok(())
    }

    fn compile_mir_index_components_any_context(
        &mut self,
        indexing: &MirIndexing,
    ) -> Result<(), CompileError> {
        for component in &indexing.components {
            let MirIndexComponent::Expr(operand) = component else {
                return Err(self.compile_error(
                    "MIR bytecode lowering for non-expression indices is not implemented yet",
                ));
            };
            self.compile_mir_operand(operand)?;
        }
        Ok(())
    }

    fn compile_mir_cell_index_components(
        &mut self,
        indexing: &MirIndexing,
        expected_context: IndexResultContext,
    ) -> Result<(), CompileError> {
        if !mir_indexing_context_matches(indexing.result_context.clone(), expected_context) {
            return Err(self.compile_error(
                "MIR bytecode lowering for this indexing form is not implemented yet",
            ));
        }
        for component in &indexing.components {
            match component {
                MirIndexComponent::Expr(operand) => self.compile_mir_operand(operand)?,
                MirIndexComponent::End { offset, .. } if *offset == 0 => {
                    self.emit(Instr::LoadConst(-0.0));
                }
                _ => {
                    return Err(self.compile_error(
                        "MIR bytecode lowering for non-expression indices is not implemented yet",
                    ))
                }
            }
        }
        Ok(())
    }

    fn compile_mir_cell_index_components_any_context(
        &mut self,
        indexing: &MirIndexing,
    ) -> Result<(), CompileError> {
        for component in &indexing.components {
            match component {
                MirIndexComponent::Expr(operand) => self.compile_mir_operand(operand)?,
                MirIndexComponent::End { offset, .. } if *offset == 0 => {
                    self.emit(Instr::LoadConst(-0.0));
                }
                _ => {
                    return Err(self.compile_error(
                        "MIR bytecode lowering for non-expression indices is not implemented yet",
                    ))
                }
            }
        }
        Ok(())
    }

    fn compile_mir_slice_index(&mut self, indexing: &MirIndexing) -> Result<(), CompileError> {
        if self.indexing_needs_slice_expr(indexing) {
            let (numeric_count, colon_mask, end_mask, end_numeric_exprs) =
                self.compile_mir_slice_expr_components(indexing)?;
            self.emit(Instr::IndexSliceExpr {
                dims: indexing.components.len(),
                numeric_count,
                colon_mask,
                end_mask,
                range_dims: Vec::new(),
                range_has_step: Vec::new(),
                range_start_exprs: Vec::new(),
                range_step_exprs: Vec::new(),
                range_end_exprs: Vec::new(),
                end_numeric_exprs,
            });
            return Ok(());
        }

        let (numeric_count, colon_mask, end_mask) = self.compile_mir_slice_components(indexing)?;
        self.emit(Instr::IndexSlice(
            indexing.components.len(),
            numeric_count,
            colon_mask,
            end_mask,
        ));
        Ok(())
    }

    fn indexing_needs_slice_expr(&self, indexing: &MirIndexing) -> bool {
        indexing.components.iter().any(|component| match component {
            MirIndexComponent::End { offset, .. } => *offset != 0,
            MirIndexComponent::Expr(operand) => self.mir_operand_end_expr(operand).is_some(),
            _ => false,
        })
    }

    fn compile_mir_slice_components(
        &mut self,
        indexing: &MirIndexing,
    ) -> Result<(usize, u32, u32), CompileError> {
        let mut colon_mask = 0u32;
        let mut end_mask = 0u32;
        let mut numeric_count = 0usize;

        for (dim, component) in indexing.components.iter().enumerate() {
            match component {
                MirIndexComponent::Colon => colon_mask |= 1u32 << dim,
                MirIndexComponent::End { offset, .. } if *offset == 0 => end_mask |= 1u32 << dim,
                MirIndexComponent::Expr(MirOperand::Local(local))
                    if self.mir_local_is_colon(*local) =>
                {
                    colon_mask |= 1u32 << dim;
                }
                MirIndexComponent::Expr(MirOperand::Local(local))
                    if self.mir_local_is_end(*local) =>
                {
                    end_mask |= 1u32 << dim;
                }
                MirIndexComponent::Expr(operand) => {
                    self.compile_mir_operand(operand)?;
                    numeric_count += 1;
                }
                MirIndexComponent::Logical(_) | MirIndexComponent::End { .. } => {
                    return Err(self.compile_error(
                        "MIR bytecode lowering for this slice index is not implemented yet",
                    ))
                }
            }
        }

        Ok((numeric_count, colon_mask, end_mask))
    }

    fn compile_mir_slice_expr_components(
        &mut self,
        indexing: &MirIndexing,
    ) -> Result<(usize, u32, u32, Vec<(usize, EndExpr)>), CompileError> {
        let mut colon_mask = 0u32;
        let end_mask = 0u32;
        let mut numeric_count = 0usize;
        let mut end_numeric_exprs = Vec::new();

        for (dim, component) in indexing.components.iter().enumerate() {
            match component {
                MirIndexComponent::Colon => colon_mask |= 1u32 << dim,
                MirIndexComponent::End { offset, .. } => {
                    if *offset == 0 {
                        self.emit(Instr::LoadConst(0.0));
                        end_numeric_exprs.push((numeric_count, EndExpr::End));
                    } else {
                        self.emit(Instr::LoadConst(0.0));
                        end_numeric_exprs.push((numeric_count, end_expr_with_offset(*offset)));
                    }
                    numeric_count += 1;
                }
                MirIndexComponent::Expr(MirOperand::Local(local))
                    if self.mir_local_is_colon(*local) =>
                {
                    colon_mask |= 1u32 << dim;
                }
                MirIndexComponent::Expr(operand)
                    if self.mir_operand_end_expr(operand).is_some() =>
                {
                    self.emit(Instr::LoadConst(0.0));
                    end_numeric_exprs.push((
                        numeric_count,
                        self.mir_operand_end_expr(operand).ok_or_else(|| {
                            self.compile_error("MIR end expression disappeared during lowering")
                        })?,
                    ));
                    numeric_count += 1;
                }
                MirIndexComponent::Expr(operand) => {
                    self.compile_mir_operand(operand)?;
                    numeric_count += 1;
                }
                MirIndexComponent::Logical(_) => {
                    return Err(self.compile_error(
                        "MIR bytecode lowering for this slice index is not implemented yet",
                    ))
                }
            }
        }

        Ok((numeric_count, colon_mask, end_mask, end_numeric_exprs))
    }

    fn mir_local_is_colon(&self, local: runmat_mir::MirLocalId) -> bool {
        self.mir_local_matches_rvalue(local, |value| matches!(value, MirRvalue::Colon))
    }

    fn mir_local_is_end(&self, local: runmat_mir::MirLocalId) -> bool {
        self.mir_local_matches_rvalue(local, |value| matches!(value, MirRvalue::End))
    }

    fn mir_operand_end_expr(&self, operand: &MirOperand) -> Option<EndExpr> {
        self.mir_operand_end_expr_internal(operand)
            .and_then(|(expr, has_end)| has_end.then_some(expr))
    }

    fn mir_operand_end_expr_internal(&self, operand: &MirOperand) -> Option<(EndExpr, bool)> {
        match operand {
            MirOperand::Local(local) => self.mir_local_end_expr_internal(*local),
            MirOperand::Constant(MirConstant::Number(value)) => value
                .parse::<f64>()
                .ok()
                .map(|value| (EndExpr::Const(value), false)),
            _ => None,
        }
    }

    fn mir_local_end_expr_internal(
        &self,
        local: runmat_mir::MirLocalId,
    ) -> Option<(EndExpr, bool)> {
        let body = self.body.as_ref()?;
        body.blocks
            .iter()
            .flat_map(|block| block.statements.iter())
            .find_map(|stmt| match &stmt.kind {
                MirStmtKind::Assign {
                    place: MirPlace::Local(candidate),
                    value,
                } if *candidate == local => self.mir_rvalue_end_expr_internal(value),
                _ => None,
            })
    }

    fn mir_rvalue_end_expr_internal(&self, value: &MirRvalue) -> Option<(EndExpr, bool)> {
        match value {
            MirRvalue::End => Some((EndExpr::End, true)),
            MirRvalue::Use(operand) => self.mir_operand_end_expr_internal(operand),
            MirRvalue::Unary(op, operand) => {
                let (expr, has_end) = self.mir_operand_end_expr_internal(operand)?;
                match op {
                    OperatorKind::UnaryPlus => Some((EndExpr::Pos(Box::new(expr)), has_end)),
                    OperatorKind::UnaryMinus => Some((EndExpr::Neg(Box::new(expr)), has_end)),
                    _ => None,
                }
            }
            MirRvalue::Binary(left, op, right) => {
                let (left, left_has_end) = self.mir_operand_end_expr_internal(left)?;
                let (right, right_has_end) = self.mir_operand_end_expr_internal(right)?;
                let has_end = left_has_end || right_has_end;
                let expr = match op {
                    OperatorKind::Add => EndExpr::Add(Box::new(left), Box::new(right)),
                    OperatorKind::Subtract => EndExpr::Sub(Box::new(left), Box::new(right)),
                    OperatorKind::MatrixMultiply => EndExpr::Mul(Box::new(left), Box::new(right)),
                    OperatorKind::Mrdivide => EndExpr::Div(Box::new(left), Box::new(right)),
                    OperatorKind::Mldivide => EndExpr::LeftDiv(Box::new(left), Box::new(right)),
                    OperatorKind::MatrixPower => EndExpr::Pow(Box::new(left), Box::new(right)),
                    _ => return None,
                };
                Some((expr, has_end))
            }
            _ => None,
        }
    }

    fn mir_local_matches_rvalue(
        &self,
        local: runmat_mir::MirLocalId,
        predicate: impl Fn(&MirRvalue) -> bool,
    ) -> bool {
        let Some(body) = &self.body else {
            return false;
        };
        body.blocks.iter().any(|block| {
            block.statements.iter().any(|stmt| {
                matches!(
                    &stmt.kind,
                    MirStmtKind::Assign {
                        place: MirPlace::Local(candidate),
                        value,
                    } if *candidate == local && predicate(value)
                )
            })
        })
    }

    fn compile_mir_operand(&mut self, operand: &MirOperand) -> Result<(), CompileError> {
        match operand {
            MirOperand::Local(local) => {
                let slot = self.mir_local_slot(*local)?;
                self.emit(Instr::LoadVar(slot));
                Ok(())
            }
            MirOperand::Constant(MirConstant::Number(value)) => {
                let value = value
                    .parse()
                    .map_err(|_| self.compile_error(format!("invalid number literal {value:?}")))?;
                self.emit(Instr::LoadConst(value));
                Ok(())
            }
            MirOperand::Constant(MirConstant::String(value)) => {
                emit_string_literal(self, &value.0);
                Ok(())
            }
            MirOperand::Constant(MirConstant::Bool(value)) => {
                self.emit(Instr::LoadBool(*value));
                Ok(())
            }
            MirOperand::Constant(MirConstant::Symbol(name)) => {
                let name = &name.0;
                let constants = runmat_builtins::constants();
                let constant = constants
                    .iter()
                    .find(|constant| constant.name == name)
                    .ok_or_else(|| self.compile_error(format!("unknown constant {name}")))?;
                match &constant.value {
                    runmat_builtins::Value::Num(value) => self.emit(Instr::LoadConst(*value)),
                    runmat_builtins::Value::Complex(re, im) => {
                        self.emit(Instr::LoadComplex(*re, *im))
                    }
                    runmat_builtins::Value::Bool(value) => self.emit(Instr::LoadBool(*value)),
                    _ => {
                        return Err(self.compile_error(format!(
                            "constant {name} is not supported in primary MIR lowering yet"
                        )))
                    }
                };
                Ok(())
            }
            MirOperand::FunctionHandle(target) => self.compile_mir_function_handle(target),
            MirOperand::Constant(MirConstant::EmptyArray) => {
                self.emit(Instr::CreateMatrix(0, 0));
                Ok(())
            }
            MirOperand::Temp(_) => {
                Err(self
                    .compile_error("MIR bytecode lowering for this operand is not implemented yet"))
            }
        }
    }

    fn compile_mir_function_handle(
        &mut self,
        target: &runmat_hir::FunctionHandleTarget,
    ) -> Result<(), CompileError> {
        match target {
            runmat_hir::FunctionHandleTarget::Builtin(builtin) => {
                self.emit(Instr::CreateFunctionHandle(builtin.0.clone()));
                Ok(())
            }
            runmat_hir::FunctionHandleTarget::DynamicName(name) => {
                self.emit(Instr::CreateFunctionHandle(name.0.clone()));
                Ok(())
            }
            runmat_hir::FunctionHandleTarget::Anonymous(function)
            | runmat_hir::FunctionHandleTarget::Function(function) => {
                let (captures, display_name) = self
                    .layout
                    .as_ref()
                    .and_then(|layout| layout.functions.get(function))
                    .ok_or_else(|| {
                        self.compile_error(format!(
                            "missing VM layout for function handle target {function:?}"
                        ))
                    })
                    .map(|layout| (layout.captures.clone(), layout.display_name.clone()))?;
                for capture in &captures {
                    self.emit(Instr::LoadVar(capture.slot.0));
                }
                self.emit(Instr::CreateSemanticClosure(
                    *function,
                    display_name,
                    captures.len(),
                ));
                Ok(())
            }
            _ => Err(self.compile_error(
                "MIR bytecode lowering for this function handle target is not implemented yet",
            )),
        }
    }

    fn mir_place_slot(&self, place: &MirPlace) -> Result<usize, CompileError> {
        match place {
            MirPlace::Local(local) => self.mir_local_slot(*local),
            MirPlace::Binding(binding) => self.binding_slot(*binding),
            _ => Err(CompileError::new(
                "MIR bytecode lowering for this assignment place is not implemented yet",
            )),
        }
    }

    fn mir_local_slot(&self, local: runmat_mir::MirLocalId) -> Result<usize, CompileError> {
        let function = self
            .function
            .ok_or_else(|| CompileError::new("compiler missing selected function"))?;
        self.layout
            .as_ref()
            .and_then(|layout| layout.functions.get(&function))
            .and_then(|layout| layout.mir_local_slots.get(&local))
            .map(|slot| slot.0)
            .ok_or_else(|| CompileError::new(format!("missing VM slot for MIR local {local:?}")))
    }

    fn binding_slot(&self, binding: BindingId) -> Result<usize, CompileError> {
        let function = self
            .function
            .ok_or_else(|| CompileError::new("compiler missing selected function"))?;
        self.layout
            .as_ref()
            .and_then(|layout| layout.functions.get(&function))
            .and_then(|layout| layout.binding_slots.get(&binding))
            .map(|slot| slot.0)
            .ok_or_else(|| CompileError::new(format!("missing VM slot for binding {binding:?}")))
    }

    fn ensure_var(&mut self, id: usize) {
        if id + 1 > self.var_count {
            self.var_count = id + 1;
        }
        while self.var_types.len() <= id {
            self.var_types.push(Type::Unknown);
        }
    }

    pub(crate) fn alloc_temp(&mut self) -> usize {
        let id = self.var_count;
        self.var_count += 1;
        if self.var_types.len() <= id {
            self.var_types.push(Type::Unknown);
        }
        id
    }

    pub fn emit(&mut self, instr: Instr) -> usize {
        match &instr {
            Instr::LoadVar(id) | Instr::StoreVar(id) => self.ensure_var(*id),
            _ => {}
        }
        let pc = self.instructions.len();
        self.instructions.push(instr);
        let span = self.current_span.unwrap_or_default();
        self.instr_spans.push(span);
        self.call_arg_spans.push(None);
        pc
    }

    pub(crate) fn emit_call_with_arg_spans(
        &mut self,
        instr: Instr,
        arg_spans: &[runmat_hir::Span],
    ) -> usize {
        let pc = self.emit(instr);
        if !arg_spans.is_empty() {
            if let Some(slot) = self.call_arg_spans.get_mut(pc) {
                *slot = Some(arg_spans.to_vec());
            }
        }
        pc
    }

    pub fn patch(&mut self, idx: usize, instr: Instr) {
        self.instructions[idx] = instr;
    }

    pub(crate) fn compile_error(&self, message: impl Into<String>) -> CompileError {
        let mut err = CompileError::new(message);
        if let Some(span) = self.current_span {
            err = err.with_span(span);
        }
        err
    }

    pub fn compile_program_legacy(&mut self, prog: &HirProgram) -> Result<(), CompileError> {
        // Validate imports early for duplicate/specific-name ambiguities
        runmat_hir::validate_imports(prog)?;
        // Validate class definitions for attribute correctness and name conflicts
        runmat_hir::validate_classdefs(prog)?;
        // Pre-collect imports (both wildcard and specific) for name resolution
        for stmt in &prog.body {
            let _span_guard = SpanGuard::new(self, stmt.span());
            if let HirStmt::Import { path, wildcard, .. } = stmt {
                self.imports.push((path.clone(), *wildcard));
                self.emit(Instr::RegisterImport {
                    path: path.clone(),
                    wildcard: *wildcard,
                });
            }
            if let HirStmt::Global(vars, _) = stmt {
                let ids: Vec<usize> = vars.iter().map(|(v, _n)| v.0).collect();
                let names: Vec<String> = vars.iter().map(|(_v, n)| n.clone()).collect();
                self.emit(Instr::DeclareGlobalNamed(ids, names));
            }
            if let HirStmt::Persistent(vars, _) = stmt {
                let ids: Vec<usize> = vars.iter().map(|(v, _n)| v.0).collect();
                let names: Vec<String> = vars.iter().map(|(_v, n)| n.clone()).collect();
                self.emit(Instr::DeclarePersistentNamed(ids, names));
            }
        }
        for stmt in &prog.body {
            if !matches!(
                stmt,
                HirStmt::Import { .. } | HirStmt::Global(_, _) | HirStmt::Persistent(_, _)
            ) {
                self.compile_stmt(stmt)?;
            }
        }
        Ok(())
    }

    pub fn compile_stmt(&mut self, stmt: &HirStmt) -> Result<(), CompileError> {
        let _span_guard = SpanGuard::new(self, stmt.span());
        self.compile_stmt_impl(stmt)
    }

    pub fn compile_expr(&mut self, expr: &HirExpr) -> Result<(), CompileError> {
        let _span_guard = SpanGuard::new(self, expr.span);
        self.compile_expr_impl(expr)
    }
}

fn emit_string_literal(compiler: &mut Compiler, value: &str) {
    if value.starts_with('"') && value.ends_with('"') && value.len() >= 2 {
        let inner = &value[1..value.len() - 1];
        compiler.emit(Instr::LoadString(inner.replace("\"\"", "\"")));
    } else if value.starts_with('\'') && value.ends_with('\'') && value.len() >= 2 {
        let inner = &value[1..value.len() - 1];
        compiler.emit(Instr::LoadCharRow(inner.replace("''", "'")));
    } else {
        compiler.emit(Instr::LoadString(value.to_string()));
    }
}
