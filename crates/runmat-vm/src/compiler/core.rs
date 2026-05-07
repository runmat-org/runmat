use crate::compiler::CompileError;
use crate::functions::UserFunction;
use crate::instr::{EmitLabel, Instr};
use crate::layout::VmAssemblyLayout;
use runmat_builtins::{self, Type};
use runmat_hir::{
    BindingId, EntrypointId, FunctionId, HirAssembly, HirCallableRef, IndexKind,
    IndexResultContext, LegacyHirExpr as HirExpr, LegacyHirExprKind as HirExprKind,
    LegacyHirProgram as HirProgram, LegacyHirStmt as HirStmt, OperatorKind, RequestedOutputCount,
};
use runmat_mir::{
    BasicBlockId, MirAggregateKind, MirAssembly, MirBody, MirCall, MirCallArg, MirConstant,
    MirIndexComponent, MirIndexing, MirOperand, MirPlace, MirRvalue, MirStmt, MirStmtKind,
    MirTerminatorKind,
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
            MirStmtKind::MultiAssign { .. } => Err(self
                .compile_error("MIR bytecode lowering for this statement is not implemented yet")),
        }
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
                self.compile_mir_index_components(indexing, IndexResultContext::AssignmentTarget)?;
                self.compile_mir_rvalue(value)?;
                self.emit(Instr::StoreIndex(indexing.components.len()));
                self.emit(Instr::StoreVar(base_slot));
                Ok(())
            }
            _ => Err(self.compile_error(
                "MIR bytecode lowering for this assignment place is not implemented yet",
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
                _ => Err(CompileError::new(
                    "MIR bytecode lowering for multi-value returns is not implemented yet",
                )),
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
            _ => {
                Err(self
                    .compile_error("MIR bytecode lowering for this rvalue is not implemented yet"))
            }
        }
    }

    fn compile_mir_call(&mut self, call: &MirCall) -> Result<(), CompileError> {
        let name = self.mir_builtin_call_name(call)?;
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

        for arg in &call.args {
            let MirCallArg::Single(operand) = arg else {
                return Err(self.compile_error(
                    "MIR bytecode lowering for expanded call arguments is not implemented yet",
                ));
            };
            self.compile_mir_operand(operand)?;
        }
        self.emit(Instr::CallBuiltin(name, call.args.len()));
        Ok(())
    }

    fn mir_builtin_call_name(&self, call: &MirCall) -> Result<String, CompileError> {
        let candidate = match &call.callee {
            HirCallableRef::Builtin(id) => id.0.clone(),
            HirCallableRef::Unresolved(name) if name.0.len() == 1 => name.0[0].0.clone(),
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
                self.compile_mir_index_components(indexing, IndexResultContext::ReadSingle)?;
                self.emit(Instr::Index(indexing.components.len()));
            }
            IndexKind::Brace => {
                self.compile_mir_index_components(indexing, indexing.result_context.clone())?;
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
        if indexing.result_context != expected_context {
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
                self.emit(Instr::LoadString(value.0.clone()));
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
            MirOperand::Constant(MirConstant::EmptyArray)
            | MirOperand::FunctionHandle(_)
            | MirOperand::Temp(_) => {
                Err(self
                    .compile_error("MIR bytecode lowering for this operand is not implemented yet"))
            }
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
