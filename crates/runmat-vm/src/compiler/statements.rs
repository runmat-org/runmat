//! Statement lowering.

use crate::compiler::core::{Compiler, LoopLabels};
use crate::compiler::idioms;
use crate::compiler::CompileError;
use crate::instr::{EmitLabel, Instr};
use runmat_hir::{HirExpr, HirExprKind, HirStmt};

fn expr_supports_inline_emit(expr: &HirExpr) -> bool {
    match &expr.kind {
        HirExprKind::FuncCall(name, _) => !runmat_builtins::suppresses_auto_output(name),
        _ => true,
    }
}

fn label_for_expr(expr: &HirExpr) -> EmitLabel {
    if let HirExprKind::Var(var_id) = &expr.kind {
        EmitLabel::Var(var_id.0)
    } else {
        EmitLabel::Ans
    }
}

impl Compiler {
    pub(crate) fn compile_stmt_impl(&mut self, stmt: &HirStmt) -> Result<(), CompileError> {
        if idioms::try_lower_stmt_idiom(self, stmt)? {
            return Ok(());
        }
        match stmt {
            HirStmt::ExprStmt(expr, suppressed, _) => {
                self.compile_expr(expr)?;
                if !suppressed && expr_supports_inline_emit(expr) {
                    let label = label_for_expr(expr);
                    self.emit(Instr::EmitStackTop { label });
                }
                self.emit(Instr::Pop);
            }
            HirStmt::Assign(id, expr, suppressed, _) => {
                self.compile_expr(expr)?;
                self.emit(Instr::StoreVar(id.0));
                if !suppressed {
                    self.emit(Instr::EmitVar {
                        var_index: id.0,
                        label: EmitLabel::Var(id.0),
                    });
                }
            }
            HirStmt::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
                ..
            } => {
                self.compile_expr(cond)?;
                let mut last_jump = self.emit(Instr::JumpIfFalse(usize::MAX));
                for s in then_body {
                    self.compile_stmt(s)?;
                }
                let mut end_jumps = Vec::new();
                end_jumps.push(self.emit(Instr::Jump(usize::MAX)));
                for (c, b) in elseif_blocks {
                    self.patch(last_jump, Instr::JumpIfFalse(self.instructions.len()));
                    self.compile_expr(c)?;
                    last_jump = self.emit(Instr::JumpIfFalse(usize::MAX));
                    for s in b {
                        self.compile_stmt(s)?;
                    }
                    end_jumps.push(self.emit(Instr::Jump(usize::MAX)));
                }
                self.patch(last_jump, Instr::JumpIfFalse(self.instructions.len()));
                if let Some(body) = else_body {
                    for s in body {
                        self.compile_stmt(s)?;
                    }
                }
                let end = self.instructions.len();
                for j in end_jumps {
                    self.patch(j, Instr::Jump(end));
                }
            }
            HirStmt::While { cond, body, .. } => {
                let start = self.instructions.len();
                self.compile_expr(cond)?;
                let jump_end = self.emit(Instr::JumpIfFalse(usize::MAX));
                let labels = LoopLabels {
                    break_jumps: Vec::new(),
                    continue_jumps: Vec::new(),
                };
                self.loop_stack.push(labels);
                for s in body {
                    self.compile_stmt(s)?;
                }
                let labels = self.loop_stack.pop().unwrap();
                for j in labels.continue_jumps {
                    self.patch(j, Instr::Jump(start));
                }
                self.emit(Instr::Jump(start));
                let end = self.instructions.len();
                self.patch(jump_end, Instr::JumpIfFalse(end));
                for j in labels.break_jumps {
                    self.patch(j, Instr::Jump(end));
                }
            }
            HirStmt::For {
                var, expr, body, ..
            } => {
                if let HirExprKind::Range(start, step, end) = &expr.kind {
                    self.compile_expr(start)?;
                    self.emit(Instr::StoreVar(var.0));
                    self.compile_expr(end)?;
                    let end_var = self.alloc_temp();
                    self.emit(Instr::StoreVar(end_var));
                    let step_var = self.alloc_temp();
                    if let Some(step_expr) = step {
                        self.compile_expr(step_expr)?;
                        self.emit(Instr::StoreVar(step_var));
                    } else {
                        self.emit(Instr::LoadConst(1.0));
                        self.emit(Instr::StoreVar(step_var));
                    }

                    let loop_start = self.instructions.len();

                    self.emit(Instr::LoadVar(step_var));
                    self.emit(Instr::LoadConst(0.0));
                    self.emit(Instr::Equal);
                    let j_step_zero_skip = self.emit(Instr::JumpIfFalse(usize::MAX));
                    let jump_end = self.emit(Instr::Jump(usize::MAX));
                    let after_step_zero_check = self.instructions.len();
                    self.patch(j_step_zero_skip, Instr::JumpIfFalse(after_step_zero_check));

                    self.emit(Instr::LoadVar(step_var));
                    self.emit(Instr::LoadConst(0.0));
                    self.emit(Instr::GreaterEqual);
                    let j_neg_branch = self.emit(Instr::JumpIfFalse(usize::MAX));
                    self.emit(Instr::LoadVar(var.0));
                    self.emit(Instr::LoadVar(end_var));
                    self.emit(Instr::LessEqual);
                    let j_exit_pos = self.emit(Instr::JumpIfFalse(usize::MAX));
                    let j_cond_done = self.emit(Instr::Jump(usize::MAX));
                    let neg_branch = self.instructions.len();
                    self.patch(j_neg_branch, Instr::JumpIfFalse(neg_branch));
                    self.emit(Instr::LoadVar(var.0));
                    self.emit(Instr::LoadVar(end_var));
                    self.emit(Instr::GreaterEqual);
                    let j_exit_neg = self.emit(Instr::JumpIfFalse(usize::MAX));
                    let cond_done = self.instructions.len();
                    self.patch(j_cond_done, Instr::Jump(cond_done));

                    self.loop_stack.push(LoopLabels {
                        break_jumps: Vec::new(),
                        continue_jumps: Vec::new(),
                    });
                    for s in body {
                        self.compile_stmt(s)?;
                    }
                    let labels = self.loop_stack.pop().unwrap();
                    for j in labels.continue_jumps {
                        self.patch(j, Instr::Jump(self.instructions.len()));
                    }
                    self.emit(Instr::LoadVar(var.0));
                    self.emit(Instr::LoadVar(step_var));
                    self.emit(Instr::Add);
                    self.emit(Instr::StoreVar(var.0));
                    self.emit(Instr::Jump(loop_start));

                    let end_pc = self.instructions.len();
                    self.patch(jump_end, Instr::Jump(end_pc));
                    self.patch(j_exit_pos, Instr::JumpIfFalse(end_pc));
                    self.patch(j_exit_neg, Instr::JumpIfFalse(end_pc));
                    for j in labels.break_jumps {
                        self.patch(j, Instr::Jump(end_pc));
                    }
                } else {
                    return Err(self.compile_error("for loop expects range"));
                }
            }
            HirStmt::Break(_) => {
                if self.loop_stack.is_empty() {
                    return Err(self.compile_error("break outside loop"));
                }
                let idx = self.emit(Instr::Jump(usize::MAX));
                if let Some(labels) = self.loop_stack.last_mut() {
                    labels.break_jumps.push(idx);
                }
            }
            HirStmt::Continue(_) => {
                if self.loop_stack.is_empty() {
                    return Err(self.compile_error("continue outside loop"));
                }
                let idx = self.emit(Instr::Jump(usize::MAX));
                if let Some(labels) = self.loop_stack.last_mut() {
                    labels.continue_jumps.push(idx);
                }
            }
            HirStmt::Return(_) => {
                self.emit(Instr::Return);
            }
            HirStmt::Function {
                name,
                params,
                outputs,
                body,
                has_varargin,
                has_varargout,
                ..
            } => self.compile_function_stmt(
                name,
                params,
                outputs,
                body,
                *has_varargin,
                *has_varargout,
            )?,
            HirStmt::Switch {
                expr,
                cases,
                otherwise,
                ..
            } => {
                let temp_id = self.alloc_temp();
                self.compile_expr(expr)?;
                self.emit(Instr::StoreVar(temp_id));
                let mut end_jumps: Vec<usize> = Vec::new();
                let mut next_case_jump_to_here: Option<usize> = None;
                for (case_expr, body) in cases {
                    if let Some(j) = next_case_jump_to_here.take() {
                        self.patch(j, Instr::JumpIfFalse(self.instructions.len()));
                    }
                    self.emit(Instr::LoadVar(temp_id));
                    self.compile_expr(case_expr)?;
                    self.emit(Instr::Equal);
                    let jmp = self.emit(Instr::JumpIfFalse(usize::MAX));
                    for s in body {
                        self.compile_stmt(s)?;
                    }
                    end_jumps.push(self.emit(Instr::Jump(usize::MAX)));
                    next_case_jump_to_here = Some(jmp);
                }
                let otherwise_start = self.instructions.len();
                if let Some(j) = next_case_jump_to_here.take() {
                    self.patch(j, Instr::JumpIfFalse(otherwise_start));
                }
                if let Some(body) = otherwise {
                    for s in body {
                        self.compile_stmt(s)?;
                    }
                }
                let end = self.instructions.len();
                for j in end_jumps {
                    self.patch(j, Instr::Jump(end));
                }
            }
            HirStmt::TryCatch {
                try_body,
                catch_var,
                catch_body,
                ..
            } => {
                let enter_idx = self.emit(Instr::EnterTry(usize::MAX, catch_var.map(|v| v.0)));
                for s in try_body {
                    self.compile_stmt(s)?;
                }
                self.emit(Instr::PopTry);
                let jmp_end = self.emit(Instr::Jump(usize::MAX));
                let catch_pc = self.instructions.len();
                self.patch(enter_idx, Instr::EnterTry(catch_pc, catch_var.map(|v| v.0)));
                for s in catch_body {
                    self.compile_stmt(s)?;
                }
                let end_pc = self.instructions.len();
                self.patch(jmp_end, Instr::Jump(end_pc));
            }
            HirStmt::AssignLValue(lv, rhs, _, _) => self.compile_assign_lvalue(lv, rhs)?,
            HirStmt::Global(vars, _) => {
                let ids: Vec<usize> = vars.iter().map(|(v, _n)| v.0).collect();
                let names: Vec<String> = vars.iter().map(|(_v, n)| n.clone()).collect();
                self.emit(Instr::DeclareGlobalNamed(ids, names));
            }
            HirStmt::Persistent(vars, _) => {
                let ids: Vec<usize> = vars.iter().map(|(v, _n)| v.0).collect();
                let names: Vec<String> = vars.iter().map(|(_v, n)| n.clone()).collect();
                self.emit(Instr::DeclarePersistentNamed(ids, names));
            }
            HirStmt::Import { path, wildcard, .. } => {
                self.emit(Instr::RegisterImport {
                    path: path.clone(),
                    wildcard: *wildcard,
                });
            }
            HirStmt::ClassDef {
                name,
                super_class,
                members,
                ..
            } => self.compile_class_def(name, super_class, members)?,
            HirStmt::MultiAssign(vars, expr, suppressed, _) => {
                match &expr.kind {
                    HirExprKind::FuncCall(name, args) => {
                        let call_arg_spans: Vec<runmat_hir::Span> =
                            args.iter().map(|a| a.span).collect();
                        if self.functions.contains_key(name) {
                            for arg in args {
                                self.compile_expr(arg)?;
                            }
                            self.emit_call_with_arg_spans(
                                Instr::CallFunctionMulti(name.clone(), args.len(), vars.len()),
                                &call_arg_spans,
                            );
                            self.emit(Instr::Unpack(vars.len()));
                            for (_i, var) in vars.iter().enumerate().rev() {
                                if let Some(v) = var {
                                    self.emit(Instr::StoreVar(v.0));
                                } else {
                                    self.emit(Instr::Pop);
                                }
                            }
                        } else {
                            for arg in args {
                                self.compile_expr(arg)?;
                            }
                            self.emit_call_with_arg_spans(
                                Instr::CallBuiltin(name.clone(), args.len()),
                                &call_arg_spans,
                            );
                            self.emit(Instr::Unpack(vars.len()));
                            for (_i, var) in vars.iter().enumerate().rev() {
                                if let Some(v) = var {
                                    self.emit(Instr::StoreVar(v.0));
                                } else {
                                    self.emit(Instr::Pop);
                                }
                            }
                        }
                    }
                    HirExprKind::IndexCell(base, indices) => {
                        self.compile_expr(base)?;
                        for index in indices {
                            self.compile_expr(index)?;
                        }
                        self.emit(Instr::IndexCellExpand(indices.len(), vars.len()));
                        for (_i, var) in vars.iter().enumerate().rev() {
                            if let Some(v) = var {
                                self.emit(Instr::StoreVar(v.0));
                            } else {
                                self.emit(Instr::Pop);
                            }
                        }
                    }
                    _ => {
                        let first_real = vars.iter().position(|v| v.is_some());
                        if let Some(first_idx) = first_real {
                            self.compile_expr(expr)?;
                            if let Some(Some(first_var)) = vars.get(first_idx) {
                                self.emit(Instr::StoreVar(first_var.0));
                            }
                        }
                        for (i, var) in vars.iter().enumerate() {
                            if Some(i) == first_real {
                                continue;
                            }
                            if let Some(v) = var {
                                self.emit(Instr::LoadConst(0.0));
                                self.emit(Instr::StoreVar(v.0));
                            }
                        }
                    }
                }
                if !suppressed {
                    self.emit_multiassign_outputs(vars);
                }
            }
        }
        Ok(())
    }
}
