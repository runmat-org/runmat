use crate::call::builtins::is_vm_intrinsic_counter_builtin;
use crate::call::shared::strict_callable_display_name;
use crate::compiler::CompileError;
use crate::instr::{ArgSpec, EndExpr, Instr};
use crate::layout::VmAssemblyLayout;
use runmat_builtins::{self, Type};
use runmat_hir::{
    BindingId, CallSyntax, CallableIdentity, EntrypointId, FunctionId, HirAssembly, IndexKind,
    IndexResultContext, OperatorKind, RequestedOutputCount,
};
use runmat_mir::{
    BasicBlockId, MirAggregateKind, MirAssembly, MirBody, MirCall, MirCallArg, MirCallee,
    MirConstant, MirIndexComponent, MirIndexPlan, MirIndexing, MirOperand, MirOutputTarget,
    MirPlace, MirPlaceMutation, MirRvalue, MirStmt, MirStmtKind, MirTerminatorKind,
};
use std::collections::{HashMap, HashSet};

type ClassRegistration = (
    String,
    Option<String>,
    Vec<(String, bool, String, String)>,
    Vec<(String, String, bool, String)>,
);

pub struct Compiler {
    pub instructions: Vec<Instr>,
    pub instr_spans: Vec<runmat_hir::Span>,
    pub call_arg_spans: Vec<Option<Vec<runmat_hir::Span>>>,
    pub var_count: usize,
    pub imports: Vec<(Vec<String>, bool)>,
    pub var_types: Vec<Type>,
    pub layout: Option<VmAssemblyLayout>,
    pub function: Option<FunctionId>,
    pub body: Option<MirBody>,
    pub class_registrations: Vec<ClassRegistration>,
    current_span: Option<runmat_hir::Span>,
    pending_place_mutation: Option<MirPlaceMutation>,
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

struct MirSliceExprComponents {
    numeric_count: usize,
    colon_mask: u32,
    end_mask: u32,
    range_dims: Vec<usize>,
    range_has_step: Vec<bool>,
    range_start_exprs: Vec<Option<EndExpr>>,
    range_step_exprs: Vec<Option<EndExpr>>,
    range_end_exprs: Vec<EndExpr>,
    end_numeric_exprs: Vec<(usize, EndExpr)>,
}

#[derive(Clone, Copy)]
enum MirRangeParamOrder {
    BeforeNumeric,
    AfterNumeric,
}

struct MirRangeEndSpec {
    start_expr: Option<EndExpr>,
    step_expr: Option<EndExpr>,
    end_expr: EndExpr,
    has_step: bool,
}

struct MirStochasticEvolutionPlan {
    state: runmat_mir::MirLocalId,
    drift: MirOperand,
    scale: MirOperand,
    steps: MirOperand,
}

const CELL_END_PLUS_TAG_VALUE: u64 = 0x7ff8_c311_0000_0000;
const CELL_END_PLUS_OFFSET_MASK: u64 = 0x0000_0000_ffff_ffff;
const IDENT_MIR_CELL_EXPAND_PLAN_INVALID: &str = "RunMat:MirCellExpandPlanInvalid";
const IDENT_MIR_PAREN_CELL_PLAN_INVALID: &str = "RunMat:MirParenCellPlanInvalid";
const IDENT_MIR_SCALAR_INDEX_PLAN_INVALID: &str = "RunMat:MirScalarIndexPlanInvalid";
const IDENT_MIR_SLICE_INDEX_PLAN_INVALID: &str = "RunMat:MirSliceIndexPlanInvalid";
const IDENT_MIR_CELL_INDEX_PLAN_INVALID: &str = "RunMat:MirCellIndexPlanInvalid";
const IDENT_MIR_CELL_INDEX_CONTEXT_INVALID: &str = "RunMat:MirCellIndexContextInvalid";
const IDENT_MIR_INDEX_CONTEXT_INVALID: &str = "RunMat:MirIndexContextInvalid";
const IDENT_MIR_MULTI_ASSIGN_OUTPUT_COUNT_MISMATCH: &str =
    "RunMat:MirMultiAssignOutputCountMismatch";
const IDENT_MIR_DELETE_ASSIGNMENT_RHS_INVALID: &str = "RunMat:MirDeleteAssignmentRhsInvalid";
const IDENT_MIR_DELETE_ASSIGNMENT_PLACE_MISMATCH: &str = "RunMat:MirDeleteAssignmentPlaceMismatch";
const IDENT_MIR_DELETE_ASSIGNMENT_TARGET_INVALID: &str = "RunMat:MirDeleteAssignmentTargetInvalid";
const IDENT_MIR_DELETE_ASSIGNMENT_INDEX_KIND_INVALID: &str =
    "RunMat:MirDeleteAssignmentIndexKindInvalid";
const IDENT_MIR_DELETE_ASSIGNMENT_CONTEXT_INVALID: &str =
    "RunMat:MirDeleteAssignmentContextInvalid";
const IDENT_MIR_DELETION_CONTEXT_WITHOUT_DELETE_INVALID: &str =
    "RunMat:MirDeletionContextWithoutDeleteInvalid";
const IDENT_MIR_DELETE_ASSIGNMENT_CREATION_POLICY_INVALID: &str =
    "RunMat:MirDeleteAssignmentCreationPolicyInvalid";
const IDENT_MIR_AGGREGATE_SHAPE_INVALID: &str = "RunMat:MirAggregateShapeInvalid";
const IDENT_MIR_OPERATOR_UNSUPPORTED: &str = "RunMat:MirOperatorUnsupported";
const IDENT_MIR_BUILTIN_UNKNOWN: &str = "RunMat:MirBuiltinUnknown";
const IDENT_MIR_NUMBER_LITERAL_INVALID: &str = "RunMat:MirNumberLiteralInvalid";
const IDENT_MIR_CONSTANT_UNKNOWN: &str = "RunMat:MirConstantUnknown";
const IDENT_MIR_FUNCTION_HANDLE_NAME_MISSING: &str = "RunMat:MirFunctionHandleNameMissing";
const IDENT_MIR_FUNCTION_HANDLE_TARGET_UNSUPPORTED: &str =
    "RunMat:MirFunctionHandleTargetUnsupported";
const IDENT_MIR_CALL_TARGET_NAME_INVALID: &str = "RunMat:MirCallTargetNameInvalid";
const IDENT_MIR_CALL_FALLBACK_POLICY_UNSUPPORTED: &str = "RunMat:MirCallFallbackPolicyUnsupported";
const IDENT_MIR_METHOD_FALLBACK_POLICY_UNSUPPORTED: &str =
    "RunMat:MirMethodFallbackPolicyUnsupported";
const IDENT_MIR_METHOD_CALL_CALLEE_INVALID: &str = "RunMat:MirMethodCallCalleeInvalid";
const IDENT_MIR_METHOD_CALL_RECEIVER_MISSING: &str = "RunMat:MirMethodCallReceiverMissing";

fn encode_cell_end_offset(offset: isize) -> f64 {
    if offset <= 0 {
        if offset == 0 {
            -0.0
        } else {
            offset as f64
        }
    } else {
        f64::from_bits(CELL_END_PLUS_TAG_VALUE | ((offset as u64) & CELL_END_PLUS_OFFSET_MASK))
    }
}

fn stochastic_evolution_disabled() -> bool {
    std::env::var("RUNMAT_DISABLE_STOCHASTIC_EVOLUTION")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes"
            )
        })
        .unwrap_or(false)
}

fn mir_range_starts_at_one(iterable: &MirRvalue) -> bool {
    let MirRvalue::Range { start, step, .. } = iterable else {
        return false;
    };
    mir_operand_is_one(start) && step.as_ref().is_none_or(mir_operand_is_one)
}

fn mir_operand_is_one(operand: &MirOperand) -> bool {
    matches!(operand, MirOperand::Constant(MirConstant::Number(value)) if value == "1" || value == "1.0")
}

fn call_name(call: &MirCall) -> Option<&str> {
    match &call.callee {
        MirCallee::Static(CallableIdentity::Builtin(id)) => Some(id.0.as_str()),
        MirCallee::Static(CallableIdentity::ExternalName(name)) if name.0.len() == 1 => {
            Some(name.0[0].0.as_str())
        }
        MirCallee::Static(CallableIdentity::DynamicName(name)) => Some(name.0.as_str()),
        _ => None,
    }
}

fn imported_handle_runtime_name(path: &runmat_hir::DefPath) -> Option<String> {
    let module_name = path.module.display_name()?;
    let item_name = path.item.last().map(|item| item.display_name())?;
    if item_name.is_empty() {
        return None;
    }
    let module_leaf = path.module.0.last()?;
    (module_leaf.0 == item_name).then_some(module_name)
}

fn randn_assignment(stmt: &MirStmt) -> Option<(runmat_mir::MirLocalId, &MirCall)> {
    let MirStmtKind::Assign {
        place: MirPlace::Local(local),
        value: MirRvalue::Call(call),
    } = &stmt.kind
    else {
        return None;
    };
    call_name(call)
        .is_some_and(|name| name.eq_ignore_ascii_case("randn"))
        .then_some((*local, call))
}

fn assigned_rvalue(statements: &[MirStmt], local: runmat_mir::MirLocalId) -> Option<&MirRvalue> {
    statements.iter().find_map(|stmt| {
        let MirStmtKind::Assign {
            place: MirPlace::Local(candidate),
            value,
        } = &stmt.kind
        else {
            return None;
        };
        (*candidate == local).then_some(value)
    })
}

fn local_operand(operand: &MirOperand) -> Option<runmat_mir::MirLocalId> {
    match operand {
        MirOperand::Local(local) => Some(*local),
        _ => None,
    }
}

fn exp_call_arg(statements: &[MirStmt], exp_local: runmat_mir::MirLocalId) -> Option<&MirOperand> {
    let MirRvalue::Call(call) = assigned_rvalue(statements, exp_local)? else {
        return None;
    };
    if !call_name(call).is_some_and(|name| name.eq_ignore_ascii_case("exp")) || call.args.len() != 1
    {
        return None;
    }
    match &call.args[0] {
        MirCallArg::Single(arg) => Some(arg),
        _ => None,
    }
}

fn add_with_scale_term(
    statements: &[MirStmt],
    arg_local: runmat_mir::MirLocalId,
    z_local: runmat_mir::MirLocalId,
) -> Option<(&MirOperand, &MirOperand)> {
    let MirRvalue::Binary(left, OperatorKind::Add, right) = assigned_rvalue(statements, arg_local)?
    else {
        return None;
    };
    if local_operand(left)
        .and_then(|local| scale_term(statements, local, z_local))
        .is_some()
    {
        Some((right, left))
    } else if local_operand(right)
        .and_then(|local| scale_term(statements, local, z_local))
        .is_some()
    {
        Some((left, right))
    } else {
        None
    }
}

fn scale_term(
    statements: &[MirStmt],
    scale_mul_local: runmat_mir::MirLocalId,
    z_local: runmat_mir::MirLocalId,
) -> Option<&MirOperand> {
    let MirRvalue::Binary(left, OperatorKind::ElementwiseMultiply, right) =
        assigned_rvalue(statements, scale_mul_local)?
    else {
        return None;
    };
    if matches!(left, MirOperand::Local(local) if *local == z_local) {
        Some(right)
    } else if matches!(right, MirOperand::Local(local) if *local == z_local) {
        Some(left)
    } else {
        None
    }
}

fn mir_indexing_context_matches(actual: IndexResultContext, expected: IndexResultContext) -> bool {
    actual == expected
        || (expected == IndexResultContext::AssignmentTarget
            && actual == IndexResultContext::DeletionTarget)
}

fn hir_function_imports(hir: &HirAssembly, function: FunctionId) -> Vec<(Vec<String>, bool)> {
    let Some(hir_function) = hir
        .functions
        .iter()
        .find(|candidate| candidate.id == function)
    else {
        return Vec::new();
    };
    hir.modules
        .get(hir_function.module.0)
        .map(|module| {
            module
                .imports
                .iter()
                .map(|import| {
                    (
                        import.path.0.iter().map(|part| part.0.clone()).collect(),
                        import.wildcard,
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

fn hir_class_registrations(hir: &HirAssembly) -> Vec<ClassRegistration> {
    hir.classes
        .iter()
        .map(|class| {
            let name = class
                .name
                .0
                .iter()
                .map(|part| part.0.clone())
                .collect::<Vec<_>>()
                .join(".");
            let super_class = class.super_class.and_then(|class_id| {
                hir.classes
                    .iter()
                    .find(|candidate| candidate.id == class_id)
                    .map(|super_class| {
                        super_class
                            .name
                            .0
                            .iter()
                            .map(|part| part.0.clone())
                            .collect::<Vec<_>>()
                            .join(".")
                    })
            });
            let properties = class
                .properties
                .iter()
                .map(|property| {
                    let name = if property.attributes.is_dependent {
                        format!("@dep:{}", property.name.0)
                    } else {
                        property.name.0.clone()
                    };
                    (
                        name,
                        property.attributes.is_static,
                        member_access_name(property.attributes.get_access.clone()).to_string(),
                        member_access_name(property.attributes.set_access.clone()).to_string(),
                    )
                })
                .collect();
            let methods = class
                .methods
                .iter()
                .map(|method| {
                    (
                        method.name.0.clone(),
                        method.name.0.clone(),
                        method.is_static,
                        member_access_name(method.attributes.access.clone()).to_string(),
                    )
                })
                .collect();
            (name, super_class, properties, methods)
        })
        .collect()
}

fn member_access_name(access: runmat_hir::MemberAccess) -> &'static str {
    match access {
        runmat_hir::MemberAccess::Private => "private",
        _ => "public",
    }
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
            imports: hir_function_imports(hir, function),
            var_types,
            layout: Some(layout),
            function: Some(function),
            body: Some(body),
            class_registrations: hir_class_registrations(hir),
            current_span: None,
            pending_place_mutation: None,
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
            imports: hir_function_imports(hir, function),
            var_types,
            layout: Some(layout),
            function: Some(function),
            body: Some(body),
            class_registrations: hir_class_registrations(hir),
            current_span: None,
            pending_place_mutation: None,
        })
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

        for (name, super_class, properties, methods) in self.class_registrations.clone() {
            self.emit(Instr::RegisterClass {
                name,
                super_class,
                properties,
                methods,
            });
        }
        for (path, wildcard) in self.imports.clone() {
            self.emit(Instr::RegisterImport { path, wildcard });
        }
        self.compile_mir_body(&body)?;
        Ok(())
    }

    fn compile_mir_body(&mut self, body: &MirBody) -> Result<(), CompileError> {
        let mut blocks = body.blocks.clone();
        blocks.sort_by_key(|block| block.id.0);

        let mut block_starts = HashMap::new();
        let mut pending_jumps: Vec<(usize, BasicBlockId, bool)> = Vec::new();
        let mut pending_try_entries: Vec<(usize, BasicBlockId, Option<usize>)> = Vec::new();
        let try_entry_blocks: HashSet<BasicBlockId> = blocks
            .iter()
            .filter_map(|block| match block.terminator.kind {
                MirTerminatorKind::TryCatch { try_block, .. } => Some(try_block),
                _ => None,
            })
            .collect();

        for (block_index, block) in blocks.iter().enumerate() {
            self.pending_place_mutation = None;
            block_starts.insert(block.id, self.instructions.len());
            for stmt in &block.statements {
                self.compile_mir_stmt(stmt)?;
            }
            let exits_try_scope = try_entry_blocks.contains(&block.id);
            match &block.terminator.kind {
                MirTerminatorKind::Goto(target) => {
                    if exits_try_scope {
                        self.emit(Instr::PopTry);
                    }
                    let pc = self.emit(Instr::Jump(usize::MAX));
                    pending_jumps.push((pc, *target, false));
                }
                MirTerminatorKind::Branch {
                    cond,
                    then_block,
                    else_block,
                } => {
                    if exits_try_scope {
                        self.emit(Instr::PopTry);
                    }
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
                    if exits_try_scope {
                        self.emit(Instr::PopTry);
                    }
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
                MirTerminatorKind::TryCatch {
                    try_block,
                    catch_block,
                    catch_binding,
                } => {
                    let catch_var = catch_binding
                        .map(|local| self.mir_local_slot(local))
                        .transpose()?;
                    let enter_pc = self.emit(Instr::EnterTry(usize::MAX, catch_var));
                    pending_try_entries.push((enter_pc, *catch_block, catch_var));
                    let try_pc = self.emit(Instr::Jump(usize::MAX));
                    pending_jumps.push((try_pc, *try_block, false));
                }
                MirTerminatorKind::For {
                    binding,
                    iterable,
                    body_block,
                    exit_block,
                    ..
                } => {
                    if self.try_compile_mir_stochastic_evolution(
                        iterable,
                        *body_block,
                        *exit_block,
                        &mut pending_jumps,
                    )? {
                        continue;
                    }
                    self.compile_mir_for_terminator(
                        *binding,
                        iterable,
                        *body_block,
                        *exit_block,
                        &mut pending_jumps,
                    )?;
                }
                MirTerminatorKind::Return(values) => {
                    if exits_try_scope {
                        self.emit(Instr::PopTry);
                    }
                    if values.is_empty() && block_index + 1 < blocks.len() {
                        self.emit(Instr::Return);
                    } else {
                        self.compile_mir_return(values)?;
                    }
                }
                MirTerminatorKind::Unreachable => {
                    if exits_try_scope {
                        self.emit(Instr::PopTry);
                    }
                    self.emit(Instr::Return);
                }
                MirTerminatorKind::Await {
                    future,
                    result,
                    resume,
                } => {
                    if exits_try_scope {
                        self.emit(Instr::PopTry);
                    }
                    self.compile_mir_operand(future)?;
                    self.emit(Instr::Await);
                    if let Some(place) = result {
                        let tmp = self.alloc_temp();
                        self.emit(Instr::StoreVar(tmp));
                        self.compile_mir_assign_from_slot(place, tmp)?;
                    } else {
                        self.emit(Instr::Pop);
                    }
                    let resume_pc = self.emit(Instr::Jump(usize::MAX));
                    pending_jumps.push((resume_pc, *resume, false));
                }
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

        for (pc, target, catch_var) in pending_try_entries {
            let target_pc = *block_starts
                .get(&target)
                .ok_or_else(|| CompileError::new(format!("missing MIR catch block {target:?}")))?;
            self.patch(pc, Instr::EnterTry(target_pc, catch_var));
        }

        Ok(())
    }

    fn try_compile_mir_stochastic_evolution(
        &mut self,
        iterable: &MirRvalue,
        body_block: BasicBlockId,
        exit_block: BasicBlockId,
        pending_jumps: &mut Vec<(usize, BasicBlockId, bool)>,
    ) -> Result<bool, CompileError> {
        if stochastic_evolution_disabled() || !mir_range_starts_at_one(iterable) {
            return Ok(false);
        }
        let Some(plan) = self.detect_mir_stochastic_evolution(body_block, iterable) else {
            return Ok(false);
        };
        let state_slot = self.mir_local_slot(plan.state)?;
        self.emit(Instr::LoadVar(state_slot));
        self.compile_mir_operand(&plan.drift)?;
        self.compile_mir_operand(&plan.scale)?;
        self.compile_mir_operand(&plan.steps)?;
        self.emit(Instr::StochasticEvolution);
        self.emit(Instr::StoreVar(state_slot));
        let done = self.emit(Instr::Jump(usize::MAX));
        pending_jumps.push((done, exit_block, false));
        Ok(true)
    }

    fn detect_mir_stochastic_evolution(
        &self,
        body_block: BasicBlockId,
        iterable: &MirRvalue,
    ) -> Option<MirStochasticEvolutionPlan> {
        let body = self.body.as_ref()?;
        let block = body.blocks.iter().find(|block| block.id == body_block)?;
        let statements = block.statements.as_slice();
        let (z_local, _) = statements.iter().find_map(randn_assignment)?;
        let (state, exp_operand) = statements.iter().rev().find_map(|stmt| {
            let MirStmtKind::Assign {
                place: MirPlace::Local(state),
                value: MirRvalue::Binary(left, OperatorKind::ElementwiseMultiply, right),
            } = &stmt.kind
            else {
                return None;
            };
            if matches!(left, MirOperand::Local(local) if local == state) {
                Some((*state, right))
            } else if matches!(right, MirOperand::Local(local) if local == state) {
                Some((*state, left))
            } else {
                None
            }
        })?;
        let exp_local = local_operand(exp_operand)?;
        let exp_arg = exp_call_arg(statements, exp_local)?;
        let arg_local = local_operand(exp_arg)?;
        let (drift, scale_mul_operand) = add_with_scale_term(statements, arg_local, z_local)?;
        let scale = scale_term(statements, local_operand(scale_mul_operand)?, z_local)?;
        let MirRvalue::Range { end, .. } = iterable else {
            return None;
        };
        Some(MirStochasticEvolutionPlan {
            state,
            drift: drift.clone(),
            scale: scale.clone(),
            steps: end.clone(),
        })
    }

    fn compile_mir_for_terminator(
        &mut self,
        binding: runmat_mir::MirLocalId,
        iterable: &MirRvalue,
        body_block: BasicBlockId,
        exit_block: BasicBlockId,
        pending_jumps: &mut Vec<(usize, BasicBlockId, bool)>,
    ) -> Result<(), CompileError> {
        let MirRvalue::Range { start, step, end } = iterable else {
            return self.compile_mir_for_columns_terminator(
                binding,
                iterable,
                body_block,
                exit_block,
                pending_jumps,
            );
        };
        let binding_slot = self.mir_local_slot(binding)?;
        let init_flag = self.alloc_temp();
        let end_var = self.alloc_temp();
        let step_var = self.alloc_temp();

        self.emit(Instr::LoadVar(init_flag));
        self.emit(Instr::LoadConst(0.0));
        self.emit(Instr::Equal);
        let already_initialized = self.emit(Instr::JumpIfFalse(usize::MAX));
        self.compile_mir_operand(start)?;
        self.emit(Instr::StoreVar(binding_slot));
        if let Some(step) = step {
            self.compile_mir_operand(step)?;
        } else {
            self.emit(Instr::LoadConst(1.0));
        }
        self.emit(Instr::StoreVar(step_var));
        self.compile_mir_operand(end)?;
        self.emit(Instr::StoreVar(end_var));
        self.emit(Instr::LoadConst(1.0));
        self.emit(Instr::StoreVar(init_flag));
        let after_update = self.emit(Instr::Jump(usize::MAX));
        let increment_pc = self.instructions.len();
        self.patch(already_initialized, Instr::JumpIfFalse(increment_pc));
        self.emit(Instr::LoadVar(binding_slot));
        self.emit(Instr::LoadVar(step_var));
        self.emit(Instr::Add);
        self.emit(Instr::StoreVar(binding_slot));
        let condition_pc = self.instructions.len();
        self.patch(after_update, Instr::Jump(condition_pc));

        self.emit(Instr::LoadVar(step_var));
        self.emit(Instr::LoadConst(0.0));
        self.emit(Instr::Equal);
        let nonzero_step = self.emit(Instr::JumpIfFalse(usize::MAX));
        self.emit(Instr::LoadConst(0.0));
        self.emit(Instr::StoreVar(init_flag));
        let zero_step_exit = self.emit(Instr::Jump(usize::MAX));
        pending_jumps.push((zero_step_exit, exit_block, false));
        let after_zero_step = self.instructions.len();
        self.patch(nonzero_step, Instr::JumpIfFalse(after_zero_step));

        self.emit(Instr::LoadVar(step_var));
        self.emit(Instr::LoadConst(0.0));
        self.emit(Instr::GreaterEqual);
        let negative_step_branch = self.emit(Instr::JumpIfFalse(usize::MAX));
        self.emit(Instr::LoadVar(binding_slot));
        self.emit(Instr::LoadVar(end_var));
        self.emit(Instr::LessEqual);
        let positive_step_exit = self.emit(Instr::JumpIfFalse(usize::MAX));
        let condition_done = self.emit(Instr::Jump(usize::MAX));
        let negative_branch = self.instructions.len();
        self.patch(negative_step_branch, Instr::JumpIfFalse(negative_branch));
        self.emit(Instr::LoadVar(binding_slot));
        self.emit(Instr::LoadVar(end_var));
        self.emit(Instr::GreaterEqual);
        let negative_step_exit = self.emit(Instr::JumpIfFalse(usize::MAX));
        let body_jump_pc = self.instructions.len();
        self.patch(condition_done, Instr::Jump(body_jump_pc));

        let body_jump = self.emit(Instr::Jump(usize::MAX));
        pending_jumps.push((body_jump, body_block, false));

        let exit_pc = self.instructions.len();
        self.patch(positive_step_exit, Instr::JumpIfFalse(exit_pc));
        self.patch(negative_step_exit, Instr::JumpIfFalse(exit_pc));
        self.emit(Instr::LoadConst(0.0));
        self.emit(Instr::StoreVar(init_flag));
        let done = self.emit(Instr::Jump(usize::MAX));
        pending_jumps.push((done, exit_block, false));

        Ok(())
    }

    fn compile_mir_for_columns_terminator(
        &mut self,
        binding: runmat_mir::MirLocalId,
        iterable: &MirRvalue,
        body_block: BasicBlockId,
        exit_block: BasicBlockId,
        pending_jumps: &mut Vec<(usize, BasicBlockId, bool)>,
    ) -> Result<(), CompileError> {
        let binding_slot = self.mir_local_slot(binding)?;
        let init_flag = self.alloc_temp();
        let iterable_slot = self.alloc_temp();
        let col_slot = self.alloc_temp();
        let col_count_slot = self.alloc_temp();

        self.emit(Instr::LoadVar(init_flag));
        self.emit(Instr::LoadConst(0.0));
        self.emit(Instr::Equal);
        let already_initialized = self.emit(Instr::JumpIfFalse(usize::MAX));

        self.compile_mir_rvalue(iterable)?;
        self.emit(Instr::StoreVar(iterable_slot));
        self.emit(Instr::LoadVar(iterable_slot));
        self.emit(Instr::LoadConst(2.0));
        self.emit(Instr::CallBuiltinMulti("size".to_string(), 2, 1));
        self.emit(Instr::StoreVar(col_count_slot));
        self.emit(Instr::LoadConst(1.0));
        self.emit(Instr::StoreVar(col_slot));
        self.emit(Instr::LoadConst(1.0));
        self.emit(Instr::StoreVar(init_flag));
        let after_update = self.emit(Instr::Jump(usize::MAX));

        let increment_pc = self.instructions.len();
        self.patch(already_initialized, Instr::JumpIfFalse(increment_pc));
        self.emit(Instr::LoadVar(col_slot));
        self.emit(Instr::LoadConst(1.0));
        self.emit(Instr::Add);
        self.emit(Instr::StoreVar(col_slot));

        let condition_pc = self.instructions.len();
        self.patch(after_update, Instr::Jump(condition_pc));
        self.emit(Instr::LoadVar(col_slot));
        self.emit(Instr::LoadVar(col_count_slot));
        self.emit(Instr::LessEqual);
        let exhausted = self.emit(Instr::JumpIfFalse(usize::MAX));

        self.emit(Instr::LoadVar(iterable_slot));
        self.emit(Instr::LoadVar(col_slot));
        self.emit(Instr::IndexSlice(2, 1, 1u32 << 0, 0));
        self.emit(Instr::StoreVar(binding_slot));
        let body_jump = self.emit(Instr::Jump(usize::MAX));
        pending_jumps.push((body_jump, body_block, false));

        let exit_pc = self.instructions.len();
        self.patch(exhausted, Instr::JumpIfFalse(exit_pc));
        self.emit(Instr::LoadConst(0.0));
        self.emit(Instr::StoreVar(init_flag));
        let done = self.emit(Instr::Jump(usize::MAX));
        pending_jumps.push((done, exit_block, false));

        Ok(())
    }

    fn take_assign_delete_flag(&mut self, place: &MirPlace) -> Result<bool, CompileError> {
        let Some(mutation) = self.pending_place_mutation.take() else {
            return Ok(false);
        };
        if !matches!(mutation.kind, runmat_hir::PlaceMutationKind::Delete) {
            return Ok(false);
        }
        if mutation.place != *place {
            return Err(self
                .compile_error(
                    "MIR delete assignment invariant violated: place mutation target must match assign target",
                )
                .with_identifier(IDENT_MIR_DELETE_ASSIGNMENT_PLACE_MISMATCH));
        }
        if mutation.creation_policy != runmat_hir::AssignmentCreationPolicy::ExistingOnly {
            return Err(self
                .compile_error(
                    "MIR delete assignment invariant violated: delete mutation requires ExistingOnly creation policy",
                )
                .with_identifier(IDENT_MIR_DELETE_ASSIGNMENT_CREATION_POLICY_INVALID));
        }
        Ok(true)
    }

    fn compile_mir_stmt(&mut self, stmt: &MirStmt) -> Result<(), CompileError> {
        let _span_guard = SpanGuard::new(self, stmt.span);
        match &stmt.kind {
            MirStmtKind::Assign { place, value } => {
                let delete = self.take_assign_delete_flag(place)?;
                self.compile_mir_assign(place, value, delete)
            }
            MirStmtKind::Expr(value) => {
                self.pending_place_mutation = None;
                self.compile_mir_rvalue(value)?;
                self.emit(Instr::Pop);
                Ok(())
            }
            MirStmtKind::WorkspaceEffect { effect, bindings } => {
                self.pending_place_mutation = None;
                self.compile_mir_workspace_effect(effect, bindings)
            }
            MirStmtKind::EnvironmentEffect(_) => {
                self.pending_place_mutation = None;
                Ok(())
            }
            MirStmtKind::PlaceMutation(mutation) => {
                self.pending_place_mutation = Some(mutation.clone());
                Ok(())
            }
            MirStmtKind::MultiAssign { targets, value } => {
                self.pending_place_mutation = None;
                self.compile_mir_multi_assign(targets, value)
            }
        }
    }

    fn compile_mir_workspace_effect(
        &mut self,
        effect: &runmat_hir::WorkspaceEffect,
        bindings: &[runmat_mir::MirLocalId],
    ) -> Result<(), CompileError> {
        let (ids, names) = self.mir_workspace_effect_names(bindings)?;
        match effect {
            runmat_hir::WorkspaceEffect::MutatesGlobal => {
                self.emit(Instr::DeclareGlobalNamed(ids, names));
            }
            runmat_hir::WorkspaceEffect::MutatesPersistent => {
                self.emit(Instr::DeclarePersistentNamed(ids, names));
            }
            _ => {}
        }
        Ok(())
    }

    fn mir_workspace_effect_names(
        &self,
        bindings: &[runmat_mir::MirLocalId],
    ) -> Result<(Vec<usize>, Vec<String>), CompileError> {
        let layout = self
            .layout
            .as_ref()
            .ok_or_else(|| CompileError::new("compiler missing VM layout"))?;
        let function = self
            .function
            .ok_or_else(|| CompileError::new("compiler missing selected function"))?;
        let function_layout = layout.functions.get(&function).ok_or_else(|| {
            CompileError::new(format!("missing VM layout for function {function:?}"))
        })?;
        let mut ids = Vec::with_capacity(bindings.len());
        let mut names = Vec::with_capacity(bindings.len());
        for binding in bindings {
            let slot = function_layout
                .mir_local_slots
                .get(binding)
                .ok_or_else(|| {
                    CompileError::new(format!("missing VM slot for MIR local {binding:?}"))
                })?;
            let binding_id = function_layout
                .binding_slots
                .iter()
                .find_map(|(binding_id, binding_slot)| {
                    (*binding_slot == *slot).then_some(*binding_id)
                })
                .ok_or_else(|| {
                    CompileError::new(format!("missing binding for VM slot {:?}", slot))
                })?;
            let name = layout
                .storage_bindings
                .get(&binding_id)
                .map(|binding| binding.name.clone())
                .ok_or_else(|| {
                    CompileError::new(format!("missing binding name for {binding_id:?}"))
                })?;
            ids.push(slot.0);
            names.push(name);
        }
        Ok((ids, names))
    }

    fn compile_mir_multi_assign(
        &mut self,
        targets: &runmat_mir::MirOutputTargetList,
        value: &MirRvalue,
    ) -> Result<(), CompileError> {
        let output_count = self.output_count_for_targets(targets)?;
        match value {
            MirRvalue::Call(call) => self.compile_mir_call_for_multi_assign(call, output_count)?,
            MirRvalue::Index { base, indexing }
                if indexing.kind == IndexKind::Brace
                    && matches!(indexing.result_context, IndexResultContext::ReadCommaList) =>
            {
                self.compile_mir_cell_expand_for_multi_assign(base, indexing, output_count)?
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
            self.compile_mir_output_target_store(target)?;
        }
        Ok(())
    }

    fn compile_mir_output_target_store(
        &mut self,
        target: &MirOutputTarget,
    ) -> Result<(), CompileError> {
        match target {
            MirOutputTarget::Place(place @ (MirPlace::Local(_) | MirPlace::Binding(_))) => {
                let slot = self.mir_place_slot(place)?;
                self.emit(Instr::StoreVar(slot));
                Ok(())
            }
            MirOutputTarget::Place(place) => {
                let tmp = self.alloc_temp();
                self.emit(Instr::StoreVar(tmp));
                self.compile_mir_assign_from_slot(place, tmp)
            }
            MirOutputTarget::Discard => {
                self.emit(Instr::Pop);
                Ok(())
            }
        }
    }

    fn compile_mir_assign_from_slot(
        &mut self,
        place: &MirPlace,
        value_slot: usize,
    ) -> Result<(), CompileError> {
        match place {
            MirPlace::Local(_) | MirPlace::Binding(_) => {
                self.emit(Instr::LoadVar(value_slot));
                let slot = self.mir_place_slot(place)?;
                self.emit(Instr::StoreVar(slot));
                Ok(())
            }
            MirPlace::Index(base, indexing) => {
                if let Ok(base_slot) = self.mir_place_slot(base) {
                    self.emit(Instr::LoadVar(base_slot));
                    self.compile_mir_store_indexed_value_from_temp(indexing, value_slot, false)?;
                    self.emit(Instr::StoreVar(base_slot));
                    return Ok(());
                }
                self.compile_mir_place_read(base)?;
                self.compile_mir_store_indexed_value_from_temp(indexing, value_slot, false)?;
                self.emit_store_back_mir_member_chain(base)
            }
            MirPlace::Member(base, member) => {
                self.compile_mir_member_base_for_assignment(base)?;
                self.emit(Instr::LoadVar(value_slot));
                self.emit(Instr::StoreMemberOrInit(member.0.clone()));
                self.emit_store_back_mir_member_chain(base)
            }
            MirPlace::DynamicMember(base, member) => {
                self.compile_mir_member_base_for_assignment(base)?;
                self.compile_mir_operand(member)?;
                self.emit(Instr::LoadVar(value_slot));
                self.emit(Instr::StoreMemberDynamicOrInit);
                self.emit_store_back_mir_member_chain(base)
            }
        }
    }

    fn compile_mir_cell_expand_for_multi_assign(
        &mut self,
        base: &MirOperand,
        indexing: &MirIndexing,
        output_count: usize,
    ) -> Result<(), CompileError> {
        self.compile_mir_operand(base)?;
        let (index_count, expand_all, end_offsets, end_exprs) =
            self.compile_mir_cell_selector_operands(indexing)?;
        if expand_all {
            self.emit(Instr::IndexCellExpand {
                num_indices: 0,
                out_count: output_count,
                end_offsets,
                end_exprs,
            });
        } else {
            self.emit(Instr::IndexCellExpand {
                num_indices: index_count,
                out_count: output_count,
                end_offsets,
                end_exprs,
            });
        }
        Ok(())
    }

    fn compile_mir_cell_list(
        &mut self,
        base: &MirOperand,
        indexing: &MirIndexing,
    ) -> Result<(), CompileError> {
        self.compile_mir_operand(base)?;
        let (index_count, expand_all, end_offsets, end_exprs) =
            self.compile_mir_cell_selector_operands(indexing)?;
        self.emit(Instr::IndexCellList {
            num_indices: if expand_all { 0 } else { index_count },
            end_offsets,
            end_exprs,
        });
        Ok(())
    }

    fn compile_mir_cell_selector_operands(
        &mut self,
        indexing: &MirIndexing,
    ) -> Result<(usize, bool, Vec<(usize, isize)>, Vec<(usize, EndExpr)>), CompileError> {
        let expand_all = indexing.cell_expand_all;
        let mut index_count = 0usize;
        let mut end_offsets = Vec::new();
        let mut end_exprs = Vec::new();
        for component in &indexing.components {
            match component {
                MirIndexComponent::Colon => {
                    if !expand_all {
                        self.emit(Instr::LoadString(":".to_string()));
                        index_count += 1;
                    }
                }
                MirIndexComponent::Expr(operand) => {
                    self.compile_mir_operand(operand)?;
                    if let Some(expr) = self.mir_operand_end_expr(operand) {
                        if let Some(offset) = Self::mir_cell_end_offset_from_expr(&expr) {
                            end_offsets.push((index_count, offset));
                        } else {
                            end_exprs.push((index_count, expr));
                        }
                    }
                    index_count += 1;
                }
                MirIndexComponent::End { offset, .. } => {
                    self.emit(Instr::LoadConst(encode_cell_end_offset(*offset)));
                    end_offsets.push((index_count, *offset));
                    index_count += 1;
                }
            }
        }
        if expand_all
            && indexing
                .components
                .iter()
                .any(|component| !matches!(component, MirIndexComponent::Colon))
        {
            return Err(
                self.compile_error(
                    "MIR cell expansion invariant violated: expand_all requires all-colon selectors",
                )
                .with_identifier(IDENT_MIR_CELL_EXPAND_PLAN_INVALID),
            );
        }
        Ok((index_count, expand_all, end_offsets, end_exprs))
    }

    fn compile_mir_call_for_multi_assign(
        &mut self,
        call: &MirCall,
        output_count: usize,
    ) -> Result<(), CompileError> {
        match call.requested_outputs {
            RequestedOutputCount::Zero if output_count == 0 => {}
            RequestedOutputCount::One if output_count == 1 => {}
            RequestedOutputCount::Exactly(count) if count == output_count => {}
            _ => {
                return Err(self
                    .compile_error("MIR multi-assign call output count does not match targets")
                    .with_identifier(IDENT_MIR_MULTI_ASSIGN_OUTPUT_COUNT_MISMATCH))
            }
        }
        let (specs, has_expansion) = self.mir_call_arg_specs(&call.args);
        if matches!(call.syntax, CallSyntax::Method | CallSyntax::DottedInvoke) {
            match &call.callee {
                MirCallee::Static(CallableIdentity::SemanticFunction(_))
                | MirCallee::Dynamic(_) => {
                    return Err(self
                        .compile_error(
                            "MIR method-call lowering expected a non-semantic static callee",
                        )
                        .with_identifier(IDENT_MIR_METHOD_CALL_CALLEE_INVALID));
                }
                MirCallee::Static(_) => return self.compile_mir_method_call(call, has_expansion),
            }
        }
        match &call.callee {
            MirCallee::Static(CallableIdentity::SemanticFunction(function)) => {
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
                self.compile_mir_operand(match &call.callee {
                    MirCallee::Dynamic(callee) => callee,
                    _ => unreachable!(),
                })?;
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                if has_expansion {
                    self.emit(Instr::CallFevalExpandMultiOutput(specs, output_count));
                } else {
                    self.emit(Instr::CallFevalMulti(call.args.len(), output_count));
                }
            }
            MirCallee::Static(CallableIdentity::Builtin(id)) => {
                let name = self.mir_builtin_call_name(id)?;
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                if has_expansion {
                    self.emit(Instr::CallBuiltinExpandMultiOutput(
                        name,
                        specs,
                        output_count,
                    ));
                } else {
                    self.emit(Instr::CallBuiltinMulti(name, call.args.len(), output_count));
                }
            }
            MirCallee::Static(identity) => {
                let fallback_policy = call.fallback_policy;
                if !fallback_policy.supports_vm_static_call() {
                    return Err(self
                        .compile_error(format!(
                            "MIR call fallback policy {:?} is not supported for static callee {:?}",
                            fallback_policy, identity
                        ))
                        .with_identifier(IDENT_MIR_CALL_FALLBACK_POLICY_UNSUPPORTED));
                }
                if self.mir_runtime_name_callee(identity).is_none() {
                    return Err(self
                        .compile_error(format!(
                            "MIR static call callee identity {:?} is missing a valid runtime name shape",
                            identity
                        ))
                        .with_identifier(IDENT_MIR_CALL_TARGET_NAME_INVALID));
                }
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                if has_expansion {
                    self.emit(Instr::CallFunctionExpandMultiOutput {
                        identity: identity.clone(),
                        fallback_policy,
                        specs,
                        out_count: output_count,
                    });
                } else {
                    self.emit(Instr::CallFunctionMulti {
                        identity: identity.clone(),
                        fallback_policy,
                        arg_count: call.args.len(),
                        out_count: output_count,
                    });
                }
            }
        }
        Ok(())
    }

    fn compile_mir_assign(
        &mut self,
        place: &MirPlace,
        value: &MirRvalue,
        delete: bool,
    ) -> Result<(), CompileError> {
        if !delete {
            if let MirPlace::Index(_, indexing) = place {
                if indexing.result_context == IndexResultContext::DeletionTarget {
                    return Err(self
                        .compile_error(
                            "MIR assignment invariant violated: DeletionTarget index context requires delete mutation",
                        )
                        .with_identifier(IDENT_MIR_DELETION_CONTEXT_WITHOUT_DELETE_INVALID));
                }
            }
        }
        if delete {
            let MirPlace::Index(_, indexing) = place else {
                return Err(self
                    .compile_error(
                        "MIR delete assignment invariant violated: delete mutation requires indexed assignment target",
                    )
                    .with_identifier(IDENT_MIR_DELETE_ASSIGNMENT_TARGET_INVALID));
            };
            if indexing.kind != IndexKind::Paren {
                return Err(self
                    .compile_error(
                        "MIR delete assignment invariant violated: delete mutation currently requires paren indexing",
                    )
                    .with_identifier(IDENT_MIR_DELETE_ASSIGNMENT_INDEX_KIND_INVALID));
            }
            if indexing.result_context != IndexResultContext::DeletionTarget {
                return Err(self
                    .compile_error(
                        "MIR delete assignment invariant violated: delete mutation requires DeletionTarget index context",
                    )
                    .with_identifier(IDENT_MIR_DELETE_ASSIGNMENT_CONTEXT_INVALID));
            }
            if !self.mir_delete_rhs_is_empty_tensor_literal(value) {
                return Err(self
                    .compile_error(
                        "MIR delete assignment requires an empty tensor literal RHS at compile boundary",
                    )
                    .with_identifier(IDENT_MIR_DELETE_ASSIGNMENT_RHS_INVALID));
            }
        }
        match place {
            MirPlace::Local(_) | MirPlace::Binding(_) => {
                self.compile_mir_rvalue(value)?;
                let slot = self.mir_place_slot(place)?;
                self.emit(Instr::StoreVar(slot));
                Ok(())
            }
            MirPlace::Index(base, indexing) => {
                if let Ok(base_slot) = self.mir_place_slot(base) {
                    self.emit(Instr::LoadVar(base_slot));
                    self.compile_mir_index_assignment_after_base(indexing, value, delete)?;
                    self.emit(Instr::StoreVar(base_slot));
                    return Ok(());
                }
                self.compile_mir_place_read(base)?;
                self.compile_mir_index_assignment_after_base(indexing, value, delete)?;
                self.emit_store_back_mir_member_chain(base)
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

    fn compile_mir_index_assignment_after_base(
        &mut self,
        indexing: &MirIndexing,
        value: &MirRvalue,
        delete: bool,
    ) -> Result<(), CompileError> {
        match indexing.kind {
            IndexKind::Paren => match indexing.plan {
                MirIndexPlan::Scalar => {
                    self.compile_mir_scalar_index_components(indexing)?;
                    self.compile_mir_rvalue(value)?;
                    self.emit(if delete {
                        Instr::StoreIndexDelete(indexing.components.len())
                    } else {
                        Instr::StoreIndex(indexing.components.len())
                    });
                }
                MirIndexPlan::SliceExpr => {
                    let components = self.compile_mir_slice_expr_components(
                        indexing,
                        MirRangeParamOrder::AfterNumeric,
                    )?;
                    self.compile_mir_rvalue(value)?;
                    if delete {
                        self.emit(Instr::StoreSliceExprDelete {
                            dims: indexing.components.len(),
                            numeric_count: components.numeric_count,
                            colon_mask: components.colon_mask,
                            end_mask: components.end_mask,
                            range_dims: components.range_dims,
                            range_has_step: components.range_has_step,
                            range_start_exprs: components.range_start_exprs,
                            range_step_exprs: components.range_step_exprs,
                            range_end_exprs: components.range_end_exprs,
                            end_numeric_exprs: components.end_numeric_exprs,
                        });
                    } else {
                        self.emit(Instr::StoreSliceExpr {
                            dims: indexing.components.len(),
                            numeric_count: components.numeric_count,
                            colon_mask: components.colon_mask,
                            end_mask: components.end_mask,
                            range_dims: components.range_dims,
                            range_has_step: components.range_has_step,
                            range_start_exprs: components.range_start_exprs,
                            range_step_exprs: components.range_step_exprs,
                            range_end_exprs: components.range_end_exprs,
                            end_numeric_exprs: components.end_numeric_exprs,
                        });
                    }
                }
                MirIndexPlan::Slice => {
                    let (numeric_count, colon_mask, end_mask) =
                        self.compile_mir_slice_components(indexing)?;
                    self.compile_mir_rvalue(value)?;
                    self.emit(if delete {
                        Instr::StoreSliceDelete(
                            indexing.components.len(),
                            numeric_count,
                            colon_mask,
                            end_mask,
                        )
                    } else {
                        Instr::StoreSlice(
                            indexing.components.len(),
                            numeric_count,
                            colon_mask,
                            end_mask,
                        )
                    });
                }
                MirIndexPlan::Cell => {
                    return Err(self
                        .compile_error("MIR paren assignment lowering received cell index plan")
                        .with_identifier(IDENT_MIR_PAREN_CELL_PLAN_INVALID));
                }
            },
            IndexKind::Brace => {
                let (end_offsets, end_exprs) = self.compile_mir_cell_index_components(
                    indexing,
                    IndexResultContext::AssignmentTarget,
                )?;
                self.compile_mir_rvalue(value)?;
                self.emit(if delete {
                    Instr::StoreIndexCellDelete {
                        num_indices: indexing.components.len(),
                        end_offsets: end_offsets.clone(),
                        end_exprs: end_exprs.clone(),
                    }
                } else {
                    Instr::StoreIndexCell {
                        num_indices: indexing.components.len(),
                        end_offsets,
                        end_exprs,
                    }
                });
            }
        };
        Ok(())
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
                self.compile_mir_store_indexed_value_from_temp(indexing, tmp, false)?;
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
            IndexKind::Paren => self.compile_mir_slice_index(indexing)?,
            IndexKind::Brace => {
                let (end_offsets, end_exprs) =
                    self.compile_mir_cell_index_components_any_context(indexing)?;
                self.emit(Instr::IndexCell {
                    num_indices: indexing.components.len(),
                    end_offsets,
                    end_exprs,
                });
            }
        }
        Ok(())
    }

    fn compile_mir_store_indexed_value_from_temp(
        &mut self,
        indexing: &MirIndexing,
        tmp: usize,
        delete: bool,
    ) -> Result<(), CompileError> {
        match indexing.kind {
            IndexKind::Paren => {
                match indexing.plan {
                    MirIndexPlan::Scalar => {
                        self.compile_mir_scalar_index_components(indexing)?;
                        self.emit(Instr::LoadVar(tmp));
                        self.emit(if delete {
                            Instr::StoreIndexDelete(indexing.components.len())
                        } else {
                            Instr::StoreIndex(indexing.components.len())
                        });
                    }
                    MirIndexPlan::SliceExpr => {
                        let components = self.compile_mir_slice_expr_components(
                            indexing,
                            MirRangeParamOrder::AfterNumeric,
                        )?;
                        self.emit(Instr::LoadVar(tmp));
                        if delete {
                            self.emit(Instr::StoreSliceExprDelete {
                                dims: indexing.components.len(),
                                numeric_count: components.numeric_count,
                                colon_mask: components.colon_mask,
                                end_mask: components.end_mask,
                                range_dims: components.range_dims,
                                range_has_step: components.range_has_step,
                                range_start_exprs: components.range_start_exprs,
                                range_step_exprs: components.range_step_exprs,
                                range_end_exprs: components.range_end_exprs,
                                end_numeric_exprs: components.end_numeric_exprs,
                            });
                        } else {
                            self.emit(Instr::StoreSliceExpr {
                                dims: indexing.components.len(),
                                numeric_count: components.numeric_count,
                                colon_mask: components.colon_mask,
                                end_mask: components.end_mask,
                                range_dims: components.range_dims,
                                range_has_step: components.range_has_step,
                                range_start_exprs: components.range_start_exprs,
                                range_step_exprs: components.range_step_exprs,
                                range_end_exprs: components.range_end_exprs,
                                end_numeric_exprs: components.end_numeric_exprs,
                            });
                        }
                    }
                    MirIndexPlan::Slice => {
                        let (numeric_count, colon_mask, end_mask) =
                            self.compile_mir_slice_components(indexing)?;
                        self.emit(Instr::LoadVar(tmp));
                        self.emit(if delete {
                            Instr::StoreSliceDelete(
                                indexing.components.len(),
                                numeric_count,
                                colon_mask,
                                end_mask,
                            )
                        } else {
                            Instr::StoreSlice(
                                indexing.components.len(),
                                numeric_count,
                                colon_mask,
                                end_mask,
                            )
                        });
                    }
                    MirIndexPlan::Cell => {
                        return Err(self
                            .compile_error("MIR paren assignment lowering received cell index plan")
                            .with_identifier(IDENT_MIR_PAREN_CELL_PLAN_INVALID));
                    }
                }
                Ok(())
            }
            IndexKind::Brace => {
                let (end_offsets, end_exprs) =
                    self.compile_mir_cell_index_components_any_context(indexing)?;
                self.emit(Instr::LoadVar(tmp));
                self.emit(if delete {
                    Instr::StoreIndexCellDelete {
                        num_indices: indexing.components.len(),
                        end_offsets: end_offsets.clone(),
                        end_exprs: end_exprs.clone(),
                    }
                } else {
                    Instr::StoreIndexCell {
                        num_indices: indexing.components.len(),
                        end_offsets,
                        end_exprs,
                    }
                });
                Ok(())
            }
        }
    }

    fn compile_mir_return(&mut self, values: &[MirOperand]) -> Result<(), CompileError> {
        match values.len() {
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
                    OperatorKind::Not => self.emit(Instr::LogicalNot),
                    OperatorKind::Transpose => self.emit(Instr::Transpose),
                    OperatorKind::ConjugateTranspose => self.emit(Instr::ConjugateTranspose),
                    _ => {
                        return Err(self
                            .compile_error(format!("operator {op:?} is not a MIR unary operator"))
                            .with_identifier(IDENT_MIR_OPERATOR_UNSUPPORTED))
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
                    OperatorKind::ElementwiseAnd => self.emit(Instr::LogicalAnd),
                    OperatorKind::ElementwiseOr => self.emit(Instr::LogicalOr),
                    _ => {
                        return Err(self
                            .compile_error(format!(
                                "operator {op:?} is not supported in primary MIR lowering yet"
                            ))
                            .with_identifier(IDENT_MIR_OPERATOR_UNSUPPORTED))
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
            MirRvalue::Future {
                function,
                args,
                requested_outputs,
                ..
            } => {
                let (specs, has_expansion) = self.mir_call_arg_specs(args);
                for arg in args {
                    self.compile_mir_call_arg(arg)?;
                }
                let out_count = requested_outputs.fixed_count();
                if has_expansion {
                    self.emit(Instr::CreateSemanticFutureExpandMultiOutput(
                        *function, specs, out_count,
                    ));
                } else {
                    self.emit(Instr::CreateSemanticFuture(
                        *function,
                        args.len(),
                        out_count,
                    ));
                }
                Ok(())
            }
            MirRvalue::Spawn(operand) => {
                self.compile_mir_operand(operand)?;
                self.emit(Instr::Spawn);
                Ok(())
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
        let requested_outputs = self.resolved_call_output_count(call)?;

        let (specs, has_expansion) = self.mir_call_arg_specs(&call.args);
        if matches!(call.syntax, CallSyntax::Method | CallSyntax::DottedInvoke) {
            match &call.callee {
                MirCallee::Static(CallableIdentity::SemanticFunction(_))
                | MirCallee::Dynamic(_) => {
                    return Err(self
                        .compile_error(
                            "MIR method-call lowering expected a non-semantic static callee",
                        )
                        .with_identifier(IDENT_MIR_METHOD_CALL_CALLEE_INVALID));
                }
                MirCallee::Static(_) => return self.compile_mir_method_call(call, has_expansion),
            }
        }
        match &call.callee {
            MirCallee::Static(CallableIdentity::SemanticFunction(function)) => {
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                if has_expansion {
                    self.emit(Instr::CallSemanticFunctionExpandMultiOutput(
                        *function,
                        specs,
                        requested_outputs,
                    ));
                } else {
                    self.emit(Instr::CallSemanticFunctionMulti(
                        *function,
                        call.args.len(),
                        requested_outputs,
                    ));
                }
            }
            MirCallee::Dynamic(callee) => {
                self.compile_mir_operand(callee)?;
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                if has_expansion {
                    self.emit(Instr::CallFevalExpandMultiOutput(specs, requested_outputs));
                } else {
                    self.emit(Instr::CallFevalMulti(call.args.len(), requested_outputs));
                }
            }
            MirCallee::Static(CallableIdentity::Builtin(id)) => {
                let name = self.mir_builtin_call_name(id)?;
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                if has_expansion {
                    self.emit(Instr::CallBuiltinExpandMultiOutput(
                        name,
                        specs,
                        requested_outputs,
                    ));
                } else {
                    self.emit(Instr::CallBuiltinMulti(
                        name,
                        call.args.len(),
                        requested_outputs,
                    ));
                }
            }
            MirCallee::Static(identity) => {
                let fallback_policy = call.fallback_policy;
                if !fallback_policy.supports_vm_static_call() {
                    return Err(self
                        .compile_error(format!(
                            "MIR call fallback policy {:?} is not supported for static callee {:?}",
                            fallback_policy, identity
                        ))
                        .with_identifier(IDENT_MIR_CALL_FALLBACK_POLICY_UNSUPPORTED));
                }
                if self.mir_runtime_name_callee(identity).is_none() {
                    return Err(self
                        .compile_error(format!(
                            "MIR static call callee identity {:?} is missing a valid runtime name shape",
                            identity
                        ))
                        .with_identifier(IDENT_MIR_CALL_TARGET_NAME_INVALID));
                }
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                if has_expansion {
                    self.emit(Instr::CallFunctionExpandMultiOutput {
                        identity: identity.clone(),
                        fallback_policy,
                        specs,
                        out_count: requested_outputs,
                    });
                } else {
                    self.emit(Instr::CallFunctionMulti {
                        identity: identity.clone(),
                        fallback_policy,
                        arg_count: call.args.len(),
                        out_count: requested_outputs,
                    });
                }
            }
        }
        Ok(())
    }

    fn call_requested_output_count(&self, call: &MirCall) -> Result<usize, CompileError> {
        Ok(call.requested_outputs.fixed_count())
    }

    fn resolved_call_output_count(&self, call: &MirCall) -> Result<usize, CompileError> {
        self.call_requested_output_count(call)
    }

    fn output_count_for_targets(
        &self,
        targets: &runmat_mir::MirOutputTargetList,
    ) -> Result<usize, CompileError> {
        targets
            .validate_fixed_arity("MIR multi-assign")
            .map_err(|message| self.compile_error(message))
    }

    fn mir_runtime_name_callee(&self, callee: &CallableIdentity) -> Option<String> {
        match callee {
            CallableIdentity::ExternalName(runmat_hir::QualifiedName(segments))
                if segments.len() <= 1 =>
            {
                None
            }
            CallableIdentity::Imported(path) => imported_handle_runtime_name(path),
            _ => strict_callable_display_name(callee),
        }
    }

    fn compile_mir_method_call(
        &mut self,
        call: &MirCall,
        has_expansion: bool,
    ) -> Result<(), CompileError> {
        let MirCallee::Static(identity) = &call.callee else {
            return Err(self
                .compile_error("internal error: method-call lowering expected a static callee")
                .with_identifier(IDENT_MIR_METHOD_CALL_CALLEE_INVALID));
        };
        let identity = identity.clone();
        let fallback_policy = call.fallback_policy;
        if !fallback_policy.supports_vm_method_or_member_call() {
            return Err(self
                .compile_error(format!(
                    "MIR method-call fallback policy {:?} is not supported for callee {:?}",
                    fallback_policy, identity
                ))
                .with_identifier(IDENT_MIR_METHOD_FALLBACK_POLICY_UNSUPPORTED));
        }
        if call.args.is_empty() {
            return Err(self
                .compile_error("MIR method calls require a base receiver")
                .with_identifier(IDENT_MIR_METHOD_CALL_RECEIVER_MISSING));
        }
        for arg in &call.args {
            self.compile_mir_call_arg(arg)?;
        }
        if has_expansion {
            let (specs, _) = self.mir_call_arg_specs(&call.args);
            let output_count = self.resolved_call_output_count(call)?;
            self.emit(Instr::CallMethodOrMemberIndexExpandMultiOutput {
                identity,
                fallback_policy,
                specs,
                out_count: output_count,
            });
            return Ok(());
        }
        let argc = call.args.len().saturating_sub(1);
        let output_count = self.resolved_call_output_count(call)?;
        self.emit(Instr::CallMethodOrMemberIndexMulti {
            identity,
            fallback_policy,
            arg_count: argc,
            out_count: output_count,
        });
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
                    if let Some(offset) = self.mir_operand_cell_end_offset(index) {
                        self.emit(Instr::LoadConst(encode_cell_end_offset(offset)));
                    } else {
                        self.compile_mir_operand(index)?;
                    }
                }
                Ok(())
            }
        }
    }

    fn mir_builtin_call_name(
        &self,
        builtin: &runmat_hir::BuiltinId,
    ) -> Result<String, CompileError> {
        let candidate = builtin.0.clone();
        if is_vm_intrinsic_counter_builtin(&candidate) {
            return Ok(candidate);
        }
        if let Some(builtin) = runmat_builtins::builtin_function_by_name(&candidate) {
            return Ok(builtin.name.to_string());
        }
        Err(CompileError::new(format!("unknown builtin id {candidate}"))
            .with_identifier(IDENT_MIR_BUILTIN_UNKNOWN))
    }

    fn compile_mir_aggregate(
        &mut self,
        kind: &MirAggregateKind,
        rows: usize,
        cols: usize,
        elements: &[MirOperand],
    ) -> Result<(), CompileError> {
        if rows.checked_mul(cols) != Some(elements.len()) {
            return Err(self
                .compile_error("MIR aggregate shape does not match aggregate element count")
                .with_identifier(IDENT_MIR_AGGREGATE_SHAPE_INVALID));
        }

        match kind {
            MirAggregateKind::Tensor if self.mir_aggregate_needs_dynamic_concat(elements) => {
                for element in elements {
                    self.compile_mir_operand(element)?;
                }
                for _ in 0..rows {
                    self.emit(Instr::LoadConst(cols as f64));
                }
                self.emit(Instr::CreateMatrixDynamic(rows));
            }
            MirAggregateKind::Tensor => {
                for element in elements {
                    self.compile_mir_operand(element)?;
                }
                self.emit(Instr::CreateMatrix(rows, cols));
            }
            MirAggregateKind::Cell => {
                for element in elements {
                    self.compile_mir_operand(element)?;
                }
                self.emit(Instr::CreateCell2D(rows, cols));
            }
        };
        Ok(())
    }

    fn mir_aggregate_needs_dynamic_concat(&self, elements: &[MirOperand]) -> bool {
        elements
            .iter()
            .any(|element| self.mir_operand_needs_dynamic_concat(element))
    }

    fn mir_delete_rhs_is_empty_tensor_literal(&self, value: &MirRvalue) -> bool {
        matches!(
            value,
            MirRvalue::Aggregate {
                kind: MirAggregateKind::Tensor,
                rows: 0,
                cols: 0,
                elements,
            } if elements.is_empty()
        )
    }

    fn mir_operand_needs_dynamic_concat(&self, operand: &MirOperand) -> bool {
        match operand {
            MirOperand::Constant(MirConstant::String(_)) => true,
            MirOperand::Local(local) => self
                .mir_local_rvalue(*local)
                .is_some_and(|value| self.mir_rvalue_needs_dynamic_concat(&value)),
            _ => false,
        }
    }

    fn mir_rvalue_needs_dynamic_concat(&self, value: &MirRvalue) -> bool {
        matches!(
            value,
            MirRvalue::Range { .. }
                | MirRvalue::Call(_)
                | MirRvalue::Aggregate { .. }
                | MirRvalue::Index { .. }
                | MirRvalue::Member { .. }
                | MirRvalue::DynamicMember { .. }
        )
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
            return Err(self
                .compile_error(
                    "MIR index lowering expected ReadSingle/ReadCommaList result context",
                )
                .with_identifier(IDENT_MIR_INDEX_CONTEXT_INVALID));
        }

        if indexing.kind == IndexKind::Brace
            && matches!(indexing.result_context, IndexResultContext::ReadCommaList)
        {
            return self.compile_mir_cell_list(base, indexing);
        }

        self.compile_mir_operand(base)?;
        match indexing.kind {
            IndexKind::Paren => self.compile_mir_slice_index(indexing)?,
            IndexKind::Brace => {
                let (end_offsets, end_exprs) = self
                    .compile_mir_cell_index_components(indexing, indexing.result_context.clone())?;
                self.emit(Instr::IndexCell {
                    num_indices: indexing.components.len(),
                    end_offsets,
                    end_exprs,
                });
            }
        };
        Ok(())
    }

    fn compile_mir_cell_index_components(
        &mut self,
        indexing: &MirIndexing,
        expected_context: IndexResultContext,
    ) -> Result<(Vec<(usize, isize)>, Vec<(usize, EndExpr)>), CompileError> {
        if !mir_indexing_context_matches(indexing.result_context.clone(), expected_context) {
            return Err(self
                .compile_error("MIR cell index lowering received mismatched index result context")
                .with_identifier(IDENT_MIR_CELL_INDEX_CONTEXT_INVALID));
        }
        let mut end_offsets = Vec::new();
        let mut end_exprs = Vec::new();
        let mut index_position = 0usize;
        for component in &indexing.components {
            match component {
                MirIndexComponent::Expr(operand) => {
                    self.compile_mir_operand(operand)?;
                    if let Some(expr) = self.mir_operand_end_expr(operand) {
                        if let Some(offset) = Self::mir_cell_end_offset_from_expr(&expr) {
                            end_offsets.push((index_position, offset));
                        } else {
                            end_exprs.push((index_position, expr));
                        }
                    }
                    index_position += 1;
                }
                MirIndexComponent::End { offset, .. } => {
                    self.emit(Instr::LoadConst(encode_cell_end_offset(*offset)));
                    end_offsets.push((index_position, *offset));
                    index_position += 1;
                }
                _ => {
                    return Err(self
                        .compile_error(
                            "MIR cell index lowering expects expression selectors or end-relative selectors",
                        )
                        .with_identifier(IDENT_MIR_CELL_INDEX_PLAN_INVALID))
                }
            }
        }
        Ok((end_offsets, end_exprs))
    }

    fn compile_mir_cell_index_components_any_context(
        &mut self,
        indexing: &MirIndexing,
    ) -> Result<(Vec<(usize, isize)>, Vec<(usize, EndExpr)>), CompileError> {
        let mut end_offsets = Vec::new();
        let mut end_exprs = Vec::new();
        let mut index_position = 0usize;
        for component in &indexing.components {
            match component {
                MirIndexComponent::Expr(operand) => {
                    self.compile_mir_operand(operand)?;
                    if let Some(expr) = self.mir_operand_end_expr(operand) {
                        if let Some(offset) = Self::mir_cell_end_offset_from_expr(&expr) {
                            end_offsets.push((index_position, offset));
                        } else {
                            end_exprs.push((index_position, expr));
                        }
                    }
                    index_position += 1;
                }
                MirIndexComponent::End { offset, .. } => {
                    self.emit(Instr::LoadConst(encode_cell_end_offset(*offset)));
                    end_offsets.push((index_position, *offset));
                    index_position += 1;
                }
                _ => {
                    return Err(self
                        .compile_error(
                            "MIR cell index lowering expects expression selectors or end-relative selectors",
                        )
                        .with_identifier(IDENT_MIR_CELL_INDEX_PLAN_INVALID))
                }
            }
        }
        Ok((end_offsets, end_exprs))
    }

    fn compile_mir_slice_index(&mut self, indexing: &MirIndexing) -> Result<(), CompileError> {
        match indexing.plan {
            MirIndexPlan::Scalar => {
                self.compile_mir_scalar_index_components(indexing)?;
                self.emit(Instr::Index(indexing.components.len()));
                Ok(())
            }
            MirIndexPlan::SliceExpr => {
                let components = self.compile_mir_slice_expr_components(
                    indexing,
                    MirRangeParamOrder::BeforeNumeric,
                )?;
                self.emit(Instr::IndexSliceExpr {
                    dims: indexing.components.len(),
                    numeric_count: components.numeric_count,
                    colon_mask: components.colon_mask,
                    end_mask: components.end_mask,
                    range_dims: components.range_dims,
                    range_has_step: components.range_has_step,
                    range_start_exprs: components.range_start_exprs,
                    range_step_exprs: components.range_step_exprs,
                    range_end_exprs: components.range_end_exprs,
                    end_numeric_exprs: components.end_numeric_exprs,
                });
                Ok(())
            }
            MirIndexPlan::Slice => {
                let (numeric_count, colon_mask, end_mask) =
                    self.compile_mir_slice_components(indexing)?;
                self.emit(Instr::IndexSlice(
                    indexing.components.len(),
                    numeric_count,
                    colon_mask,
                    end_mask,
                ));
                Ok(())
            }
            MirIndexPlan::Cell => Err(self
                .compile_error("MIR paren index lowering received cell index plan")
                .with_identifier(IDENT_MIR_PAREN_CELL_PLAN_INVALID)),
        }
    }

    fn compile_mir_scalar_index_components(
        &mut self,
        indexing: &MirIndexing,
    ) -> Result<(), CompileError> {
        for component in &indexing.components {
            let MirIndexComponent::Expr(operand) = component else {
                return Err(self
                    .compile_error("scalar index lowering expects expression selectors only")
                    .with_identifier(IDENT_MIR_SCALAR_INDEX_PLAN_INVALID));
            };
            if self.mir_operand_range_end_spec(operand).is_some()
                || self.mir_operand_end_expr(operand).is_some()
            {
                return Err(self
                    .compile_error(
                        "scalar index lowering invariant violated: range/end selectors must lower through IndexSliceExpr",
                    )
                    .with_identifier(IDENT_MIR_SCALAR_INDEX_PLAN_INVALID));
            }
            self.compile_mir_operand(operand)?;
        }
        Ok(())
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
                MirIndexComponent::Expr(operand)
                    if self.mir_operand_range_needs_slice_expr(operand)
                        || self.mir_operand_end_expr(operand).is_some() =>
                {
                    return Err(self
                        .compile_error(
                            "MIR slice lowering invariant violated: range/end selectors must lower through IndexSliceExpr",
                        )
                        .with_identifier(IDENT_MIR_SLICE_INDEX_PLAN_INVALID))
                }
                MirIndexComponent::Expr(operand) => {
                    self.compile_mir_operand(operand)?;
                    numeric_count += 1;
                }
                MirIndexComponent::End { .. } => {
                    return Err(self
                        .compile_error(
                            "MIR slice lowering invariant violated: nonzero end offset must lower through IndexSliceExpr",
                        )
                        .with_identifier(IDENT_MIR_SLICE_INDEX_PLAN_INVALID))
                }
            }
        }

        Ok((numeric_count, colon_mask, end_mask))
    }

    fn compile_mir_slice_expr_components(
        &mut self,
        indexing: &MirIndexing,
        range_order: MirRangeParamOrder,
    ) -> Result<MirSliceExprComponents, CompileError> {
        let mut colon_mask = 0u32;
        let end_mask = 0u32;
        let mut numeric_count = 0usize;
        let mut numeric_operands = Vec::new();
        let mut range_params = Vec::new();
        let mut range_dims = Vec::new();
        let mut range_has_step = Vec::new();
        let mut range_start_exprs = Vec::new();
        let mut range_step_exprs = Vec::new();
        let mut range_end_exprs = Vec::new();
        let mut end_numeric_exprs = Vec::new();

        for (dim, component) in indexing.components.iter().enumerate() {
            match component {
                MirIndexComponent::Colon => colon_mask |= 1u32 << dim,
                MirIndexComponent::End { offset, .. } => {
                    numeric_operands.push(None);
                    if *offset == 0 {
                        end_numeric_exprs.push((numeric_count, EndExpr::End));
                    } else {
                        end_numeric_exprs.push((numeric_count, end_expr_with_offset(*offset)));
                    }
                    numeric_count += 1;
                }
                MirIndexComponent::Expr(operand)
                    if self.mir_operand_range_end_spec(operand).is_some() =>
                {
                    let spec = self.mir_operand_range_end_spec(operand).ok_or_else(|| {
                        self.compile_error("MIR range end expression disappeared during lowering")
                    })?;
                    range_dims.push(dim);
                    range_has_step.push(spec.has_step);
                    range_start_exprs.push(spec.start_expr.clone());
                    range_step_exprs.push(spec.step_expr.clone());
                    range_end_exprs.push(spec.end_expr.clone());
                    range_params.push((operand.clone(), spec));
                }
                MirIndexComponent::Expr(operand)
                    if self.mir_operand_end_expr(operand).is_some() =>
                {
                    numeric_operands.push(None);
                    end_numeric_exprs.push((
                        numeric_count,
                        self.mir_operand_end_expr(operand).ok_or_else(|| {
                            self.compile_error("MIR end expression disappeared during lowering")
                        })?,
                    ));
                    numeric_count += 1;
                }
                MirIndexComponent::Expr(operand) => {
                    numeric_operands.push(Some(operand.clone()));
                    numeric_count += 1;
                }
            }
        }

        if matches!(range_order, MirRangeParamOrder::AfterNumeric) {
            self.emit_mir_slice_numeric_operands(&numeric_operands)?;
        }
        self.emit_mir_slice_range_params(&range_params)?;
        if matches!(range_order, MirRangeParamOrder::BeforeNumeric) {
            self.emit_mir_slice_numeric_operands(&numeric_operands)?;
        }

        Ok(MirSliceExprComponents {
            numeric_count,
            colon_mask,
            end_mask,
            range_dims,
            range_has_step,
            range_start_exprs,
            range_step_exprs,
            range_end_exprs,
            end_numeric_exprs,
        })
    }

    fn emit_mir_slice_numeric_operands(
        &mut self,
        operands: &[Option<MirOperand>],
    ) -> Result<(), CompileError> {
        for operand in operands {
            if let Some(operand) = operand {
                self.compile_mir_operand(operand)?;
            } else {
                self.emit(Instr::LoadConst(0.0));
            }
        }
        Ok(())
    }

    fn emit_mir_slice_range_params(
        &mut self,
        params: &[(MirOperand, MirRangeEndSpec)],
    ) -> Result<(), CompileError> {
        for (operand, spec) in params {
            let Some(MirRvalue::Range { start, step, .. }) = self.mir_operand_rvalue(operand)
            else {
                return Err(self.compile_error("MIR range index disappeared during lowering"));
            };
            if spec.start_expr.is_some() {
                self.emit(Instr::LoadConst(0.0));
            } else {
                self.compile_mir_operand(&start)?;
            }
            if let Some(step) = step {
                if spec.step_expr.is_some() {
                    self.emit(Instr::LoadConst(0.0));
                } else {
                    self.compile_mir_operand(&step)?;
                }
            }
        }
        Ok(())
    }

    fn mir_operand_end_expr(&self, operand: &MirOperand) -> Option<EndExpr> {
        self.mir_operand_end_expr_internal(operand)
            .and_then(|(expr, has_end)| has_end.then_some(expr))
    }

    fn mir_operand_cell_end_offset(&self, operand: &MirOperand) -> Option<isize> {
        let (expr, has_end) = self.mir_operand_end_expr_internal(operand)?;
        has_end
            .then(|| Self::mir_cell_end_offset_from_expr(&expr))
            .flatten()
    }

    fn mir_cell_end_offset_from_expr(expr: &EndExpr) -> Option<isize> {
        match expr {
            EndExpr::End => Some(0),
            EndExpr::Pos(inner) => Self::mir_cell_end_offset_from_expr(inner),
            EndExpr::Add(left, right) => {
                if matches!(left.as_ref(), EndExpr::End) {
                    Some(Self::end_expr_nonnegative_int(right)?)
                } else if matches!(right.as_ref(), EndExpr::End) {
                    Some(Self::end_expr_nonnegative_int(left)?)
                } else {
                    None
                }
            }
            EndExpr::Sub(left, right) if matches!(left.as_ref(), EndExpr::End) => {
                Some(-Self::end_expr_nonnegative_int(right)?)
            }
            _ => None,
        }
    }

    fn end_expr_nonnegative_int(expr: &EndExpr) -> Option<isize> {
        let EndExpr::Const(value) = expr else {
            return None;
        };
        if !value.is_finite() || *value < 0.0 {
            return None;
        }
        let rounded = value.round();
        if (rounded - value).abs() > f64::EPSILON {
            return None;
        }
        if rounded > isize::MAX as f64 {
            return None;
        }
        Some(rounded as isize)
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
        self.mir_local_rvalue(local)
            .and_then(|value| self.mir_rvalue_end_expr_internal(&value))
    }

    fn mir_operand_range_end_spec(&self, operand: &MirOperand) -> Option<MirRangeEndSpec> {
        let MirRvalue::Range { start, step, end } = self.mir_operand_rvalue(operand)? else {
            return None;
        };
        let start_expr = self.mir_operand_end_expr(&start);
        let step_expr = step
            .as_ref()
            .and_then(|step| self.mir_operand_end_expr(step));
        let end_expr = self.mir_operand_end_expr(&end);
        let end_expr = end_expr.or_else(|| self.mir_operand_any_end_expr(&end))?;
        Some(MirRangeEndSpec {
            start_expr,
            step_expr,
            end_expr,
            has_step: step.is_some(),
        })
    }

    fn mir_operand_range_needs_slice_expr(&self, operand: &MirOperand) -> bool {
        let Some(MirRvalue::Range { start, step, end }) = self.mir_operand_rvalue(operand) else {
            return false;
        };
        self.mir_operand_end_expr_internal(&start)
            .map(|(_, has_end)| has_end)
            .unwrap_or(false)
            || step
                .as_ref()
                .and_then(|value| self.mir_operand_end_expr_internal(value))
                .map(|(_, has_end)| has_end)
                .unwrap_or(false)
            || self
                .mir_operand_end_expr_internal(&end)
                .map(|(_, has_end)| has_end)
                .unwrap_or(false)
    }

    fn mir_operand_any_end_expr(&self, operand: &MirOperand) -> Option<EndExpr> {
        self.mir_operand_end_expr_internal(operand)
            .map(|(expr, _)| expr)
    }

    fn mir_operand_rvalue(&self, operand: &MirOperand) -> Option<MirRvalue> {
        match operand {
            MirOperand::Local(local) => self.mir_local_rvalue(*local),
            MirOperand::Constant(_) | MirOperand::FunctionHandle(_) => None,
        }
    }

    fn mir_local_rvalue(&self, local: runmat_mir::MirLocalId) -> Option<MirRvalue> {
        let body = self.body.as_ref()?;
        body.blocks
            .iter()
            .flat_map(|block| block.statements.iter())
            .find_map(|stmt| match &stmt.kind {
                MirStmtKind::Assign {
                    place: MirPlace::Local(candidate),
                    value,
                } if *candidate == local => Some(value.clone()),
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
                    OperatorKind::MatrixMultiply | OperatorKind::ElementwiseMultiply => {
                        EndExpr::Mul(Box::new(left), Box::new(right))
                    }
                    OperatorKind::Mrdivide | OperatorKind::ElementwiseDivide => {
                        EndExpr::Div(Box::new(left), Box::new(right))
                    }
                    OperatorKind::Mldivide | OperatorKind::ElementwiseLeftDivide => {
                        EndExpr::LeftDiv(Box::new(left), Box::new(right))
                    }
                    OperatorKind::MatrixPower | OperatorKind::ElementwisePower => {
                        EndExpr::Pow(Box::new(left), Box::new(right))
                    }
                    _ => return None,
                };
                Some((expr, has_end))
            }
            MirRvalue::Call(call) => self.mir_call_end_expr_internal(call),
            _ => None,
        }
    }

    fn mir_call_end_expr_internal(&self, call: &MirCall) -> Option<(EndExpr, bool)> {
        let identity = match &call.callee {
            MirCallee::Static(identity) => identity.clone(),
            MirCallee::Dynamic(_) => return None,
        };
        let mut args = Vec::with_capacity(call.args.len());
        let mut has_end = false;
        for arg in &call.args {
            let MirCallArg::Single(operand) = arg else {
                return None;
            };
            let (expr, arg_has_end) = self.mir_operand_end_expr_internal(operand)?;
            args.push(expr);
            has_end |= arg_has_end;
        }
        let expr = EndExpr::ResolvedCall {
            identity,
            fallback_policy: call.fallback_policy,
            args,
        };
        Some((expr, has_end))
    }

    fn compile_mir_operand(&mut self, operand: &MirOperand) -> Result<(), CompileError> {
        match operand {
            MirOperand::Local(local) => {
                let slot = self.mir_local_slot(*local)?;
                self.emit(Instr::LoadVar(slot));
                Ok(())
            }
            MirOperand::Constant(MirConstant::Number(value)) => {
                let value = value.parse().map_err(|_| {
                    self.compile_error(format!("invalid number literal {value:?}"))
                        .with_identifier(IDENT_MIR_NUMBER_LITERAL_INVALID)
                })?;
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
                    .ok_or_else(|| {
                        self.compile_error(format!("unknown constant {name}"))
                            .with_identifier(IDENT_MIR_CONSTANT_UNKNOWN)
                    })?;
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
        }
    }

    fn compile_mir_function_handle(
        &mut self,
        target: &CallableIdentity,
    ) -> Result<(), CompileError> {
        match target {
            CallableIdentity::Method(_) => Err(self
                .compile_error(
                    "method function-handle lowering requires explicit semantic/defpath ABI",
                )
                .with_identifier(IDENT_MIR_FUNCTION_HANDLE_TARGET_UNSUPPORTED)),
            CallableIdentity::Builtin(_)
            | CallableIdentity::DynamicName(_)
            | CallableIdentity::ExternalName(_)
            | CallableIdentity::Imported(_) => {
                let name = self.mir_runtime_name_callee(target).ok_or_else(|| {
                    self.compile_error(format!(
                        "missing runtime name for function handle target {target:?}"
                    ))
                    .with_identifier(IDENT_MIR_FUNCTION_HANDLE_NAME_MISSING)
                })?;
                if matches!(
                    target,
                    CallableIdentity::ExternalName(_) | CallableIdentity::Imported(_)
                ) {
                    self.emit(Instr::CreateExternalFunctionHandle(name));
                } else {
                    self.emit(Instr::CreateFunctionHandle(name));
                }
                Ok(())
            }
            CallableIdentity::AnonymousFunction(function) => {
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
                    let slot = self.binding_slot(capture.binding)?;
                    self.emit(Instr::LoadVar(slot));
                }
                self.emit(Instr::CreateSemanticClosure(
                    *function,
                    display_name,
                    captures.len(),
                ));
                Ok(())
            }
            CallableIdentity::SemanticFunction(function) => {
                let Some((captures, display_name)) = self
                    .layout
                    .as_ref()
                    .and_then(|layout| layout.functions.get(function))
                    .map(|layout| (layout.captures.clone(), layout.display_name.clone()))
                else {
                    // External semantic function identities may not have a local VM layout in the
                    // current compilation unit. Keep the identity and emit a simple semantic handle.
                    self.emit(Instr::CreateSemanticFunctionHandle(
                        *function,
                        format!("semantic_function_{}", function.0),
                    ));
                    return Ok(());
                };
                if captures.is_empty() {
                    self.emit(Instr::CreateSemanticFunctionHandle(*function, display_name));
                    return Ok(());
                }
                for capture in &captures {
                    let slot = self.binding_slot(capture.binding)?;
                    self.emit(Instr::LoadVar(slot));
                }
                self.emit(Instr::CreateSemanticClosure(
                    *function,
                    display_name,
                    captures.len(),
                ));
                Ok(())
            }
        }
    }

    fn mir_place_slot(&self, place: &MirPlace) -> Result<usize, CompileError> {
        match place {
            MirPlace::Local(local) => self.mir_local_slot(*local),
            MirPlace::Binding(binding) => self.binding_slot(*binding),
            _ => Err(CompileError::new(format!(
                "expected local or binding place for slot lookup, got {place:?}",
            ))),
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
