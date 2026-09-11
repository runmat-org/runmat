use crate::bytecode::instr::{BytecodeClassMethod, BytecodeClassProperty, PropertyDefaultLiteral};
use crate::call::builtins::is_vm_intrinsic_builtin;
use crate::compiler::CompileError;
use crate::instr::Instr;
use crate::layout::VmAssemblyLayout;
use runmat_builtins::{self, Type};
use runmat_hir::{
    BindingId, CallSyntax, CallableFallbackPolicy, CallableIdentity, EntrypointId, FunctionId,
    HirAssembly, IndexKind, IndexResultContext, OperatorKind, RequestedOutputCount,
};
use runmat_mir::{
    BasicBlockId, MirAggregateKind, MirAssembly, MirBody, MirCall, MirCallArg, MirCallee,
    MirConstant, MirIndexComponent, MirIndexPlan, MirIndexing, MirOperand, MirOutputTarget,
    MirPlace, MirPlaceMutation, MirRvalue, MirShortCircuitOp, MirStmt, MirStmtKind,
    MirTerminatorKind,
};
use runmat_runtime::call::arguments::{ArgumentExpansionSpec, ArgumentSpec};
use std::collections::HashMap;

mod subscript_path;

#[derive(Clone)]
pub struct ClassRegistration {
    name: runmat_types::ClassIdentity,
    super_class: Option<runmat_types::ClassIdentity>,
    is_sealed: bool,
    is_abstract: bool,
    properties: Vec<BytecodeClassProperty>,
    methods: Vec<BytecodeClassMethod>,
    enumerations: Vec<String>,
}
type MirCellSelectorCompileResult = (usize, bool);

#[derive(Clone, Copy)]
enum ResolvedCallOutputCount {
    Fixed(usize),
    FromSlot(usize),
}

impl ResolvedCallOutputCount {
    fn require_fixed(self, compiler: &Compiler, message: &str) -> Result<usize, CompileError> {
        match self {
            ResolvedCallOutputCount::Fixed(count) => Ok(count),
            ResolvedCallOutputCount::FromSlot(_) => Err(compiler.compile_error(message)),
        }
    }
}

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
    prepared_index_component: Option<usize>,
    contextual_index_component: Option<usize>,
    subscript_end_component: Option<(usize, usize)>,
}

struct SpanGuard {
    compiler: *mut Compiler,
    prev: Option<runmat_hir::Span>,
}

struct MirStochasticEvolutionPlan {
    state: runmat_mir::MirLocalId,
    drift: MirOperand,
    scale: MirOperand,
    steps: MirOperand,
}

const IDENT_MIR_CELL_EXPAND_PLAN_INVALID: &str = "RunMat:MirCellExpandPlanInvalid";
const IDENT_MIR_PAREN_CELL_PLAN_INVALID: &str = "RunMat:MirParenCellPlanInvalid";
const IDENT_MIR_SCALAR_INDEX_PLAN_INVALID: &str = "RunMat:MirScalarIndexPlanInvalid";
const IDENT_MIR_SLICE_INDEX_PLAN_INVALID: &str = "RunMat:MirSliceIndexPlanInvalid";
const IDENT_MIR_CELL_INDEX_PLAN_INVALID: &str = "RunMat:MirCellIndexPlanInvalid";
const IDENT_MIR_CELL_INDEX_CONTEXT_INVALID: &str = "RunMat:MirCellIndexContextInvalid";
const IDENT_MIR_INDEX_CONTEXT_INVALID: &str = "RunMat:MirIndexContextInvalid";
const IDENT_MIR_SUBSCRIPT_CHAIN_INVALID: &str = "RunMat:MirSubscriptChainInvalid";
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
const IDENT_MIR_CALL_TARGET_NAME_INVALID: &str = "RunMat:MirCallTargetNameInvalid";
const IDENT_MIR_CALL_FALLBACK_POLICY_UNSUPPORTED: &str = "RunMat:MirCallFallbackPolicyUnsupported";
const IDENT_MIR_METHOD_FALLBACK_POLICY_UNSUPPORTED: &str =
    "RunMat:MirMethodFallbackPolicyUnsupported";
const IDENT_MIR_METHOD_CALL_CALLEE_INVALID: &str = "RunMat:MirMethodCallCalleeInvalid";
const IDENT_MIR_METHOD_CALL_RECEIVER_MISSING: &str = "RunMat:MirMethodCallReceiverMissing";

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
    match operand {
        MirOperand::Constant(MirConstant::Number(value)) => value == "1" || value == "1.0",
        MirOperand::Constant(MirConstant::IntegerLiteral(value)) => {
            runmat_value::IntValue::from(value).try_to_u64() == Some(1)
        }
        _ => false,
    }
}

pub(super) fn call_name(call: &MirCall) -> Option<&str> {
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
    let item_name = path.item.last().map(|item| item.display_name())?;
    let item_name = item_name.trim();
    if item_name.is_empty() {
        return None;
    }
    let module_leaf = path.module.0.last()?;
    let module_leaf_name = module_leaf.0.trim();
    if module_leaf_name.is_empty() || module_leaf_name != item_name {
        return None;
    }
    let mut segments = Vec::with_capacity(path.module.0.len());
    for segment in &path.module.0 {
        let normalized = segment.0.trim();
        if normalized.is_empty() {
            return None;
        }
        segments.push(normalized.to_string());
    }
    Some(segments.join("."))
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
            let name = runmat_types::ClassIdentity::from_qualified_name(&class.declaration.name)
                .expect("HIR class declaration must have a canonical identity");
            let mut super_class = class
                .declaration
                .inheritance
                .builtin_super_class
                .as_ref()
                .or(class.declaration.inheritance.declared_super_class.as_ref())
                .map(|name| {
                    runmat_types::ClassIdentity::new(name.clone())
                        .expect("HIR superclass must have a canonical identity")
                })
                .or_else(|| {
                    class
                        .declaration
                        .inheritance
                        .resolved_super_class
                        .and_then(|class_id| {
                            hir.classes
                                .iter()
                                .find(|candidate| candidate.declaration.id == class_id)
                                .map(|super_class| {
                                    runmat_types::ClassIdentity::from_qualified_name(
                                        &super_class.declaration.name,
                                    )
                                    .expect("resolved superclass must have a canonical identity")
                                })
                        })
                });
            if super_class.is_none()
                && matches!(class.declaration.kind, runmat_hir::ClassKind::Handle)
            {
                super_class = Some(runmat_types::standard::HANDLE.owned());
            }
            let properties = class
                .declaration
                .properties
                .iter()
                .map(|property| {
                    let default = class
                        .property_default(&property.name)
                        .and_then(hir_property_default_to_value);
                    BytecodeClassProperty {
                        name: property.name.clone(),
                        is_static: property.attributes.is_static,
                        is_constant: property.attributes.is_constant,
                        is_dependent: property.attributes.is_dependent,
                        default_literal: default,
                        get_access: property.attributes.get_access,
                        set_access: property.attributes.set_access,
                    }
                })
                .collect();
            let methods = class
                .declaration
                .methods
                .iter()
                .map(|method| {
                    let function_name = hir
                        .functions
                        .iter()
                        .find(|function| function.id == method.function)
                        .map(|function| function.name.0.clone())
                        .unwrap_or_else(|| method.name.0.clone());
                    BytecodeClassMethod {
                        name: method.name.clone(),
                        function_name,
                        is_static: method.is_static,
                        is_abstract: method.attributes.is_abstract,
                        is_sealed: method.attributes.is_sealed,
                        access: method.attributes.access,
                    }
                })
                .collect();
            let enumerations = class
                .declaration
                .enumerations
                .iter()
                .map(|enumeration| enumeration.name.0.clone())
                .collect();
            ClassRegistration {
                name,
                super_class,
                is_sealed: class.declaration.is_sealed,
                is_abstract: class.declaration.is_abstract,
                properties,
                methods,
                enumerations,
            }
        })
        .collect()
}

fn hir_property_default_to_value(expr: &runmat_hir::HirExpr) -> Option<PropertyDefaultLiteral> {
    fn eval_numeric(expr: &runmat_hir::HirExpr) -> Option<f64> {
        match &expr.kind {
            runmat_hir::HirExprKind::Number(text) => text.parse::<f64>().ok(),
            runmat_hir::HirExprKind::Unary(runmat_hir::OperatorKind::UnaryPlus, inner) => {
                eval_numeric(inner)
            }
            runmat_hir::HirExprKind::Unary(runmat_hir::OperatorKind::UnaryMinus, inner) => {
                eval_numeric(inner).map(|value| -value)
            }
            runmat_hir::HirExprKind::Binary(left, op, right) => {
                let left = eval_numeric(left)?;
                let right = eval_numeric(right)?;
                match op {
                    runmat_hir::OperatorKind::Add => Some(left + right),
                    runmat_hir::OperatorKind::Subtract => Some(left - right),
                    runmat_hir::OperatorKind::MatrixMultiply
                    | runmat_hir::OperatorKind::ElementwiseMultiply => Some(left * right),
                    runmat_hir::OperatorKind::Mrdivide
                    | runmat_hir::OperatorKind::Mldivide
                    | runmat_hir::OperatorKind::ElementwiseDivide
                    | runmat_hir::OperatorKind::ElementwiseLeftDivide => Some(left / right),
                    runmat_hir::OperatorKind::MatrixPower
                    | runmat_hir::OperatorKind::ElementwisePower => Some(left.powf(right)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    match &expr.kind {
        runmat_hir::HirExprKind::Number(text) => {
            text.parse::<f64>().ok().map(PropertyDefaultLiteral::Num)
        }
        runmat_hir::HirExprKind::IntegerLiteral(value) => Some(PropertyDefaultLiteral::Int(
            runmat_value::IntValue::from(value),
        )),
        runmat_hir::HirExprKind::String(text) => {
            Some(PropertyDefaultLiteral::String(text.0.clone()))
        }
        runmat_hir::HirExprKind::Constant(name) if name.0.eq_ignore_ascii_case("true") => {
            Some(PropertyDefaultLiteral::Bool(true))
        }
        runmat_hir::HirExprKind::Constant(name) if name.0.eq_ignore_ascii_case("false") => {
            Some(PropertyDefaultLiteral::Bool(false))
        }
        _ => eval_numeric(expr).map(PropertyDefaultLiteral::Num),
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
            prepared_index_component: None,
            contextual_index_component: None,
            subscript_end_component: None,
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
            prepared_index_component: None,
            contextual_index_component: None,
            subscript_end_component: None,
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
        body.validate_expression_regions().map_err(|error| {
            CompileError::new(format!("invalid MIR expression region: {error}"))
        })?;

        for registration in self.class_registrations.clone() {
            self.emit(Instr::RegisterClass {
                name: registration.name,
                super_class: registration.super_class,
                is_sealed: registration.is_sealed,
                is_abstract: registration.is_abstract,
                properties: registration.properties,
                methods: registration.methods,
                enumerations: registration.enumerations,
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
        let mut pending_try_entries: Vec<(usize, usize, BasicBlockId, Option<usize>)> = Vec::new();
        let exception_scopes = crate::compiler::exceptions::ExceptionScopes::analyze(body);

        for (block_index, block) in blocks.iter().enumerate() {
            self.pending_place_mutation = None;
            block_starts.insert(block.id, self.instructions.len());
            for scope in exception_scopes.leaving_at(block.id) {
                self.emit(Instr::LeaveTry(*scope));
            }
            for (position, stmt) in block.statements.iter().enumerate() {
                self.record_resume_point(block.id, position)?;
                self.compile_mir_stmt(stmt)?;
            }
            self.record_resume_point(block.id, block.statements.len())?;
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
                MirTerminatorKind::TryCatch {
                    try_block,
                    catch_block,
                    catch_binding,
                } => {
                    let catch_var = catch_binding
                        .map(|local| self.mir_local_slot(local))
                        .transpose()?;
                    let scope = block.id.0;
                    let enter_pc = self.emit(Instr::EnterTry {
                        scope,
                        catch_pc: usize::MAX,
                        catch_var,
                    });
                    pending_try_entries.push((enter_pc, scope, *catch_block, catch_var));
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
                MirTerminatorKind::ParFor {
                    region,
                    iterable,
                    maximum_workers,
                    ..
                } => {
                    self.compile_mir_rvalue(iterable)?;
                    if let Some(maximum_workers) = maximum_workers {
                        self.compile_mir_rvalue(maximum_workers)?;
                    }
                    self.emit(Instr::ExecuteParfor {
                        region: *region,
                        has_maximum_workers: maximum_workers.is_some(),
                    });
                }
                MirTerminatorKind::Spmd { region, header, .. } => {
                    let header = match header.as_ref() {
                        runmat_mir::parallel::MirSpmdHeader::Default => {
                            crate::BytecodeSpmdHeader::Default
                        }
                        runmat_mir::parallel::MirSpmdHeader::One(value) => {
                            self.compile_mir_rvalue(value)?;
                            crate::BytecodeSpmdHeader::Exact
                        }
                        runmat_mir::parallel::MirSpmdHeader::Two(minimum, maximum) => {
                            self.compile_mir_rvalue(minimum)?;
                            self.compile_mir_rvalue(maximum)?;
                            crate::BytecodeSpmdHeader::Range
                        }
                        runmat_mir::parallel::MirSpmdHeader::Three(pool, minimum, maximum) => {
                            self.compile_mir_rvalue(pool)?;
                            self.compile_mir_rvalue(minimum)?;
                            self.compile_mir_rvalue(maximum)?;
                            crate::BytecodeSpmdHeader::PoolRange
                        }
                    };
                    self.emit(Instr::ExecuteSpmd {
                        region: *region,
                        header,
                    });
                }
                MirTerminatorKind::Return(values) => {
                    for scope in exception_scopes.active_at(block.id) {
                        self.emit(Instr::LeaveTry(scope));
                    }
                    if values.is_empty() && block_index + 1 < blocks.len() {
                        self.emit(Instr::Return);
                    } else {
                        self.compile_mir_return(values)?;
                    }
                }
                MirTerminatorKind::Unreachable => {
                    for scope in exception_scopes.active_at(block.id) {
                        self.emit(Instr::LeaveTry(scope));
                    }
                    self.emit(Instr::Return);
                }
                MirTerminatorKind::Await {
                    future,
                    result,
                    resume,
                } => {
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

        for (pc, scope, target, catch_var) in pending_try_entries {
            let target_pc = *block_starts
                .get(&target)
                .ok_or_else(|| CompileError::new(format!("missing MIR catch block {target:?}")))?;
            self.patch(
                pc,
                Instr::EnterTry {
                    scope,
                    catch_pc: target_pc,
                    catch_var,
                },
            );
        }

        Ok(())
    }

    fn record_resume_point(
        &mut self,
        block: BasicBlockId,
        position: usize,
    ) -> Result<(), CompileError> {
        let function = self
            .function
            .ok_or_else(|| CompileError::new("compiler missing selected function"))?;
        let function = u32::try_from(function.0)
            .map(runmat_types::ProgramFunctionId)
            .map_err(|_| CompileError::new("function identity exceeds resume schema"))?;
        let position = u32::try_from(position)
            .map_err(|_| CompileError::new("MIR position exceeds resume schema"))?;
        let point = runmat_types::ProgramPointId {
            function,
            block: u32::try_from(block.0)
                .map_err(|_| CompileError::new("MIR block exceeds resume schema"))?,
            position,
        };
        let layout = self
            .layout
            .as_mut()
            .and_then(|layout| {
                layout
                    .functions
                    .get_mut(&runmat_hir::FunctionId(function.0 as usize))
            })
            .ok_or_else(|| CompileError::new("compiler missing function resume layout"))?;
        if layout
            .resume_points
            .insert(point, self.instructions.len())
            .is_some()
        {
            return Err(CompileError::new("duplicate MIR resume point"));
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
                if !matches!(
                    value,
                    MirRvalue::Member {
                        sequence_use: runmat_types::SequenceUse::Discard,
                        ..
                    } | MirRvalue::DynamicMember {
                        sequence_use: runmat_types::SequenceUse::Discard,
                        ..
                    }
                ) {
                    self.emit(Instr::Pop);
                }
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
            MirStmtKind::SequenceAssign { target, value } => {
                self.pending_place_mutation = None;
                self.compile_mir_sequence_assign(target, value)
            }
            MirStmtKind::CaptureSequence {
                destination,
                source,
            } => {
                self.pending_place_mutation = None;
                self.compile_mir_sequence_capture(destination.0, source)
            }
        }
    }

    fn compile_mir_sequence_capture(
        &mut self,
        sequence_slot: usize,
        source: &runmat_mir::MirExpansionSource,
    ) -> Result<(), CompileError> {
        match source {
            runmat_mir::MirExpansionSource::SubscriptChain(chain) => {
                self.compile_subscript_chain(chain, Some(sequence_slot))?;
            }
            runmat_mir::MirExpansionSource::Member { base, member } => {
                self.compile_mir_operand(base)?;
                self.emit(Instr::CaptureMemberSequence {
                    member: member.clone(),
                    sequence_slot,
                });
            }
            runmat_mir::MirExpansionSource::DynamicMember { base, member } => {
                self.compile_mir_operand(base)?;
                self.compile_mir_operand(member)?;
                self.emit(Instr::CaptureMemberDynamicSequence { sequence_slot });
            }
            runmat_mir::MirExpansionSource::CellContents { base, indexing } => {
                self.compile_mir_operand(base)?;
                let (index_count, expand_all) =
                    self.compile_mir_cell_selector_operands(indexing)?;
                self.emit(Instr::CaptureCellContentsSequence {
                    sequence_slot,
                    num_indices: if expand_all { 0 } else { index_count },
                    expand_all,
                });
            }
            runmat_mir::MirExpansionSource::ReturnedOutputs(base) => {
                self.compile_mir_operand(base)?;
                self.emit(Instr::CaptureReturnedOutputsSequence { sequence_slot });
            }
        }
        Ok(())
    }

    fn compile_mir_sequence_assign(
        &mut self,
        target: &runmat_mir::MirSequenceTarget,
        value: &MirRvalue,
    ) -> Result<(), CompileError> {
        let (base, dynamic_member) = match target {
            runmat_mir::MirSequenceTarget::Member { base, .. } => (base, None),
            runmat_mir::MirSequenceTarget::DynamicMember { base, member } => (base, Some(member)),
            runmat_mir::MirSequenceTarget::CellContents { .. } => {
                return Err(self.compile_error(
                    "legacy sequence-assignment statements cannot target brace contents",
                ));
            }
        };
        self.compile_mir_place_read(base)?;
        self.emit(Instr::MemberSequenceCardinality);
        let count_slot = self.alloc_temp();
        self.emit(Instr::StoreVar(count_slot));

        // Materialize the already-evaluated destination before producing the
        // transient value sequence. The sequence register is deliberately a
        // single-instruction channel: its producer must be followed
        // immediately by the member-sequence store that consumes it.
        self.compile_mir_member_base_for_assignment(base, false)?;
        if let Some(member) = dynamic_member {
            self.compile_mir_operand(member)?;
        }

        let sequence_capture = match value {
            MirRvalue::Member { base, member, .. } => {
                self.compile_mir_operand(base)?;
                self.emit(Instr::LoadMemberSequenceUsingOutputSlot {
                    member: member.clone(),
                    output_count_slot: count_slot,
                });
                None
            }
            MirRvalue::DynamicMember { base, member, .. } => {
                self.compile_mir_operand(base)?;
                self.compile_mir_operand(member)?;
                self.emit(Instr::LoadMemberDynamicSequenceUsingOutputSlot {
                    output_count_slot: count_slot,
                });
                None
            }
            MirRvalue::Call(call) => {
                if call.requested_outputs != RequestedOutputCount::DestinationSequenceCardinality {
                    return Err(self.compile_error(
                        "sequence assignment call is missing its destination-cardinality request",
                    ));
                }
                self.compile_mir_call_with_output_count(
                    call,
                    ResolvedCallOutputCount::FromSlot(count_slot),
                )?;
                Some(Instr::CaptureCallOutputSequence)
            }
            _ => {
                self.compile_mir_rvalue(value)?;
                Some(Instr::CaptureScalarSequence)
            }
        };
        if let Some(capture) = sequence_capture {
            self.emit(capture);
        }

        if dynamic_member.is_some() {
            self.emit(Instr::StoreMemberDynamicSequence);
        } else if let runmat_mir::MirSequenceTarget::Member { member, .. } = target {
            self.emit(Instr::StoreMemberSequence(member.clone()));
        }
        self.emit_store_back_mir_member_chain(base, false)
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
        if targets
            .targets
            .iter()
            .any(|target| matches!(target, MirOutputTarget::Sequence(_)))
        {
            return self.compile_mir_dynamic_multi_assign(targets, value);
        }
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
        let emits_requested_values = matches!(value, MirRvalue::Call(_))
            || matches!(
                value,
                MirRvalue::Index { indexing, .. }
                    if indexing.kind == IndexKind::Brace
                        && matches!(indexing.result_context, IndexResultContext::ReadCommaList)
            )
            || matches!(
                value,
                MirRvalue::Member {
                    sequence_use: runmat_types::SequenceUse::SelectPrefix { count },
                    ..
                } | MirRvalue::DynamicMember {
                    sequence_use: runmat_types::SequenceUse::SelectPrefix { count },
                    ..
                } if *count == output_count
            )
            || matches!(
                value,
                MirRvalue::SubscriptChain(chain)
                    if matches!(chain.sequence_use, runmat_types::SequenceUse::SelectPrefix { count } if count == output_count)
            );
        if !emits_requested_values {
            self.emit(Instr::Unpack(targets.targets.len()));
        }
        for target in targets.targets.iter().rev() {
            self.compile_mir_output_target_store(target)?;
        }
        Ok(())
    }

    fn compile_mir_dynamic_multi_assign(
        &mut self,
        targets: &runmat_mir::MirOutputTargetList,
        value: &MirRvalue,
    ) -> Result<(), CompileError> {
        if targets.requested_outputs != RequestedOutputCount::DestinationSequenceCardinality {
            return Err(self.compile_error(
                "runtime-cardinality output targets require destination-cardinality output selection",
            ));
        }
        self.emit(Instr::BeginOutputAssignment {
            target_count: targets.targets.len(),
        });
        for target in &targets.targets {
            match target {
                MirOutputTarget::Place(_) => {
                    self.emit(Instr::PrepareFixedOutputTarget);
                }
                MirOutputTarget::Discard => {
                    self.emit(Instr::PrepareDiscardOutputTarget);
                }
                MirOutputTarget::Sequence(target) => {
                    self.compile_mir_sequence_output_target(target)?;
                }
            }
        }
        self.emit(Instr::LoadPreparedOutputCardinality);
        let count_slot = self.alloc_temp();
        self.emit(Instr::StoreVar(count_slot));

        match value {
            MirRvalue::Member { base, member, .. } => {
                self.compile_mir_operand(base)?;
                self.emit(Instr::LoadMemberSequenceUsingOutputSlot {
                    member: member.clone(),
                    output_count_slot: count_slot,
                });
            }
            MirRvalue::DynamicMember { base, member, .. } => {
                self.compile_mir_operand(base)?;
                self.compile_mir_operand(member)?;
                self.emit(Instr::LoadMemberDynamicSequenceUsingOutputSlot {
                    output_count_slot: count_slot,
                });
            }
            MirRvalue::Index { base, indexing }
                if indexing.kind == IndexKind::Brace
                    && indexing.result_context == IndexResultContext::ReadCommaList =>
            {
                self.compile_mir_cell_list(base, indexing)?;
            }
            MirRvalue::SubscriptChain(chain) => {
                self.compile_subscript_chain_to_register(chain)?;
            }
            MirRvalue::Call(call) => {
                if call.requested_outputs != RequestedOutputCount::DestinationSequenceCardinality {
                    return Err(self.compile_error(
                        "dynamic output assignment call is missing its destination-cardinality request",
                    ));
                }
                self.compile_mir_call_with_output_count(
                    call,
                    ResolvedCallOutputCount::FromSlot(count_slot),
                )?;
                self.emit(Instr::CaptureCallOutputSequence);
            }
            _ => {
                self.compile_mir_rvalue(value)?;
                self.emit(Instr::CaptureScalarSequence);
            }
        }

        let retained_outputs = targets
            .targets
            .iter()
            .filter(|target| !matches!(target, MirOutputTarget::Sequence(_)))
            .count();
        self.emit(Instr::CommitPreparedOutputTargets { retained_outputs });
        for target in targets.targets.iter().rev() {
            if !matches!(target, MirOutputTarget::Sequence(_)) {
                self.compile_mir_output_target_store(target)?;
            }
        }
        Ok(())
    }

    fn compile_mir_sequence_output_target(
        &mut self,
        target: &runmat_mir::MirSequenceTarget,
    ) -> Result<(), CompileError> {
        let root_slot = self.mir_place_root_slot(target.base())?;
        self.emit(Instr::BeginSequenceOutputTarget { root_slot });
        self.compile_mir_sequence_target_path(target.base())?;
        match target {
            runmat_mir::MirSequenceTarget::Member { member, .. } => {
                self.emit(Instr::FinishMemberSequenceOutputTarget(member.clone()));
            }
            runmat_mir::MirSequenceTarget::DynamicMember { member, .. } => {
                self.compile_mir_operand(member)?;
                self.emit(Instr::FinishDynamicMemberSequenceOutputTarget);
            }
            runmat_mir::MirSequenceTarget::CellContents { indexing, .. } => {
                let (component_count, selectors) =
                    self.compile_prepared_index_components(indexing)?;
                self.emit(Instr::FinishCellContentsSequenceOutputTarget {
                    component_count,
                    selectors,
                    expand_all: indexing.cell_expand_all,
                });
            }
        }
        Ok(())
    }

    fn compile_mir_sequence_target_path(&mut self, place: &MirPlace) -> Result<(), CompileError> {
        match place {
            MirPlace::Local(_) | MirPlace::Binding(_) => Ok(()),
            MirPlace::Member(base, member) => {
                self.compile_mir_sequence_target_path(base)?;
                self.emit(Instr::PrepareMemberPathStep(member.clone()));
                Ok(())
            }
            MirPlace::DynamicMember(base, member) => {
                self.compile_mir_sequence_target_path(base)?;
                self.compile_mir_operand(member)?;
                self.emit(Instr::PrepareDynamicMemberPathStep);
                Ok(())
            }
            MirPlace::Index(base, indexing) => {
                self.compile_mir_sequence_target_path(base)?;
                let (component_count, selectors) =
                    self.compile_prepared_index_components(indexing)?;
                match indexing.kind {
                    IndexKind::Paren => self.emit(Instr::PrepareParenthesesPathStep {
                        component_count,
                        selectors,
                    }),
                    IndexKind::Brace => self.emit(Instr::PrepareBracesPathStep {
                        component_count,
                        selectors,
                        expand_all: indexing.cell_expand_all,
                    }),
                };
                Ok(())
            }
        }
    }

    fn mir_place_root_slot(&self, place: &MirPlace) -> Result<usize, CompileError> {
        match place {
            MirPlace::Local(_) | MirPlace::Binding(_) => self.mir_place_slot(place),
            MirPlace::Member(base, _)
            | MirPlace::DynamicMember(base, _)
            | MirPlace::Index(base, _) => self.mir_place_root_slot(base),
        }
    }

    fn compile_prepared_index_components(
        &mut self,
        indexing: &MirIndexing,
    ) -> Result<(usize, Vec<crate::bytecode::BytecodeSubscriptSelector>), CompileError> {
        let component_count = if indexing.cell_expand_all {
            0
        } else {
            indexing.components.len()
        };
        self.emit(Instr::BeginPreparedIndexSelectors { component_count });
        if indexing.cell_expand_all {
            return Ok((0, Vec::new()));
        }
        let mut selectors = Vec::with_capacity(indexing.components.len());
        for (component_index, component) in indexing.components.iter().enumerate() {
            match component {
                MirIndexComponent::Colon => {
                    selectors.push(crate::bytecode::BytecodeSubscriptSelector::Colon)
                }
                MirIndexComponent::Expr(operand) => {
                    self.compile_mir_operand(operand)?;
                    selectors.push(crate::bytecode::BytecodeSubscriptSelector::Value);
                }
                MirIndexComponent::ContextualExpr(region) => {
                    let previous = self.prepared_index_component.replace(component_index);
                    self.compile_mir_expression_region(region)?;
                    self.prepared_index_component = previous;
                    selectors.push(crate::bytecode::BytecodeSubscriptSelector::Value);
                }
            }
        }
        Ok((component_count, selectors))
    }

    fn compile_mir_expression_region(
        &mut self,
        region: &runmat_mir::MirExpressionRegion,
    ) -> Result<(), CompileError> {
        region
            .validate()
            .map_err(|message| self.compile_error(message))?;
        for step in region.steps() {
            let statement = match step {
                runmat_mir::MirExpressionStep::Let { local, value, span } => MirStmt {
                    kind: MirStmtKind::Assign {
                        place: MirPlace::Local(*local),
                        value: value.clone(),
                    },
                    span: *span,
                },
                runmat_mir::MirExpressionStep::CaptureSequence {
                    destination,
                    source,
                    span,
                } => MirStmt {
                    kind: MirStmtKind::CaptureSequence {
                        destination: *destination,
                        source: source.clone(),
                    },
                    span: *span,
                },
            };
            self.compile_mir_stmt(&statement)?;
        }
        self.compile_mir_operand(region.result())
    }

    fn compile_contextual_slice_components(
        &mut self,
        indexing: &MirIndexing,
    ) -> Result<(usize, u32, u32), CompileError> {
        self.emit(Instr::BeginContextualIndexSelectors {
            component_count: indexing.components.len(),
        });
        let mut numeric_count = 0usize;
        let mut colon_mask = 0u32;
        let end_mask = 0u32;
        for (component_index, component) in indexing.components.iter().enumerate() {
            match component {
                MirIndexComponent::Colon => self.set_selector_mask_bit(
                    &mut colon_mask,
                    component_index,
                    IDENT_MIR_SLICE_INDEX_PLAN_INVALID,
                    "contextual selector dimension exceeds mask width",
                )?,
                MirIndexComponent::Expr(operand) => {
                    self.compile_mir_operand(operand)?;
                    numeric_count += 1;
                }
                MirIndexComponent::ContextualExpr(region) => {
                    let previous = self.contextual_index_component.replace(component_index);
                    self.compile_mir_expression_region(region)?;
                    self.contextual_index_component = previous;
                    numeric_count += 1;
                }
            }
        }
        self.emit(Instr::FinishContextualIndexSelectors {
            component_count: indexing.components.len(),
        });
        Ok((numeric_count, colon_mask, end_mask))
    }

    fn compile_contextual_index_values(
        &mut self,
        indexing: &MirIndexing,
    ) -> Result<(), CompileError> {
        self.emit(Instr::BeginContextualIndexSelectors {
            component_count: indexing.components.len(),
        });
        for (component_index, component) in indexing.components.iter().enumerate() {
            match component {
                MirIndexComponent::Colon => {
                    self.emit(Instr::LoadString(":".into()));
                }
                MirIndexComponent::Expr(operand) => {
                    self.compile_mir_operand(operand)?;
                }
                MirIndexComponent::ContextualExpr(region) => {
                    let previous = self.contextual_index_component.replace(component_index);
                    self.compile_mir_expression_region(region)?;
                    self.contextual_index_component = previous;
                }
            }
        }
        self.emit(Instr::FinishContextualIndexSelectors {
            component_count: indexing.components.len(),
        });
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
                if let MirPlace::Index(_, indexing) = place {
                    if indexing.result_context != IndexResultContext::AssignmentTarget {
                        return Err(self
                            .compile_error(
                                "MIR multi-assign output target index lowering expected AssignmentTarget context",
                            )
                            .with_identifier(IDENT_MIR_INDEX_CONTEXT_INVALID));
                    }
                }
                let tmp = self.alloc_temp();
                self.emit(Instr::StoreVar(tmp));
                self.compile_mir_assign_from_slot(place, tmp)
            }
            MirOutputTarget::Sequence(_) => Err(self.compile_error(
                "runtime-cardinality output targets require prepared multi-assignment lowering",
            )),
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
                    self.emit(Instr::LoadVarForIndexAssignment(base_slot));
                    self.compile_mir_store_indexed_value_from_temp(
                        indexing, value_slot, false, false,
                    )?;
                    self.emit(Instr::StoreVar(base_slot));
                    return Ok(());
                }
                self.compile_mir_place_read(base)?;
                self.compile_mir_store_indexed_value_from_temp(indexing, value_slot, false, false)?;
                self.emit_store_back_mir_member_chain(base, false)
            }
            MirPlace::Member(base, member) => {
                self.compile_mir_member_base_for_assignment(base, false)?;
                self.emit(Instr::LoadVar(value_slot));
                self.emit(Instr::StoreMemberOrInit(member.clone()));
                self.emit_store_back_mir_member_chain(base, false)
            }
            MirPlace::DynamicMember(base, member) => {
                self.compile_mir_member_base_for_assignment(base, false)?;
                self.compile_mir_operand(member)?;
                self.emit(Instr::LoadVar(value_slot));
                self.emit(Instr::StoreMemberDynamicOrInit);
                self.emit_store_back_mir_member_chain(base, false)
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
        let (index_count, expand_all) = self.compile_mir_cell_selector_operands(indexing)?;
        if expand_all {
            self.emit(Instr::IndexCellExpand {
                num_indices: 0,
                out_count: output_count,
            });
        } else {
            self.emit(Instr::IndexCellExpand {
                num_indices: index_count,
                out_count: output_count,
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
        let (index_count, expand_all) = self.compile_mir_cell_selector_operands(indexing)?;
        self.emit(Instr::IndexCellList {
            num_indices: if expand_all { 0 } else { index_count },
        });
        Ok(())
    }

    fn compile_mir_cell_selector_operands(
        &mut self,
        indexing: &MirIndexing,
    ) -> Result<MirCellSelectorCompileResult, CompileError> {
        if indexing
            .components
            .iter()
            .any(|component| matches!(component, MirIndexComponent::ContextualExpr(_)))
        {
            self.compile_contextual_index_values(indexing)?;
            return Ok((indexing.components.len(), false));
        }
        let expand_all = indexing.cell_expand_all;
        let mut index_count = 0usize;
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
                    index_count += 1;
                }
                MirIndexComponent::ContextualExpr(_) => {
                    return Err(self.compile_error(
                        "contextual brace selectors require prepared indexing lowering",
                    ));
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
        Ok((index_count, expand_all))
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
                    .with_identifier(IDENT_MIR_MULTI_ASSIGN_OUTPUT_COUNT_MISMATCH));
            }
        }
        if self.try_compile_parallel_call(call)? {
            return Ok(());
        }
        let (specs, has_expansion) = self.mir_call_arg_specs(&call.args)?;
        if matches!(call.syntax, CallSyntax::Method | CallSyntax::DottedInvoke) {
            match &call.callee {
                MirCallee::Static(
                    CallableIdentity::BoundFunction(_) | CallableIdentity::ExternalFunction { .. },
                ) => {}
                MirCallee::SuperMethod { .. } => {}
                MirCallee::SuperConstructor { .. } => {
                    return Err(self
                        .compile_error("MIR method-call lowering found super-constructor callee")
                        .with_identifier(IDENT_MIR_METHOD_CALL_CALLEE_INVALID));
                }
                MirCallee::Dynamic(_) => {
                    return Err(self
                        .compile_error(
                            "MIR method-call lowering expected a non-semantic static callee",
                        )
                        .with_identifier(IDENT_MIR_METHOD_CALL_CALLEE_INVALID));
                }
                MirCallee::Static(_) => return self.compile_mir_method_call(call, has_expansion),
            }
        }
        if let Some(name) = call.workspace_first_name.as_ref() {
            let MirCallee::Static(identity) = &call.callee else {
                return Err(self
                    .compile_error("workspace-first call lowering expected a static callee")
                    .with_identifier(IDENT_MIR_CALL_TARGET_NAME_INVALID));
            };
            self.validate_workspace_first_static_call_callee(identity, call.fallback_policy)?;
            for arg in &call.args {
                self.compile_mir_call_arg(arg)?;
            }
            if has_expansion {
                self.emit_call(
                    Instr::CallWorkspaceFirstExpandMultiOutput {
                        name: name.0.clone(),
                        identity: identity.clone(),
                        fallback_policy: call.fallback_policy,
                        bare_identifier: call.bare_identifier,
                        specs,
                        out_count: output_count,
                    },
                    call,
                );
            } else {
                self.emit_call(
                    Instr::CallWorkspaceFirstMulti {
                        name: name.0.clone(),
                        identity: identity.clone(),
                        fallback_policy: call.fallback_policy,
                        bare_identifier: call.bare_identifier,
                        arg_count: call.args.len(),
                        out_count: output_count,
                    },
                    call,
                );
            }
            return Ok(());
        }
        match &call.callee {
            MirCallee::Static(CallableIdentity::ExternalFunction {
                function,
                display_name,
            }) => {
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                let identity = CallableIdentity::ExternalFunction {
                    function: *function,
                    display_name: display_name.clone(),
                };
                if has_expansion {
                    self.emit_call(
                        Instr::CallFunctionExpandMultiOutput {
                            identity,
                            fallback_policy: call.fallback_policy,
                            specs,
                            out_count: output_count,
                        },
                        call,
                    );
                    return Ok(());
                }
                self.emit_call(
                    Instr::CallFunctionMulti {
                        identity,
                        fallback_policy: call.fallback_policy,
                        arg_count: call.args.len(),
                        out_count: output_count,
                    },
                    call,
                );
            }
            MirCallee::Static(CallableIdentity::BoundFunction(function)) => {
                // Session-resolved semantic calls can target functions compiled in prior
                // submissions; those do not exist in the current assembly layout and should
                // compile as regular semantic calls without nested-capture wiring.
                let captures = self
                    .layout
                    .as_ref()
                    .and_then(|layout| layout.functions.get(function))
                    .map(|layout| layout.captures.clone())
                    .unwrap_or_default();
                if let Some(capture_slots) =
                    self.semantic_capture_slots_for_call(*function, &captures)?
                {
                    for arg in &call.args {
                        self.compile_mir_call_arg(arg)?;
                    }
                    if has_expansion {
                        self.emit_call(
                            Instr::CallSemanticNestedFunctionExpandMultiOutput {
                                function: *function,
                                capture_slots,
                                specs,
                                out_count: output_count,
                            },
                            call,
                        );
                    } else {
                        self.emit_call(
                            Instr::CallSemanticNestedFunctionMulti {
                                function: *function,
                                capture_slots,
                                arg_count: call.args.len(),
                                out_count: output_count,
                            },
                            call,
                        );
                    }
                    return Ok(());
                }
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                if has_expansion {
                    self.emit_call(
                        Instr::CallSemanticFunctionExpandMultiOutput(
                            *function,
                            specs,
                            output_count,
                        ),
                        call,
                    );
                    return Ok(());
                }
                self.emit_call(
                    Instr::CallSemanticFunctionMulti(*function, call.args.len(), output_count),
                    call,
                );
            }
            MirCallee::Dynamic(_) => {
                self.compile_mir_dynamic_callee_operand(match &call.callee {
                    MirCallee::Dynamic(callee) => callee,
                    _ => unreachable!(),
                })?;
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                if has_expansion {
                    self.emit_call(Instr::CallFevalExpandMultiOutput(specs, output_count), call);
                } else {
                    self.emit_call(Instr::CallFevalMulti(call.args.len(), output_count), call);
                }
            }
            MirCallee::SuperConstructor {
                current_class,
                super_class,
            } => {
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                if has_expansion {
                    self.emit_call(
                        Instr::CallSuperConstructorExpandMultiOutput {
                            current_class: current_class.clone(),
                            super_class: super_class.clone(),
                            specs,
                            out_count: output_count,
                        },
                        call,
                    );
                } else {
                    self.emit_call(
                        Instr::CallSuperConstructorMulti {
                            current_class: current_class.clone(),
                            super_class: super_class.clone(),
                            arg_count: call.args.len(),
                            out_count: output_count,
                        },
                        call,
                    );
                }
            }
            MirCallee::SuperMethod {
                current_class,
                super_class,
                method,
            } => {
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                if has_expansion {
                    self.emit_call(
                        Instr::CallSuperMethodExpandMultiOutput {
                            current_class: current_class.clone(),
                            super_class: super_class.clone(),
                            method: method.clone(),
                            specs,
                            out_count: output_count,
                        },
                        call,
                    );
                } else {
                    self.emit_call(
                        Instr::CallSuperMethodMulti {
                            current_class: current_class.clone(),
                            super_class: super_class.clone(),
                            method: method.clone(),
                            arg_count: call.args.len(),
                            out_count: output_count,
                        },
                        call,
                    );
                }
            }
            MirCallee::Static(CallableIdentity::Builtin(id)) => {
                let name = self.mir_builtin_call_name(id)?;
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                if has_expansion {
                    self.emit_call(
                        Instr::CallBuiltinExpandMultiOutput(name, specs, output_count),
                        call,
                    );
                } else {
                    self.emit_call(
                        Instr::CallBuiltinMulti(name, call.args.len(), output_count),
                        call,
                    );
                }
            }
            MirCallee::Static(identity) => {
                let fallback_policy = call.fallback_policy;
                self.validate_static_call_callee(identity, fallback_policy)?;
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                if has_expansion {
                    self.emit_call(
                        Instr::CallFunctionExpandMultiOutput {
                            identity: identity.clone(),
                            fallback_policy,
                            specs,
                            out_count: output_count,
                        },
                        call,
                    );
                } else {
                    self.emit_call(
                        Instr::CallFunctionMulti {
                            identity: identity.clone(),
                            fallback_policy,
                            arg_count: call.args.len(),
                            out_count: output_count,
                        },
                        call,
                    );
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
                if indexing.result_context != IndexResultContext::AssignmentTarget {
                    return Err(self
                        .compile_error(
                            "MIR assignment invariant violated: indexed assignment target requires AssignmentTarget context",
                        )
                        .with_identifier(IDENT_MIR_INDEX_CONTEXT_INVALID));
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
                    if delete {
                        self.emit(Instr::LoadVar(base_slot));
                    } else {
                        self.emit(Instr::LoadVarForIndexAssignment(base_slot));
                    }
                    self.compile_mir_index_assignment_after_base(indexing, value, delete)?;
                    self.emit(Instr::StoreVar(base_slot));
                    return Ok(());
                }
                self.compile_mir_place_read(base)?;
                self.compile_mir_index_assignment_after_base(indexing, value, delete)?;
                self.emit_store_back_mir_member_chain(base, delete)
            }
            MirPlace::Member(base, member) => {
                self.compile_mir_member_base_for_assignment(base, false)?;
                self.compile_mir_rvalue(value)?;
                self.emit(Instr::StoreMemberOrInit(member.clone()));
                self.emit_store_back_mir_member_chain(base, false)
            }
            MirPlace::DynamicMember(base, member) => {
                self.compile_mir_member_base_for_assignment(base, false)?;
                self.compile_mir_operand(member)?;
                self.compile_mir_rvalue(value)?;
                self.emit(Instr::StoreMemberDynamicOrInit);
                self.emit_store_back_mir_member_chain(base, false)
            }
        }
    }

    fn compile_mir_member_base_for_assignment(
        &mut self,
        base: &MirPlace,
        allow_deletion_context: bool,
    ) -> Result<(), CompileError> {
        match base {
            MirPlace::Index(parent, indexing) => {
                self.compile_mir_place_read(parent)?;
                self.compile_mir_index_after_base(indexing, allow_deletion_context)
            }
            MirPlace::Member(parent, field) => {
                self.compile_mir_member_base_for_assignment(parent, allow_deletion_context)?;
                self.emit(Instr::LoadMemberOrInit(field.clone()));
                Ok(())
            }
            MirPlace::DynamicMember(parent, name) => {
                self.compile_mir_member_base_for_assignment(parent, allow_deletion_context)?;
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
        if indexing.kind == IndexKind::Paren
            && indexing
                .components
                .iter()
                .any(|component| matches!(component, MirIndexComponent::ContextualExpr(_)))
        {
            let (numeric_count, colon_mask, end_mask) =
                self.compile_contextual_slice_components(indexing)?;
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
            return Ok(());
        }
        match indexing.kind {
            IndexKind::Paren => match indexing.plan {
                MirIndexPlan::Scalar => {
                    if indexing.components.len() > 2 {
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
                    } else {
                        self.compile_mir_scalar_index_components(indexing)?;
                        self.compile_mir_rvalue(value)?;
                        self.emit(if delete {
                            Instr::StoreIndexDelete(indexing.components.len())
                        } else {
                            Instr::StoreIndex(indexing.components.len())
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
                self.compile_mir_cell_index_components(
                    indexing,
                    IndexResultContext::AssignmentTarget,
                )?;
                self.compile_mir_rvalue(value)?;
                self.emit(if delete {
                    Instr::StoreIndexCellDelete {
                        num_indices: indexing.components.len(),
                    }
                } else {
                    Instr::StoreIndexCell {
                        num_indices: indexing.components.len(),
                    }
                });
            }
        };
        Ok(())
    }

    fn emit_store_back_mir_member_chain(
        &mut self,
        base: &MirPlace,
        allow_deletion_context: bool,
    ) -> Result<(), CompileError> {
        match base {
            MirPlace::Local(_) | MirPlace::Binding(_) => {
                let slot = self.mir_place_slot(base)?;
                self.emit(Instr::StoreVar(slot));
                Ok(())
            }
            MirPlace::Member(parent, field) => {
                self.compile_mir_member_base_for_assignment(parent, allow_deletion_context)?;
                self.emit(Instr::Swap);
                self.emit(Instr::StoreMemberOrInit(field.clone()));
                self.emit_store_back_mir_member_chain(parent, allow_deletion_context)
            }
            MirPlace::DynamicMember(parent, name) => {
                let tmp = self.alloc_temp();
                self.emit(Instr::StoreVar(tmp));
                self.compile_mir_member_base_for_assignment(parent, allow_deletion_context)?;
                self.compile_mir_operand(name)?;
                self.emit(Instr::LoadVar(tmp));
                self.emit(Instr::StoreMemberDynamicOrInit);
                self.emit_store_back_mir_member_chain(parent, allow_deletion_context)
            }
            MirPlace::Index(parent, indexing) => {
                let tmp = self.alloc_temp();
                self.emit(Instr::StoreVar(tmp));
                self.compile_mir_place_read(parent)?;
                self.compile_mir_store_indexed_value_from_temp(
                    indexing,
                    tmp,
                    false,
                    allow_deletion_context,
                )?;
                self.emit_store_back_mir_member_chain(parent, allow_deletion_context)
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
                self.emit(Instr::LoadMember(member.clone()));
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
                self.compile_mir_index_after_base(indexing, true)
            }
        }
    }

    fn compile_mir_index_after_base(
        &mut self,
        indexing: &MirIndexing,
        allow_deletion_context: bool,
    ) -> Result<(), CompileError> {
        let context_ok = if allow_deletion_context {
            mir_indexing_context_matches(
                indexing.result_context,
                IndexResultContext::AssignmentTarget,
            )
        } else {
            indexing.result_context == IndexResultContext::AssignmentTarget
        };
        if !context_ok {
            return Err(self
                .compile_error(
                    "MIR indexed helper-read invariant violated: lvalue base indexing requires AssignmentTarget context",
                )
                .with_identifier(IDENT_MIR_INDEX_CONTEXT_INVALID));
        }
        match indexing.kind {
            IndexKind::Paren => self.compile_mir_slice_index(indexing)?,
            IndexKind::Brace => {
                self.compile_mir_cell_index_components(
                    indexing,
                    IndexResultContext::AssignmentTarget,
                )?;
                self.emit(Instr::IndexCell {
                    num_indices: indexing.components.len(),
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
        allow_deletion_context: bool,
    ) -> Result<(), CompileError> {
        let context_ok = if allow_deletion_context {
            mir_indexing_context_matches(
                indexing.result_context,
                IndexResultContext::AssignmentTarget,
            )
        } else {
            indexing.result_context == IndexResultContext::AssignmentTarget
        };
        if !context_ok {
            return Err(self
                .compile_error(
                    "MIR indexed helper store-back invariant violated: assignment-index context must be AssignmentTarget",
                )
                .with_identifier(IDENT_MIR_INDEX_CONTEXT_INVALID));
        }
        if indexing.kind == IndexKind::Paren
            && indexing
                .components
                .iter()
                .any(|component| matches!(component, MirIndexComponent::ContextualExpr(_)))
        {
            let (numeric_count, colon_mask, end_mask) =
                self.compile_contextual_slice_components(indexing)?;
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
            return Ok(());
        }
        match indexing.kind {
            IndexKind::Paren => {
                match indexing.plan {
                    MirIndexPlan::Scalar => {
                        if indexing.components.len() > 2 {
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
                        } else {
                            self.compile_mir_scalar_index_components(indexing)?;
                            self.emit(Instr::LoadVar(tmp));
                            self.emit(if delete {
                                Instr::StoreIndexDelete(indexing.components.len())
                            } else {
                                Instr::StoreIndex(indexing.components.len())
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
                self.compile_mir_cell_index_components(
                    indexing,
                    IndexResultContext::AssignmentTarget,
                )?;
                self.emit(Instr::LoadVar(tmp));
                self.emit(if delete {
                    Instr::StoreIndexCellDelete {
                        num_indices: indexing.components.len(),
                    }
                } else {
                    Instr::StoreIndexCell {
                        num_indices: indexing.components.len(),
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
            MirRvalue::SubscriptChain(chain) => self.compile_subscript_chain(chain, None),
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
                            .with_identifier(IDENT_MIR_OPERATOR_UNSUPPORTED));
                    }
                };
                Ok(())
            }
            MirRvalue::Binary(left, op, right) => {
                match op {
                    OperatorKind::ShortCircuitAnd => {
                        return self.compile_mir_short_circuit_and(left, &[], right);
                    }
                    OperatorKind::ShortCircuitOr => {
                        return self.compile_mir_short_circuit_or(left, &[], right);
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
                            .with_identifier(IDENT_MIR_OPERATOR_UNSUPPORTED));
                    }
                };
                Ok(())
            }
            MirRvalue::ShortCircuit {
                left,
                op,
                right_temps,
                right,
            } => match op {
                MirShortCircuitOp::And => {
                    self.compile_mir_short_circuit_and(left, right_temps, right)
                }
                MirShortCircuitOp::Or => {
                    self.compile_mir_short_circuit_or(left, right_temps, right)
                }
            },
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
                row_lengths,
                elements,
            } => self.compile_mir_aggregate(kind, row_lengths, elements),
            MirRvalue::StructLiteral { fields } => self.compile_mir_struct_literal(fields),
            MirRvalue::ObjectLiteral { class_name, fields } => {
                self.compile_mir_object_literal(class_name, fields)
            }
            MirRvalue::Index { base, indexing } => self.compile_mir_index(base, indexing),
            MirRvalue::Member {
                base,
                member,
                sequence_use,
            } => {
                self.compile_mir_operand(base)?;
                self.emit(Instr::LoadMemberSequence {
                    member: member.clone(),
                    selection: *sequence_use,
                });
                Ok(())
            }
            MirRvalue::DynamicMember {
                base,
                member,
                sequence_use,
            } => {
                self.compile_mir_operand(base)?;
                self.compile_mir_operand(member)?;
                self.emit(Instr::LoadMemberDynamicSequence {
                    selection: *sequence_use,
                });
                Ok(())
            }
            MirRvalue::WorkspaceFirstStaticProperty {
                workspace_name,
                class_name,
                property,
            } => {
                self.emit(Instr::LoadWorkspaceFirstStaticProperty {
                    name: workspace_name.0.clone(),
                    class_name: class_name.clone(),
                    property: property.clone(),
                });
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
                if let Some((component, component_count)) = self.subscript_end_component {
                    self.emit(Instr::LoadSubscriptEnd {
                        component,
                        component_count,
                    });
                } else if let Some(component) = self.prepared_index_component {
                    self.emit(Instr::LoadPreparedIndexEnd { component });
                } else if let Some(component) = self.contextual_index_component {
                    self.emit(Instr::LoadContextualIndexEnd { component });
                } else {
                    return Err(self.compile_error(
                        "MIR end expression is not enclosed by a contextual index component",
                    ));
                }
                Ok(())
            }
            MirRvalue::Future {
                function,
                args,
                requested_outputs,
                ..
            } => {
                let (specs, has_expansion) = self.mir_call_arg_specs(args)?;
                for arg in args {
                    self.compile_mir_call_arg(arg)?;
                }
                let out_count = requested_outputs.executor_carrier_count().ok_or_else(|| {
                    self.compile_error(
                        "future output cardinality cannot be derived from an assignment destination",
                    )
                })?;
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
            MirRvalue::Distributed(operation) => {
                use runmat_mir::parallel::MirDistributedOp;
                let instruction = match operation {
                    MirDistributedOp::Create {
                        id,
                        owner,
                        input,
                        scheme,
                    } => {
                        self.compile_mir_operand(input)?;
                        crate::BytecodeDistributedOp::Create {
                            id: *id,
                            owner: *owner,
                            scheme: scheme.clone(),
                        }
                    }
                    MirDistributedOp::Codistributed {
                        id,
                        owner,
                        input,
                        overload,
                        coordination,
                    } => {
                        self.compile_mir_operand(input)?;
                        let overload = match overload {
                            runmat_mir::parallel::MirCodistributedOverload::ReplicatedInputDefault => {
                                crate::BytecodeCodistributedOverload::ReplicatedInputDefault
                            }
                            runmat_mir::parallel::MirCodistributedOverload::CodistributorOrDesignatedWorker { operand } => {
                                self.compile_mir_operand(operand)?;
                                crate::BytecodeCodistributedOverload::CodistributorOrDesignatedWorker
                            }
                            runmat_mir::parallel::MirCodistributedOverload::DesignatedWorkerWithCodistributor { worker, codistributor } => {
                                self.compile_mir_operand(worker)?;
                                self.compile_mir_operand(codistributor)?;
                                crate::BytecodeCodistributedOverload::DesignatedWorkerWithCodistributor
                            }
                        };
                        crate::BytecodeDistributedOp::Codistributed {
                            id: *id,
                            owner: *owner,
                            overload,
                            coordination: *coordination,
                        }
                    }
                    MirDistributedOp::Build {
                        id,
                        owner,
                        local_part,
                        codistributor,
                        validation,
                        coordination,
                    } => {
                        self.compile_mir_operand(local_part)?;
                        if let Some(codistributor) = codistributor {
                            self.compile_mir_operand(codistributor)?;
                        }
                        let validation = match validation {
                            runmat_mir::parallel::MirDistributedBuildValidation::ValidateAcrossWorkers => {
                                crate::BytecodeDistributedBuildValidation::ValidateAcrossWorkers
                            }
                            runmat_mir::parallel::MirDistributedBuildValidation::NoCommunication => {
                                crate::BytecodeDistributedBuildValidation::NoCommunication
                            }
                            runmat_mir::parallel::MirDistributedBuildValidation::RuntimeOption(option) => {
                                self.compile_mir_operand(option)?;
                                crate::BytecodeDistributedBuildValidation::RuntimeOption
                            }
                        };
                        crate::BytecodeDistributedOp::Build {
                            id: *id,
                            owner: *owner,
                            has_codistributor: codistributor.is_some(),
                            validation,
                            coordination: *coordination,
                        }
                    }
                    MirDistributedOp::LocalPart { value } => {
                        self.compile_mir_operand(value)?;
                        crate::BytecodeDistributedOp::LocalPart
                    }
                    MirDistributedOp::Materialize { value } => {
                        self.compile_mir_operand(value)?;
                        crate::BytecodeDistributedOp::Materialize
                    }
                    MirDistributedOp::Codistributor { value } => {
                        self.compile_mir_operand(value)?;
                        crate::BytecodeDistributedOp::Codistributor
                    }
                    MirDistributedOp::GlobalIndices {
                        value,
                        dimension,
                        lab,
                        requested_outputs,
                    } => {
                        self.compile_mir_operand(value)?;
                        self.compile_mir_operand(dimension)?;
                        if let Some(lab) = lab {
                            self.compile_mir_operand(lab)?;
                        }
                        crate::BytecodeDistributedOp::GlobalIndices {
                            has_lab: lab.is_some(),
                            requested_outputs: *requested_outputs,
                        }
                    }
                    MirDistributedOp::Redistribute {
                        value,
                        codistributor,
                    } => {
                        self.compile_mir_operand(value)?;
                        self.compile_mir_operand(codistributor)?;
                        crate::BytecodeDistributedOp::Redistribute
                    }
                };
                self.emit(Instr::Distributed(instruction));
                Ok(())
            }
            MirRvalue::Collective(operation) => {
                for operand in operation.operands() {
                    self.compile_mir_operand(operand)?;
                }
                use runmat_mir::parallel::MirCollectiveOp;
                let (id, operation) = match operation {
                    MirCollectiveOp::Barrier { id } => (*id, crate::BytecodeCollectiveOp::Barrier),
                    MirCollectiveOp::Broadcast { id, input, .. } => (
                        *id,
                        crate::BytecodeCollectiveOp::Broadcast {
                            has_input: input.is_some(),
                        },
                    ),
                    MirCollectiveOp::Gather { id, .. } => {
                        (*id, crate::BytecodeCollectiveOp::Gather)
                    }
                    MirCollectiveOp::Scatter { id, .. } => {
                        (*id, crate::BytecodeCollectiveOp::Scatter)
                    }
                    MirCollectiveOp::AllGather { id, .. } => {
                        (*id, crate::BytecodeCollectiveOp::AllGather)
                    }
                    MirCollectiveOp::Reduce { id, operator, .. } => (
                        *id,
                        crate::BytecodeCollectiveOp::Reduce {
                            operator: *operator,
                        },
                    ),
                    MirCollectiveOp::AllReduce { id, operator, .. } => (
                        *id,
                        crate::BytecodeCollectiveOp::AllReduce {
                            operator: *operator,
                        },
                    ),
                    MirCollectiveOp::Cat { id, root, .. } => (
                        *id,
                        crate::BytecodeCollectiveOp::Cat {
                            has_root: root.is_some(),
                        },
                    ),
                    MirCollectiveOp::FunctionalReduce { id, root, .. } => (
                        *id,
                        crate::BytecodeCollectiveOp::FunctionalReduce {
                            has_root: root.is_some(),
                        },
                    ),
                    MirCollectiveOp::Send { id, tag, .. } => (
                        *id,
                        crate::BytecodeCollectiveOp::Send {
                            has_tag: tag.is_some(),
                        },
                    ),
                    MirCollectiveOp::Receive {
                        id,
                        source,
                        tag,
                        requested_outputs,
                    } => (
                        *id,
                        crate::BytecodeCollectiveOp::Receive {
                            has_source: source.is_some(),
                            has_tag: tag.is_some(),
                            requested_outputs: *requested_outputs,
                        },
                    ),
                    MirCollectiveOp::SendReceive { id, tag, .. } => (
                        *id,
                        crate::BytecodeCollectiveOp::SendReceive {
                            has_tag: tag.is_some(),
                        },
                    ),
                    MirCollectiveOp::Probe {
                        id, source, tag, ..
                    } => (
                        *id,
                        crate::BytecodeCollectiveOp::Probe {
                            has_source: source.is_some(),
                            has_tag: tag.is_some(),
                        },
                    ),
                };
                self.emit(Instr::Collective { id, operation });
                Ok(())
            }
        }
    }

    fn compile_mir_short_circuit_and(
        &mut self,
        left: &MirOperand,
        right_temps: &[MirStmt],
        right: &MirOperand,
    ) -> Result<(), CompileError> {
        self.compile_mir_operand(left)?;
        let lhs_false = self.emit(Instr::JumpIfFalse(usize::MAX));
        for stmt in right_temps {
            self.compile_mir_stmt(stmt)?;
        }
        self.compile_mir_operand(right)?;
        let rhs_false = self.emit(Instr::JumpIfFalse(usize::MAX));
        self.emit(Instr::LoadConst(1.0));
        let end = self.emit(Instr::Jump(usize::MAX));
        let false_pc = self.instructions.len();
        self.emit(Instr::LoadConst(0.0));
        let end_pc = self.instructions.len();
        self.patch(lhs_false, Instr::JumpIfFalse(false_pc));
        self.patch(rhs_false, Instr::JumpIfFalse(false_pc));
        self.patch(end, Instr::Jump(end_pc));
        Ok(())
    }

    fn compile_mir_short_circuit_or(
        &mut self,
        left: &MirOperand,
        right_temps: &[MirStmt],
        right: &MirOperand,
    ) -> Result<(), CompileError> {
        self.compile_mir_operand(left)?;
        let lhs_false = self.emit(Instr::JumpIfFalse(usize::MAX));
        self.emit(Instr::LoadConst(1.0));
        let end = self.emit(Instr::Jump(usize::MAX));
        let right_pc = self.instructions.len();
        self.patch(lhs_false, Instr::JumpIfFalse(right_pc));
        for stmt in right_temps {
            self.compile_mir_stmt(stmt)?;
        }
        self.compile_mir_operand(right)?;
        let rhs_false = self.emit(Instr::JumpIfFalse(usize::MAX));
        self.emit(Instr::LoadConst(1.0));
        let rhs_end = self.emit(Instr::Jump(usize::MAX));
        let rhs_false_pc = self.instructions.len();
        self.emit(Instr::LoadConst(0.0));
        let end_pc = self.instructions.len();
        self.patch(rhs_false, Instr::JumpIfFalse(rhs_false_pc));
        self.patch(rhs_end, Instr::Jump(end_pc));
        self.patch(end, Instr::Jump(end_pc));
        Ok(())
    }

    fn compile_mir_call(&mut self, call: &MirCall) -> Result<(), CompileError> {
        if self.try_compile_parallel_call(call)? {
            return Ok(());
        }
        let requested_outputs = self.resolved_call_output_count(call)?;

        self.compile_mir_call_with_output_count(call, requested_outputs)
    }

    fn compile_mir_call_with_output_count(
        &mut self,
        call: &MirCall,
        requested_outputs: ResolvedCallOutputCount,
    ) -> Result<(), CompileError> {
        let (specs, has_expansion) = self.mir_call_arg_specs(&call.args)?;
        if matches!(call.syntax, CallSyntax::Method | CallSyntax::DottedInvoke) {
            match &call.callee {
                MirCallee::Static(
                    CallableIdentity::BoundFunction(_) | CallableIdentity::ExternalFunction { .. },
                ) => {}
                MirCallee::SuperMethod { .. } => {}
                MirCallee::SuperConstructor { .. } => {
                    return Err(self
                        .compile_error("MIR method-call lowering found super-constructor callee")
                        .with_identifier(IDENT_MIR_METHOD_CALL_CALLEE_INVALID));
                }
                MirCallee::Dynamic(_) => {
                    return Err(self
                        .compile_error(
                            "MIR method-call lowering expected a non-semantic static callee",
                        )
                        .with_identifier(IDENT_MIR_METHOD_CALL_CALLEE_INVALID));
                }
                MirCallee::Static(_) => return self.compile_mir_method_call(call, has_expansion),
            }
        }
        if let Some(name) = call.workspace_first_name.as_ref() {
            let MirCallee::Static(identity) = &call.callee else {
                return Err(self
                    .compile_error("workspace-first call lowering expected a static callee")
                    .with_identifier(IDENT_MIR_CALL_TARGET_NAME_INVALID));
            };
            self.validate_workspace_first_static_call_callee(identity, call.fallback_policy)?;
            for arg in &call.args {
                self.compile_mir_call_arg(arg)?;
            }
            if has_expansion {
                match requested_outputs {
                    ResolvedCallOutputCount::Fixed(out_count) => {
                        self.emit_call(
                            Instr::CallWorkspaceFirstExpandMultiOutput {
                                name: name.0.clone(),
                                identity: identity.clone(),
                                fallback_policy: call.fallback_policy,
                                bare_identifier: call.bare_identifier,
                                specs,
                                out_count,
                            },
                            call,
                        );
                    }
                    ResolvedCallOutputCount::FromSlot(out_count_slot) => {
                        self.emit_call(
                            Instr::CallWorkspaceFirstExpandMultiOutputUsingOutputSlot {
                                name: name.0.clone(),
                                identity: identity.clone(),
                                fallback_policy: call.fallback_policy,
                                bare_identifier: call.bare_identifier,
                                specs,
                                out_count_slot,
                            },
                            call,
                        );
                    }
                }
            } else {
                match requested_outputs {
                    ResolvedCallOutputCount::Fixed(out_count) => {
                        self.emit_call(
                            Instr::CallWorkspaceFirstMulti {
                                name: name.0.clone(),
                                identity: identity.clone(),
                                fallback_policy: call.fallback_policy,
                                bare_identifier: call.bare_identifier,
                                arg_count: call.args.len(),
                                out_count,
                            },
                            call,
                        );
                    }
                    ResolvedCallOutputCount::FromSlot(out_count_slot) => {
                        self.emit_call(
                            Instr::CallWorkspaceFirstMultiUsingOutputSlot {
                                name: name.0.clone(),
                                identity: identity.clone(),
                                fallback_policy: call.fallback_policy,
                                bare_identifier: call.bare_identifier,
                                arg_count: call.args.len(),
                                out_count_slot,
                            },
                            call,
                        );
                    }
                }
            }
            return Ok(());
        }
        match &call.callee {
            MirCallee::Static(CallableIdentity::ExternalFunction {
                function,
                display_name,
            }) => {
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                let identity = CallableIdentity::ExternalFunction {
                    function: *function,
                    display_name: display_name.clone(),
                };
                if has_expansion {
                    let out_count = requested_outputs.require_fixed(
                        self,
                        "dynamic output count is not supported for expanded semantic calls",
                    )?;
                    self.emit_call(
                        Instr::CallFunctionExpandMultiOutput {
                            identity,
                            fallback_policy: call.fallback_policy,
                            specs,
                            out_count,
                        },
                        call,
                    );
                } else {
                    match requested_outputs {
                        ResolvedCallOutputCount::Fixed(out_count) => {
                            self.emit_call(
                                Instr::CallFunctionMulti {
                                    identity,
                                    fallback_policy: call.fallback_policy,
                                    arg_count: call.args.len(),
                                    out_count,
                                },
                                call,
                            );
                        }
                        ResolvedCallOutputCount::FromSlot(out_count_slot) => {
                            self.emit_call(
                                Instr::CallFunctionMultiUsingOutputSlot {
                                    identity,
                                    fallback_policy: call.fallback_policy,
                                    arg_count: call.args.len(),
                                    out_count_slot,
                                },
                                call,
                            );
                        }
                    }
                }
            }
            MirCallee::Static(CallableIdentity::BoundFunction(function)) => {
                // Session-resolved semantic calls can target functions compiled in prior
                // submissions; those do not exist in the current assembly layout and should
                // compile as regular semantic calls without nested-capture wiring.
                let captures = self
                    .layout
                    .as_ref()
                    .and_then(|layout| layout.functions.get(function))
                    .map(|layout| layout.captures.clone())
                    .unwrap_or_default();
                if let Some(capture_slots) =
                    self.semantic_capture_slots_for_call(*function, &captures)?
                {
                    for arg in &call.args {
                        self.compile_mir_call_arg(arg)?;
                    }
                    if has_expansion {
                        let out_count = requested_outputs.require_fixed(
                            self,
                            "dynamic output count is not supported for expanded nested semantic calls",
                        )?;
                        self.emit_call(
                            Instr::CallSemanticNestedFunctionExpandMultiOutput {
                                function: *function,
                                capture_slots,
                                specs,
                                out_count,
                            },
                            call,
                        );
                    } else {
                        match requested_outputs {
                            ResolvedCallOutputCount::Fixed(out_count) => {
                                self.emit_call(
                                    Instr::CallSemanticNestedFunctionMulti {
                                        function: *function,
                                        capture_slots,
                                        arg_count: call.args.len(),
                                        out_count,
                                    },
                                    call,
                                );
                            }
                            ResolvedCallOutputCount::FromSlot(out_count_slot) => {
                                self.emit_call(
                                    Instr::CallSemanticNestedFunctionMultiUsingOutputSlot {
                                        function: *function,
                                        capture_slots,
                                        arg_count: call.args.len(),
                                        out_count_slot,
                                    },
                                    call,
                                );
                            }
                        }
                    }
                    return Ok(());
                }
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                if has_expansion {
                    let out_count = requested_outputs.require_fixed(
                        self,
                        "dynamic output count is not supported for expanded semantic calls",
                    )?;
                    self.emit_call(
                        Instr::CallSemanticFunctionExpandMultiOutput(*function, specs, out_count),
                        call,
                    );
                } else {
                    match requested_outputs {
                        ResolvedCallOutputCount::Fixed(out_count) => {
                            self.emit_call(
                                Instr::CallSemanticFunctionMulti(
                                    *function,
                                    call.args.len(),
                                    out_count,
                                ),
                                call,
                            );
                        }
                        ResolvedCallOutputCount::FromSlot(out_count_slot) => {
                            self.emit_call(
                                Instr::CallSemanticFunctionMultiUsingOutputSlot(
                                    *function,
                                    call.args.len(),
                                    out_count_slot,
                                ),
                                call,
                            );
                        }
                    }
                }
            }
            MirCallee::SuperConstructor {
                current_class,
                super_class,
            } => {
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                let requested_outputs = requested_outputs.require_fixed(
                    self,
                    "dynamic output count is not supported for super constructor calls",
                )?;
                if has_expansion {
                    self.emit_call(
                        Instr::CallSuperConstructorExpandMultiOutput {
                            current_class: current_class.clone(),
                            super_class: super_class.clone(),
                            specs,
                            out_count: requested_outputs,
                        },
                        call,
                    );
                } else {
                    self.emit_call(
                        Instr::CallSuperConstructorMulti {
                            current_class: current_class.clone(),
                            super_class: super_class.clone(),
                            arg_count: call.args.len(),
                            out_count: requested_outputs,
                        },
                        call,
                    );
                }
            }
            MirCallee::SuperMethod {
                current_class,
                super_class,
                method,
            } => {
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                let requested_outputs = requested_outputs.require_fixed(
                    self,
                    "dynamic output count is not supported for super method calls",
                )?;
                if has_expansion {
                    self.emit_call(
                        Instr::CallSuperMethodExpandMultiOutput {
                            current_class: current_class.clone(),
                            super_class: super_class.clone(),
                            method: method.clone(),
                            specs,
                            out_count: requested_outputs,
                        },
                        call,
                    );
                } else {
                    self.emit_call(
                        Instr::CallSuperMethodMulti {
                            current_class: current_class.clone(),
                            super_class: super_class.clone(),
                            method: method.clone(),
                            arg_count: call.args.len(),
                            out_count: requested_outputs,
                        },
                        call,
                    );
                }
            }
            MirCallee::Dynamic(callee) => {
                self.compile_mir_dynamic_callee_operand(callee)?;
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                if has_expansion {
                    match requested_outputs {
                        ResolvedCallOutputCount::Fixed(out_count) => {
                            self.emit_call(
                                Instr::CallFevalExpandMultiOutput(specs, out_count),
                                call,
                            );
                        }
                        ResolvedCallOutputCount::FromSlot(out_count_slot) => {
                            self.emit_call(
                                Instr::CallFevalExpandMultiOutputUsingOutputSlot(
                                    specs,
                                    out_count_slot,
                                ),
                                call,
                            );
                        }
                    }
                } else {
                    match requested_outputs {
                        ResolvedCallOutputCount::Fixed(out_count) => {
                            self.emit_call(Instr::CallFevalMulti(call.args.len(), out_count), call);
                        }
                        ResolvedCallOutputCount::FromSlot(out_count_slot) => {
                            self.emit_call(
                                Instr::CallFevalMultiUsingOutputSlot(
                                    call.args.len(),
                                    out_count_slot,
                                ),
                                call,
                            );
                        }
                    }
                }
            }
            MirCallee::Static(CallableIdentity::Builtin(id)) => {
                let name = self.mir_builtin_call_name(id)?;
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                if has_expansion {
                    let requested_outputs = requested_outputs.require_fixed(
                        self,
                        "dynamic output count is not supported for expanded builtin calls",
                    )?;
                    self.emit_call(
                        Instr::CallBuiltinExpandMultiOutput(name, specs, requested_outputs),
                        call,
                    );
                } else {
                    match requested_outputs {
                        ResolvedCallOutputCount::Fixed(out_count) => {
                            self.emit_call(
                                Instr::CallBuiltinMulti(name, call.args.len(), out_count),
                                call,
                            );
                        }
                        ResolvedCallOutputCount::FromSlot(out_count_slot) => {
                            self.emit_call(
                                Instr::CallBuiltinMultiUsingOutputSlot(
                                    name,
                                    call.args.len(),
                                    out_count_slot,
                                ),
                                call,
                            );
                        }
                    }
                }
            }
            MirCallee::Static(identity) => {
                let fallback_policy = call.fallback_policy;
                self.validate_static_call_callee(identity, fallback_policy)?;
                for arg in &call.args {
                    self.compile_mir_call_arg(arg)?;
                }
                if has_expansion {
                    let requested_outputs = requested_outputs.require_fixed(
                        self,
                        "dynamic output count is not supported for expanded static calls",
                    )?;
                    self.emit_call(
                        Instr::CallFunctionExpandMultiOutput {
                            identity: identity.clone(),
                            fallback_policy,
                            specs,
                            out_count: requested_outputs,
                        },
                        call,
                    );
                } else {
                    match requested_outputs {
                        ResolvedCallOutputCount::Fixed(out_count) => {
                            self.emit_call(
                                Instr::CallFunctionMulti {
                                    identity: identity.clone(),
                                    fallback_policy,
                                    arg_count: call.args.len(),
                                    out_count,
                                },
                                call,
                            );
                        }
                        ResolvedCallOutputCount::FromSlot(out_count_slot) => {
                            self.emit_call(
                                Instr::CallFunctionMultiUsingOutputSlot {
                                    identity: identity.clone(),
                                    fallback_policy,
                                    arg_count: call.args.len(),
                                    out_count_slot,
                                },
                                call,
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn call_requested_output_count(
        &self,
        call: &MirCall,
    ) -> Result<ResolvedCallOutputCount, CompileError> {
        match call.requested_outputs {
            RequestedOutputCount::CurrentFunctionNargout => {
                let slot = self.current_function_nargout_slot()?;
                Ok(ResolvedCallOutputCount::FromSlot(slot))
            }
            RequestedOutputCount::DestinationSequenceCardinality => Err(self.compile_error(
                "destination-cardinality output request requires a sequence assignment",
            )),
            _ => Ok(ResolvedCallOutputCount::Fixed(
                call.requested_outputs.known_count().ok_or_else(|| {
                    self.compile_error("call output count is not statically known")
                })?,
            )),
        }
    }

    fn resolved_call_output_count(
        &self,
        call: &MirCall,
    ) -> Result<ResolvedCallOutputCount, CompileError> {
        self.call_requested_output_count(call)
    }

    fn current_function_nargout_slot(&self) -> Result<usize, CompileError> {
        let layout = self
            .layout
            .as_ref()
            .ok_or_else(|| self.compile_error("compiler missing VM layout"))?;
        let function = self
            .function
            .ok_or_else(|| self.compile_error("compiler missing selected function"))?;
        let function_layout = layout.functions.get(&function).ok_or_else(|| {
            self.compile_error(format!("missing VM layout for function {function:?}"))
        })?;
        let slot = function_layout.frame_abi.implicit_nargout.ok_or_else(|| {
            self.compile_error(
                "dynamic requested output count requires function implicit nargout slot",
            )
        })?;
        Ok(slot.0)
    }

    fn output_count_for_targets(
        &self,
        targets: &runmat_mir::MirOutputTargetList,
    ) -> Result<usize, CompileError> {
        targets
            .validate_fixed_arity("MIR multi-assign")
            .map_err(|message| self.compile_error(message))
    }

    fn validate_static_call_callee(
        &self,
        identity: &CallableIdentity,
        fallback_policy: CallableFallbackPolicy,
    ) -> Result<(), CompileError> {
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
        Ok(())
    }

    fn validate_workspace_first_static_call_callee(
        &self,
        identity: &CallableIdentity,
        fallback_policy: CallableFallbackPolicy,
    ) -> Result<(), CompileError> {
        if matches!(fallback_policy, CallableFallbackPolicy::None) {
            return match identity {
                CallableIdentity::Builtin(runmat_hir::BuiltinId(name)) if !name.trim().is_empty() => {
                    Ok(())
                }
                _ => Err(self
                    .compile_error(format!(
                        "MIR workspace-first call fallback policy {:?} is not supported for static callee {:?}",
                        fallback_policy, identity
                    ))
                    .with_identifier(IDENT_MIR_CALL_FALLBACK_POLICY_UNSUPPORTED)),
            };
        }
        self.validate_static_call_callee(identity, fallback_policy)
    }

    fn mir_runtime_name_callee(&self, callee: &CallableIdentity) -> Option<String> {
        match callee {
            CallableIdentity::Builtin(runmat_hir::BuiltinId(name)) => {
                let trimmed = name.trim();
                (!trimmed.is_empty()).then_some(trimmed.to_string())
            }
            CallableIdentity::DynamicName(runmat_hir::SymbolName(name)) => {
                let trimmed = name.trim();
                (!trimmed.is_empty()).then_some(trimmed.to_string())
            }
            CallableIdentity::ExternalName(runmat_hir::QualifiedName(segments))
                if segments.len() > 1
                    && segments.iter().all(|segment| !segment.0.trim().is_empty()) =>
            {
                Some(
                    segments
                        .iter()
                        .map(|segment| segment.0.trim())
                        .collect::<Vec<_>>()
                        .join("."),
                )
            }
            CallableIdentity::Imported(path) => imported_handle_runtime_name(path),
            _ => None,
        }
    }

    fn mir_method_or_member_callee_supported(&self, callee: &CallableIdentity) -> bool {
        match callee {
            CallableIdentity::DynamicName(runmat_hir::SymbolName(name))
            | CallableIdentity::Method(runmat_hir::MethodId(name)) => !name.trim().is_empty(),
            CallableIdentity::ExternalName(runmat_hir::QualifiedName(segments)) => {
                segments.len() == 1 && !segments[0].0.trim().is_empty()
            }
            _ => false,
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
        if !self.mir_method_or_member_callee_supported(&identity) {
            return Err(self
                .compile_error(format!(
                    "MIR method-call callee identity {:?} is not supported for method/member dispatch",
                    identity
                ))
                .with_identifier(IDENT_MIR_METHOD_CALL_CALLEE_INVALID));
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
            let (specs, _) = self.mir_call_arg_specs(&call.args)?;
            let output_count = self.resolved_call_output_count(call)?.require_fixed(
                self,
                "dynamic output count is not supported for expanded method/member calls",
            )?;
            self.emit_call(
                Instr::CallMethodOrMemberIndexExpandMultiOutput {
                    identity,
                    fallback_policy,
                    specs,
                    out_count: output_count,
                },
                call,
            );
            return Ok(());
        }
        let argc = call.args.len().saturating_sub(1);
        let output_count = self.resolved_call_output_count(call)?.require_fixed(
            self,
            "dynamic output count is not supported for method/member calls",
        )?;
        self.emit_call(
            Instr::CallMethodOrMemberIndexMulti {
                identity,
                fallback_policy,
                arg_count: argc,
                out_count: output_count,
            },
            call,
        );
        Ok(())
    }

    fn mir_call_arg_specs(
        &self,
        args: &[MirCallArg],
    ) -> Result<(Vec<ArgumentSpec>, bool), CompileError> {
        let mut has_expansion = false;
        let specs = args
            .iter()
            .map(|arg| {
                Ok(match arg {
                    MirCallArg::Single(_) => ArgumentSpec::Single,
                    MirCallArg::Expansion(source) => {
                        has_expansion = true;
                        ArgumentSpec::Expansion(match source {
                            runmat_mir::MirExpansionSource::SubscriptChain(_) => {
                                return Err(self.compile_error(
                                "subscript-chain expansions must be captured before call lowering",
                            ));
                            }
                            runmat_mir::MirExpansionSource::CellContents { indexing, .. } => {
                                ArgumentExpansionSpec::CellContents {
                                    num_indices: if indexing.cell_expand_all {
                                        0
                                    } else {
                                        indexing.components.len()
                                    },
                                    expand_all: indexing.cell_expand_all,
                                }
                            }
                            runmat_mir::MirExpansionSource::ReturnedOutputs(_) => {
                                ArgumentExpansionSpec::ReturnedOutputs
                            }
                            runmat_mir::MirExpansionSource::Member { member, .. } => {
                                ArgumentExpansionSpec::Member(member.clone())
                            }
                            runmat_mir::MirExpansionSource::DynamicMember { .. } => {
                                ArgumentExpansionSpec::DynamicMember
                            }
                        })
                    }
                    MirCallArg::CapturedSequence(slot) => {
                        has_expansion = true;
                        ArgumentSpec::CapturedSequence { slot: slot.0 }
                    }
                })
            })
            .collect::<Result<Vec<_>, CompileError>>()?;
        Ok((specs, has_expansion))
    }

    pub(super) fn compile_mir_call_arg(&mut self, arg: &MirCallArg) -> Result<(), CompileError> {
        match arg {
            MirCallArg::Single(operand) => self.compile_mir_operand(operand),
            MirCallArg::Expansion(source) => match source {
                runmat_mir::MirExpansionSource::SubscriptChain(_) => Err(self.compile_error(
                    "subscript-chain expansions must be captured before call lowering",
                )),
                runmat_mir::MirExpansionSource::CellContents { base, indexing } => {
                    self.compile_mir_operand(base)?;
                    self.compile_mir_cell_selector_operands(indexing)?;
                    Ok(())
                }
                runmat_mir::MirExpansionSource::ReturnedOutputs(base)
                | runmat_mir::MirExpansionSource::Member { base, .. } => {
                    self.compile_mir_operand(base)
                }
                runmat_mir::MirExpansionSource::DynamicMember { base, member } => {
                    self.compile_mir_operand(base)?;
                    self.compile_mir_operand(member)
                }
            },
            MirCallArg::CapturedSequence(_) => Ok(()),
        }
    }

    fn mir_builtin_call_name(
        &self,
        builtin: &runmat_hir::BuiltinId,
    ) -> Result<String, CompileError> {
        let candidate = builtin.0.clone();
        if is_vm_intrinsic_builtin(&candidate) {
            return Ok(candidate);
        }
        if runmat_builtins::builtin_name_is_known(&candidate) {
            return Ok(candidate);
        }
        Err(CompileError::new(format!("unknown builtin id {candidate}"))
            .with_identifier(IDENT_MIR_BUILTIN_UNKNOWN))
    }

    fn compile_mir_aggregate(
        &mut self,
        kind: &MirAggregateKind,
        row_lengths: &[usize],
        elements: &[runmat_mir::MirAggregateElement],
    ) -> Result<(), CompileError> {
        let Some(element_count) = row_lengths
            .iter()
            .try_fold(0usize, |total, length| total.checked_add(*length))
        else {
            return Err(self
                .compile_error("MIR aggregate row element count overflows the platform")
                .with_identifier(IDENT_MIR_AGGREGATE_SHAPE_INVALID));
        };
        if element_count != elements.len() {
            return Err(self
                .compile_error("MIR aggregate shape does not match aggregate element count")
                .with_identifier(IDENT_MIR_AGGREGATE_SHAPE_INVALID));
        }
        let rows = row_lengths.len();
        let rectangular_columns = row_lengths
            .first()
            .copied()
            .filter(|columns| row_lengths.iter().all(|candidate| candidate == columns));

        if elements.iter().any(|element| {
            matches!(
                element,
                runmat_mir::MirAggregateElement::CapturedSequence(_)
            )
        }) {
            let specs = elements
                .iter()
                .map(|element| match element {
                    runmat_mir::MirAggregateElement::Single(operand) => {
                        self.compile_mir_operand(operand)?;
                        Ok(crate::bytecode::AggregateElementSpec::Single)
                    }
                    runmat_mir::MirAggregateElement::CapturedSequence(sequence) => {
                        Ok(crate::bytecode::AggregateElementSpec::CapturedSequence {
                            slot: sequence.0,
                        })
                    }
                })
                .collect::<Result<Vec<_>, CompileError>>()?;
            self.emit(match kind {
                MirAggregateKind::Tensor => Instr::CreateMatrixFromSequences {
                    rows,
                    row_lengths: row_lengths.to_vec(),
                    elements: specs,
                },
                MirAggregateKind::Cell => Instr::CreateCellFromSequences {
                    rows,
                    row_lengths: row_lengths.to_vec(),
                    elements: specs,
                },
            });
            return Ok(());
        }

        match kind {
            MirAggregateKind::Tensor
                if self.mir_aggregate_needs_dynamic_concat(elements)
                    || rectangular_columns.is_none() =>
            {
                for element in elements {
                    self.compile_mir_aggregate_element(element)?;
                }
                for columns in row_lengths {
                    self.emit(Instr::LoadConst(*columns as f64));
                }
                self.emit(Instr::CreateMatrixDynamic(rows));
            }
            MirAggregateKind::Tensor => {
                for element in elements {
                    self.compile_mir_aggregate_element(element)?;
                }
                self.emit(Instr::CreateMatrix(rows, rectangular_columns.unwrap_or(0)));
            }
            MirAggregateKind::Cell => {
                let Some(columns) = rectangular_columns else {
                    return Err(self
                        .compile_error("cell literal rows realize different widths")
                        .with_identifier(IDENT_MIR_AGGREGATE_SHAPE_INVALID));
                };
                for element in elements {
                    self.compile_mir_aggregate_element(element)?;
                }
                self.emit(Instr::CreateCell2D(rows, columns));
            }
        };
        Ok(())
    }

    fn compile_mir_aggregate_element(
        &mut self,
        element: &runmat_mir::MirAggregateElement,
    ) -> Result<(), CompileError> {
        match element {
            runmat_mir::MirAggregateElement::Single(operand) => self.compile_mir_operand(operand),
            runmat_mir::MirAggregateElement::CapturedSequence(_) => Ok(()),
        }
    }

    fn compile_mir_struct_literal(
        &mut self,
        fields: &[(runmat_hir::MemberName, MirOperand)],
    ) -> Result<(), CompileError> {
        let mut names = Vec::with_capacity(fields.len());
        for (name, value) in fields {
            self.compile_mir_operand(value)?;
            names.push(name.0.clone());
        }
        self.emit(Instr::CreateStructLiteral(names));
        Ok(())
    }

    fn compile_mir_object_literal(
        &mut self,
        class_name: &runmat_hir::QualifiedName,
        fields: &[(runmat_hir::MemberName, MirOperand)],
    ) -> Result<(), CompileError> {
        let class_name =
            runmat_types::ClassIdentity::from_qualified_name(class_name).map_err(|error| {
                self.compile_error(format!("invalid MIR object literal class name: {error}"))
                    .with_identifier(IDENT_MIR_CALL_TARGET_NAME_INVALID)
            })?;
        let mut names = Vec::with_capacity(fields.len());
        for (name, value) in fields {
            self.compile_mir_operand(value)?;
            names.push(name.0.clone());
        }
        self.emit(Instr::CreateObjectLiteral {
            class_name,
            fields: names,
        });
        Ok(())
    }

    fn mir_aggregate_needs_dynamic_concat(
        &self,
        elements: &[runmat_mir::MirAggregateElement],
    ) -> bool {
        elements.iter().any(|element| {
            element
                .operand()
                .is_none_or(|operand| self.mir_operand_needs_dynamic_concat(operand))
        })
    }

    fn mir_delete_rhs_is_empty_tensor_literal(&self, value: &MirRvalue) -> bool {
        matches!(
            value,
            MirRvalue::Aggregate {
                kind: MirAggregateKind::Tensor,
                row_lengths,
                elements,
            } if row_lengths.is_empty() && elements.is_empty()
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
                | MirRvalue::WorkspaceFirstStaticProperty { .. }
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

        self.compile_mir_operand(base)?;
        match indexing.kind {
            IndexKind::Paren => self.compile_mir_slice_index(indexing)?,
            IndexKind::Brace => {
                // A generic rvalue has a scalar stack contract even when the HIR
                // records that brace syntax can denote a comma-separated list.
                // Sequence-aware owners (argument capture and prepared output
                // assignment) lower that syntax through their explicit carriers;
                // emitting IndexCellList here would leave the transient sequence
                // register live across the scalar consumer's store instructions.
                self.compile_mir_cell_index_components(indexing, indexing.result_context)?;
                self.emit(Instr::IndexCell {
                    num_indices: indexing.components.len(),
                });
            }
        };
        Ok(())
    }

    fn compile_mir_cell_index_components(
        &mut self,
        indexing: &MirIndexing,
        expected_context: IndexResultContext,
    ) -> Result<(), CompileError> {
        if !mir_indexing_context_matches(indexing.result_context, expected_context) {
            return Err(self
                .compile_error("MIR cell index lowering received mismatched index result context")
                .with_identifier(IDENT_MIR_CELL_INDEX_CONTEXT_INVALID));
        }
        if indexing
            .components
            .iter()
            .any(|component| matches!(component, MirIndexComponent::ContextualExpr(_)))
        {
            self.compile_contextual_index_values(indexing)?;
            return Ok(());
        }
        for component in &indexing.components {
            match component {
                MirIndexComponent::Expr(operand) => {
                    self.compile_mir_operand(operand)?;
                }
                _ => {
                    return Err(self
                        .compile_error("MIR cell index lowering expects expression selectors")
                        .with_identifier(IDENT_MIR_CELL_INDEX_PLAN_INVALID))
                }
            }
        }
        Ok(())
    }

    fn compile_mir_slice_index(&mut self, indexing: &MirIndexing) -> Result<(), CompileError> {
        if indexing
            .components
            .iter()
            .any(|component| matches!(component, MirIndexComponent::ContextualExpr(_)))
        {
            let (numeric_count, colon_mask, end_mask) =
                self.compile_contextual_slice_components(indexing)?;
            self.emit(Instr::IndexSlice(
                indexing.components.len(),
                numeric_count,
                colon_mask,
                end_mask,
            ));
            return Ok(());
        }
        match indexing.plan {
            MirIndexPlan::Scalar => {
                if indexing.components.len() > 2 {
                    let (numeric_count, colon_mask, end_mask) =
                        self.compile_mir_slice_components(indexing)?;
                    self.emit(Instr::IndexSlice(
                        indexing.components.len(),
                        numeric_count,
                        colon_mask,
                        end_mask,
                    ));
                } else {
                    self.compile_mir_scalar_index_components(indexing)?;
                    self.emit(Instr::Index(indexing.components.len()));
                }
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
            self.compile_mir_operand(operand)?;
        }
        Ok(())
    }

    fn compile_mir_slice_components(
        &mut self,
        indexing: &MirIndexing,
    ) -> Result<(usize, u32, u32), CompileError> {
        let mut colon_mask = 0u32;
        let end_mask = 0u32;
        let mut numeric_count = 0usize;

        for (dim, component) in indexing.components.iter().enumerate() {
            match component {
                MirIndexComponent::Colon => self.set_selector_mask_bit(
                    &mut colon_mask,
                    dim,
                    IDENT_MIR_SLICE_INDEX_PLAN_INVALID,
                    "MIR slice lowering invariant violated: selector dimension exceeds mask width",
                )?,
                MirIndexComponent::Expr(operand) => {
                    self.compile_mir_operand(operand)?;
                    numeric_count += 1;
                }
                MirIndexComponent::ContextualExpr(_) => {
                    return Err(self
                        .compile_error("contextual selectors require contextual slice lowering"));
                }
            }
        }

        Ok((numeric_count, colon_mask, end_mask))
    }

    fn set_selector_mask_bit(
        &self,
        mask: &mut u32,
        dim: usize,
        identifier: &'static str,
        message: &str,
    ) -> Result<(), CompileError> {
        if dim >= u32::BITS as usize {
            return Err(self.compile_error(message).with_identifier(identifier));
        }
        *mask |= 1u32 << dim;
        Ok(())
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
            MirOperand::Constant(MirConstant::IntegerLiteral(value)) => {
                self.emit(Instr::LoadInt(runmat_value::IntValue::from(value)));
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
                    runmat_value::Value::Num(value) => self.emit(Instr::LoadConst(*value)),
                    runmat_value::Value::Complex(re, im) => self.emit(Instr::LoadComplex(*re, *im)),
                    runmat_value::Value::Bool(value) => self.emit(Instr::LoadBool(*value)),
                    _ => {
                        return Err(self.compile_error(format!(
                            "constant {name} is not supported in primary MIR lowering yet"
                        )));
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

    fn compile_mir_dynamic_callee_operand(
        &mut self,
        operand: &MirOperand,
    ) -> Result<(), CompileError> {
        if let Some(name) = callback_name_from_mir_operand(operand) {
            if self.compile_semantic_function_handle_for_name(&name)? {
                return Ok(());
            }
        }
        self.compile_mir_operand(operand)
    }

    fn compile_mir_function_handle(
        &mut self,
        target: &CallableIdentity,
    ) -> Result<(), CompileError> {
        match target {
            CallableIdentity::Method(runmat_hir::MethodId(name)) => {
                let trimmed = name.trim();
                if trimmed.is_empty() {
                    return Err(self
                        .compile_error(format!(
                            "missing runtime name for function handle target {target:?}"
                        ))
                        .with_identifier(IDENT_MIR_FUNCTION_HANDLE_NAME_MISSING));
                }
                self.emit(Instr::CreateMethodFunctionHandle(trimmed.to_string()));
                Ok(())
            }
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
            CallableIdentity::ExternalFunction {
                function,
                display_name,
            } => {
                self.emit(Instr::CreateExternalBoundFunctionHandle(
                    *function,
                    display_name.clone(),
                ));
                Ok(())
            }
            CallableIdentity::BoundFunction(function) => {
                let Some((captures, display_name)) = self
                    .layout
                    .as_ref()
                    .and_then(|layout| layout.functions.get(function))
                    .map(|layout| (layout.captures.clone(), layout.display_name.clone()))
                else {
                    // External semantic function identities may not have a local VM layout in the
                    // current compilation unit. Keep the identity and emit a simple semantic handle.
                    self.emit(Instr::CreateBoundFunctionHandle(
                        *function,
                        format!("bound_function_{}", function.0),
                    ));
                    return Ok(());
                };
                if captures.is_empty() {
                    self.emit(Instr::CreateBoundFunctionHandle(*function, display_name));
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

    fn compile_semantic_function_handle_for_name(
        &mut self,
        name: &str,
    ) -> Result<bool, CompileError> {
        let Some((function, captures, display_name)) =
            self.resolve_visible_semantic_function_layout(name)
        else {
            return Ok(false);
        };
        if captures.is_empty() {
            self.emit(Instr::CreateBoundFunctionHandle(function, display_name));
            return Ok(true);
        }
        for capture in &captures {
            let slot = self.binding_slot(capture.binding)?;
            self.emit(Instr::LoadVar(slot));
        }
        self.emit(Instr::CreateSemanticClosure(
            function,
            display_name,
            captures.len(),
        ));
        Ok(true)
    }

    fn resolve_visible_semantic_function_layout(
        &self,
        name: &str,
    ) -> Option<(FunctionId, Vec<crate::layout::VmCaptureSlot>, String)> {
        let layout = self.layout.as_ref()?;
        let current_function = self.function;
        if let Some(current) = current_function {
            let owner_scope = layout
                .functions
                .get(&current)
                .map(|function| function.private_owner_scope.as_str())
                .unwrap_or_default();
            if !owner_scope.is_empty() && !name.contains('.') {
                let scoped_name = format!("{owner_scope}.__private__.{name}");
                if let Some((function, function_layout)) = layout
                    .functions
                    .iter()
                    .find(|(_, function_layout)| function_layout.display_name == scoped_name)
                {
                    return Some((
                        *function,
                        function_layout.captures.clone(),
                        function_layout.display_name.clone(),
                    ));
                }
            }
        }
        let mut ids: Vec<_> = layout.functions.keys().copied().collect();
        ids.sort_by_key(|id| id.0);
        ids.into_iter().find_map(|function| {
            let function_layout = layout.functions.get(&function)?;
            if function_layout.display_name != name {
                return None;
            }
            let visible = function_layout.captures.is_empty()
                || current_function.is_some_and(|current| {
                    function_layout
                        .captures
                        .iter()
                        .all(|capture| capture.from_function == current)
                });
            visible.then(|| {
                (
                    function,
                    function_layout.captures.clone(),
                    function_layout.display_name.clone(),
                )
            })
        })
    }

    fn semantic_capture_slots_for_call(
        &self,
        target_function: FunctionId,
        captures: &[crate::layout::VmCaptureSlot],
    ) -> Result<Option<Vec<usize>>, CompileError> {
        if captures.is_empty() {
            return Ok(None);
        }

        let current_function = self
            .function
            .ok_or_else(|| self.compile_error("compiler missing selected function"))?;
        let captures_are_parent_to_child = captures
            .iter()
            .all(|capture| capture.from_function == current_function);
        let captures_are_self_recursive = target_function == current_function;
        if !captures_are_parent_to_child && !captures_are_self_recursive {
            return Ok(None);
        }

        captures
            .iter()
            .map(|capture| self.binding_slot(capture.binding))
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
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
            Instr::LoadVar(id) | Instr::LoadVarForIndexAssignment(id) | Instr::StoreVar(id) => {
                self.ensure_var(*id)
            }
            _ => {}
        }
        let pc = self.instructions.len();
        self.instructions.push(instr);
        let span = self.current_span.unwrap_or_default();
        self.instr_spans.push(span);
        self.call_arg_spans.push(None);
        pc
    }

    fn emit_call(&mut self, instr: Instr, call: &MirCall) -> usize {
        let pc = self.emit(instr);
        self.call_arg_spans[pc] = Some(call.arg_spans.clone());
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

fn callback_name_from_mir_operand(operand: &MirOperand) -> Option<String> {
    let MirOperand::Constant(MirConstant::String(value)) = operand else {
        return None;
    };
    let text = string_literal_runtime_text(&value.0);
    let name = text.trim().strip_prefix('@').unwrap_or(text.trim()).trim();
    (!name.is_empty()).then(|| name.to_string())
}

fn string_literal_runtime_text(value: &str) -> String {
    runmat_hir::StringLiteral(value.to_string()).runtime_text()
}

fn emit_string_literal(compiler: &mut Compiler, value: &str) {
    let literal = runmat_hir::StringLiteral(value.to_string());
    let text = literal.runtime_text();
    if literal.is_character_row() {
        compiler.emit(Instr::LoadCharRow(text));
    } else {
        compiler.emit(Instr::LoadString(text));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{
        VmAssemblyLayout, VmFrameAbi, VmFunctionLayout, VmSlotId, VmStorageBinding,
    };
    use runmat_hir::{FunctionAbi, FunctionId};
    use runmat_mir::{BasicBlock, BasicBlockId, MirLocal, MirLocalId, MirLocalKind, MirTerminator};

    fn empty_function_abi() -> FunctionAbi {
        FunctionAbi {
            fixed_inputs: Vec::new(),
            varargin: None,
            fixed_outputs: Vec::new(),
            varargout: None,
            implicit_nargin: None,
            implicit_nargout: None,
        }
    }

    fn compiler_with_local_assignments(assignments: Vec<MirRvalue>) -> Compiler {
        compiler_with_assignments(
            assignments
                .into_iter()
                .map(|value| (MirLocalId(0), value))
                .collect(),
        )
    }

    fn compiler_with_assignments(assignments: Vec<(MirLocalId, MirRvalue)>) -> Compiler {
        let function = FunctionId(0);
        let target_local = MirLocalId(0);
        let mut mir_local_slots = HashMap::new();
        let mut locals = Vec::new();
        for local in assignments
            .iter()
            .map(|(local, _)| *local)
            .chain(std::iter::once(target_local))
        {
            if mir_local_slots.contains_key(&local) {
                continue;
            }
            let slot = VmSlotId(7 + local.0);
            mir_local_slots.insert(local, slot);
            locals.push(MirLocal {
                id: local,
                binding: None,
                kind: MirLocalKind::Temporary,
                span: runmat_hir::Span::default(),
            });
        }
        let local_count = mir_local_slots
            .values()
            .map(|slot| slot.0 + 1)
            .max()
            .unwrap_or(0);
        let mut functions = HashMap::new();
        functions.insert(
            function,
            VmFunctionLayout {
                function,
                display_name: "test".into(),
                private_owner_scope: String::new(),
                frame_abi: VmFrameAbi {
                    fixed_inputs: Vec::new(),
                    varargin: None,
                    fixed_outputs: Vec::new(),
                    varargout: None,
                    implicit_nargin: None,
                    implicit_nargout: None,
                },
                binding_slots: HashMap::new(),
                mir_local_slots,
                captures: Vec::new(),
                local_count,
                resume_points: std::collections::BTreeMap::new(),
            },
        );
        let span = runmat_hir::Span::default();
        let statements = assignments
            .into_iter()
            .map(|(local, value)| MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(local),
                    value,
                },
                span,
            })
            .collect();
        Compiler {
            instructions: Vec::new(),
            instr_spans: Vec::new(),
            call_arg_spans: Vec::new(),
            var_count: local_count,
            imports: Vec::new(),
            var_types: vec![Type::Unknown; local_count],
            layout: Some(VmAssemblyLayout {
                functions,
                entrypoints: HashMap::new(),
                storage_bindings: HashMap::<BindingId, VmStorageBinding>::new(),
            }),
            function: Some(function),
            body: Some(MirBody {
                function,
                abi: empty_function_abi(),
                locals,
                blocks: vec![BasicBlock {
                    id: BasicBlockId(0),
                    statements,
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::Return(Vec::new()),
                        span,
                    },
                }],
            }),
            class_registrations: Vec::new(),
            current_span: None,
            pending_place_mutation: None,
            prepared_index_component: None,
            contextual_index_component: None,
            subscript_end_component: None,
        }
    }

    #[test]
    fn compiler_records_exact_empty_stack_mir_resume_boundaries() {
        let mut compiler = compiler_with_local_assignments(vec![MirRvalue::Use(
            MirOperand::Constant(MirConstant::Number("1".into())),
        )]);
        compiler.compile().unwrap();
        let points = &compiler
            .layout
            .as_ref()
            .unwrap()
            .functions
            .get(&FunctionId(0))
            .unwrap()
            .resume_points;
        assert_eq!(
            points.get(&runmat_types::ProgramPointId {
                function: runmat_types::ProgramFunctionId(0),
                block: 0,
                position: 0,
            }),
            Some(&0)
        );
        let terminator = points
            .get(&runmat_types::ProgramPointId {
                function: runmat_types::ProgramFunctionId(0),
                block: 0,
                position: 1,
            })
            .copied()
            .unwrap();
        assert!(terminator > 0);
        assert!(terminator <= compiler.instructions.len());
    }
}
