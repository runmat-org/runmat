#[cfg(feature = "native-accel")]
use crate::accel::graph::build_accel_graph;
#[cfg(feature = "native-accel")]
use crate::accel::stack_layout::annotate_fusion_groups_with_stack_layout;
use crate::bytecode::instr::Instr;
use crate::layout::VmAssemblyLayout;
#[cfg(feature = "native-accel")]
use runmat_accelerate::graph::AccelGraph;
#[cfg(feature = "native-accel")]
use runmat_accelerate::FusionGroup;
use runmat_builtins::Type;
use runmat_hir::FunctionId;
use runmat_types::{FunctionArgDefaultValue, FunctionArgSizeSpec, FunctionArgValidator};
use runmat_value::Value;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use super::{
    parallel::{BytecodeParforRegion, BytecodeSpmdRegion},
    region::BytecodeRegion,
};

#[derive(Debug, Clone)]
pub struct CallFrame {
    pub function_name: String,
    pub return_address: usize,
    pub locals_start: usize,
    pub locals_count: usize,
    pub expected_outputs: usize,
}

#[derive(Debug)]
pub struct ExecutionContext {
    pub call_stack: Vec<CallFrame>,
    pub locals: Vec<Value>,
    pub instruction_pointer: usize,
    pub runtime: runmat_runtime::context::RuntimeContext,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            call_stack: Vec::new(),
            locals: Vec::new(),
            instruction_pointer: 0,
            runtime: runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionBytecode {
    pub function: FunctionId,
    pub display_name: String,
    pub class_method_owner: Option<runmat_types::ClassMethodOwner>,
    #[serde(default)]
    pub private_owner_scope: String,
    #[serde(default)]
    pub source_id: Option<runmat_hir::SourceId>,
    /// Whole-function semantic requirements, including reachable static
    /// callees, captured by MIR analysis before bytecode serialization.
    #[serde(default)]
    pub capabilities: runmat_types::CapabilitySet,
    pub instructions: Vec<Instr>,
    #[serde(default)]
    pub instr_spans: Vec<runmat_hir::Span>,
    #[serde(default)]
    pub call_arg_spans: Vec<Option<Vec<runmat_hir::Span>>>,
    #[serde(default)]
    pub coverage_sites: Vec<Vec<u64>>,
    pub var_count: usize,
    pub input_slots: Vec<usize>,
    #[serde(default)]
    pub varargin_slot: Option<usize>,
    #[serde(default)]
    pub implicit_nargin_slot: Option<usize>,
    pub output_slots: Vec<usize>,
    #[serde(default)]
    pub varargout_slot: Option<usize>,
    #[serde(default)]
    pub implicit_nargout_slot: Option<usize>,
    pub capture_slots: Vec<usize>,
    #[serde(default)]
    pub var_names: HashMap<usize, String>,
    #[serde(default)]
    pub initially_unassigned_slots: HashSet<usize>,
    #[serde(default)]
    pub argument_validations: Vec<FunctionArgumentValidation>,
    #[serde(default, with = "crate::layout::resume_point_map_serde")]
    pub resume_points: std::collections::BTreeMap<runmat_types::ProgramPointId, usize>,
    #[serde(default)]
    pub regions: Vec<BytecodeRegion>,
    #[serde(default)]
    pub parfor_regions: Vec<BytecodeParforRegion>,
    #[serde(default)]
    pub spmd_regions: Vec<BytecodeSpmdRegion>,
    #[serde(default)]
    pub distributed_values: Vec<runmat_types::DistributedValueContract>,
    #[serde(default)]
    pub collective_contracts: Vec<runmat_types::CollectiveContract>,
}

impl Default for FunctionBytecode {
    fn default() -> Self {
        Self {
            function: FunctionId(0),
            display_name: String::new(),
            class_method_owner: None,
            private_owner_scope: String::new(),
            source_id: None,
            capabilities: Default::default(),
            instructions: Vec::new(),
            instr_spans: Vec::new(),
            call_arg_spans: Vec::new(),
            coverage_sites: Vec::new(),
            var_count: 0,
            input_slots: Vec::new(),
            varargin_slot: None,
            implicit_nargin_slot: None,
            output_slots: Vec::new(),
            varargout_slot: None,
            implicit_nargout_slot: None,
            capture_slots: Vec::new(),
            var_names: HashMap::new(),
            initially_unassigned_slots: HashSet::new(),
            argument_validations: Vec::new(),
            resume_points: std::collections::BTreeMap::new(),
            regions: Vec::new(),
            parfor_regions: Vec::new(),
            spmd_regions: Vec::new(),
            distributed_values: Vec::new(),
            collective_contracts: Vec::new(),
        }
    }
}

impl FunctionBytecode {
    /// Materializes the function-local executable view consumed by the VM.
    /// All compiler-owned metadata travels with the instruction stream here so
    /// function dispatch cannot accidentally discard executable contracts.
    pub(crate) fn execution_bytecode(&self, registry: &FunctionRegistry) -> Bytecode {
        let mut bytecode = Bytecode::with_instructions(self.instructions.clone(), self.var_count);
        bytecode.instr_spans = self.instr_spans.clone();
        bytecode.call_arg_spans = self.call_arg_spans.clone();
        bytecode.coverage_sites = self.coverage_sites.clone();
        bytecode.source_id = self.source_id;
        bytecode.active_function = Some(self.function);
        bytecode.active_class_method_owner = self.class_method_owner.clone();
        bytecode.var_names = self.var_names.clone();
        bytecode.initially_unassigned_slots = self.initially_unassigned_slots.clone();
        bytecode.regions = self.regions.clone();
        bytecode.parfor_regions = self.parfor_regions.clone();
        bytecode.spmd_regions = self.spmd_regions.clone();
        bytecode.distributed_values = self.distributed_values.clone();
        bytecode.collective_contracts = self.collective_contracts.clone();
        bytecode.bound_functions = registry.functions.clone();
        bytecode.function_registry = registry.clone();
        bytecode
    }

    /// Rebinds every compiler-owned identity when a unit-local function is
    /// installed into a session registry. Control-flow and parallel metadata
    /// must move atomically with the function instruction stream.
    pub fn rebind_identity(
        &mut self,
        function: FunctionId,
        program_function: runmat_types::ProgramFunctionId,
    ) {
        self.function = function;
        self.resume_points = std::mem::take(&mut self.resume_points)
            .into_iter()
            .map(|(mut point, pc)| {
                point.function = program_function;
                (point, pc)
            })
            .collect();
        for region in &mut self.regions {
            region.rebind_owner(program_function);
        }
        for region in &mut self.parfor_regions {
            region.rebind_owner(program_function);
        }
        for region in &mut self.spmd_regions {
            region.rebind_owner(program_function);
        }
        for distributed in &mut self.distributed_values {
            distributed.id.function = program_function;
            match &mut distributed.owner {
                runmat_types::DistributedOwner::Client(function) => *function = program_function,
                runmat_types::DistributedOwner::Region(region) => {
                    region.0.function = program_function
                }
            }
        }
        for collective in &mut self.collective_contracts {
            collective.id.region.0.function = program_function;
        }
        for instruction in &mut self.instructions {
            match instruction {
                Instr::ExecuteParfor { region, .. } | Instr::ExecuteSpmd { region, .. } => {
                    region.0.function = program_function;
                }
                _ => {}
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionArgumentValidation {
    pub input_slot: usize,
    pub size: Option<FunctionArgSizeSpec>,
    pub class_name: Option<String>,
    #[serde(default)]
    pub validators: Vec<FunctionArgValidator>,
    #[serde(default)]
    pub default_value: Option<FunctionArgDefaultValue>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FunctionRegistry {
    pub functions: HashMap<FunctionId, FunctionBytecode>,
    #[serde(default)]
    pub names: HashMap<String, FunctionId>,
    #[serde(default)]
    pub source_functions: HashMap<runmat_hir::SourceId, Vec<FunctionId>>,
    #[serde(skip)]
    execution_stacks: OnceLock<HashMap<FunctionId, runmat_types::ExecutionStackRequirement>>,
}

impl FunctionRegistry {
    pub fn new(functions: HashMap<FunctionId, FunctionBytecode>) -> Self {
        let mut names = HashMap::new();
        let mut source_functions: HashMap<runmat_hir::SourceId, Vec<FunctionId>> = HashMap::new();
        let mut ids: Vec<_> = functions.keys().copied().collect();
        ids.sort_by_key(|id| id.0);
        for id in ids {
            if let Some(function) = functions.get(&id) {
                names.entry(function.display_name.clone()).or_insert(id);
                if let Some(source_id) = function.source_id {
                    source_functions.entry(source_id).or_default().push(id);
                }
            }
        }
        Self {
            functions,
            names,
            source_functions,
            execution_stacks: OnceLock::new(),
        }
    }

    pub fn get(&self, function: FunctionId) -> Option<&FunctionBytecode> {
        self.functions.get(&function)
    }

    pub fn resolve_name(&self, name: &str) -> Option<FunctionId> {
        self.names.get(name).copied()
    }

    pub fn resolve_name_in_private_scope(
        &self,
        private_owner_scope: &str,
        name: &str,
    ) -> Option<FunctionId> {
        if private_owner_scope.is_empty() || name.contains('.') {
            return None;
        }
        let scoped_name = format!("{private_owner_scope}.__private__.{name}");
        self.names.get(&scoped_name).copied()
    }

    pub fn insert_replacing_name(&mut self, function: FunctionBytecode) {
        self.execution_stacks.take();
        if let Some(previous) = self
            .names
            .insert(function.display_name.clone(), function.function)
        {
            self.remove(previous);
        }
        let function_id = function.function;
        if let Some(source_id) = function.source_id {
            let functions = self.source_functions.entry(source_id).or_default();
            if !functions.contains(&function_id) {
                functions.push(function_id);
            }
        }
        self.functions.insert(function_id, function);
    }

    pub fn remove(&mut self, function: FunctionId) -> Option<FunctionBytecode> {
        self.execution_stacks.take();
        let removed = self.functions.remove(&function)?;
        if self.names.get(&removed.display_name) == Some(&function) {
            self.names.remove(&removed.display_name);
        }
        if let Some(source_id) = removed.source_id {
            if let Some(functions) = self.source_functions.get_mut(&source_id) {
                functions.retain(|id| *id != function);
                if functions.is_empty() {
                    self.source_functions.remove(&source_id);
                }
            }
        }
        Some(removed)
    }

    pub fn remove_source(&mut self, source: runmat_hir::SourceId) -> Vec<FunctionBytecode> {
        self.execution_stacks.take();
        let ids = self.source_functions.remove(&source).unwrap_or_default();
        let mut removed = Vec::new();
        for id in ids {
            if let Some(function) = self.functions.remove(&id) {
                if self.names.get(&function.display_name) == Some(&id) {
                    self.names.remove(&function.display_name);
                }
                removed.push(function);
            }
        }
        removed
    }

    pub fn functions_for_source(&self, source: runmat_hir::SourceId) -> &[FunctionId] {
        self.source_functions
            .get(&source)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Returns the stack required by this function and its reachable callees.
    pub fn execution_stack(&self, function: FunctionId) -> runmat_types::ExecutionStackRequirement {
        *self
            .execution_stacks
            .get_or_init(|| {
                self.functions
                    .keys()
                    .copied()
                    .map(|function| {
                        (
                            function,
                            self.execution_stack_inner(function, &mut HashSet::new()),
                        )
                    })
                    .collect()
            })
            .get(&function)
            .unwrap_or(&runmat_types::ExecutionStackRequirement::Process)
    }

    fn execution_stack_inner(
        &self,
        function: FunctionId,
        visiting: &mut HashSet<FunctionId>,
    ) -> runmat_types::ExecutionStackRequirement {
        if !visiting.insert(function) {
            return runmat_types::ExecutionStackRequirement::Any;
        }
        let required = self
            .functions
            .get(&function)
            .map(|bytecode| {
                bytecode
                    .instructions
                    .iter()
                    .map(|instruction| self.instruction_execution_stack(instruction, visiting))
                    .max()
                    .unwrap_or(runmat_types::ExecutionStackRequirement::Any)
            })
            .unwrap_or(runmat_types::ExecutionStackRequirement::Process);
        visiting.remove(&function);
        required
    }

    fn instruction_execution_stack(
        &self,
        instruction: &Instr,
        visiting: &mut HashSet<FunctionId>,
    ) -> runmat_types::ExecutionStackRequirement {
        use runmat_types::CallableIdentity;

        let identity_stack =
            |identity: &CallableIdentity, registry: &Self, visiting: &mut HashSet<FunctionId>| {
                match identity {
                    CallableIdentity::BoundFunction(function)
                    | CallableIdentity::AnonymousFunction(function)
                    | CallableIdentity::ExternalFunction { function, .. } => {
                        registry.execution_stack_inner(*function, visiting)
                    }
                    CallableIdentity::Builtin(builtin) => builtin_execution_stack(&builtin.0),
                    CallableIdentity::DynamicName(name) => {
                        if runmat_builtins::builtin_name_is_known(&name.0) {
                            builtin_execution_stack(&name.0)
                        } else {
                            runmat_types::ExecutionStackRequirement::Process
                        }
                    }
                    CallableIdentity::Method(method) => registry
                        .registered_named_method_execution_stack(
                            &runmat_types::MethodName::from(method.0.as_str()),
                            visiting,
                        )
                        .unwrap_or(runmat_types::ExecutionStackRequirement::Process),
                    CallableIdentity::ExternalName(_) | CallableIdentity::Imported(_) => {
                        runmat_types::ExecutionStackRequirement::Process
                    }
                }
            };

        match instruction {
            Instr::CallBuiltinMulti(name, ..)
            | Instr::CallBuiltinMultiUsingOutputSlot(name, ..)
            | Instr::CallBuiltinExpandMultiOutput(name, ..) => builtin_execution_stack(name),
            Instr::CallFunctionMulti { identity, .. }
            | Instr::CallFunctionMultiUsingOutputSlot { identity, .. }
            | Instr::CallWorkspaceFirstMulti { identity, .. }
            | Instr::CallWorkspaceFirstMultiUsingOutputSlot { identity, .. }
            | Instr::CallFunctionExpandMultiOutput { identity, .. }
            | Instr::CallWorkspaceFirstExpandMultiOutput { identity, .. }
            | Instr::CallWorkspaceFirstExpandMultiOutputUsingOutputSlot { identity, .. } => {
                identity_stack(identity, self, visiting)
            }
            Instr::CallSemanticFunctionMulti(function, ..)
            | Instr::CallSemanticFunctionMultiUsingOutputSlot(function, ..)
            | Instr::CallSemanticFunctionExpandMultiOutput(function, ..)
            | Instr::CreateSemanticFuture(function, ..)
            | Instr::CreateSemanticFutureExpandMultiOutput(function, ..) => {
                self.execution_stack_inner(*function, visiting)
            }
            Instr::CallSemanticNestedFunctionMulti { function, .. }
            | Instr::CallSemanticNestedFunctionMultiUsingOutputSlot { function, .. }
            | Instr::CallSemanticNestedFunctionExpandMultiOutput { function, .. } => {
                self.execution_stack_inner(*function, visiting)
            }
            Instr::CallMethodOrMemberIndexMulti { identity, .. }
            | Instr::CallMethodOrMemberIndexExpandMultiOutput { identity, .. } => identity
                .display_name()
                .and_then(|name| self.registered_member_or_method_execution_stack(&name, visiting))
                .unwrap_or_else(|| identity_stack(identity, self, visiting)),
            Instr::CallFevalMulti(..)
            | Instr::CallFevalMultiUsingOutputSlot(..)
            | Instr::CallFevalExpandMultiOutput(..)
            | Instr::CallFevalExpandMultiOutputUsingOutputSlot(..)
            | Instr::CallSuperConstructorMulti { .. }
            | Instr::CallSuperMethodMulti { .. } => {
                runmat_types::ExecutionStackRequirement::Process
            }
            instruction => self.registered_method_execution_stack(instruction, visiting),
        }
    }

    fn registered_method_execution_stack(
        &self,
        instruction: &Instr,
        visiting: &mut HashSet<FunctionId>,
    ) -> runmat_types::ExecutionStackRequirement {
        if !instruction_has_dynamic_dispatch(instruction) {
            return runmat_types::ExecutionStackRequirement::Any;
        }
        let mut targets = HashSet::new();
        for function in self.functions.values() {
            for candidate in &function.instructions {
                let Instr::RegisterClass { methods, .. } = candidate else {
                    continue;
                };
                for method in methods {
                    if instruction_may_dispatch_method(instruction, &method.name) {
                        targets.insert(method.function_name.as_str());
                    }
                }
            }
        }
        self.registered_targets_execution_stack(targets, visiting)
            .unwrap_or(runmat_types::ExecutionStackRequirement::Any)
    }

    fn registered_named_method_execution_stack(
        &self,
        requested_name: &runmat_types::MethodName,
        visiting: &mut HashSet<FunctionId>,
    ) -> Option<runmat_types::ExecutionStackRequirement> {
        let mut targets = HashSet::new();
        for function in self.functions.values() {
            for candidate in &function.instructions {
                let Instr::RegisterClass { methods, .. } = candidate else {
                    continue;
                };
                for method in methods {
                    if &method.name == requested_name {
                        targets.insert(method.function_name.as_str());
                    }
                }
            }
        }
        self.registered_targets_execution_stack(targets, visiting)
    }

    fn registered_member_or_method_execution_stack(
        &self,
        requested_name: &str,
        visiting: &mut HashSet<FunctionId>,
    ) -> Option<runmat_types::ExecutionStackRequirement> {
        if let Some(requirement) = self.registered_named_method_execution_stack(
            &runmat_types::MethodName::from(requested_name),
            visiting,
        ) {
            return Some(requirement);
        }
        let mut property_is_registered = false;
        let mut targets = HashSet::new();
        for function in self.functions.values() {
            for candidate in &function.instructions {
                let Instr::RegisterClass {
                    properties,
                    methods,
                    ..
                } = candidate
                else {
                    continue;
                };
                if !properties
                    .iter()
                    .any(|property| property.name.display_name() == requested_name)
                {
                    continue;
                }
                property_is_registered = true;
                let getter = runmat_types::MethodName::property_getter(&runmat_types::MemberName(
                    requested_name.to_owned(),
                ));
                for method in methods {
                    if runmat_runtime::OBJECT_SUBSREF_METHOD.is(&method.name)
                        || method.name == getter
                    {
                        targets.insert(method.function_name.as_str());
                    }
                }
            }
        }
        property_is_registered.then(|| {
            self.registered_targets_execution_stack(targets, visiting)
                .unwrap_or(runmat_types::ExecutionStackRequirement::Any)
        })
    }

    fn registered_targets_execution_stack(
        &self,
        targets: HashSet<&str>,
        visiting: &mut HashSet<FunctionId>,
    ) -> Option<runmat_types::ExecutionStackRequirement> {
        targets
            .into_iter()
            .map(|target| {
                self.resolve_name(target)
                    .map(|function| self.execution_stack_inner(function, visiting))
                    .unwrap_or_else(|| {
                        if runmat_builtins::builtin_name_is_known(target) {
                            builtin_execution_stack(target)
                        } else {
                            runmat_types::ExecutionStackRequirement::Process
                        }
                    })
            })
            .max()
    }
}

fn instruction_has_dynamic_dispatch(instruction: &Instr) -> bool {
    !dynamic_dispatch_method_names(instruction).is_empty()
        || matches!(
            instruction,
            Instr::Index(_)
                | Instr::IndexSlice(..)
                | Instr::IndexCell { .. }
                | Instr::IndexCellExpand { .. }
                | Instr::IndexCellList { .. }
                | Instr::StoreIndex(_)
                | Instr::StoreIndexCell { .. }
                | Instr::StoreIndexDelete(_)
                | Instr::StoreIndexCellDelete { .. }
                | Instr::StoreSlice(..)
                | Instr::StoreSliceDelete(..)
                | Instr::LoadMember(_)
                | Instr::LoadMemberOrInit(_)
                | Instr::LoadMemberDynamic
                | Instr::LoadMemberDynamicOrInit
                | Instr::LoadMemberSequence { .. }
                | Instr::LoadMemberDynamicSequence { .. }
                | Instr::StoreMember(_)
                | Instr::StoreMemberOrInit(_)
                | Instr::StoreMemberDynamic
                | Instr::StoreMemberDynamicOrInit
        )
}

fn instruction_may_dispatch_method(
    instruction: &Instr,
    method_name: &runmat_types::MethodName,
) -> bool {
    if dynamic_dispatch_method_names(instruction)
        .iter()
        .any(|candidate| candidate.is(method_name))
    {
        return true;
    }
    match instruction {
        Instr::Index(_)
        | Instr::IndexSlice(..)
        | Instr::IndexCell { .. }
        | Instr::IndexCellExpand { .. }
        | Instr::IndexCellList { .. } => runmat_runtime::OBJECT_SUBSREF_METHOD.is(method_name),
        Instr::StoreIndex(_)
        | Instr::StoreIndexCell { .. }
        | Instr::StoreIndexDelete(_)
        | Instr::StoreIndexCellDelete { .. }
        | Instr::StoreSlice(..)
        | Instr::StoreSliceDelete(..) => runmat_runtime::OBJECT_SUBSASGN_METHOD.is(method_name),
        Instr::LoadMember(name)
        | Instr::LoadMemberOrInit(name)
        | Instr::LoadMemberSequence { member: name, .. } => {
            runmat_runtime::OBJECT_SUBSREF_METHOD.is(method_name)
                || method_name == &runmat_types::MethodName::property_getter(name)
        }
        Instr::LoadMemberDynamic
        | Instr::LoadMemberDynamicOrInit
        | Instr::LoadMemberDynamicSequence { .. } => {
            runmat_runtime::OBJECT_SUBSREF_METHOD.is(method_name)
                || method_name.is_property_getter()
        }
        Instr::StoreMember(name) | Instr::StoreMemberOrInit(name) => {
            runmat_runtime::OBJECT_SUBSASGN_METHOD.is(method_name)
                || method_name == &runmat_types::MethodName::property_setter(name)
        }
        Instr::StoreMemberDynamic | Instr::StoreMemberDynamicOrInit => {
            runmat_runtime::OBJECT_SUBSASGN_METHOD.is(method_name)
                || method_name.is_property_setter()
        }
        _ => false,
    }
}

fn dynamic_dispatch_method_names(instruction: &Instr) -> &'static [runmat_types::StaticMethodName] {
    use runmat_types::StaticMethodName;

    const PLUS: &[StaticMethodName] = &[StaticMethodName::new("plus")];
    const MINUS: &[StaticMethodName] = &[StaticMethodName::new("minus")];
    const MTIMES: &[StaticMethodName] = &[StaticMethodName::new("mtimes")];
    const RIGHT_DIVIDE: &[StaticMethodName] = &[
        StaticMethodName::new("mrdivide"),
        StaticMethodName::new("rdivide"),
    ];
    const LEFT_DIVIDE: &[StaticMethodName] = &[
        StaticMethodName::new("mldivide"),
        StaticMethodName::new("ldivide"),
    ];
    const POWER: &[StaticMethodName] = &[
        StaticMethodName::new("mpower"),
        StaticMethodName::new("power"),
    ];
    const NEGATE: &[StaticMethodName] = &[
        StaticMethodName::new("uminus"),
        StaticMethodName::new("times"),
    ];
    const UPLUS: &[StaticMethodName] = &[StaticMethodName::new("uplus")];
    const TRANSPOSE: &[StaticMethodName] = &[StaticMethodName::new("transpose")];
    const CONJUGATE_TRANSPOSE: &[StaticMethodName] = &[StaticMethodName::new("ctranspose")];
    const TIMES: &[StaticMethodName] = &[StaticMethodName::new("times")];
    const RDIVIDE: &[StaticMethodName] = &[StaticMethodName::new("rdivide")];
    const ELEMENTWISE_POWER: &[StaticMethodName] = &[StaticMethodName::new("power")];
    const LDIVIDE: &[StaticMethodName] = &[StaticMethodName::new("ldivide")];
    const LESS_EQUAL: &[StaticMethodName] = &[
        StaticMethodName::new("le"),
        StaticMethodName::new("gt"),
        StaticMethodName::new("ge"),
        StaticMethodName::new("lt"),
    ];
    const LESS: &[StaticMethodName] = &[StaticMethodName::new("lt"), StaticMethodName::new("gt")];
    const GREATER: &[StaticMethodName] =
        &[StaticMethodName::new("gt"), StaticMethodName::new("lt")];
    const GREATER_EQUAL: &[StaticMethodName] = &[
        StaticMethodName::new("ge"),
        StaticMethodName::new("lt"),
        StaticMethodName::new("le"),
        StaticMethodName::new("gt"),
    ];
    const EQUAL: &[StaticMethodName] = &[StaticMethodName::new("eq")];
    const NOT_EQUAL: &[StaticMethodName] = &[StaticMethodName::new("ne")];
    const NOT: &[StaticMethodName] = &[StaticMethodName::new("not")];
    const AND: &[StaticMethodName] = &[StaticMethodName::new("and")];
    const OR: &[StaticMethodName] = &[StaticMethodName::new("or")];

    match instruction {
        Instr::Add => PLUS,
        Instr::Sub => MINUS,
        Instr::Mul => MTIMES,
        Instr::RightDiv => RIGHT_DIVIDE,
        Instr::LeftDiv => LEFT_DIVIDE,
        Instr::Pow => POWER,
        Instr::Neg => NEGATE,
        Instr::UPlus => UPLUS,
        Instr::Transpose => TRANSPOSE,
        Instr::ConjugateTranspose => CONJUGATE_TRANSPOSE,
        Instr::ElemMul => TIMES,
        Instr::ElemDiv => RDIVIDE,
        Instr::ElemPow => ELEMENTWISE_POWER,
        Instr::ElemLeftDiv => LDIVIDE,
        Instr::LessEqual => LESS_EQUAL,
        Instr::Less => LESS,
        Instr::Greater => GREATER,
        Instr::GreaterEqual => GREATER_EQUAL,
        Instr::Equal => EQUAL,
        Instr::NotEqual => NOT_EQUAL,
        Instr::LogicalNot => NOT,
        Instr::LogicalAnd => AND,
        Instr::LogicalOr => OR,
        _ => &[],
    }
}
fn builtin_execution_stack(name: &str) -> runmat_types::ExecutionStackRequirement {
    runmat_builtins::builtin_execution_stack_requirement(name)
}

impl runmat_runtime::call::descriptor::FunctionNameResolver for FunctionRegistry {
    fn resolve_function(&self, name: &str) -> Option<FunctionId> {
        self.resolve_name(name)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bytecode {
    pub instructions: Vec<Instr>,
    #[serde(skip)]
    pub active_function: Option<FunctionId>,
    #[serde(skip)]
    pub active_class_method_owner: Option<runmat_types::ClassMethodOwner>,
    #[serde(default)]
    pub instr_spans: Vec<runmat_hir::Span>,
    #[serde(default)]
    pub call_arg_spans: Vec<Option<Vec<runmat_hir::Span>>>,
    #[serde(default)]
    pub coverage_sites: Vec<Vec<u64>>,
    #[serde(default)]
    pub source_id: Option<runmat_hir::SourceId>,
    pub var_count: usize,
    #[serde(default)]
    pub bound_functions: HashMap<FunctionId, FunctionBytecode>,
    #[serde(default)]
    pub function_registry: FunctionRegistry,
    #[serde(default)]
    pub var_types: Vec<Type>,
    #[serde(default)]
    pub var_names: HashMap<usize, String>,
    #[serde(default)]
    pub initially_unassigned_slots: HashSet<usize>,
    #[serde(default)]
    pub layout: Option<VmAssemblyLayout>,
    #[serde(default)]
    pub async_metadata: AsyncMetadata,
    /// Canonical region identities mapped onto function-local bytecode PCs.
    #[serde(default)]
    pub regions: Vec<BytecodeRegion>,
    /// Compiler-bound executable records for analyzed `parfor` regions.
    #[serde(default)]
    pub parfor_regions: Vec<BytecodeParforRegion>,
    /// Compiler-bound executable records for analyzed SPMD regions.
    #[serde(default)]
    pub spmd_regions: Vec<BytecodeSpmdRegion>,
    #[serde(default)]
    pub distributed_values: Vec<runmat_types::DistributedValueContract>,
    #[serde(default)]
    pub collective_contracts: Vec<runmat_types::CollectiveContract>,
    #[cfg(feature = "native-accel")]
    #[serde(default)]
    pub accel_graph: Option<AccelGraph>,
    #[cfg(feature = "native-accel")]
    #[serde(default)]
    pub fusion_groups: Vec<FusionGroup>,
    #[cfg(feature = "native-accel")]
    #[serde(default)]
    pub fusion_metadata: FusionMetadata,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AsyncMetadata {
    pub mir_spawn_site_count: usize,
    pub mir_spawn_sites: Vec<SpawnSite>,
    pub mir_await_site_count: usize,
    pub mir_await_sites: Vec<AwaitSite>,
    #[serde(default)]
    pub runtime_model: AsyncRuntimeModel,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AsyncRuntimeModel {
    LazyFutureDescriptorLane,
}

impl Default for AsyncRuntimeModel {
    fn default() -> Self {
        Self::LazyFutureDescriptorLane
    }
}

impl AsyncRuntimeModel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LazyFutureDescriptorLane => "lazy_future_descriptor_lane",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnSite {
    pub function: runmat_hir::FunctionId,
    pub block: runmat_mir::BasicBlockId,
    pub stmt_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwaitSite {
    pub function: runmat_hir::FunctionId,
    pub block: runmat_mir::BasicBlockId,
    pub resume: runmat_mir::BasicBlockId,
}

#[cfg(feature = "native-accel")]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FusionMetadata {
    pub mir_fusion_signal_count: usize,
    pub mir_fusion_candidate_group_count: usize,
    pub mir_fusion_candidate_groups: Vec<FusionCandidateGroup>,
    pub instruction_window_count: usize,
    #[serde(default)]
    pub instruction_windows: Vec<FusionInstructionWindow>,
}

#[cfg(feature = "native-accel")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionCandidateGroup {
    pub id: usize,
    pub signal_count: usize,
    pub function: runmat_hir::FunctionId,
    pub block: runmat_mir::BasicBlockId,
    pub stmt_start: usize,
    pub stmt_end: usize,
    #[serde(default)]
    pub source_span: runmat_hir::Span,
}

#[cfg(feature = "native-accel")]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FusionInstructionKind {
    Elementwise,
    Reduction,
    Matmul,
}

#[cfg(feature = "native-accel")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FusionInstructionWindow {
    pub span: runmat_accelerate::graph::InstrSpan,
    pub kind: FusionInstructionKind,
}

#[cfg(feature = "native-accel")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeAccelGraphSource {
    NotMaterialized,
    RuntimeMaterializedFromInstructions,
}

#[cfg(feature = "native-accel")]
impl RuntimeAccelGraphSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotMaterialized => "not_materialized",
            Self::RuntimeMaterializedFromInstructions => "runtime_materialized_from_instructions",
        }
    }
}

impl Bytecode {
    pub fn empty() -> Self {
        Self {
            instructions: Vec::new(),
            active_function: None,
            active_class_method_owner: None,
            instr_spans: Vec::new(),
            call_arg_spans: Vec::new(),
            coverage_sites: Vec::new(),
            source_id: None,
            var_count: 0,
            bound_functions: HashMap::new(),
            function_registry: FunctionRegistry::default(),
            var_types: Vec::new(),
            var_names: HashMap::new(),
            initially_unassigned_slots: HashSet::new(),
            layout: None,
            async_metadata: AsyncMetadata::default(),
            regions: Vec::new(),
            parfor_regions: Vec::new(),
            spmd_regions: Vec::new(),
            distributed_values: Vec::new(),
            collective_contracts: Vec::new(),
            #[cfg(feature = "native-accel")]
            accel_graph: None,
            #[cfg(feature = "native-accel")]
            fusion_groups: Vec::new(),
            #[cfg(feature = "native-accel")]
            fusion_metadata: FusionMetadata::default(),
        }
    }

    pub fn with_instructions(instructions: Vec<Instr>, var_count: usize) -> Self {
        let instr_spans = vec![runmat_hir::Span::default(); instructions.len()];
        let call_arg_spans = vec![None; instructions.len()];
        let coverage_sites = vec![Vec::new(); instructions.len()];
        Self {
            instructions,
            instr_spans,
            call_arg_spans,
            coverage_sites,
            var_count,
            ..Self::empty()
        }
    }

    pub fn function_registry(&self) -> FunctionRegistry {
        if self.function_registry.functions.is_empty() && !self.bound_functions.is_empty() {
            return FunctionRegistry::new(self.bound_functions.clone());
        }
        self.function_registry.clone()
    }

    pub fn for_function(
        function: &FunctionBytecode,
        registry: FunctionRegistry,
        layout: VmAssemblyLayout,
    ) -> Self {
        let mut bytecode = function.execution_bytecode(&registry);
        bytecode.var_types = vec![Type::Unknown; function.var_count];
        bytecode.layout = Some(layout);
        bytecode
    }

    #[cfg(feature = "native-accel")]
    pub fn runtime_fusion_groups(&self) -> Vec<FusionGroup> {
        let metadata_present = self.fusion_metadata.mir_fusion_signal_count > 0
            || self.fusion_metadata.mir_fusion_candidate_group_count > 0
            || !self.fusion_metadata.mir_fusion_candidate_groups.is_empty()
            || self.fusion_metadata.instruction_window_count > 0
            || !self.fusion_metadata.instruction_windows.is_empty();

        if !metadata_present {
            return self.fusion_groups.clone();
        }

        if self.fusion_metadata.mir_fusion_candidate_group_count == 0
            || self.fusion_metadata.instruction_windows.is_empty()
        {
            return Vec::new();
        }
        self.fusion_metadata
            .instruction_windows
            .iter()
            .enumerate()
            .map(|(id, window)| FusionGroup {
                id,
                kind: match window.kind {
                    FusionInstructionKind::Elementwise => {
                        runmat_accelerate::fusion::FusionKind::ElementwiseChain
                    }
                    FusionInstructionKind::Reduction => {
                        runmat_accelerate::fusion::FusionKind::Reduction
                    }
                    FusionInstructionKind::Matmul => {
                        runmat_accelerate::fusion::FusionKind::MatmulEpilogue
                    }
                },
                nodes: Vec::new(),
                shape: runmat_accelerate::graph::ShapeInfo::Unknown,
                span: window.span.clone(),
                pattern: None,
                stack_layout: None,
            })
            .collect()
    }

    #[cfg(feature = "native-accel")]
    pub fn runtime_fusion_groups_for_graph(&self, graph: &AccelGraph) -> Vec<FusionGroup> {
        let mut groups = self.runtime_fusion_groups();
        if groups.is_empty() {
            return groups;
        }
        if groups.iter().any(|group| group.stack_layout.is_none()) {
            annotate_fusion_groups_with_stack_layout(&self.instructions, graph, &mut groups);
        }
        groups
    }

    #[cfg(feature = "native-accel")]
    pub fn runtime_accel_graph_for_fusion(
        &self,
        runtime_groups: &[FusionGroup],
    ) -> Option<AccelGraph> {
        self.runtime_accel_graph_for_fusion_with_source(runtime_groups)
            .0
    }

    #[cfg(feature = "native-accel")]
    pub fn runtime_accel_graph_for_fusion_with_source(
        &self,
        runtime_groups: &[FusionGroup],
    ) -> (Option<AccelGraph>, RuntimeAccelGraphSource) {
        if runtime_groups.is_empty() || self.fusion_metadata.mir_fusion_candidate_group_count == 0 {
            return (None, RuntimeAccelGraphSource::NotMaterialized);
        }
        (
            Some(build_accel_graph(&self.instructions, &self.var_types)),
            RuntimeAccelGraphSource::RuntimeMaterializedFromInstructions,
        )
    }
}

#[cfg(test)]
mod function_registry_tests {
    use super::{Bytecode, FunctionBytecode, FunctionRegistry};
    use crate::Instr;
    use runmat_hir::FunctionId;
    use runmat_types::{
        ProgramFunctionId, ProgramPointId, ProgramSourceId, ProgramSpan, RegionContract, RegionId,
        RegionProvenance, REGION_CONTRACT_SCHEMA_VERSION,
    };
    use std::collections::{HashMap, HashSet};

    fn test_function(id: usize, display_name: &str, private_owner_scope: &str) -> FunctionBytecode {
        FunctionBytecode {
            function: FunctionId(id),
            display_name: display_name.into(),
            class_method_owner: None,
            private_owner_scope: private_owner_scope.to_string(),
            source_id: None,
            capabilities: Default::default(),
            instructions: vec![Instr::Return],
            instr_spans: Vec::new(),
            call_arg_spans: Vec::new(),
            coverage_sites: Vec::new(),
            var_count: 0,
            input_slots: Vec::new(),
            varargin_slot: None,
            implicit_nargin_slot: None,
            output_slots: Vec::new(),
            varargout_slot: None,
            implicit_nargout_slot: None,
            capture_slots: Vec::new(),
            var_names: HashMap::new(),
            initially_unassigned_slots: HashSet::new(),
            argument_validations: Vec::new(),
            resume_points: std::collections::BTreeMap::new(),
            regions: Vec::new(),
            parfor_regions: Vec::new(),
            spmd_regions: Vec::new(),
            distributed_values: Vec::new(),
            collective_contracts: Vec::new(),
        }
    }

    #[test]
    fn function_registry_resolves_private_name_in_owner_scope() {
        let mut functions = HashMap::new();
        functions.insert(FunctionId(1), test_function(1, "helper", ""));
        functions.insert(FunctionId(2), test_function(2, "C.__private__.helper", "C"));
        let registry = FunctionRegistry::new(functions);

        assert_eq!(
            registry.resolve_name("helper"),
            Some(FunctionId(1)),
            "unscoped lookup should keep ordinary name resolution"
        );
        assert_eq!(
            registry.resolve_name_in_private_scope("C", "helper"),
            Some(FunctionId(2)),
            "class owner scope should prefer its synthetic private helper"
        );
        assert_eq!(
            registry.resolve_name_in_private_scope("", "helper"),
            None,
            "empty owner scope should not expose synthetic private helpers"
        );
        assert_eq!(
            registry.resolve_name_in_private_scope("C", "pkg.helper"),
            None,
            "qualified names should not be rewritten as private-folder aliases"
        );
    }

    #[test]
    fn execution_stack_requirement_propagates_through_static_calls_and_cycles() {
        let mut entry = test_function(1, "entry", "");
        entry.instructions = vec![Instr::CallSemanticFunctionMulti(FunctionId(2), 0, 1)];
        let mut helper = test_function(2, "helper", "");
        helper.instructions = vec![
            Instr::CallSemanticFunctionMulti(FunctionId(1), 0, 1),
            Instr::CallBuiltinMulti("javaObject".into(), 1, 1),
        ];
        let registry = FunctionRegistry::new(HashMap::from([
            (FunctionId(1), entry),
            (FunctionId(2), helper),
        ]));

        assert_eq!(
            registry.execution_stack(FunctionId(1)),
            runmat_types::ExecutionStackRequirement::Process
        );
        assert_eq!(
            registry.execution_stack(FunctionId(2)),
            runmat_types::ExecutionStackRequirement::Process
        );
    }

    #[test]
    fn execution_stack_requirement_keeps_pure_functions_segmentable() {
        let registry = FunctionRegistry::new(HashMap::from([(
            FunctionId(1),
            test_function(1, "pure_helper", ""),
        )]));

        assert_eq!(
            registry.execution_stack(FunctionId(1)),
            runmat_types::ExecutionStackRequirement::Any
        );
    }

    #[test]
    fn execution_stack_requirement_is_conservative_at_dynamic_boundaries() {
        let mut callback = test_function(1, "callback", "");
        callback.instructions = vec![Instr::CallFevalMulti(1, 1)];
        let registry = FunctionRegistry::new(HashMap::from([(FunctionId(1), callback)]));

        assert_eq!(
            registry.execution_stack(FunctionId(1)),
            runmat_types::ExecutionStackRequirement::Process
        );
    }

    #[test]
    fn execution_stack_requirement_follows_registered_operator_overloads() {
        let mut caller = test_function(1, "caller", "");
        caller.instructions = vec![
            Instr::RegisterClass {
                name: "Example".into(),
                super_class: None,
                is_sealed: false,
                is_abstract: false,
                properties: Vec::new(),
                methods: vec![crate::bytecode::instr::BytecodeClassMethod {
                    name: "plus".into(),
                    function_name: "Example.plus".into(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: runmat_types::MemberAccess::Public,
                }],
                enumerations: Vec::new(),
            },
            Instr::Add,
        ];
        let mut operator = test_function(2, "Example.plus", "");
        operator.instructions = vec![Instr::CallBuiltinMulti("javaObject".into(), 1, 1)];
        let registry = FunctionRegistry::new(HashMap::from([
            (FunctionId(1), caller),
            (FunctionId(2), operator),
        ]));

        assert_eq!(
            registry.execution_stack(FunctionId(1)),
            runmat_types::ExecutionStackRequirement::Process
        );
    }

    fn property_dispatch_registry(property_body: Vec<Instr>) -> (FunctionRegistry, FunctionId) {
        let mut caller = test_function(1, "caller", "");
        caller.instructions = vec![
            Instr::RegisterClass {
                name: "Example".into(),
                super_class: None,
                is_sealed: false,
                is_abstract: false,
                properties: vec![crate::bytecode::instr::BytecodeClassProperty {
                    name: "data".into(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    default_literal: None,
                    get_access: runmat_types::MemberAccess::Public,
                    set_access: runmat_types::MemberAccess::Public,
                }],
                methods: vec![crate::bytecode::instr::BytecodeClassMethod {
                    name: "get.data".into(),
                    function_name: "Example.get.data".into(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: runmat_types::MemberAccess::Public,
                }],
                enumerations: Vec::new(),
            },
            Instr::CallMethodOrMemberIndexMulti {
                identity: runmat_types::CallableIdentity::DynamicName(runmat_types::SymbolName(
                    "data".into(),
                )),
                fallback_policy: runmat_types::CallableFallbackPolicy::ObjectDispatch,
                arg_count: 1,
                out_count: 1,
            },
        ];
        let mut getter = test_function(2, "Example.get.data", "");
        getter.instructions = property_body;
        (
            FunctionRegistry::new(HashMap::from([
                (FunctionId(1), caller),
                (FunctionId(2), getter),
            ])),
            FunctionId(1),
        )
    }

    #[test]
    fn execution_stack_requirement_resolves_registered_property_dispatch() {
        let (registry, caller) = property_dispatch_registry(vec![Instr::Return]);

        assert_eq!(
            registry.execution_stack(caller),
            runmat_types::ExecutionStackRequirement::Any
        );
    }

    #[test]
    fn execution_stack_requirement_follows_registered_property_accessors() {
        let (registry, caller) =
            property_dispatch_registry(vec![Instr::CallBuiltinMulti("javaObject".into(), 1, 1)]);

        assert_eq!(
            registry.execution_stack(caller),
            runmat_types::ExecutionStackRequirement::Process
        );
    }

    #[test]
    fn execution_stack_requirement_keeps_unresolved_member_dispatch_conservative() {
        let mut caller = test_function(1, "caller", "");
        caller.instructions = vec![Instr::CallMethodOrMemberIndexMulti {
            identity: runmat_types::CallableIdentity::DynamicName(runmat_types::SymbolName(
                "unknown_member".into(),
            )),
            fallback_policy: runmat_types::CallableFallbackPolicy::ObjectDispatch,
            arg_count: 1,
            out_count: 1,
        }];
        let registry = FunctionRegistry::new(HashMap::from([(FunctionId(1), caller)]));

        assert_eq!(
            registry.execution_stack(FunctionId(1)),
            runmat_types::ExecutionStackRequirement::Process
        );
    }

    #[test]
    fn execution_stack_cache_is_invalidated_when_functions_change() {
        let mut registry = FunctionRegistry::new(HashMap::from([(
            FunctionId(1),
            test_function(1, "replaceable", ""),
        )]));
        assert_eq!(
            registry.execution_stack(FunctionId(1)),
            runmat_types::ExecutionStackRequirement::Any
        );

        let mut replacement = test_function(1, "replaceable", "");
        replacement.instructions = vec![Instr::CallBuiltinMulti("javaObject".into(), 1, 1)];
        registry.insert_replacing_name(replacement);
        assert_eq!(
            registry.execution_stack(FunctionId(1)),
            runmat_types::ExecutionStackRequirement::Process
        );
    }

    fn region_contract() -> RegionContract {
        let function = ProgramFunctionId(7);
        RegionContract {
            schema_version: REGION_CONTRACT_SCHEMA_VERSION,
            id: RegionId {
                function,
                ordinal: 2,
            },
            source: ProgramSourceId(1),
            span: ProgramSpan { start: 10, end: 20 },
            entry: ProgramPointId {
                function,
                block: 3,
                position: 1,
            },
            exits: vec![ProgramPointId {
                function,
                block: 3,
                position: 4,
            }],
            live_in: Vec::new(),
            live_out: Vec::new(),
            value_facts: Vec::new(),
            effects: Default::default(),
            capabilities: Default::default(),
            guards: Vec::new(),
            provenance: RegionProvenance::Inferred,
        }
    }

    #[test]
    fn bytecode_installs_exact_region_boundaries_for_every_function_authority() {
        let contract = region_contract();
        let mut function = test_function(7, "region_owner", "");
        function.resume_points.insert(contract.entry, 5);
        function.resume_points.insert(contract.exits[0], 11);
        let mut bytecode = Bytecode::empty();
        bytecode
            .function_registry
            .functions
            .insert(FunctionId(7), function.clone());
        bytecode
            .bound_functions
            .insert(FunctionId(7), function.clone());

        bytecode
            .install_regions(std::slice::from_ref(&contract))
            .unwrap();

        assert_eq!(bytecode.regions.len(), 1);
        assert_eq!(bytecode.regions[0].entry.pc, 5);
        assert_eq!(bytecode.regions[0].exits[0].pc, 11);
        assert_eq!(
            bytecode
                .function_registry
                .functions
                .get(&FunctionId(7))
                .unwrap()
                .regions,
            bytecode.regions
        );
        assert_eq!(
            bytecode
                .bound_functions
                .get(&FunctionId(7))
                .unwrap()
                .regions,
            bytecode.regions
        );
    }

    #[test]
    fn bytecode_omits_regions_without_an_exact_resume_boundary() {
        let contract = region_contract();
        let mut function = test_function(7, "region_owner", "");
        function.resume_points.insert(contract.entry, 5);
        function.resume_points.insert(contract.exits[0], 11);
        let mut bytecode = Bytecode::empty();
        bytecode
            .function_registry
            .functions
            .insert(FunctionId(7), function);
        bytecode
            .install_regions(std::slice::from_ref(&contract))
            .unwrap();
        bytecode
            .function_registry
            .functions
            .get_mut(&FunctionId(7))
            .unwrap()
            .resume_points
            .remove(&contract.exits[0]);

        bytecode
            .install_regions(std::slice::from_ref(&contract))
            .unwrap();

        assert!(bytecode.regions.is_empty());
        assert!(bytecode
            .function_registry
            .functions
            .get(&FunctionId(7))
            .unwrap()
            .regions
            .is_empty());
    }
}

#[cfg(all(test, feature = "native-accel"))]
mod tests {
    use super::{Bytecode, FusionInstructionKind, FusionInstructionWindow};
    use runmat_accelerate::graph::InstrSpan;
    use runmat_accelerate::graph::{AccelNodeLabel, PrimitiveOp};

    #[test]
    fn runtime_fusion_groups_fallback_to_semantic_windows_when_bytecode_groups_are_empty() {
        let mut bytecode = Bytecode::empty();
        bytecode.fusion_metadata.mir_fusion_candidate_group_count = 1;
        bytecode.fusion_metadata.instruction_windows = vec![FusionInstructionWindow {
            span: InstrSpan { start: 2, end: 4 },
            kind: FusionInstructionKind::Elementwise,
        }];

        let groups = bytecode.runtime_fusion_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].span.start, 2);
        assert_eq!(groups[0].span.end, 4);
        assert!(groups[0].nodes.is_empty());
        assert_eq!(
            groups[0].kind,
            runmat_accelerate::fusion::FusionKind::ElementwiseChain
        );
    }

    #[test]
    fn runtime_fusion_groups_use_semantic_windows_when_metadata_is_present() {
        let mut bytecode = Bytecode::empty();
        bytecode.fusion_groups = vec![runmat_accelerate::fusion::FusionGroup {
            id: 7,
            kind: runmat_accelerate::fusion::FusionKind::ElementwiseChain,
            nodes: vec![1],
            shape: runmat_accelerate::graph::ShapeInfo::Unknown,
            span: InstrSpan { start: 5, end: 5 },
            pattern: None,
            stack_layout: None,
        }];
        bytecode.fusion_metadata.mir_fusion_candidate_group_count = 1;
        bytecode.fusion_metadata.instruction_windows = vec![FusionInstructionWindow {
            span: InstrSpan { start: 10, end: 20 },
            kind: FusionInstructionKind::Elementwise,
        }];

        let groups = bytecode.runtime_fusion_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].id, 0);
        assert!(groups[0].nodes.is_empty());
        assert_eq!(groups[0].span.start, 10);
        assert_eq!(groups[0].span.end, 20);
    }

    #[test]
    fn runtime_fusion_groups_ignore_stale_compile_groups_when_semantic_candidates_are_empty() {
        let mut bytecode = Bytecode::empty();
        bytecode.fusion_groups = vec![runmat_accelerate::fusion::FusionGroup {
            id: 7,
            kind: runmat_accelerate::fusion::FusionKind::ElementwiseChain,
            nodes: vec![1],
            shape: runmat_accelerate::graph::ShapeInfo::Unknown,
            span: InstrSpan { start: 5, end: 5 },
            pattern: None,
            stack_layout: None,
        }];
        bytecode.fusion_metadata.mir_fusion_signal_count = 2;
        bytecode.fusion_metadata.mir_fusion_candidate_group_count = 0;

        let groups = bytecode.runtime_fusion_groups();
        assert!(
            groups.is_empty(),
            "semantic metadata should gate runtime fusion groups when no candidates exist"
        );
    }

    #[test]
    fn runtime_fusion_groups_fallback_to_existing_bytecode_groups_without_semantic_metadata() {
        let mut bytecode = Bytecode::empty();
        bytecode.fusion_groups = vec![runmat_accelerate::fusion::FusionGroup {
            id: 7,
            kind: runmat_accelerate::fusion::FusionKind::ElementwiseChain,
            nodes: vec![1],
            shape: runmat_accelerate::graph::ShapeInfo::Unknown,
            span: InstrSpan { start: 5, end: 5 },
            pattern: None,
            stack_layout: None,
        }];

        let groups = bytecode.runtime_fusion_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].id, 7);
        assert_eq!(groups[0].nodes, vec![1]);
        assert_eq!(groups[0].span.start, 5);
        assert_eq!(groups[0].span.end, 5);
    }

    #[test]
    fn runtime_accel_graph_materializes_when_semantic_groups_exist_and_compile_graph_is_missing() {
        let mut bytecode = Bytecode::empty();
        bytecode.instructions = vec![crate::Instr::Add];
        bytecode.var_types = vec![
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
        ];
        bytecode.fusion_metadata.mir_fusion_candidate_group_count = 1;
        bytecode.fusion_metadata.instruction_windows = vec![FusionInstructionWindow {
            span: InstrSpan { start: 0, end: 0 },
            kind: FusionInstructionKind::Elementwise,
        }];

        let runtime_groups = bytecode.runtime_fusion_groups();
        let (graph, source) = bytecode.runtime_accel_graph_for_fusion_with_source(&runtime_groups);
        assert!(
            graph.is_some(),
            "runtime graph should be materialized when semantic runtime groups exist and compile graph is missing"
        );
        assert_eq!(
            source,
            super::RuntimeAccelGraphSource::RuntimeMaterializedFromInstructions
        );
    }

    #[test]
    fn runtime_accel_graph_materializes_when_semantic_groups_exist_and_compile_graph_is_present() {
        let mut bytecode = Bytecode::empty();
        bytecode.instructions = vec![
            crate::Instr::LoadVar(0),
            crate::Instr::LoadVar(1),
            crate::Instr::Add,
        ];
        bytecode.var_types = vec![
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
        ];
        bytecode.accel_graph = Some(crate::accel::graph::build_accel_graph(
            &bytecode.instructions,
            &bytecode.var_types,
        ));
        bytecode.fusion_metadata.mir_fusion_candidate_group_count = 1;
        bytecode.fusion_metadata.instruction_windows = vec![FusionInstructionWindow {
            span: InstrSpan { start: 2, end: 2 },
            kind: FusionInstructionKind::Elementwise,
        }];

        let runtime_groups = bytecode.runtime_fusion_groups();
        let (graph, source) = bytecode.runtime_accel_graph_for_fusion_with_source(&runtime_groups);
        assert!(
            graph.is_some(),
            "runtime graph should still be materialized when compile graph metadata is present"
        );
        assert_eq!(
            source,
            super::RuntimeAccelGraphSource::RuntimeMaterializedFromInstructions
        );
    }

    #[test]
    fn runtime_accel_graph_ignores_stale_compile_graph_metadata() {
        let mut bytecode = Bytecode::empty();
        bytecode.instructions = vec![
            crate::Instr::LoadVar(0),
            crate::Instr::LoadVar(1),
            crate::Instr::Add,
        ];
        bytecode.var_types = vec![
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
        ];

        let stale_graph = crate::accel::graph::build_accel_graph(
            &[
                crate::Instr::LoadVar(0),
                crate::Instr::LoadVar(1),
                crate::Instr::Mul,
            ],
            &bytecode.var_types,
        );
        bytecode.accel_graph = Some(stale_graph);
        bytecode.fusion_metadata.mir_fusion_candidate_group_count = 1;
        bytecode.fusion_metadata.instruction_windows = vec![FusionInstructionWindow {
            span: InstrSpan { start: 2, end: 2 },
            kind: FusionInstructionKind::Elementwise,
        }];

        let runtime_groups = bytecode.runtime_fusion_groups();
        let (graph, source) = bytecode.runtime_accel_graph_for_fusion_with_source(&runtime_groups);
        let graph =
            graph.expect("runtime graph should be materialized from active bytecode instructions");
        assert!(
            graph
                .nodes
                .iter()
                .any(|node| matches!(node.label, AccelNodeLabel::Primitive(PrimitiveOp::Add))),
            "runtime graph should reflect active bytecode instructions"
        );
        assert!(
            !graph
                .nodes
                .iter()
                .any(|node| matches!(node.label, AccelNodeLabel::Primitive(PrimitiveOp::Mul))),
            "stale compile graph metadata should not be reused at runtime"
        );
        assert_eq!(
            source,
            super::RuntimeAccelGraphSource::RuntimeMaterializedFromInstructions
        );
    }

    #[test]
    fn runtime_accel_graph_is_not_materialized_when_runtime_groups_are_empty() {
        let bytecode = Bytecode::empty();
        let (graph, source) = bytecode.runtime_accel_graph_for_fusion_with_source(&[]);
        assert!(
            graph.is_none(),
            "runtime graph materialization should remain gated when semantic runtime groups are absent"
        );
        assert_eq!(source, super::RuntimeAccelGraphSource::NotMaterialized);
    }
}
