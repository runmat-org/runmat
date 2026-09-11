use runmat_hir::{CallableFallbackPolicy, CallableIdentity, FunctionId};
use runmat_runtime::call::arguments::ArgumentSpec;
use runmat_types::{ClassIdentity, MethodName};
use runmat_value::IntValue;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackEffect {
    pub pops: usize,
    pub pushes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregateElementSpec {
    Single,
    CapturedSequence { slot: usize },
}

impl AggregateElementSpec {
    pub const fn stack_operand_count(self) -> usize {
        match self {
            Self::Single => 1,
            Self::CapturedSequence { .. } => 0,
        }
    }
}

/// Source-level SPMD header form. Runtime operands are evaluated once and
/// remain on the stack in source order for typed admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BytecodeSpmdHeader {
    Default,
    Exact,
    Range,
    PoolRange,
}

impl BytecodeSpmdHeader {
    pub const fn operand_count(self) -> usize {
        match self {
            Self::Default => 0,
            Self::Exact => 1,
            Self::Range => 2,
            Self::PoolRange => 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BytecodeDistributedOp {
    Create {
        id: runmat_types::DistributedValueId,
        owner: runmat_types::DistributedOwner,
        scheme: runmat_types::DistributionScheme,
    },
    Codistributed {
        id: runmat_types::DistributedValueId,
        owner: runmat_types::DistributedOwner,
        overload: BytecodeCodistributedOverload,
        coordination: Option<runmat_types::CollectiveId>,
    },
    Build {
        id: runmat_types::DistributedValueId,
        owner: runmat_types::DistributedOwner,
        has_codistributor: bool,
        validation: BytecodeDistributedBuildValidation,
        coordination: runmat_types::CollectiveId,
    },
    LocalPart,
    Materialize,
    Codistributor,
    GlobalIndices {
        has_lab: bool,
        requested_outputs: u8,
    },
    Redistribute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BytecodeCodistributedOverload {
    ReplicatedInputDefault,
    CodistributorOrDesignatedWorker,
    DesignatedWorkerWithCodistributor,
}

impl BytecodeCodistributedOverload {
    pub const fn operand_count(self) -> usize {
        match self {
            Self::ReplicatedInputDefault => 1,
            Self::CodistributorOrDesignatedWorker => 2,
            Self::DesignatedWorkerWithCodistributor => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BytecodeDistributedBuildValidation {
    ValidateAcrossWorkers,
    NoCommunication,
    RuntimeOption,
}

impl BytecodeDistributedOp {
    pub const fn operand_count(&self) -> usize {
        match self {
            Self::Create { .. } | Self::LocalPart | Self::Materialize | Self::Codistributor => 1,
            Self::GlobalIndices { has_lab, .. } => 2 + *has_lab as usize,
            Self::Codistributed { overload, .. } => overload.operand_count(),
            Self::Build {
                has_codistributor,
                validation,
                ..
            } => {
                1 + *has_codistributor as usize
                    + matches!(
                        validation,
                        BytecodeDistributedBuildValidation::RuntimeOption
                    ) as usize
            }
            Self::Redistribute => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BytecodeCollectiveOp {
    Barrier,
    Broadcast {
        has_input: bool,
    },
    Gather,
    Scatter,
    AllGather,
    Reduce {
        operator: runmat_types::OperatorKind,
    },
    AllReduce {
        operator: runmat_types::OperatorKind,
    },
    Cat {
        has_root: bool,
    },
    FunctionalReduce {
        has_root: bool,
    },
    Send {
        has_tag: bool,
    },
    Receive {
        has_source: bool,
        has_tag: bool,
        requested_outputs: u8,
    },
    SendReceive {
        has_tag: bool,
    },
    Probe {
        has_source: bool,
        has_tag: bool,
    },
}

impl BytecodeCollectiveOp {
    pub const fn operand_count(self) -> usize {
        match self {
            Self::Barrier => 0,
            Self::Broadcast { has_input } => 1 + has_input as usize,
            Self::Gather | Self::Scatter | Self::Reduce { .. } => 2,
            Self::AllGather | Self::AllReduce { .. } => 1,
            Self::Cat { has_root } | Self::FunctionalReduce { has_root } => 2 + has_root as usize,
            Self::Send { has_tag } => 2 + has_tag as usize,
            Self::Receive {
                has_source,
                has_tag,
                ..
            }
            | Self::Probe {
                has_source,
                has_tag,
            } => has_source as usize + has_tag as usize,
            Self::SendReceive { has_tag } => 3 + has_tag as usize,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmitLabel {
    Ans,
    Var(usize),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertyDefaultLiteral {
    Num(f64),
    Int(IntValue),
    Bool(bool),
    String(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BytecodeClassProperty {
    pub name: runmat_types::MemberName,
    pub is_static: bool,
    pub is_constant: bool,
    pub is_dependent: bool,
    pub default_literal: Option<PropertyDefaultLiteral>,
    pub get_access: runmat_types::MemberAccess,
    pub set_access: runmat_types::MemberAccess,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BytecodeClassMethod {
    pub name: MethodName,
    pub function_name: String,
    pub is_static: bool,
    pub is_abstract: bool,
    pub is_sealed: bool,
    pub access: runmat_types::MemberAccess,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Instr {
    // Constant and variable loads.
    LoadConst(f64),
    LoadInt(IntValue),
    LoadComplex(f64, f64),
    LoadBool(bool),
    LoadString(String),
    LoadCharRow(String),
    LoadVar(usize),
    LoadVarForIndexAssignment(usize),
    StoreVar(usize),

    // Scalar and matrix arithmetic.
    Add,
    Sub,
    Mul,
    RightDiv,
    LeftDiv,
    Pow,
    Neg,
    UPlus,
    Transpose,
    ConjugateTranspose,
    ElemMul,
    ElemDiv,
    ElemPow,
    ElemLeftDiv,
    LessEqual,
    Less,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
    LogicalNot,
    LogicalAnd,
    LogicalOr,

    // Short-circuit logical control flow.
    AndAnd(usize),
    OrOr(usize),
    JumpIfFalse(usize),
    Jump(usize),
    Pop,

    // Expands a single value into N outputs, padding with zero values when needed.
    Unpack(usize),

    // Specialized lowering target for the stochastic evolution fast path.
    StochasticEvolution,

    // Array construction and direct indexing.
    CreateMatrix(usize, usize),
    CreateMatrixFromSequences {
        rows: usize,
        row_lengths: Vec<usize>,
        elements: Vec<AggregateElementSpec>,
    },
    CreateMatrixDynamic(usize),
    CreateRange(bool),
    Index(usize),

    // Slice indexing with compiler-encoded colon and plain `end` masks.
    IndexSlice(usize, usize, u32, u32),

    // Cell array construction and indexing.
    CreateCell2D(usize, usize),
    CreateCellFromSequences {
        rows: usize,
        row_lengths: Vec<usize>,
        elements: Vec<AggregateElementSpec>,
    },
    CreateStructLiteral(Vec<String>),
    CreateObjectLiteral {
        class_name: ClassIdentity,
        fields: Vec<String>,
    },
    IndexCell {
        num_indices: usize,
    },

    // Expands cell contents into a comma-separated list with fixed output arity.
    IndexCellExpand {
        num_indices: usize,
        out_count: usize,
    },

    // Expands cell contents into a first-class comma-separated list value.
    IndexCellList {
        num_indices: usize,
    },

    // Indexed assignment updates the base value and pushes the updated base.
    StoreIndex(usize),
    StoreIndexCell {
        num_indices: usize,
    },
    StoreIndexDelete(usize),
    StoreIndexCellDelete {
        num_indices: usize,
    },

    // Slice assignment with compiler-encoded colon and plain `end` masks.
    StoreSlice(usize, usize, u32, u32),
    StoreSliceDelete(usize, usize, u32, u32),

    // Struct, object, and class member access.
    LoadMember(runmat_types::MemberName),
    LoadMemberOrInit(runmat_types::MemberName),
    LoadMemberDynamic,
    LoadMemberDynamicOrInit,
    LoadMemberSequence {
        member: runmat_types::MemberName,
        selection: runmat_types::SequenceUse,
    },
    LoadMemberDynamicSequence {
        selection: runmat_types::SequenceUse,
    },
    MemberSequenceCardinality,
    LoadMemberSequenceUsingOutputSlot {
        member: runmat_types::MemberName,
        output_count_slot: usize,
    },
    LoadMemberDynamicSequenceUsingOutputSlot {
        output_count_slot: usize,
    },
    CaptureCallOutputSequence,
    CaptureScalarSequence,
    CaptureMemberSequence {
        member: runmat_types::MemberName,
        sequence_slot: usize,
    },
    CaptureMemberDynamicSequence {
        sequence_slot: usize,
    },
    CaptureCellContentsSequence {
        sequence_slot: usize,
        num_indices: usize,
        expand_all: bool,
    },
    CaptureReturnedOutputsSequence {
        sequence_slot: usize,
    },
    ReadSubscriptPath {
        steps: Vec<super::BytecodeSubscriptStep>,
        selection: runmat_types::SequenceUse,
        context: runmat_types::ObjectIndexingContext,
        to_sequence_register: bool,
    },
    CaptureSubscriptPath {
        steps: Vec<super::BytecodeSubscriptStep>,
        sequence_slot: usize,
        context: runmat_types::ObjectIndexingContext,
    },
    BeginSubscriptEndReceiver {
        prefix: Vec<super::BytecodeSubscriptStep>,
    },
    LoadSubscriptEnd {
        component: usize,
        component_count: usize,
    },
    FinishSubscriptEndReceiver {
        selector_count: usize,
        prefix_operand_count: usize,
    },
    BeginOutputAssignment {
        target_count: usize,
    },
    PrepareFixedOutputTarget,
    PrepareDiscardOutputTarget,
    PrepareMemberSequenceOutputTarget {
        root_slot: usize,
        member: runmat_types::MemberName,
    },
    PrepareMemberDynamicSequenceOutputTarget {
        root_slot: usize,
    },
    PrepareIndexedMemberSequenceOutputTarget {
        root_slot: usize,
        member: runmat_types::MemberName,
        num_indices: usize,
    },
    PrepareIndexedMemberDynamicSequenceOutputTarget {
        root_slot: usize,
        num_indices: usize,
    },
    PrepareCellContentsSequenceOutputTarget {
        root_slot: usize,
        num_indices: usize,
        expand_all: bool,
    },
    BeginSequenceOutputTarget {
        root_slot: usize,
    },
    PrepareMemberPathStep(runmat_types::MemberName),
    PrepareDynamicMemberPathStep,
    BeginPreparedIndexSelectors {
        component_count: usize,
    },
    LoadPreparedIndexEnd {
        component: usize,
    },
    BeginContextualIndexSelectors {
        component_count: usize,
    },
    LoadContextualIndexEnd {
        component: usize,
    },
    FinishContextualIndexSelectors {
        component_count: usize,
    },
    PrepareParenthesesPathStep {
        component_count: usize,
        selectors: Vec<super::BytecodeSubscriptSelector>,
    },
    PrepareBracesPathStep {
        component_count: usize,
        selectors: Vec<super::BytecodeSubscriptSelector>,
        expand_all: bool,
    },
    FinishMemberSequenceOutputTarget(runmat_types::MemberName),
    FinishDynamicMemberSequenceOutputTarget,
    FinishCellContentsSequenceOutputTarget {
        component_count: usize,
        selectors: Vec<super::BytecodeSubscriptSelector>,
        expand_all: bool,
    },
    LoadPreparedOutputCardinality,
    CommitPreparedOutputTargets {
        retained_outputs: usize,
    },
    StoreMemberSequence(runmat_types::MemberName),
    StoreMemberDynamicSequence,
    StoreMember(runmat_types::MemberName),
    StoreMemberOrInit(runmat_types::MemberName),
    StoreMemberDynamic,
    StoreMemberDynamicOrInit,
    LoadMethod(runmat_types::MethodName),

    // Ambiguous `obj.name(...)` shape resolved at runtime as method call or member indexing.
    CallMethodOrMemberIndexMulti {
        identity: CallableIdentity,
        fallback_policy: CallableFallbackPolicy,
        arg_count: usize,
        out_count: usize,
    },
    CallMethodOrMemberIndexExpandMultiOutput {
        identity: CallableIdentity,
        fallback_policy: CallableFallbackPolicy,
        specs: Vec<ArgumentSpec>,
        out_count: usize,
    },

    // Closure and static class dispatch.
    CreateFunctionHandle(String),
    CreateExternalFunctionHandle(String),
    CreateMethodFunctionHandle(String),
    CreateBoundFunctionHandle(FunctionId, String),
    CreateExternalBoundFunctionHandle(FunctionId, String),
    CreateClosure(String, usize),
    CreateSemanticClosure(FunctionId, String, usize),
    LoadStaticProperty(ClassIdentity, runmat_types::MemberName),
    LoadWorkspaceFirstStaticProperty {
        name: String,
        class_name: ClassIdentity,
        property: runmat_types::MemberName,
    },

    // Registers a runtime class definition produced by `classdef` lowering.
    RegisterClass {
        name: ClassIdentity,
        super_class: Option<ClassIdentity>,
        is_sealed: bool,
        is_abstract: bool,
        properties: Vec<BytecodeClassProperty>,
        methods: Vec<BytecodeClassMethod>,
        enumerations: Vec<String>,
    },

    // `feval` keeps the callable value on the stack instead of naming the target statically.
    CallFevalMulti(usize, usize),
    CallFevalMultiUsingOutputSlot(usize, usize),
    CallFevalExpandMultiOutput(Vec<ArgumentSpec>, usize),
    CallFevalExpandMultiOutputUsingOutputSlot(Vec<ArgumentSpec>, usize),
    // Create a lazy semantic-future descriptor from call arguments.
    CreateSemanticFuture(FunctionId, usize, usize),
    CreateSemanticFutureExpandMultiOutput(FunctionId, Vec<ArgumentSpec>, usize),
    // Resolve the pool/no-pool overload and schedule a MATLAB-facing invocation.
    ScheduleFeval {
        arg_count: usize,
        on_all: bool,
    },
    // Explicit async spawn boundary.
    Spawn,
    // Explicit async spawn on a validated pool handle.
    SpawnOn,
    // Explicit await boundary.
    Await,
    // Retrieve and combine outputs from a scalar or array of parallel futures.
    FetchOutputs {
        arg_count: usize,
        requested_outputs: usize,
    },
    // Atomically retrieve the next completed unread task in completion order.
    FetchNext {
        has_timeout: bool,
        requested_outputs: usize,
    },
    // Resolve or create the active execution pool from MATLAB-facing arguments.
    EnsurePool(usize),
    // Return the active pool, optionally creating it, from MATLAB-facing arguments.
    CurrentPool(usize),
    // Execute one compiler-bound parallel loop region. The iterable and optional
    // worker limit are evaluated exactly once before this instruction.
    ExecuteParfor {
        region: runmat_types::ParallelRegionId,
        has_maximum_workers: bool,
    },
    /// Execute one compiler-bound SPMD gang. Header operands are interpreted
    /// according to the exact source form after they have been evaluated.
    ExecuteSpmd {
        region: runmat_types::ParallelRegionId,
        header: BytecodeSpmdHeader,
    },
    Distributed(BytecodeDistributedOp),
    Collective {
        id: runmat_types::CollectiveId,
        operation: BytecodeCollectiveOp,
    },

    // Stack and exception-control operations.
    Swap,
    EnterTry {
        scope: usize,
        catch_pc: usize,
        catch_var: Option<usize>,
    },
    LeaveTry(usize),
    Return,
    ReturnValue,

    // User-function invocation variants.
    CallBuiltinMulti(String, usize, usize),
    CallBuiltinMultiUsingOutputSlot(String, usize, usize),
    CallSuperConstructorMulti {
        current_class: ClassIdentity,
        super_class: ClassIdentity,
        arg_count: usize,
        out_count: usize,
    },
    CallSuperMethodMulti {
        current_class: ClassIdentity,
        super_class: ClassIdentity,
        method: MethodName,
        arg_count: usize,
        out_count: usize,
    },

    // Calls a user function and shapes the result list to `out_count`.
    CallFunctionMulti {
        identity: CallableIdentity,
        fallback_policy: CallableFallbackPolicy,
        arg_count: usize,
        out_count: usize,
    },
    CallFunctionMultiUsingOutputSlot {
        identity: CallableIdentity,
        fallback_policy: CallableFallbackPolicy,
        arg_count: usize,
        out_count_slot: usize,
    },
    CallWorkspaceFirstMulti {
        name: String,
        identity: CallableIdentity,
        fallback_policy: CallableFallbackPolicy,
        bare_identifier: bool,
        arg_count: usize,
        out_count: usize,
    },
    CallWorkspaceFirstMultiUsingOutputSlot {
        name: String,
        identity: CallableIdentity,
        fallback_policy: CallableFallbackPolicy,
        bare_identifier: bool,
        arg_count: usize,
        out_count_slot: usize,
    },
    CallSemanticFunctionMulti(FunctionId, usize, usize),
    CallSemanticFunctionMultiUsingOutputSlot(FunctionId, usize, usize),
    CallSemanticNestedFunctionMulti {
        function: FunctionId,
        capture_slots: Vec<usize>,
        arg_count: usize,
        out_count: usize,
    },
    CallSemanticNestedFunctionMultiUsingOutputSlot {
        function: FunctionId,
        capture_slots: Vec<usize>,
        arg_count: usize,
        out_count_slot: usize,
    },

    CallFunctionExpandMultiOutput {
        identity: CallableIdentity,
        fallback_policy: CallableFallbackPolicy,
        specs: Vec<ArgumentSpec>,
        out_count: usize,
    },
    CallWorkspaceFirstExpandMultiOutput {
        name: String,
        identity: CallableIdentity,
        fallback_policy: CallableFallbackPolicy,
        bare_identifier: bool,
        specs: Vec<ArgumentSpec>,
        out_count: usize,
    },
    CallWorkspaceFirstExpandMultiOutputUsingOutputSlot {
        name: String,
        identity: CallableIdentity,
        fallback_policy: CallableFallbackPolicy,
        bare_identifier: bool,
        specs: Vec<ArgumentSpec>,
        out_count_slot: usize,
    },
    CallSemanticFunctionExpandMultiOutput(FunctionId, Vec<ArgumentSpec>, usize),
    CallSemanticNestedFunctionExpandMultiOutput {
        function: FunctionId,
        capture_slots: Vec<usize>,
        specs: Vec<ArgumentSpec>,
        out_count: usize,
    },
    CallBuiltinExpandMultiOutput(String, Vec<ArgumentSpec>, usize),
    CallSuperConstructorExpandMultiOutput {
        current_class: ClassIdentity,
        super_class: ClassIdentity,
        specs: Vec<ArgumentSpec>,
        out_count: usize,
    },
    CallSuperMethodExpandMultiOutput {
        current_class: ClassIdentity,
        super_class: ClassIdentity,
        method: MethodName,
        specs: Vec<ArgumentSpec>,
        out_count: usize,
    },

    // Packs the top N values into row or column tensor form.
    PackToRow(usize),
    PackToCol(usize),

    // Local scope and local variable access.
    EnterScope(usize),
    ExitScope(usize),
    LoadLocal(usize),
    StoreLocal(usize),

    // Import registration for later unqualified resolution.
    RegisterImport {
        path: Vec<String>,
        wildcard: bool,
    },

    // Global and persistent declarations, including name-stable forms across units.
    DeclareGlobal(Vec<usize>),
    DeclarePersistent(Vec<usize>),
    DeclareGlobalNamed(Vec<usize>, Vec<String>),
    DeclarePersistentNamed(Vec<usize>, Vec<String>),

    // Emission instructions used to produce visible workspace outputs.
    EmitStackTop {
        label: EmitLabel,
    },
    EmitVar {
        var_index: usize,
        label: EmitLabel,
    },
}

impl Instr {
    pub fn stack_effect(&self) -> Option<StackEffect> {
        fn effect(pops: usize, pushes: usize) -> Option<StackEffect> {
            Some(StackEffect { pops, pushes })
        }

        match self {
            Instr::LoadConst(_)
            | Instr::LoadInt(_)
            | Instr::LoadComplex(_, _)
            | Instr::LoadBool(_)
            | Instr::LoadString(_)
            | Instr::LoadCharRow(_)
            | Instr::CreateFunctionHandle(_)
            | Instr::CreateExternalFunctionHandle(_)
            | Instr::CreateMethodFunctionHandle(_)
            | Instr::CreateBoundFunctionHandle(_, _)
            | Instr::CreateExternalBoundFunctionHandle(_, _)
            | Instr::LoadVar(_)
            | Instr::LoadVarForIndexAssignment(_)
            | Instr::LoadLocal(_) => effect(0, 1),
            Instr::StoreVar(_)
            | Instr::StoreLocal(_)
            | Instr::Pop
            | Instr::JumpIfFalse(_)
            | Instr::AndAnd(_)
            | Instr::OrOr(_) => effect(1, 0),
            Instr::Add
            | Instr::Sub
            | Instr::Mul
            | Instr::RightDiv
            | Instr::LeftDiv
            | Instr::Pow
            | Instr::ElemMul
            | Instr::ElemDiv
            | Instr::ElemPow
            | Instr::ElemLeftDiv
            | Instr::LessEqual
            | Instr::Less
            | Instr::Greater
            | Instr::GreaterEqual
            | Instr::Equal
            | Instr::NotEqual
            | Instr::LogicalAnd
            | Instr::LogicalOr => effect(2, 1),
            Instr::Swap => effect(2, 2),
            Instr::Neg
            | Instr::UPlus
            | Instr::LogicalNot
            | Instr::Transpose
            | Instr::ConjugateTranspose
            | Instr::LoadMember(_)
            | Instr::LoadMemberOrInit(_)
            | Instr::LoadMethod(_) => effect(1, 1),
            Instr::LoadMemberSequence { selection, .. } => {
                selection.output_count().and_then(|count| effect(1, count))
            }
            Instr::MemberSequenceCardinality => effect(1, 1),
            Instr::LoadMemberSequenceUsingOutputSlot { .. } => effect(1, 0),
            Instr::LoadMemberDynamicSequenceUsingOutputSlot { .. } => effect(2, 0),
            Instr::CaptureCallOutputSequence => effect(0, 0),
            Instr::CaptureScalarSequence => effect(1, 0),
            Instr::ReadSubscriptPath {
                steps,
                selection,
                to_sequence_register,
                ..
            } => {
                if *to_sequence_register {
                    effect(1 + subscript_operand_count(steps), 0)
                } else {
                    selection
                        .output_count()
                        .and_then(|count| effect(1 + subscript_operand_count(steps), count))
                }
            }
            Instr::CaptureSubscriptPath { steps, .. } => {
                effect(1 + subscript_operand_count(steps), 0)
            }
            Instr::BeginSubscriptEndReceiver { .. } => effect(0, 1),
            Instr::LoadSubscriptEnd { .. } => effect(0, 1),
            Instr::FinishSubscriptEndReceiver {
                selector_count,
                prefix_operand_count,
            } => {
                let consumed = selector_count
                    .checked_add(*prefix_operand_count)?
                    .checked_add(1)?;
                effect(consumed, *selector_count)
            }
            Instr::BeginOutputAssignment { .. }
            | Instr::PrepareFixedOutputTarget
            | Instr::PrepareDiscardOutputTarget
            | Instr::PrepareMemberSequenceOutputTarget { .. } => effect(0, 0),
            Instr::LoadPreparedOutputCardinality => effect(0, 1),
            Instr::PrepareMemberDynamicSequenceOutputTarget { .. } => effect(1, 0),
            Instr::PrepareIndexedMemberSequenceOutputTarget { num_indices, .. } => {
                effect(*num_indices, 0)
            }
            Instr::PrepareIndexedMemberDynamicSequenceOutputTarget { num_indices, .. } => {
                effect(num_indices.saturating_add(1), 0)
            }
            Instr::PrepareCellContentsSequenceOutputTarget { num_indices, .. } => {
                effect(*num_indices, 0)
            }
            Instr::BeginSequenceOutputTarget { .. }
            | Instr::PrepareMemberPathStep(_)
            | Instr::BeginPreparedIndexSelectors { .. }
            | Instr::FinishMemberSequenceOutputTarget(_) => effect(0, 0),
            Instr::PrepareDynamicMemberPathStep
            | Instr::FinishDynamicMemberSequenceOutputTarget => effect(1, 0),
            Instr::LoadPreparedIndexEnd { .. } | Instr::LoadContextualIndexEnd { .. } => {
                effect(0, 1)
            }
            Instr::BeginContextualIndexSelectors { .. }
            | Instr::FinishContextualIndexSelectors { .. } => effect(0, 0),
            Instr::PrepareParenthesesPathStep { selectors, .. }
            | Instr::PrepareBracesPathStep { selectors, .. }
            | Instr::FinishCellContentsSequenceOutputTarget { selectors, .. } => {
                effect(subscript_selector_operand_count(selectors), 0)
            }
            Instr::CommitPreparedOutputTargets { retained_outputs } => effect(0, *retained_outputs),
            Instr::CaptureMemberSequence { .. } | Instr::CaptureReturnedOutputsSequence { .. } => {
                effect(1, 0)
            }
            Instr::CaptureMemberDynamicSequence { .. } => effect(2, 0),
            Instr::CaptureCellContentsSequence { num_indices, .. } => effect(1 + *num_indices, 0),
            Instr::CallBuiltinMulti(_, argc, out_count) => effect(*argc, *out_count),
            Instr::CallBuiltinMultiUsingOutputSlot(_, argc, _) => effect(*argc, 0),
            Instr::CallSuperConstructorMulti {
                arg_count,
                out_count,
                ..
            }
            | Instr::CallSuperMethodMulti {
                arg_count,
                out_count,
                ..
            } => effect(*arg_count, *out_count),
            Instr::CallFunctionMulti {
                arg_count,
                out_count,
                ..
            } => effect(*arg_count, *out_count),
            Instr::CallFunctionMultiUsingOutputSlot { arg_count, .. } => effect(*arg_count, 0),
            Instr::CallWorkspaceFirstMulti {
                arg_count,
                out_count,
                ..
            } => effect(*arg_count, *out_count),
            Instr::CallWorkspaceFirstMultiUsingOutputSlot { arg_count, .. } => {
                effect(*arg_count, 0)
            }
            Instr::CallSemanticFunctionMulti(_, argc, out_count) => effect(*argc, *out_count),
            Instr::CallSemanticFunctionMultiUsingOutputSlot(_, argc, _) => effect(*argc, 0),
            Instr::CallSemanticNestedFunctionMulti {
                arg_count,
                out_count,
                ..
            } => effect(*arg_count, *out_count),
            Instr::CallSemanticNestedFunctionMultiUsingOutputSlot { arg_count, .. } => {
                effect(*arg_count, 0)
            }
            Instr::CallMethodOrMemberIndexMulti {
                arg_count,
                out_count,
                ..
            } => effect(arg_count + 1, *out_count),
            Instr::CallFevalMulti(argc, out_count) => effect(argc + 1, *out_count),
            Instr::CallFevalMultiUsingOutputSlot(argc, _) => effect(argc + 1, 0),
            Instr::CreateSemanticFuture(_, arg_count, _) => effect(*arg_count, 1),
            Instr::ScheduleFeval { arg_count, .. } => effect(*arg_count, 1),
            Instr::FetchNext {
                has_timeout,
                requested_outputs,
            } => effect(if *has_timeout { 2 } else { 1 }, *requested_outputs),
            Instr::FetchOutputs {
                arg_count,
                requested_outputs,
            } => effect(*arg_count, *requested_outputs),
            Instr::CreateMatrix(rows, cols) | Instr::CreateCell2D(rows, cols) => {
                effect(rows * cols, 1)
            }
            Instr::CreateMatrixFromSequences { elements, .. }
            | Instr::CreateCellFromSequences { elements, .. } => effect(
                elements
                    .iter()
                    .copied()
                    .map(AggregateElementSpec::stack_operand_count)
                    .sum(),
                1,
            ),
            Instr::CreateStructLiteral(fields) => effect(fields.len(), 1),
            Instr::CreateObjectLiteral { fields, .. } => effect(fields.len(), 1),
            Instr::CreateMatrixDynamic(rows) => effect(*rows, 1),
            Instr::CreateRange(has_step) => effect(if *has_step { 3 } else { 2 }, 1),
            Instr::Unpack(_) => effect(0, 0),
            Instr::Index(n) => effect(n + 1, 1),
            Instr::IndexCell { num_indices, .. } => effect(num_indices + 1, 1),
            Instr::IndexCellList { num_indices, .. } => effect(num_indices + 1, 0),
            Instr::IndexCellExpand {
                num_indices,
                out_count,
                ..
            } => effect(num_indices + 1, *out_count),
            Instr::StoreIndex(n)
            | Instr::StoreIndexDelete(n)
            | Instr::StoreIndexCell { num_indices: n, .. }
            | Instr::StoreIndexCellDelete { num_indices: n, .. } => effect(n + 2, 1),
            Instr::IndexSlice(dims, numeric_count, _, _)
            | Instr::StoreSlice(dims, numeric_count, _, _)
            | Instr::StoreSliceDelete(dims, numeric_count, _, _) => {
                let pops = 1 + numeric_count;
                if matches!(
                    self,
                    Instr::StoreSlice(_, _, _, _) | Instr::StoreSliceDelete(_, _, _, _)
                ) {
                    effect(pops + 1, 1)
                } else {
                    let _ = dims;
                    effect(pops, 1)
                }
            }
            Instr::StoreMember(_)
            | Instr::StoreMemberOrInit(_)
            | Instr::StoreMemberDynamic
            | Instr::StoreMemberDynamicOrInit => effect(2, 1),
            Instr::StoreMemberSequence(_) => effect(1, 1),
            Instr::StoreMemberDynamicSequence => effect(2, 1),
            Instr::LoadMemberDynamic | Instr::LoadMemberDynamicOrInit => effect(2, 1),
            Instr::LoadMemberDynamicSequence { selection } => {
                selection.output_count().and_then(|count| effect(2, count))
            }
            Instr::CreateClosure(_, capture_count)
            | Instr::CreateSemanticClosure(_, _, capture_count) => effect(*capture_count, 1),
            Instr::LoadStaticProperty(_, _) | Instr::LoadWorkspaceFirstStaticProperty { .. } => {
                effect(0, 1)
            }
            Instr::RegisterClass { .. } => effect(0, 0),
            Instr::CallFevalExpandMultiOutput(specs, out_count) => {
                let operands: usize = specs.iter().map(ArgumentSpec::stack_operand_count).sum();
                effect(1 + operands, *out_count)
            }
            Instr::CallFevalExpandMultiOutputUsingOutputSlot(specs, _output_slot) => {
                let operands: usize = specs.iter().map(ArgumentSpec::stack_operand_count).sum();
                effect(1 + operands, 0)
            }
            Instr::CreateSemanticFutureExpandMultiOutput(_, specs, _) => {
                let operands: usize = specs.iter().map(ArgumentSpec::stack_operand_count).sum();
                effect(operands, 1)
            }
            Instr::CallFunctionExpandMultiOutput {
                specs, out_count, ..
            }
            | Instr::CallWorkspaceFirstExpandMultiOutput {
                specs, out_count, ..
            }
            | Instr::CallMethodOrMemberIndexExpandMultiOutput {
                specs, out_count, ..
            }
            | Instr::CallSuperConstructorExpandMultiOutput {
                specs, out_count, ..
            }
            | Instr::CallSuperMethodExpandMultiOutput {
                specs, out_count, ..
            } => {
                let operands: usize = specs.iter().map(ArgumentSpec::stack_operand_count).sum();
                effect(operands, *out_count)
            }
            Instr::CallWorkspaceFirstExpandMultiOutputUsingOutputSlot { specs, .. } => {
                let operands: usize = specs.iter().map(ArgumentSpec::stack_operand_count).sum();
                effect(operands, 0)
            }
            Instr::CallSemanticFunctionExpandMultiOutput(_, specs, out_count)
            | Instr::CallBuiltinExpandMultiOutput(_, specs, out_count) => {
                let operands: usize = specs.iter().map(ArgumentSpec::stack_operand_count).sum();
                effect(operands, *out_count)
            }
            Instr::CallSemanticNestedFunctionExpandMultiOutput {
                specs, out_count, ..
            } => {
                let operands: usize = specs.iter().map(ArgumentSpec::stack_operand_count).sum();
                effect(operands, *out_count)
            }
            Instr::PackToRow(n) | Instr::PackToCol(n) => effect(*n, 1),
            Instr::EnterScope(_) | Instr::ExitScope(_) | Instr::Jump(_) | Instr::LeaveTry(_) => {
                effect(0, 0)
            }
            Instr::EnterTry { .. } => effect(0, 0),
            Instr::Return => effect(0, 0),
            Instr::ReturnValue => effect(1, 0),
            Instr::RegisterImport { .. }
            | Instr::DeclareGlobal(_)
            | Instr::DeclarePersistent(_)
            | Instr::DeclareGlobalNamed(_, _)
            | Instr::DeclarePersistentNamed(_, _) => effect(0, 0),
            Instr::Spawn => effect(1, 1),
            Instr::SpawnOn => effect(2, 1),
            Instr::Await => effect(1, 1),
            Instr::EnsurePool(arg_count) | Instr::CurrentPool(arg_count) => effect(*arg_count, 1),
            Instr::ExecuteParfor {
                has_maximum_workers,
                ..
            } => effect(1 + usize::from(*has_maximum_workers), 0),
            Instr::ExecuteSpmd { header, .. } => effect(header.operand_count(), 0),
            Instr::Distributed(operation) => effect(
                operation.operand_count(),
                match operation {
                    BytecodeDistributedOp::GlobalIndices {
                        requested_outputs, ..
                    } => usize::from(*requested_outputs),
                    _ => 1,
                },
            ),
            Instr::Collective { operation, .. } => effect(
                operation.operand_count(),
                match operation {
                    BytecodeCollectiveOp::Receive {
                        requested_outputs, ..
                    } => usize::from(*requested_outputs),
                    _ => 1,
                },
            ),
            Instr::EmitStackTop { .. } => effect(1, 1),
            Instr::EmitVar { .. } => effect(0, 0),
            Instr::StochasticEvolution => None,
        }
    }
}

fn subscript_operand_count(steps: &[super::BytecodeSubscriptStep]) -> usize {
    steps
        .iter()
        .map(super::BytecodeSubscriptStep::operand_count)
        .sum()
}

fn subscript_selector_operand_count(selectors: &[super::BytecodeSubscriptSelector]) -> usize {
    selectors
        .iter()
        .filter(|selector| matches!(selector, super::BytecodeSubscriptSelector::Value))
        .count()
}
