#[path = "construct/expression_regions.rs"]
mod expression_regions;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum NativeLoweringClass {
    NativeOperation,
    RuntimeSlowPath,
    StructuredSuspendResume,
    CapabilityRejection,
    ProvenUnreachable,
}

pub fn effective_native_lowering_class(
    construct: MirConstructKind,
    effects: &runmat_types::EffectSet,
) -> NativeLoweringClass {
    let base = construct.native_lowering_class();
    if base == NativeLoweringClass::NativeOperation
        && effects.0.contains(&runmat_types::EffectKind::MaySuspend)
    {
        NativeLoweringClass::RuntimeSlowPath
    } else {
        base
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum MirConstructKind {
    Use,
    Unary,
    Binary,
    ShortCircuit,
    Range,
    Call,
    Aggregate,
    StructLiteral,
    ObjectLiteral,
    Index,
    SubscriptChain,
    Member,
    DynamicMember,
    WorkspaceFirstStaticProperty,
    MetaClass,
    Colon,
    End,
    Future,
    Spawn,
    DistributedCreate,
    CodistributedCreate,
    CodistributedBuild,
    DistributedLocalPart,
    DistributedMaterialize,
    DistributedCodistributor,
    DistributedGlobalIndices,
    DistributedRedistribute,
    CollectiveBarrier,
    CollectiveBroadcast,
    CollectiveGather,
    CollectiveScatter,
    CollectiveAllGather,
    CollectiveReduce,
    CollectiveAllReduce,
    CollectiveCat,
    CollectiveFunctionalReduce,
    CollectiveSend,
    CollectiveReceive,
    CollectiveSendReceive,
    CollectiveProbe,
    Assign,
    MultiAssign,
    SequenceAssign,
    CaptureSequence,
    Expr,
    PlaceMutation,
    WorkspaceEffect,
    EnvironmentEffect,
    Goto,
    Branch,
    Switch,
    For,
    ParFor,
    Spmd,
    TryCatch,
    Return,
    Await,
    Unreachable,
}

pub fn rvalue_construct_kind(value: &crate::MirRvalue) -> MirConstructKind {
    use crate::parallel::{MirCollectiveOp as C, MirDistributedOp as D};
    use crate::MirRvalue as R;
    use MirConstructKind as K;

    match value {
        R::Use(_) => K::Use,
        R::Unary(_, _) => K::Unary,
        R::Binary(_, _, _) => K::Binary,
        R::ShortCircuit { .. } => K::ShortCircuit,
        R::Range { .. } => K::Range,
        R::Call(_) => K::Call,
        R::Aggregate { .. } => K::Aggregate,
        R::StructLiteral { .. } => K::StructLiteral,
        R::ObjectLiteral { .. } => K::ObjectLiteral,
        R::Index { .. } => K::Index,
        R::SubscriptChain(_) => K::SubscriptChain,
        R::Member { .. } => K::Member,
        R::DynamicMember { .. } => K::DynamicMember,
        R::WorkspaceFirstStaticProperty { .. } => K::WorkspaceFirstStaticProperty,
        R::MetaClass(_) => K::MetaClass,
        R::Colon => K::Colon,
        R::End => K::End,
        R::Future { .. } => K::Future,
        R::Spawn(_) => K::Spawn,
        R::Distributed(operation) => match operation {
            D::Create { .. } => K::DistributedCreate,
            D::Codistributed { .. } => K::CodistributedCreate,
            D::Build { .. } => K::CodistributedBuild,
            D::LocalPart { .. } => K::DistributedLocalPart,
            D::Materialize { .. } => K::DistributedMaterialize,
            D::Codistributor { .. } => K::DistributedCodistributor,
            D::GlobalIndices { .. } => K::DistributedGlobalIndices,
            D::Redistribute { .. } => K::DistributedRedistribute,
        },
        R::Collective(operation) => match operation {
            C::Barrier { .. } => K::CollectiveBarrier,
            C::Broadcast { .. } => K::CollectiveBroadcast,
            C::Gather { .. } => K::CollectiveGather,
            C::Scatter { .. } => K::CollectiveScatter,
            C::AllGather { .. } => K::CollectiveAllGather,
            C::Reduce { .. } => K::CollectiveReduce,
            C::AllReduce { .. } => K::CollectiveAllReduce,
            C::Cat { .. } => K::CollectiveCat,
            C::FunctionalReduce { .. } => K::CollectiveFunctionalReduce,
            C::Send { .. } => K::CollectiveSend,
            C::Receive { .. } => K::CollectiveReceive,
            C::SendReceive { .. } => K::CollectiveSendReceive,
            C::Probe { .. } => K::CollectiveProbe,
        },
    }
}

pub fn statement_construct_kind(statement: &crate::MirStmtKind) -> MirConstructKind {
    use crate::MirStmtKind as S;
    use MirConstructKind as K;

    match statement {
        S::Assign { .. } => K::Assign,
        S::MultiAssign { .. } => K::MultiAssign,
        S::SequenceAssign { .. } => K::SequenceAssign,
        S::CaptureSequence { .. } => K::CaptureSequence,
        S::Expr(_) => K::Expr,
        S::PlaceMutation(_) => K::PlaceMutation,
        S::WorkspaceEffect { .. } => K::WorkspaceEffect,
        S::EnvironmentEffect(_) => K::EnvironmentEffect,
    }
}

pub fn terminator_construct_kind(terminator: &crate::MirTerminatorKind) -> MirConstructKind {
    use crate::MirTerminatorKind as T;
    use MirConstructKind as K;

    match terminator {
        T::Goto(_) => K::Goto,
        T::Branch { .. } => K::Branch,
        T::Switch { .. } => K::Switch,
        T::For { .. } => K::For,
        T::ParFor { .. } => K::ParFor,
        T::Spmd { .. } => K::Spmd,
        T::TryCatch { .. } => K::TryCatch,
        T::Return(_) => K::Return,
        T::Await { .. } => K::Await,
        T::Unreachable => K::Unreachable,
    }
}

/// Declared effects and capabilities carried by a canonical MIR rvalue.
///
/// Native backends consume this shared classification instead of rebuilding a
/// second builtin/type inference table.
pub fn rvalue_declared_requirements(
    value: &crate::MirRvalue,
) -> (runmat_types::EffectSet, runmat_types::CapabilitySet) {
    use runmat_types::{CapabilityRequirement, CapabilitySet, EffectKind, EffectSet};

    let mut effects = EffectSet::default();
    let mut capabilities = CapabilitySet::default();
    match value {
        crate::MirRvalue::Call(call) => {
            if !matches!(call.async_behavior, crate::AsyncBehaviorFact::NeverSuspends) {
                effects.0.insert(EffectKind::MaySuspend);
            }
            let declared = call.effects;
            for (enabled, effect) in [
                (declared.workspace, EffectKind::WorkspaceWrite),
                (declared.environment, EffectKind::EnvironmentWrite),
                (declared.filesystem, EffectKind::FilesystemRead),
                (declared.network, EffectKind::Network),
                (declared.ui, EffectKind::UserInterface),
                (declared.random, EffectKind::Randomness),
                (declared.time, EffectKind::Clock),
                (declared.host_callback, EffectKind::HostCallback),
                (declared.unknown, EffectKind::Unknown),
            ] {
                if enabled {
                    effects.0.insert(effect);
                }
            }
        }
        crate::MirRvalue::Future { .. } | crate::MirRvalue::Spawn(_) => {
            effects.0.insert(EffectKind::MaySuspend);
            capabilities
                .0
                .insert(CapabilityRequirement::ParallelRuntime);
        }
        crate::MirRvalue::Distributed(_) | crate::MirRvalue::Collective(_) => {
            capabilities
                .0
                .insert(CapabilityRequirement::DistributedRuntime);
        }
        crate::MirRvalue::ShortCircuit { right_temps, .. } => {
            for statement in right_temps {
                effects
                    .0
                    .extend(statement_declared_effects(&statement.kind).0);
                if let Some(value) = statement_rvalue(&statement.kind) {
                    let (nested_effects, nested_capabilities) = rvalue_declared_requirements(value);
                    effects.0.extend(nested_effects.0);
                    capabilities.0.extend(nested_capabilities.0);
                }
            }
        }
        _ => {}
    }
    value.visit_direct_expression_regions_dyn(&mut |region| {
        let (nested_effects, nested_capabilities) =
            expression_regions::declared_requirements(region);
        effects.0.extend(nested_effects.0);
        capabilities.0.extend(nested_capabilities.0);
    });
    (effects, capabilities)
}

/// Complete canonical construct inventory for one rvalue, including the
/// conditional statement region embedded by short-circuit MIR.
pub fn rvalue_construct_inventory(value: &crate::MirRvalue) -> Vec<MirConstructKind> {
    let mut constructs = rvalue_construct_inventory_without_regions(value);
    value.visit_direct_expression_regions_dyn(&mut |region| {
        constructs.extend(expression_regions::inventory(region));
    });
    constructs
}

fn rvalue_construct_inventory_without_regions(value: &crate::MirRvalue) -> Vec<MirConstructKind> {
    let mut constructs = vec![rvalue_construct_kind(value)];
    if let crate::MirRvalue::ShortCircuit { right_temps, .. } = value {
        for statement in right_temps {
            if let Some(value) = statement_rvalue(&statement.kind) {
                constructs.extend(rvalue_construct_inventory_without_regions(value));
            }
            constructs.push(statement_construct_kind(&statement.kind));
        }
    }
    constructs
}

fn statement_rvalue(statement: &crate::MirStmtKind) -> Option<&crate::MirRvalue> {
    match statement {
        crate::MirStmtKind::Assign { value, .. }
        | crate::MirStmtKind::MultiAssign { value, .. }
        | crate::MirStmtKind::SequenceAssign { value, .. }
        | crate::MirStmtKind::Expr(value) => Some(value),
        crate::MirStmtKind::PlaceMutation(_)
        | crate::MirStmtKind::CaptureSequence { .. }
        | crate::MirStmtKind::WorkspaceEffect { .. }
        | crate::MirStmtKind::EnvironmentEffect(_) => None,
    }
}

/// Declared effects carried by a canonical MIR statement.
pub fn statement_declared_effects(statement: &crate::MirStmtKind) -> runmat_types::EffectSet {
    statement_declared_requirements(statement).0
}

pub fn statement_declared_requirements(
    statement: &crate::MirStmtKind,
) -> (runmat_types::EffectSet, runmat_types::CapabilitySet) {
    use runmat_hir::WorkspaceEffect;
    use runmat_types::{CapabilitySet, EffectKind, EffectSet};

    let mut effects = EffectSet::default();
    let mut capabilities = CapabilitySet::default();
    match statement {
        crate::MirStmtKind::WorkspaceEffect { effect, .. } => match effect {
            WorkspaceEffect::None => {}
            WorkspaceEffect::ReadsWorkspace => {
                effects.0.insert(EffectKind::WorkspaceRead);
            }
            WorkspaceEffect::CreatesBinding
            | WorkspaceEffect::ClearsBinding
            | WorkspaceEffect::ClearsFunctionCache
            | WorkspaceEffect::MutatesGlobal
            | WorkspaceEffect::MutatesPersistent
            | WorkspaceEffect::LoadsExternalBindings
            | WorkspaceEffect::DynamicEval => {
                effects.0.insert(EffectKind::WorkspaceWrite);
            }
        },
        crate::MirStmtKind::EnvironmentEffect(_) => {
            effects.0.insert(EffectKind::EnvironmentWrite);
        }
        crate::MirStmtKind::CaptureSequence { .. } => {
            effects.0.insert(EffectKind::Unknown);
        }
        _ => {}
    }
    statement.visit_statement_expression_regions_dyn(&mut |region| {
        let (nested_effects, nested_capabilities) =
            expression_regions::declared_requirements(region);
        effects.0.extend(nested_effects.0);
        capabilities.0.extend(nested_capabilities.0);
    });
    (effects, capabilities)
}

pub fn statement_construct_inventory(statement: &crate::MirStmtKind) -> Vec<MirConstructKind> {
    let mut constructs = vec![statement_construct_kind(statement)];
    statement.visit_statement_expression_regions_dyn(&mut |region| {
        constructs.extend(expression_regions::inventory(region));
    });
    constructs
}

impl MirConstructKind {
    pub const ALL: [Self; 58] = [
        Self::Use,
        Self::Unary,
        Self::Binary,
        Self::ShortCircuit,
        Self::Range,
        Self::Call,
        Self::Aggregate,
        Self::StructLiteral,
        Self::ObjectLiteral,
        Self::Index,
        Self::SubscriptChain,
        Self::Member,
        Self::DynamicMember,
        Self::WorkspaceFirstStaticProperty,
        Self::MetaClass,
        Self::Colon,
        Self::End,
        Self::Future,
        Self::Spawn,
        Self::DistributedCreate,
        Self::CodistributedCreate,
        Self::CodistributedBuild,
        Self::DistributedLocalPart,
        Self::DistributedMaterialize,
        Self::DistributedCodistributor,
        Self::DistributedGlobalIndices,
        Self::DistributedRedistribute,
        Self::CollectiveBarrier,
        Self::CollectiveBroadcast,
        Self::CollectiveGather,
        Self::CollectiveScatter,
        Self::CollectiveAllGather,
        Self::CollectiveReduce,
        Self::CollectiveAllReduce,
        Self::CollectiveCat,
        Self::CollectiveFunctionalReduce,
        Self::CollectiveSend,
        Self::CollectiveReceive,
        Self::CollectiveSendReceive,
        Self::CollectiveProbe,
        Self::Assign,
        Self::MultiAssign,
        Self::SequenceAssign,
        Self::CaptureSequence,
        Self::Expr,
        Self::PlaceMutation,
        Self::WorkspaceEffect,
        Self::EnvironmentEffect,
        Self::Goto,
        Self::Branch,
        Self::Switch,
        Self::For,
        Self::ParFor,
        Self::Spmd,
        Self::TryCatch,
        Self::Return,
        Self::Await,
        Self::Unreachable,
    ];

    pub const fn native_lowering_class(self) -> NativeLoweringClass {
        use MirConstructKind as K;
        use NativeLoweringClass as C;
        match self {
            K::Use
            | K::Unary
            | K::Binary
            | K::ShortCircuit
            | K::Range
            | K::Aggregate
            | K::StructLiteral
            | K::Index
            | K::Member
            | K::Colon
            | K::End
            | K::Assign
            | K::MultiAssign
            | K::SequenceAssign
            | K::CaptureSequence
            | K::Expr
            | K::Goto
            | K::Branch
            | K::Switch
            | K::For
            | K::Return => C::NativeOperation,
            K::Call
            | K::ObjectLiteral
            | K::DynamicMember
            | K::WorkspaceFirstStaticProperty
            | K::MetaClass
            | K::PlaceMutation
            | K::WorkspaceEffect
            | K::EnvironmentEffect
            | K::TryCatch => C::RuntimeSlowPath,
            K::SubscriptChain | K::Future | K::Spawn | K::ParFor | K::Spmd | K::Await => {
                C::StructuredSuspendResume
            }
            K::DistributedCreate
            | K::CodistributedCreate
            | K::CodistributedBuild
            | K::DistributedLocalPart
            | K::DistributedMaterialize
            | K::DistributedCodistributor
            | K::DistributedGlobalIndices
            | K::DistributedRedistribute
            | K::CollectiveBarrier
            | K::CollectiveBroadcast
            | K::CollectiveGather
            | K::CollectiveScatter
            | K::CollectiveAllGather
            | K::CollectiveReduce
            | K::CollectiveAllReduce
            | K::CollectiveCat
            | K::CollectiveFunctionalReduce
            | K::CollectiveSend
            | K::CollectiveReceive
            | K::CollectiveSendReceive
            | K::CollectiveProbe => C::CapabilityRejection,
            K::Unreachable => C::ProvenUnreachable,
        }
    }
}
