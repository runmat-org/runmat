use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use runmat_native_codegen::{
    NativeBlockId, NativeEdge, NativeFunction, NativeTerminatorKind, NativeValueId,
};
use runmat_runtime::native::{NativeRoot, NativeRootKind, NativeRootSet, NativeValueRef};
use runmat_runtime::{context::RuntimeContext, RuntimeError};

use crate::deopt::{
    DeoptimizationPolicy, FaultInjection, MaterializedFrame, NativeMaterializationContext,
};
use crate::memory::ValueArena;
use crate::specialization::{GuardFailure, GuardFailureKind};
use crate::{NativeExecutorError, NativeExecutorResult};

pub(super) struct ActiveForLoop {
    pub iterator: runmat_runtime::iteration::ForColumnIterator,
    body_blocks: BTreeSet<NativeBlockId>,
}

pub(super) struct ActiveExceptionHandler {
    pub catch_edge: NativeEdge,
    protected_blocks: BTreeSet<NativeBlockId>,
}

pub(super) struct HostStateInput {
    pub function: NativeFunction,
    pub inputs: NativeFunctionInputs,
    pub requested_outputs: usize,
    pub runtime: RuntimeContext,
    pub program_capture: Option<Vec<u8>>,
    pub functions: Arc<Vec<NativeFunction>>,
    pub captures: Vec<NativeCaptureInput>,
    pub lexical_environment: Option<runmat_value::LexicalEnvironment>,
    pub deoptimization: DeoptimizationPolicy,
    pub interpreter_resume_points: BTreeMap<runmat_types::ProgramPointId, u64>,
    pub coverage_sites: BTreeMap<runmat_native_codegen::NativeMirSite, Vec<u64>>,
    pub osr_point: Option<runmat_types::ProgramPointId>,
    pub optimized_regions: Arc<Vec<crate::region::OptimizedRegionPlan>>,
    pub workspace: Option<super::workspace::NativeWorkspaceInput>,
}

// Keep the rooted prepared owner inline: boxing it would introduce a new
// allocation at retained entry. The source control preserves legacy validation
// and preparation order without copying either input forest.
pub(super) struct NativeFunctionInputs {
    source: NativeInputSource,
    prepared: Option<runmat_runtime::call::function_abi::PreparedFunctionInputs>,
}

enum NativeInputSource {
    Legacy(Vec<runmat_value::Value>),
    Prepared,
}

impl NativeFunctionInputs {
    pub(super) fn legacy(arguments: Vec<runmat_value::Value>) -> Self {
        Self {
            source: NativeInputSource::Legacy(arguments),
            prepared: None,
        }
    }

    pub(super) fn prepared(
        prepared: runmat_runtime::call::function_abi::PreparedFunctionInputs,
    ) -> Self {
        Self {
            source: NativeInputSource::Prepared,
            prepared: Some(prepared),
        }
    }

    pub(super) fn legacy_arguments(&self) -> Option<&[runmat_value::Value]> {
        match &self.source {
            NativeInputSource::Legacy(arguments) => Some(arguments),
            NativeInputSource::Prepared => None,
        }
    }

    pub(super) fn input_count(&self) -> usize {
        match &self.source {
            NativeInputSource::Legacy(arguments) => arguments.len(),
            NativeInputSource::Prepared => {
                self.prepared
                    .as_ref()
                    .expect("prepared input source owns its prepared inputs")
                    .nargin
            }
        }
    }

    /// Extract prepared inputs, returning the original source arguments when
    /// this entry still requires legacy preparation.
    pub(super) fn try_into_prepared(
        self,
    ) -> Result<runmat_runtime::call::function_abi::PreparedFunctionInputs, Vec<runmat_value::Value>>
    {
        match self.source {
            NativeInputSource::Legacy(arguments) => Err(arguments),
            NativeInputSource::Prepared => Ok(self
                .prepared
                .expect("prepared input source owns its prepared inputs")),
        }
    }
}

pub(super) enum NativeCaptureInput {
    Owned(runmat_runtime::call::lexical::LexicalCapture),
    Retained {
        binding: runmat_types::BindingId,
        local: runmat_native_codegen::NativeLocalId,
        value: runmat_runtime::call::CapturedValueLease,
    },
}

pub(super) struct HostState {
    pub liveness: Option<
        runmat_execution::resource::ExactShared<runmat_native_codegen::NativeProgramLiveness>,
    >,
    pub function: NativeFunction,
    pub functions: Arc<Vec<NativeFunction>>,
    pub runtime: RuntimeContext,
    inherited_catch: Option<std::rc::Rc<runmat_value::MException>>,
    pub locals: Vec<NativeValueRef>,
    pub values: BTreeMap<NativeValueId, NativeValueRef>,
    pub roots: roots::NativeRoots,
    pub last_error: Option<runmat_runtime::runtime_error::RuntimeErrorOwner>,
    pub host_failure: Option<NativeExecutorError>,
    pub retirement_failure: Option<NativeExecutorError>,
    pub current_source: runmat_runtime::native::NativeSourceLocation,
    pub pending_place_mutation: Option<runmat_mir::MirPlaceMutation>,
    pub pending_assign: Option<super::mutation::PendingAssign>,
    pub pending_place_prepare: Option<super::mutation::PendingPlacePrepare>,
    pub pending_plain_multi: Option<super::mutation::PendingPlainMulti>,
    pub program_capture: Option<Vec<u8>>,
    pub pending_await: Option<super::awaiting::PendingAwait>,
    pub pending_call: Option<super::call_suspension::PendingCall>,
    pub completed_call: Option<super::call_suspension::CompletedCall>,
    /// Typed one-producer/one-consumer channel between a sequence-assignment
    /// rvalue site and its immediately following statement site.
    pub sequence_assignment_register: Option<super::mutation::PreparedSequenceAssignment>,
    pub pending_sequence_rvalue: Option<super::mutation::PreparedSequenceRvalue>,
    pub pending_sequence_assign: Option<super::mutation::PendingSequenceAssign>,
    pub prepared_multi_assignment: Option<super::mutation::PreparedMultiAssignment>,
    pub pending_multi_preparation: Option<super::mutation::PendingMultiPreparation>,
    pub contextual_index_extents: Vec<usize>,
    pub subscript_end_receivers: Vec<(
        runmat_runtime::object::protocol::PreparedSubscriptReceiver,
        usize,
        usize,
    )>,
    contextual_regions: expression_regions::ExpressionRegionState,
    pub current_request: Option<runmat_runtime::native::NativeSiteRequest>,
    resume_target: Option<runmat_runtime::native::NativeSiteRequest>,
    current_block: Option<NativeBlockId>,
    global_bindings: runmat_runtime::workspace::bindings::FrameGlobalBindings,
    global_seen_revision: BTreeMap<
        usize,
        (
            runmat_runtime::workspace::GlobalBindingTarget,
            runmat_runtime::workspace::session::GlobalRevision,
        ),
    >,
    persistent_seen_revision:
        BTreeMap<usize, runmat_runtime::workspace::session::PersistentRevision>,
    persistent_bindings:
        BTreeMap<usize, runmat_runtime::workspace::session::PersistentBindingContinuation>,
    active_for_loops: BTreeMap<NativeBlockId, ActiveForLoop>,
    loop_backedges: BTreeMap<runmat_types::ProgramPointId, u64>,
    osr_point: Option<runmat_types::ProgramPointId>,
    osr_entry: Option<runmat_types::ProgramPointId>,
    active_exception_handlers: Vec<ActiveExceptionHandler>,
    catch_exceptions: BTreeMap<runmat_mir::CatchScopeId, std::rc::Rc<runmat_value::MException>>,
    next_await_continuation: u64,
    deoptimization: DeoptimizationPolicy,
    retired_guards: BTreeSet<runmat_types::RegionGuardId>,
    pending_deoptimization: Option<MaterializedFrame>,
    interpreter_resume_points: BTreeMap<runmat_types::ProgramPointId, u64>,
    coverage_sites: BTreeMap<runmat_native_codegen::NativeMirSite, Vec<u64>>,
    supplied_inputs: usize,
    requested_outputs: usize,
    missing_input_locals: Vec<runmat_native_codegen::NativeLocalId>,
    optimized_regions:
        BTreeMap<crate::region::SiteIdentity, crate::region::OptimizedRegionPlanLease>,
    pub(super) region_observation: Option<super::region::observation::GenericRegionObservation>,
    skipped_optimized_sites: BTreeSet<crate::region::SiteIdentity>,
    vectorized_regions: u64,
    workspace: Option<super::workspace::NativeWorkspaceState>,
    last_expression: Option<NativeValueRef>,
    // Fields drop in declaration order. Pending work/service destructors may
    // trace or collect values, so retain the arena's GC roots until every
    // execution-owned future, capture, receiver and workspace has been dropped.
    // The invocation's execution lease keeps runtime activation alive outside
    // this owner while its fields are destroyed.
    pub arena: ValueArena,
    pub lexical_environment: Option<runmat_value::LexicalEnvironment>,
    pub(super) _lexical_roots: Vec<runmat_gc::ExplicitRoot>,
}

impl HostState {
    pub fn hit_coverage(&self, site: &runmat_native_codegen::NativeMirSite) {
        if let Some(sites) = self.coverage_sites.get(site) {
            runmat_runtime::coverage::hit_sites_in(&self.runtime, sites);
        }
    }
}

mod captures;
mod construction;
pub(super) use construction::function_input_specs;
mod continuation;
mod control;
#[path = "contextual_regions.rs"]
mod expression_regions;
mod roots;
mod workspace;

pub(crate) use expression_regions::EmbeddedOperationIdentity;

fn optimized_site_identity(
    request: runmat_runtime::native::NativeSiteRequest,
) -> crate::region::SiteIdentity {
    crate::region::SiteIdentity {
        point: runmat_types::ProgramPointId {
            function: runmat_types::ProgramFunctionId(request.function),
            block: request.block,
            position: request.position,
        },
        phase: match request.phase {
            phase if phase == runmat_runtime::native::NativeSitePhase::RVALUE => {
                runmat_native_codegen::NativeSitePhase::Rvalue
            }
            phase if phase == runmat_runtime::native::NativeSitePhase::STATEMENT => {
                runmat_native_codegen::NativeSitePhase::Statement
            }
            phase if phase == runmat_runtime::native::NativeSitePhase::TERMINATOR_RVALUE => {
                runmat_native_codegen::NativeSitePhase::TerminatorRvalue
            }
            _ => runmat_native_codegen::NativeSitePhase::Terminator,
        },
        ordinal: request.ordinal,
    }
}

fn skip_before_target(
    expected_sites: &[runmat_native_codegen::NativeMirSite],
    target: runmat_runtime::native::NativeSiteRequest,
    request: runmat_runtime::native::NativeSiteRequest,
) -> NativeExecutorResult<bool> {
    if request.block != target.block {
        return Err(NativeExecutorError::Host(
            "generated re-entry reached a block before its resume target".into(),
        ));
    }
    let current = expected_sites
        .iter()
        .position(|site| native_site_matches(site, request));
    let target = expected_sites
        .iter()
        .position(|site| native_site_matches(site, target));
    match (current, target) {
        (Some(current), Some(target)) if current < target => Ok(true),
        (Some(_), Some(_)) => Err(NativeExecutorError::Host(
            "generated re-entry passed its exact resume target".into(),
        )),
        _ => Err(NativeExecutorError::Host(
            "generated re-entry produced an unverified site".into(),
        )),
    }
}

fn native_site_matches(
    site: &runmat_native_codegen::NativeMirSite,
    request: runmat_runtime::native::NativeSiteRequest,
) -> bool {
    site.point.block == request.block
        && site.point.position == request.position
        && native_site_phase(site.phase) == request.phase
        && site.ordinal == request.ordinal
}

fn native_site_phase(
    phase: runmat_native_codegen::NativeSitePhase,
) -> runmat_runtime::native::NativeSitePhase {
    match phase {
        runmat_native_codegen::NativeSitePhase::Rvalue => {
            runmat_runtime::native::NativeSitePhase::RVALUE
        }
        runmat_native_codegen::NativeSitePhase::Statement => {
            runmat_runtime::native::NativeSitePhase::STATEMENT
        }
        runmat_native_codegen::NativeSitePhase::TerminatorRvalue => {
            runmat_runtime::native::NativeSitePhase::TERMINATOR_RVALUE
        }
        runmat_native_codegen::NativeSitePhase::Terminator => {
            runmat_runtime::native::NativeSitePhase::TERMINATOR
        }
    }
}

fn empty_workspace_value() -> runmat_value::Value {
    runmat_value::Value::Tensor(
        runmat_value::Tensor::new(Vec::new(), vec![0, 0])
            .expect("the canonical empty workspace shape is valid"),
    )
}

fn successor_blocks(kind: &NativeTerminatorKind) -> Vec<NativeBlockId> {
    match kind {
        NativeTerminatorKind::Goto { edge } => vec![edge.target],
        NativeTerminatorKind::Branch {
            then_edge,
            else_edge,
            ..
        } => vec![then_edge.target, else_edge.target],
        NativeTerminatorKind::Switch {
            cases, otherwise, ..
        } => cases
            .iter()
            .map(|(_, edge)| edge.target)
            .chain(std::iter::once(otherwise.target))
            .collect(),
        NativeTerminatorKind::For { body, exit, .. }
        | NativeTerminatorKind::ParFor { body, exit, .. }
        | NativeTerminatorKind::Spmd { body, exit, .. } => vec![body.target, exit.target],
        NativeTerminatorKind::TryCatch {
            try_edge,
            catch_edge,
            ..
        } => vec![try_edge.target, catch_edge.target],
        NativeTerminatorKind::Await { resume, .. } => vec![resume.target],
        NativeTerminatorKind::Return { .. } | NativeTerminatorKind::Unreachable => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_native_codegen::{NativeMirSite, NativeSitePhase};
    use runmat_runtime::native::{NativeSitePhase as RuntimePhase, NativeSiteRequest};
    use runmat_types::{ProgramFunctionId, ProgramPointId};

    fn site(position: u32, phase: NativeSitePhase, ordinal: u32) -> NativeMirSite {
        NativeMirSite {
            point: ProgramPointId {
                function: ProgramFunctionId(7),
                block: 4,
                position,
            },
            phase,
            ordinal,
            construct: runmat_mir::MirConstructKind::Use,
        }
    }

    fn request(position: u32, phase: RuntimePhase, ordinal: u32) -> NativeSiteRequest {
        NativeSiteRequest {
            function: 7,
            block: 4,
            position,
            phase,
            ordinal,
            reserved: 0,
        }
    }

    #[test]
    fn exact_resume_skips_only_verified_predecessor_sites() {
        let expected = vec![
            site(0, NativeSitePhase::Rvalue, 0),
            site(0, NativeSitePhase::Statement, 1),
            site(1, NativeSitePhase::Terminator, 0),
        ];
        let target = request(0, RuntimePhase::STATEMENT, 1);
        assert!(
            skip_before_target(&expected, target, request(0, RuntimePhase::RVALUE, 0)).unwrap()
        );
        assert!(
            skip_before_target(&expected, target, request(1, RuntimePhase::TERMINATOR, 0)).is_err()
        );
        let wrong_block = NativeSiteRequest {
            block: 3,
            ..request(0, RuntimePhase::RVALUE, 0)
        };
        assert!(skip_before_target(&expected, target, wrong_block).is_err());
    }
}
