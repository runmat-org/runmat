use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use runmat_native_codegen::{
    NativeBlockId, NativeEdge, NativeFunction, NativeLocalKind, NativeTerminatorKind, NativeValueId,
};
use runmat_runtime::call::function_abi::{prepare_function_inputs, FunctionInputSpec};
use runmat_runtime::native::{NativeRoot, NativeRootKind, NativeRootSet, NativeValueRef};
use runmat_runtime::{context::RuntimeContext, RuntimeError};

use crate::deopt::{
    DeoptimizationPolicy, FaultInjection, MaterializedFrame, NativeMaterializationContext,
};
use crate::memory::{ScopedValueRoots, ValueArena};
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
    pub arguments: Vec<runmat_value::Value>,
    pub requested_outputs: usize,
    pub runtime: RuntimeContext,
    pub program_capture: Option<Vec<u8>>,
    pub functions: Arc<Vec<NativeFunction>>,
    pub captures: Vec<runmat_runtime::call::lexical::LexicalCapture>,
    pub deoptimization: DeoptimizationPolicy,
    pub interpreter_resume_points: BTreeMap<runmat_types::ProgramPointId, u64>,
    pub coverage_sites: BTreeMap<runmat_native_codegen::NativeMirSite, Vec<u64>>,
    pub osr_point: Option<runmat_types::ProgramPointId>,
    pub optimized_regions: Arc<Vec<crate::region::OptimizedRegionPlan>>,
    pub workspace: Option<super::workspace::NativeWorkspaceInput>,
}

pub(super) struct HostState {
    pub function: NativeFunction,
    pub functions: Arc<Vec<NativeFunction>>,
    pub runtime: RuntimeContext,
    pub arena: ValueArena,
    pub locals: Vec<NativeValueRef>,
    pub values: BTreeMap<NativeValueId, NativeValueRef>,
    pub roots: Vec<NativeRoot>,
    pub last_error: Option<RuntimeError>,
    pub host_failure: Option<NativeExecutorError>,
    pub current_source: runmat_runtime::native::NativeSourceLocation,
    pub pending_place_mutation: Option<runmat_mir::MirPlaceMutation>,
    pub program_capture: Option<Vec<u8>>,
    pub pending_await: Option<super::awaiting::PendingAwait>,
    resume_target: Option<runmat_runtime::native::NativeSiteRequest>,
    current_block: Option<NativeBlockId>,
    global_bindings: BTreeMap<usize, String>,
    persistent_bindings: BTreeMap<usize, String>,
    active_for_loops: BTreeMap<NativeBlockId, ActiveForLoop>,
    loop_backedges: BTreeMap<runmat_types::ProgramPointId, u64>,
    osr_point: Option<runmat_types::ProgramPointId>,
    osr_entry: Option<runmat_types::ProgramPointId>,
    active_exception_handlers: Vec<ActiveExceptionHandler>,
    next_await_continuation: u64,
    deoptimization: DeoptimizationPolicy,
    retired_guards: BTreeSet<runmat_types::RegionGuardId>,
    pending_deoptimization: Option<MaterializedFrame>,
    interpreter_resume_points: BTreeMap<runmat_types::ProgramPointId, u64>,
    coverage_sites: BTreeMap<runmat_native_codegen::NativeMirSite, Vec<u64>>,
    supplied_inputs: usize,
    requested_outputs: usize,
    missing_input_locals: Vec<runmat_native_codegen::NativeLocalId>,
    optimized_regions: BTreeMap<crate::region::SiteIdentity, crate::region::OptimizedRegionPlan>,
    disabled_optimized_regions: BTreeSet<runmat_types::RegionId>,
    skipped_optimized_sites: BTreeSet<crate::region::SiteIdentity>,
    vectorized_regions: u64,
    workspace: Option<super::workspace::NativeWorkspaceState>,
    last_expression: Option<NativeValueRef>,
}

impl HostState {
    pub fn new(input: HostStateInput) -> NativeExecutorResult<(Self, Vec<NativeValueRef>)> {
        let HostStateInput {
            function,
            arguments,
            requested_outputs,
            runtime,
            program_capture,
            functions,
            captures,
            deoptimization,
            interpreter_resume_points,
            coverage_sites,
            osr_point,
            optimized_regions,
            workspace,
        } = input;
        let construction_values = arguments
            .iter()
            .cloned()
            .chain(captures.iter().map(|capture| capture.value.clone()))
            .collect();
        let _construction_roots =
            ScopedValueRoots::register(construction_values, "native_invocation_construction")?;
        let input_specs = function
            .argument_validations
            .iter()
            .map(|validation| {
                let input_index = function
                    .abi
                    .fixed_inputs
                    .iter()
                    .position(|local| *local == validation.input)
                    .ok_or_else(|| {
                        NativeExecutorError::Host(
                            "native argument validation does not name a fixed input".into(),
                        )
                    })?;
                Ok(FunctionInputSpec {
                    input_index,
                    size: validation.size.as_ref(),
                    class_name: validation.class_name.as_deref(),
                    validators: &validation.validators,
                    default_value: validation.default_value.as_ref(),
                })
            })
            .collect::<NativeExecutorResult<Vec<_>>>()?;
        let supplied_inputs = arguments.len();
        let prepared = prepare_function_inputs(
            &function.name,
            &arguments,
            function.abi.fixed_inputs.len(),
            function.abi.varargin.is_some(),
            &input_specs,
        )?;
        let missing_input_locals = function
            .abi
            .fixed_inputs
            .iter()
            .zip(&prepared.fixed)
            .filter_map(|(local, value)| value.is_none().then_some(*local))
            .collect();
        let workspace = if let Some(input) = workspace {
            super::workspace::NativeWorkspaceState::new(&function, input)?
        } else {
            super::workspace::NativeWorkspaceState::function_frame(&function)
        };
        let ports = runtime
            .service_ports()
            .clone()
            .with_workspace(workspace.service());
        let runtime = runtime.with_service_ports(ports);
        let mut arena = ValueArena::new()?;
        let argument_refs = arguments
            .into_iter()
            .map(|value| arena.insert(value))
            .collect::<Vec<_>>();
        let mut locals = vec![NativeValueRef::NULL; function.local_count()];
        let capture_locals = function
            .locals
            .iter()
            .filter(|local| local.kind == NativeLocalKind::Capture)
            .collect::<Vec<_>>();
        if capture_locals.len() != captures.len() {
            return Err(NativeExecutorError::Host(format!(
                "native lexical entry expected {} captures but received {}",
                capture_locals.len(),
                captures.len()
            )));
        }
        for local in capture_locals {
            let binding = local.binding.ok_or_else(|| {
                NativeExecutorError::Host("native capture local has no semantic binding".into())
            })?;
            let capture = captures
                .iter()
                .find(|capture| capture.binding == binding)
                .ok_or_else(|| {
                    NativeExecutorError::Host("native lexical capture binding is missing".into())
                })?;
            locals[local.id.0 as usize] = arena.insert(capture.value.clone());
        }
        for (local, value) in function.abi.fixed_inputs.iter().zip(&prepared.fixed) {
            let slot = locals.get_mut(local.0 as usize).ok_or_else(|| {
                NativeExecutorError::Host("function ABI input local is out of bounds".into())
            })?;
            if let Some(value) = value {
                *slot = arena.insert(value.clone());
            }
        }
        if let (Some(local), Some(varargin)) = (function.abi.varargin, prepared.varargin) {
            locals[local.0 as usize] = arena.insert(runmat_value::Value::Cell(varargin));
        }
        if let Some(local) = function.abi.varargout {
            let empty = runmat_value::CellArray::new(Vec::new(), 1, 0)
                .map_err(|error| NativeExecutorError::Host(format!("varargout: {error}")))?;
            locals[local.0 as usize] = arena.insert(runmat_value::Value::Cell(empty));
        }
        if let Some(local) = function.abi.implicit_nargin {
            locals[local.0 as usize] =
                arena.insert(runmat_value::Value::Num(prepared.nargin as f64));
        }
        if let Some(local) = function.abi.implicit_nargout {
            locals[local.0 as usize] =
                arena.insert(runmat_value::Value::Num(requested_outputs as f64));
        }
        for (slot, local) in locals.iter_mut().enumerate() {
            if let Some(value) = workspace.initial_value(slot) {
                *local = arena.insert(value);
            }
            let reference = *local;
            if !reference.is_null() {
                workspace.synchronize_local(slot, arena.get(reference)?.clone());
            }
        }
        let roots = locals
            .iter()
            .enumerate()
            .map(|(slot, value)| NativeRoot {
                value: *value,
                kind: NativeRootKind::LOCAL,
                slot: slot as u32,
            })
            .collect::<Vec<_>>();
        let source = function.source.0;
        let mut state = Self {
            function,
            functions,
            runtime,
            arena,
            locals,
            values: BTreeMap::new(),
            roots,
            last_error: None,
            host_failure: None,
            current_source: runmat_runtime::native::NativeSourceLocation {
                source,
                ..runmat_runtime::native::NativeSourceLocation::default()
            },
            active_for_loops: BTreeMap::new(),
            loop_backedges: BTreeMap::new(),
            osr_point,
            osr_entry: None,
            pending_place_mutation: None,
            program_capture,
            pending_await: None,
            resume_target: None,
            current_block: None,
            global_bindings: BTreeMap::new(),
            persistent_bindings: BTreeMap::new(),
            active_exception_handlers: Vec::new(),
            next_await_continuation: 1,
            deoptimization,
            retired_guards: BTreeSet::new(),
            pending_deoptimization: None,
            interpreter_resume_points,
            coverage_sites,
            supplied_inputs,
            requested_outputs,
            missing_input_locals,
            optimized_regions: optimized_regions
                .iter()
                .cloned()
                .map(|plan| (plan.entry, plan))
                .collect(),
            disabled_optimized_regions: BTreeSet::new(),
            skipped_optimized_sites: BTreeSet::new(),
            vectorized_regions: 0,
            workspace: Some(workspace),
            last_expression: None,
        };
        state.enter_block(state.function.entry)?;
        Ok((state, argument_refs))
    }

    pub fn hit_coverage(&self, site: &runmat_native_codegen::NativeMirSite) {
        if let Some(sites) = self.coverage_sites.get(site) {
            runmat_runtime::coverage::hit_sites_in(&self.runtime, sites);
        }
    }
}

mod captures;
mod continuation;
mod control;
mod workspace;

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
