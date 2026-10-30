use super::*;
use crate::execute::state::HostStateInput;
use crate::region::placement_test_support::{choose, plan, runtime, Probe};
use crate::region::RegionPlacement;
use runmat_native_codegen::{
    NativeBlock, NativeBlockId, NativeFrameState, NativeFunction, NativeFunctionAbi, NativeMirSite,
    NativeSitePhase, NativeSourceLocation, NativeTerminator, NativeTerminatorKind, NativeValueId,
};
use runmat_types::{ProgramFunctionId, ProgramPointId, ProgramSourceId, ProgramSpan};
use std::{rc::Rc, sync::Arc};

// Failure retires an original observation; it does not prove a physical sample.
// Keep that lifecycle observable without authorizing feedback or fake costs.
struct Retirement {
    retired: Rc<std::cell::Cell<usize>>,
    _program: runmat_runtime::numeric_region::service::NativeNumericRegionProgramOwner,
    _inputs: runmat_runtime::execution::RootedValues,
}
impl Drop for Retirement {
    fn drop(&mut self) {
        self.retired.set(self.retired.get() + 1);
    }
}
impl runmat_runtime::numeric_region::service::NativeOriginalRegionObservation for Retirement {
    fn audit(&self) -> Option<runmat_runtime::numeric_region::service::SourceRegionRouteAudit> {
        None
    }
    fn complete(
        self: Box<Self>,
        _: runmat_runtime::context::RuntimeContext,
        _: runmat_runtime::numeric_region::service::NativeNumericRegionRight,
    ) -> Result<
        runmat_runtime::execution::RuntimeContinuationFuture<'static, ()>,
        runmat_runtime::RuntimeError,
    > {
        panic!("failed native execution must not complete an observation")
    }
}

pub(in crate::execute) fn observed_state_with_retirement(
    probe: &Rc<Probe>,
    retired: Rc<std::cell::Cell<usize>>,
) -> HostState {
    let mut state = observed_state(probe);
    let plan = plan();
    let site = runmat_runtime::native::NativeSiteRequest {
        function: plan.entry.point.function.0,
        block: plan.entry.point.block,
        position: plan.entry.point.position,
        phase: runmat_runtime::native::NativeSitePhase::RVALUE,
        ordinal: plan.entry.ordinal,
        reserved: 0,
    };
    let (program, inputs, right) =
        runmat_runtime::numeric_region::service::NativeNumericRegionRequest::capture_verified(
            &state.runtime,
            plan.region,
            site,
            &plan.program,
            &[&runmat_value::Value::Num(1.)],
        )
        .unwrap()
        .unwrap()
        .into_parts();
    assert!(right.is_current(&state.runtime));
    assert!(right.matches(plan.region, site));
    state.region_observation = Some(GenericRegionObservation::new(
        crate::region::OptimizedRegionPlanLease::from_test_plan(plan),
        RegionObservation::original(
            state.runtime.clone(),
            right,
            Some(Box::new(Retirement {
                retired,
                _program: program,
                _inputs: inputs,
            })),
        ),
    ));
    state
}

pub(in crate::execute) fn observed_state(probe: &Rc<Probe>) -> HostState {
    let runtime = runtime(probe);
    let function = NativeFunction {
        spawn_safety: runmat_types::SpawnSafetyFact::SpawnSafe,
        lexical_bindings: Vec::new(),
        inherited_storage_bindings: Vec::new(),
        creates_lexical_environment: false,
        id: runmat_types::FunctionId(0),
        portable_id: ProgramFunctionId(0),
        mir_body_digest: runmat_execution::Digest::sha256(b"fixture:native-observation"),
        source: ProgramSourceId(0),
        name: "observation".into(),
        source_role: runmat_types::FunctionSourceRole::Unknown,
        lexical_parent: None,
        class_method_owner: None,
        capabilities: Default::default(),
        abi: NativeFunctionAbi {
            fixed_inputs: vec![],
            varargin: None,
            fixed_outputs: vec![],
            varargout: None,
            implicit_nargin: None,
            implicit_nargout: None,
        },
        argument_validations: vec![],
        binding_locals: vec![],
        closure_captures: vec![],
        locals: vec![],
        entry: NativeBlockId(0),
        blocks: vec![entry_block()],
        expression_continuation_geometry:
            runmat_native_codegen::NativeExpressionContinuationGeometry {
                region_capacity: 16,
                retained_input_capacity: 16,
            },
        operation_contracts: vec![],
        expected_sites: vec![],
    };
    let (mut state, _) = HostState::new(HostStateInput {
        functions: Arc::new(vec![function.clone()]),
        function,
        inputs: crate::execute::state::NativeFunctionInputs::legacy(vec![]),
        requested_outputs: 0,
        runtime: runtime.clone(),
        program_capture: None,
        captures: vec![],
        lexical_environment: None,
        deoptimization: Default::default(),
        interpreter_resume_points: Default::default(),
        coverage_sites: Default::default(),
        osr_point: None,
        optimized_regions: Default::default(),
        workspace: None,
    })
    .unwrap();
    let plan = plan();
    let RegionPlacement::Generic(ticket) = choose(&runtime, &plan, &runmat_value::Value::Num(1.))
    else {
        panic!("generic")
    };
    state.region_observation = Some(GenericRegionObservation::new(
        crate::region::OptimizedRegionPlanLease::from_test_plan(plan),
        ticket,
    ));
    state
}

fn entry_block() -> NativeBlock {
    let point = ProgramPointId {
        function: ProgramFunctionId(0),
        block: 0,
        position: 0,
    };
    let source = NativeSourceLocation {
        source: ProgramSourceId(0),
        span: ProgramSpan { start: 0, end: 0 },
    };
    NativeBlock {
        id: NativeBlockId(0),
        catch_scopes: vec![],
        parameters: vec![],
        side_effect_epoch: NativeValueId(0),
        region_boundaries: vec![],
        instructions: vec![],
        terminator: NativeTerminator {
            site: NativeMirSite {
                point,
                phase: NativeSitePhase::Terminator,
                ordinal: 0,
                construct: runmat_mir::MirConstructKind::Return,
            },
            source: source.clone(),
            class: runmat_mir::NativeLoweringClass::NativeOperation,
            effects: Default::default(),
            capabilities: Default::default(),
            safepoint: None,
            kind: NativeTerminatorKind::Return { values: vec![] },
            frame_state: NativeFrameState {
                point,
                source,
                locals: vec![],
                operands: vec![],
                side_effect_epoch: NativeValueId(0),
            },
        },
    }
}
