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

pub(in crate::execute) fn observed_state(probe: &Rc<Probe>) -> HostState {
    let runtime = runtime(probe);
    let function = NativeFunction {
        id: ProgramFunctionId(0),
        source: ProgramSourceId(0),
        name: "observation".into(),
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
        locals: vec![],
        entry: NativeBlockId(0),
        blocks: vec![entry_block()],
        expected_sites: vec![],
    };
    let (mut state, _) = HostState::new(HostStateInput {
        functions: Arc::new(vec![function.clone()]),
        function,
        arguments: vec![],
        requested_outputs: 0,
        runtime: runtime.clone(),
        program_capture: None,
        captures: vec![],
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
    state.region_observation = Some(GenericRegionObservation::new(&plan, ticket));
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
