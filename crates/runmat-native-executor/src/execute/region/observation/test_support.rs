use super::*;
use crate::execute::state::HostStateInput;
use crate::region::placement_test_support::{choose, plan, runtime, Probe};
use crate::region::RegionPlacement;
use runmat_native_codegen::{NativeBlockId, NativeFunction, NativeFunctionAbi};
use runmat_types::{ProgramFunctionId, ProgramSourceId};
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
        blocks: vec![],
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
