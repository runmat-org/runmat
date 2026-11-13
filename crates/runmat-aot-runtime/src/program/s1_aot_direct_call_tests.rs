//! Exact closed-world AOT program plus a test-linked NativeEntryPoint ABI driver.
//! The driver forwards verified Native IR sites; the executor still owns every
//! site operation and canonical builtin call. No Runtime shortcut is invoked.
use super::*;
use runmat_execution::{
    CandidateObservationEvidence, Digest, PlacementDecision, PlacementFeedback,
    PlacementInvalidation, PlacementPlanRequest, PlacementRevision, WholeWorkDecision,
    WholeWorkObservationOutcome, WholeWorkPlacementRequest,
};
use runmat_runtime::context::{
    RuntimeContext, RuntimePlacementService, RuntimeServicePorts, RuntimeWholeWorkSelection,
};
use runmat_runtime::native::{
    NativeCall, NativeExit, NativeHostStatus, NativeSiteOutcome, NativeSiteOutcomeKind,
    NativeSitePhase, NativeSiteRequest,
};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    ffi::c_void,
    sync::OnceLock,
};

struct RecordingPlacement {
    actual: runmat_accelerate::placement::PlacementSession,
    decisions: RefCell<Vec<WholeWorkDecision>>,
}

impl RuntimePlacementService for RecordingPlacement {
    fn whole_work_runtime_binding_session_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        binding: runmat_builtins::BuiltinBindingIdentity,
        registration: Digest,
    ) -> Option<PlacementRevision> {
        RuntimePlacementService::whole_work_runtime_binding_session_revision(
            &self.actual,
            program,
            binding,
            registration,
        )
    }

    fn select_whole_work(
        &self,
        request: WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, runmat_runtime::RuntimeError> {
        let selected = RuntimePlacementService::select_whole_work(&self.actual, request)?;
        self.decisions.borrow_mut().push(selected.decision.clone());
        Ok(selected)
    }

    fn observe_selected_whole_work(
        &self,
        selected: &RuntimeWholeWorkSelection,
        elapsed_ns: u64,
        outcome: WholeWorkObservationOutcome,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        RuntimePlacementService::observe_selected_whole_work(
            &self.actual,
            selected,
            elapsed_ns,
            outcome,
        )
    }

    fn plan(
        &self,
        request: PlacementPlanRequest,
    ) -> Result<PlacementDecision, runmat_runtime::RuntimeError> {
        RuntimePlacementService::plan(&self.actual, request)
    }

    fn observe(&self, feedback: PlacementFeedback) -> Result<(), runmat_runtime::RuntimeError> {
        RuntimePlacementService::observe(&self.actual, feedback)
    }

    fn invalidate(&self, invalidation: PlacementInvalidation) {
        RuntimePlacementService::invalidate(&self.actual, invalidation)
    }
}

thread_local! {
    static LINKED_SITES: RefCell<Vec<NativeSiteRequest>> = const { RefCell::new(Vec::new()) };
}

unsafe extern "C" fn linked_entry(
    call: *mut NativeCall,
    exit: *mut NativeExit,
) -> NativeHostStatus {
    if call.is_null() || exit.is_null() {
        return NativeHostStatus::INVALID_ARGUMENT;
    }
    // SAFETY: NativeExecutor calls this linked entrypoint with a checked
    // transaction, host table, and writable exit for the invocation lifetime.
    let host = unsafe { (*call).host.as_ref() };
    let Some(host) = host else {
        return NativeHostStatus::INVALID_ARGUMENT;
    };
    let Some(execute_site) = host.execute_site else {
        return NativeHostStatus::INVALID_ARGUMENT;
    };
    let sites = LINKED_SITES.with(|sites| sites.borrow().clone());
    for request in &sites {
        let mut outcome = NativeSiteOutcome::continue_execution();
        // SAFETY: every site was projected from this exact verified single
        // Native IR block, and all ABI pointers remain borrowed for this call.
        let status = unsafe { execute_site(host.context, call, request, &mut outcome, exit) };
        if status != NativeHostStatus::OK {
            return status;
        }
        if outcome.kind == NativeSiteOutcomeKind::EXIT {
            return NativeHostStatus::OK;
        }
        if outcome.kind != NativeSiteOutcomeKind::CONTINUE {
            return NativeHostStatus::HOST_FAILURE;
        }
    }
    NativeHostStatus::HOST_FAILURE
}

fn site_phase(phase: runmat_native_codegen::NativeSitePhase) -> NativeSitePhase {
    match phase {
        runmat_native_codegen::NativeSitePhase::Rvalue => NativeSitePhase::RVALUE,
        runmat_native_codegen::NativeSitePhase::Statement => NativeSitePhase::STATEMENT,
        runmat_native_codegen::NativeSitePhase::TerminatorRvalue => {
            NativeSitePhase::TERMINATOR_RVALUE
        }
        runmat_native_codegen::NativeSitePhase::Terminator => NativeSitePhase::TERMINATOR,
    }
}

static SELECTED_BINDING: OnceLock<runmat_runtime::builtin::RuntimeBuiltinBinding> = OnceLock::new();

unsafe extern "C" fn binding_resolver(index: u32) -> *const c_void {
    if index != 0 {
        return std::ptr::null();
    }
    SELECTED_BINDING.get().map_or(std::ptr::null(), |binding| {
        std::ptr::from_ref(binding).cast::<c_void>()
    })
}

#[test]
fn closed_world_native_callback_reuses_local_strcmp_feedback() {
    let mut compiler =
        runmat_core::RunMatSession::with_options(false, false).expect("Core source compiler");
    let source = runmat_core::ExecutableSource::new(
        "s1-aot-strcmp@1",
        "s1AotCompare.m",
        "function y = s1AotCompare(a,b)\ny = strcmp(a,b);\nend\n",
    );
    let unit = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(compiler.compile_executable_unit(source, None))
        .expect("compiled frontend product");
    let input = unit
        .prepare_native_compilation_for(Some("s1AotCompare"))
        .expect("exact native input");
    let entrypoint = input.entrypoint();
    let input = input
        .retain_functions(&BTreeSet::from([entrypoint]))
        .expect("retain exact selected function");
    let assembly = input
        .lower(runmat_native_codegen::NativeTarget::current())
        .expect("lower exact Native IR");
    assert_eq!(assembly.functions.len(), 1);
    let function = &assembly.functions[0];
    assert_eq!(
        function.blocks.len(),
        1,
        "test-linked driver only accepts one block"
    );
    let mut sites = function.expected_sites.clone();
    sites.sort_by_key(|site| (site.point, site.phase, site.ordinal));
    assert!(!sites.is_empty());
    assert!(sites
        .iter()
        .all(|site| site.point.function == function.portable_id));
    LINKED_SITES.with(|slot| {
        *slot.borrow_mut() = sites
            .iter()
            .map(|site| NativeSiteRequest {
                function: function.portable_id.0,
                block: site.point.block,
                position: site.point.position,
                phase: site_phase(site.phase),
                ordinal: site.ordinal,
                reserved: 0,
            })
            .collect();
    });

    let binding = runmat_runtime::builtin::runtime_builtin_bindings_by_name("strcmp")[0];
    SELECTED_BINDING.get_or_init(|| binding);
    let object_data = input
        .aot_object_data(
            &assembly,
            runmat_core::AotObjectDataOptions {
                clock_inputs: None,
                local_service_inputs: None,
                runtime_binding_mode:
                    runmat_native_codegen::aot::AotRuntimeBindingMode::ClosedWorld,
                builtin_bindings: vec![runmat_native_codegen::aot::AotBuiltinBinding {
                    name: binding.identity.builtin.name.to_owned(),
                    variant: binding.identity.variant.to_owned(),
                    native_symbol: runmat_builtins::native_binding_symbol(
                        binding.identity.builtin.name,
                        binding.identity.variant,
                    ),
                }],
                foreign_policy: None,
                native_interfaces: br#"{"schema_version":1,"interfaces":[]}"#.to_vec(),
                mex_artifacts: br#"{"schema_version":1,"artifacts":[]}"#.to_vec(),
                java_artifacts: br#"{"schema_version":1,"environment":null,"artifacts":[]}"#
                    .to_vec(),
                python_artifacts: br#"{"schema_version":1,"artifacts":[]}"#.to_vec(),
            },
        )
        .expect("exact closed-world object data");
    let manifest_bytes = &object_data
        .iter()
        .find(|data| data.symbol == runmat_native_codegen::aot::AOT_PROGRAM_SYMBOL)
        .expect("embedded program manifest")
        .bytes;
    let manifest =
        runmat_native_codegen::aot::AotProgramManifest::from_canonical_bytes(manifest_bytes)
            .expect("decode embedded closed-world program");
    let builtins = crate::builtin::resolve(&manifest, binding_resolver)
        .expect("resolve linked builtin")
        .expect("closed-world service");
    let placement = Rc::new(RecordingPlacement {
        actual: runmat_accelerate::placement::PlacementSession::default(),
        decisions: RefCell::new(Vec::new()),
    });
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_program_revision(Some(assembly.program.clone()));
    let runtime =
        runmat_runtime::source_context::bind_portable_source_map_v2(runtime, &manifest.source_map)
            .expect("bind exact source map");
    let runtime = runtime.with_service_ports(
        RuntimeServicePorts::default()
            .with_builtin(builtins)
            .with_placement(placement.clone()),
    );
    let executable = runmat_native_executor::NativeExecutable::linked(BTreeMap::from([(
        entrypoint,
        linked_entry as runmat_runtime::native::NativeEntryPoint,
    )]))
    .expect("link exact callback");
    let executor = Rc::new(
        runmat_native_executor::NativeExecutor::bind(
            assembly,
            executable,
            runmat_native_executor::NativeExecutorOptions {
                program_capture: Some(input.program_capture().to_vec()),
                interpreter_resume_points: input.interpreter_resume_points().clone(),
                ..Default::default()
            },
            runmat_native_codegen::NativeLivenessAdmission::capture(&runtime)
                .expect("test liveness admission"),
        )
        .expect("bind Native IR and callback"),
    );
    let async_runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    for call in 0..3 {
        let output = async_runtime
            .block_on(runtime.clone().scope(crate::program::invoke(
                executor.clone(),
                &manifest,
                entrypoint,
                vec![Value::String("same".into()), Value::String("same".into())],
                1,
                runtime.clone(),
            )))
            .expect("AOT callback invocation");
        assert_eq!(output.outputs, vec![Value::Bool(true)]);
        assert_eq!(placement.decisions.borrow().len(), call + 1);
    }
    let decisions = placement.decisions.borrow();
    let first = &decisions[0];
    assert_eq!(
        first.kind,
        runmat_execution::WholeWorkDecisionKind::Exploratory
    );
    assert!(matches!(
        placement.actual.observed_candidate_duration(
            &first.signature,
            &first.selected.candidate,
            first.selected.kind,
        ),
        CandidateObservationEvidence::Observed(_)
    ));
    assert_eq!(decisions[2].signature, first.signature);
    assert!(decisions[2].predicted_total_ns.is_some());
}
