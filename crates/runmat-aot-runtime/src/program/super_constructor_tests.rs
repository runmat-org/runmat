//! A linked AOT entry uses the same retained class body and session object.

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
    sync::Mutex,
};

use runmat_runtime::native::{
    NativeCall, NativeExit, NativeHostStatus, NativeSiteOutcome, NativeSiteOutcomeKind,
    NativeSitePhase, NativeSiteRequest,
};
use runmat_value::Value;

static CWD_LOCK: Mutex<()> = Mutex::new(());
thread_local! {
    static SITES: RefCell<BTreeMap<u32, Vec<NativeSiteRequest>>> = const { RefCell::new(BTreeMap::new()) };
}

unsafe extern "C" fn linked_entry(
    call: *mut NativeCall,
    exit: *mut NativeExit,
) -> NativeHostStatus {
    if call.is_null() || exit.is_null() {
        return NativeHostStatus::INVALID_ARGUMENT;
    }
    // SAFETY: the checked NativeExecutor transaction owns the frame, resume
    // state, host table, and exit for this callback's lifetime.
    let Some(frame) = (unsafe { (*call).frame.as_ref() }) else {
        return NativeHostStatus::INVALID_ARGUMENT;
    };
    let Some(resume) = (unsafe { frame.resume.as_ref() }) else {
        return NativeHostStatus::INVALID_ARGUMENT;
    };
    let Some(host) = (unsafe { (*call).host.as_ref() }) else {
        return NativeHostStatus::INVALID_ARGUMENT;
    };
    let Some(execute_site) = host.execute_site else {
        return NativeHostStatus::INVALID_ARGUMENT;
    };
    let Some(sites) = SITES.with(|sites| sites.borrow().get(&resume.function).cloned()) else {
        return NativeHostStatus::HOST_FAILURE;
    };
    for request in &sites {
        let mut outcome = NativeSiteOutcome::continue_execution();
        // SAFETY: every request was projected from this exact verified function.
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

fn phase(phase: runmat_native_codegen::NativeSitePhase) -> NativeSitePhase {
    match phase {
        runmat_native_codegen::NativeSitePhase::Rvalue => NativeSitePhase::RVALUE,
        runmat_native_codegen::NativeSitePhase::Statement => NativeSitePhase::STATEMENT,
        runmat_native_codegen::NativeSitePhase::TerminatorRvalue => {
            NativeSitePhase::TERMINATOR_RVALUE
        }
        runmat_native_codegen::NativeSitePhase::Terminator => NativeSitePhase::TERMINATOR,
    }
}

#[test]
fn standalone_derived_constructor_runs_retained_base_on_same_handle() {
    runmat_gc::gc_test_context(|| {
        let _cwd_lock = CWD_LOCK.lock().expect("cwd lock");
        let directory = tempfile::TempDir::new().expect("class source directory");
        std::fs::write(directory.path().join("AotBase.m"), "classdef AotBase < handle\n properties\n baseValue\n end\n methods\n function obj = AotBase(v)\n global aot_base_instance;\n aot_base_instance = obj;\n obj.baseValue = v;\n end\n end\nend").expect("base class");
        std::fs::write(directory.path().join("AotDerived.m"), "classdef AotDerived < AotBase\n properties\n childValue\n end\n methods\n function obj = AotDerived(v)\n obj@AotBase(v);\n obj.childValue = v;\n end\n end\nend").expect("derived class");
        let prior_cwd = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(directory.path()).expect("set fixture cwd");
        struct RestoreCwd(std::path::PathBuf);
        impl Drop for RestoreCwd {
            fn drop(&mut self) {
                std::env::set_current_dir(&self.0).expect("restore cwd");
            }
        }
        let _restore = RestoreCwd(prior_cwd);
        let driver = "function obj = makeAotDerived()\n obj = AotDerived(7);\nend";
        std::fs::write(directory.path().join("makeAotDerived.m"), driver).expect("driver");
        let async_runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        let mut session =
            runmat_core::RunMatSession::with_options(false, false).expect("Core compiler");
        let unit = async_runtime
            .block_on(session.compile_executable_unit(
                runmat_core::ExecutableSource::frozen(
                    "aot-super@1",
                    "makeAotDerived.m",
                    directory.path().join("makeAotDerived.m").to_string_lossy(),
                    driver,
                ),
                None,
            ))
            .expect("compile class product");
        let input = unit
            .prepare_native_compilation_for(Some("makeAotDerived"))
            .expect("native input");
        let entry = input.entrypoint();
        let retained = unit
            .reachability_analysis()
            .1
            .into_iter()
            .collect::<BTreeSet<_>>();
        let input = input.retain_functions(&retained).expect("retained closure");
        let assembly = input
            .lower(runmat_native_codegen::NativeTarget::current())
            .expect("Native IR");
        let derived = assembly
            .class_declarations
            .iter()
            .find(|class| class.class.display_name() == "AotDerived")
            .expect("derived class");
        let base = assembly
            .class_declarations
            .iter()
            .find(|class| class.class.display_name() == "AotBase")
            .expect("base class");
        assert_eq!(derived.parent.as_ref(), Some(&base.class));
        assert!(derived.constructor.is_some() && base.constructor.is_some());
        let mut missing_base = assembly.clone();
        missing_base
            .class_declarations
            .retain(|class| class.class != base.class);
        assert_eq!(
            missing_base.verify().unwrap_err().code,
            "native.ir.super_constructor",
            "a custom superclass cannot fall back to a mutable registry name"
        );
        assert!(base
            .properties
            .iter()
            .any(|property| property.name.display_name() == "baseValue"));
        assert!(derived
            .properties
            .iter()
            .any(|property| property.name.display_name() == "childValue"));
        let object_data = input
            .aot_object_data(
                &assembly,
                runmat_core::AotObjectDataOptions {
                    clock_inputs: None,
                    local_service_inputs: None,
                    foreign_policy: None,
                    runtime_binding_mode:
                        runmat_native_codegen::aot::AotRuntimeBindingMode::ClosedWorld,
                    builtin_bindings: Vec::new(),
                    native_interfaces: br#"{"schema_version":1,"interfaces":[]}"#.to_vec(),
                    mex_artifacts: br#"{"schema_version":1,"artifacts":[]}"#.to_vec(),
                    java_artifacts: br#"{"schema_version":1,"environment":null,"artifacts":[]}"#
                        .to_vec(),
                    python_artifacts: br#"{"schema_version":1,"artifacts":[]}"#.to_vec(),
                },
            )
            .expect("AOT product");
        let manifest = runmat_native_codegen::aot::AotProgramManifest::from_canonical_bytes(
            &object_data
                .iter()
                .find(|data| data.symbol == runmat_native_codegen::aot::AOT_PROGRAM_SYMBOL)
                .expect("manifest")
                .bytes,
        )
        .expect("decode manifest");
        SITES.with(|slot| {
            *slot.borrow_mut() = assembly
                .functions
                .iter()
                .map(|function| {
                    assert_eq!(
                        function.blocks.len(),
                        1,
                        "linked test driver admits one block per function"
                    );
                    let mut sites = function.expected_sites.clone();
                    sites.sort_by_key(|site| (site.point, site.phase, site.ordinal));
                    assert!(sites
                        .iter()
                        .all(|site| site.point.function == function.portable_id));
                    (
                        function.portable_id.0,
                        sites
                            .into_iter()
                            .map(|site| NativeSiteRequest {
                                function: function.portable_id.0,
                                block: site.point.block,
                                position: site.point.position,
                                phase: phase(site.phase),
                                ordinal: site.ordinal,
                                reserved: 0,
                            })
                            .collect(),
                    )
                })
                .collect();
        });
        let executable = runmat_native_executor::NativeExecutable::linked(
            assembly
                .functions
                .iter()
                .map(|function| {
                    (
                        function.id,
                        linked_entry as runmat_runtime::native::NativeEntryPoint,
                    )
                })
                .collect(),
        )
        .expect("linked callbacks");
        let executor = Rc::new(
            runmat_native_executor::NativeExecutor::bind(
                assembly.clone(),
                executable,
                runmat_native_executor::NativeExecutorOptions {
                    program_capture: Some(input.program_capture().to_vec()),
                    interpreter_resume_points: input.interpreter_resume_points().clone(),
                    ..Default::default()
                },
                runmat_native_codegen::NativeLivenessAdmission::AuthorizedUnbounded,
            )
            .expect("bind Native IR"),
        );
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_program_revision(Some(assembly.program.clone()));
        let runtime = runmat_runtime::source_context::bind_portable_source_map_v2(
            runtime,
            &manifest.source_map,
        )
        .expect("source leases");
        let local_tasks = Rc::new(runmat_runtime::context::RuntimeLocalTaskSupervisor::new());
        let ports = runtime
            .service_ports()
            .clone()
            .with_local_tasks(local_tasks.submission_port(runtime.local_identity()));
        let runtime = runtime.with_service_ports(ports);
        let output = async_runtime
            .block_on(local_tasks.drive_until(runtime.clone().scope(super::invoke(
                executor,
                &manifest,
                entry,
                Vec::new(),
                1,
                runtime.clone(),
            ))))
            .expect("standalone local task host")
            .expect("standalone constructor");
        let [Value::HandleObject(instance)] = output.outputs.as_slice() else {
            panic!("expected managed derived handle")
        };
        assert_eq!(instance.class_name.display_name(), "AotDerived");
        let Some(Value::HandleObject(observed)) = runtime
            .session_global_value("aot_base_instance")
            .expect("session global")
        else {
            panic!("retained Base body did not execute")
        };
        assert_eq!(instance.target, observed.target);
        let Some(Value::Object(properties)) = runmat_gc::gc_clone_value(&instance.target).ok()
        else {
            panic!("returned handle target is not live")
        };
        assert_eq!(
            properties.properties().get("baseValue"),
            Some(&Value::Num(7.0))
        );
        assert_eq!(
            properties.properties().get("childValue"),
            Some(&Value::Num(7.0))
        );
    });
}
