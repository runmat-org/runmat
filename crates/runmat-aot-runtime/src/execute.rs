use std::{collections::BTreeMap, rc::Rc};

use runmat_execution::ExecutableEntrypointKind;
use runmat_execution_artifact::target_test::{
    NativePairedProductHeader, NativeTargetTestFailure, NativeTargetTestRequest,
    NativeTargetTestResult, NativeTargetTestStatus,
};
use runmat_native_executor::execute::NativeWorkspaceInput;
use runmat_types::ProgramPointId;

use crate::{
    input::{LinkedAotProcessInput, LinkedAotProcessParts},
    AotProcessInput,
};

#[path = "execute/acceleration.rs"]
mod acceleration;
mod failure;
#[path = "execute/linked_selected.rs"]
mod linked_selected;
#[path = "execute/target_test.rs"]
mod target_test;
pub(crate) use failure::ProcessFailure;

enum InvocationMode<'a> {
    Application,
    Test(target_test::TestInvocation<'a>),
}

impl InvocationMode<'_> {
    fn is_test(&self) -> bool {
        matches!(self, Self::Test(_))
    }

    fn test(&self) -> Option<&target_test::TestInvocation<'_>> {
        match self {
            Self::Application => None,
            Self::Test(test) => Some(test),
        }
    }
}

pub fn execute(input: AotProcessInput) -> Result<(), String> {
    execute_with_linked_authority(
        input,
        false,
        std::ptr::null(),
        std::ptr::null(),
        crate::input::LinkedExternalFactories::absent(),
        None,
    )
    .map(|_| ())
    .map_err(|error| error.to_string())
}

pub(crate) fn execute_linked(input: LinkedAotProcessInput) -> Result<(), ProcessFailure> {
    let LinkedAotProcessParts {
        input,
        source_service_factory,
        network_service_factory,
        external_factories,
    } = input.into_parts();
    execute_with_linked_authority(
        input,
        true,
        source_service_factory,
        network_service_factory,
        external_factories,
        None,
    )
    .map(|_| ())
}

pub(crate) fn execute_linked_test(
    input: LinkedAotProcessInput,
    header: &NativePairedProductHeader,
    request: &NativeTargetTestRequest,
) -> Result<NativeTargetTestResult, String> {
    let LinkedAotProcessParts {
        input,
        source_service_factory,
        network_service_factory,
        external_factories,
    } = input.into_parts();
    execute_with_linked_authority(
        input,
        true,
        source_service_factory,
        network_service_factory,
        external_factories,
        Some((header, request)),
    )
    .map_err(|error| error.to_string())?
    .ok_or_else(|| "target-test execution omitted its receiver result".into())
}

fn execute_with_linked_authority(
    input: AotProcessInput,
    verified_link: bool,
    source_service_factory: *const std::ffi::c_void,
    network_service_factory: *const std::ffi::c_void,
    external_factories: crate::input::LinkedExternalFactories,
    test: Option<(&NativePairedProductHeader, &NativeTargetTestRequest)>,
) -> Result<Option<NativeTargetTestResult>, ProcessFailure> {
    let _console_forwarder = if test.is_some() {
        crate::output::install_test_console_forwarder()
    } else {
        crate::output::install_console_forwarder()
    };
    let assembly: runmat_native_codegen::NativeAssembly = serde_json::from_slice(&input.native_ir)
        .map_err(|error| format!("standalone Native IR is invalid: {error}"))?;
    assembly
        .verify()
        .map_err(|error| format!("standalone Native IR failed verification: {error}"))?;
    assembly
        .target
        .require_host_executable()
        .map_err(|error| format!("standalone target is not executable by this runtime: {error}"))?;
    let interop = &assembly.requirements.interop;
    let program =
        runmat_native_codegen::aot::AotProgramManifest::from_canonical_bytes(&input.program)
            .map_err(|error| format!("standalone program manifest is invalid: {error}"))?;
    if program.executable != assembly.executable_identity
        || program.native_ir_digest != runmat_execution::Digest::sha256(&input.native_ir)
        || assembly.selected_entrypoint != program.entrypoint
    {
        return Err("standalone program manifest does not match its Native IR".into());
    }
    let mode = if let Some((header, request)) = test {
        InvocationMode::Test(target_test::TestInvocation::select(
            header, request, &assembly, &program,
        )?)
    } else {
        InvocationMode::Application
    };
    let linked_selected = linked_selected::admit(
        &program,
        &assembly,
        verified_link,
        source_service_factory,
        network_service_factory,
        external_factories,
    )?;
    let local_service_inputs = mode.test().map_or(&program.local_service_inputs, |test| {
        &test.request.local_service_inputs
    });
    let selected_invocation = linked_selected
        .as_ref()
        .map(|_| match mode.test() {
            Some(test) => test
                .request
                .digest_for(&test.header.binding)
                .map_err(|error| error.to_string()),
            None => Ok(runmat_execution::Digest::sha256(&input.program)),
        })
        .transpose()?;
    if program.foreign_policy.is_none()
        && (!interop.foreign_types.is_empty()
            || !interop.adapters.is_empty()
            || !interop.adapter_contracts.is_empty())
    {
        return Err("standalone foreign program has no captured policy".into());
    }
    program
        .validate_function_inventory(&assembly)
        .map_err(|error| error.to_string())?;
    let ordered_resume_points: Vec<(ProgramPointId, u64)> =
        serde_json::from_slice(&input.resume_points)
            .map_err(|error| format!("standalone resume-point map is invalid: {error}"))?;
    let resume_point_count = ordered_resume_points.len();
    let resume_points = ordered_resume_points
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    if resume_points.len() != resume_point_count {
        return Err("standalone resume-point map contains duplicate identities".into());
    }
    let entrypoint = mode.test().map_or(program.entrypoint, |test| test.function);
    let function = assembly
        .functions
        .iter()
        .find(|function| function.id == entrypoint)
        .ok_or_else(|| "standalone entrypoint is absent from Native IR".to_string())?;
    if !mode.is_test() && (!function.abi.fixed_inputs.is_empty() || function.abi.varargin.is_some())
    {
        return Err("standalone entrypoint requires unsupported command-line inputs".into());
    }
    acceleration::initialize_for_program(&program, &assembly, linked_selected.as_ref())?;
    let requested_outputs = function.abi.fixed_outputs.len();
    let native_id = runmat_types::ForeignAdapterId::new(runmat_native_ffi::NATIVE_FFI_ADAPTER_ID)
        .map_err(|error| error.to_string())?;
    let mex_id = runmat_types::ForeignAdapterId::new(runmat_mex::MEX_ADAPTER_ID)
        .map_err(|error| error.to_string())?;
    let java_id = runmat_types::ForeignAdapterId::new(runmat_java::JAVA_ADAPTER_ID)
        .map_err(|error| error.to_string())?;
    let python_id = runmat_types::ForeignAdapterId::new(runmat_python::PYTHON_ADAPTER_ID)
        .map_err(|error| error.to_string())?;
    let mut policy = crate::foreign_policy::AdmittedForeignPolicy::admit(
        program.foreign_policy.as_ref(),
        interop,
        &input,
        &native_id,
        &mex_id,
        &java_id,
        &python_id,
    )?;
    let async_runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("standalone async runtime initialization failed: {error}"))?;
    let mut runtime = runmat_runtime::source_context::bind_portable_source_map_v2(
        standalone_runtime(
            &assembly.program,
            linked_selected
                .as_ref()
                .map(|selected| selected.product.binding()),
            local_service_inputs.as_ref(),
        )?,
        &program.source_map,
    )
    .map_err(|error| format!("standalone source binding failed: {error}"))?;
    let mut _external_host_owners = Vec::new();
    if let Some(selected) = &linked_selected {
        let issuers = selected
            .services
            .iter()
            .filter(|issuer| issuer.binding().kind.is_local_effect_service())
            .cloned()
            .collect();
        runtime = crate::local_service::install(
            runtime,
            selected.product.binding(),
            selected_invocation.expect("selected invocation was derived"),
            issuers,
            local_service_inputs,
        )?;
        if let Some(random) = local_service_inputs
            .as_ref()
            .and_then(|inputs| inputs.random.as_ref())
        {
            runtime
                .install_selected_random_inputs(random)
                .map_err(|error| format!("selected random stream installation failed: {error}"))?;
        }
        if let Some(issuer) = selected.services.iter().find(|issuer| {
            issuer.binding().kind
                == runmat_execution::host::NativeDeploymentServiceKind::Effect(
                    runmat_types::EffectKind::Network,
                )
        }) {
            runtime = crate::network_service::install(
                runtime,
                selected.product.binding(),
                selected_invocation.expect("selected invocation was derived"),
                issuer,
                network_service_factory,
                async_runtime.handle(),
            )?;
        }
    }
    let test_ledger = mode.test().map(|test| {
        let (managed, ledger) = test.install_resources(runtime.clone());
        runtime = managed;
        ledger
    });
    let local_tasks = Rc::new(runmat_runtime::context::RuntimeLocalTaskSupervisor::new());
    let ports = runtime
        .service_ports()
        .clone()
        .with_local_tasks(local_tasks.submission_port(runtime.local_identity()))
        .with_call(Rc::new(runmat_runtime::context::RuntimeCallRouter));
    runtime = runtime.with_service_ports(ports);
    // Application inputs are authenticated by the linked program bytes;
    // tests bind their explicit override to the full per-attempt request.
    let clock_inputs = mode.test().map_or_else(
        || program.clock_inputs.clone(),
        |test| test.request.clock_inputs.clone(),
    );
    let clock_issuer = linked_selected.as_ref().and_then(|selected| {
        selected.services.iter().find(|issuer| {
            issuer.binding().kind
                == runmat_execution::host::NativeDeploymentServiceKind::Effect(
                    runmat_types::EffectKind::Clock,
                )
        })
    });
    let clock_invocation = if let Some(issuer) = clock_issuer {
        let invocation = selected_invocation.expect("selected clock has an invocation");
        let identity = runmat_execution::host::NativeClockInvocationIdentity::new(
            invocation,
            clock_inputs.as_ref(),
        )
        .map_err(|error| error.to_string())?;
        runtime =
            crate::clock_service::install(runtime, issuer.clone(), identity.clone(), clock_inputs)?;
        Some(identity)
    } else {
        None
    };
    #[cfg(feature = "source-evaluation")]
    let (mut runtime, source_evaluation_owner) = if let Some(issuer) =
        linked_selected.as_ref().and_then(|selected| {
            selected
                .services
                .iter()
                .find(|issuer| {
                    issuer.binding().kind
                        == runmat_execution::host::NativeDeploymentServiceKind::SourceEvaluation
                })
                .cloned()
        }) {
        let (runtime, owner) = crate::source_service::install(runtime, issuer)?;
        (runtime, Some(owner))
    } else {
        (runtime, None)
    };
    let lifecycle_host =
        runmat_runtime::context::RuntimeLifecycleHost::new(&runtime, local_tasks.clone())
            .map_err(|error| format!("standalone cleanup lifecycle host failed: {error}"))?;
    if let Some(builtins) = crate::builtin::resolve(&program, input.builtin_resolver)? {
        let ports = runtime.service_ports().clone().with_builtin(builtins);
        runtime = runtime.with_service_ports(ports);
    }
    if let Some(selected) = &linked_selected {
        let invocation = selected_invocation.expect("selected invocation was derived");
        if let Some(issuer) = selected.services.iter().find(|issuer| {
            issuer.binding().kind
                == runmat_execution::host::NativeDeploymentServiceKind::Effect(
                    runmat_types::EffectKind::UserInterface,
                )
        }) {
            let (installed, owner) = crate::external_service::install_host(
                runtime,
                selected.product.binding(),
                invocation,
                issuer,
                external_factories.ui,
            )?;
            runtime = installed;
            _external_host_owners.push(owner);
        }
        if let Some(issuer) = selected.services.iter().find(|issuer| {
            issuer.binding().kind
                == runmat_execution::host::NativeDeploymentServiceKind::Effect(
                    runmat_types::EffectKind::HostCallback,
                )
        }) {
            runtime = crate::external_service::install_callback(
                runtime,
                selected.product.binding(),
                invocation,
                issuer,
                external_factories.callback,
            )?;
        }
        if let Some(issuer) = selected.services.iter().find(|issuer| {
            issuer.binding().kind
                == runmat_execution::host::NativeDeploymentServiceKind::HostedDynamicDispatch
        }) {
            runtime = crate::external_service::install_dynamic(
                runtime,
                selected.product.binding(),
                invocation,
                issuer,
                external_factories.dynamic,
            )?;
        }
    }
    let foreign = Rc::new(runmat_runtime::foreign::ForeignRuntime::new(
        runmat_runtime::foreign::ForeignPlatform::Native,
    ));
    let native_ffi = runmat_runtime::foreign::NativeFfiAdapter::new(foreign.handles().clone())
        .map_err(|error| error.to_string())?;
    let java = runmat_runtime::foreign::JavaAdapter::new(foreign.handles().clone())
        .map_err(|error| error.to_string())?;
    let python = runmat_runtime::foreign::PythonAdapter::new(foreign.handles().clone())
        .map_err(|error| error.to_string())?;
    let mex = Rc::new(runmat_runtime::foreign::MexRuntimeSession::new());
    foreign
        .register_adapter(native_ffi.clone())
        .map_err(|error| error.to_string())?;
    foreign
        .register_adapter(java.clone())
        .map_err(|error| error.to_string())?;
    foreign
        .register_adapter(python.clone())
        .map_err(|error| error.to_string())?;
    foreign
        .register_adapter(mex.clone())
        .map_err(|error| error.to_string())?;
    policy.apply(&native_ffi, &mex, &java, &python)?;
    let _native_interfaces =
        crate::native_interfaces::install_admitted(&policy.native_bundle, &native_ffi)?;
    let _mex_artifacts = crate::mex_artifacts::install_admitted(&policy.mex_bundle, &mex)?;
    let _java_artifacts = crate::java_artifacts::install_admitted(&policy.java_bundle, &java)?;
    crate::python_artifacts::install_admitted(&policy.python_bundle, &python)?;
    foreign
        .admit(&assembly.requirements.interop)
        .map_err(|error| format!("standalone interop admission failed: {error}"))?;
    let ports = runtime.service_ports().clone().with_foreign(foreign);
    runtime = runtime.with_service_ports(ports);
    let mut mex_guard = crate::mex::install(&runtime, mex, local_tasks.clone())?;
    let mut entrypoints = BTreeMap::new();
    for function in &assembly.functions {
        let function_index = u32::try_from(function.id.0).map_err(|_| {
            format!(
                "compiled function {} exceeds the standalone resolver ABI",
                function.id.0
            )
        })?;
        // SAFETY: the generated resolver has the exact declared C ABI. It
        // returns only addresses of functions emitted from this same verified
        // assembly, or null when the object is inconsistent.
        let address = unsafe { (input.function_resolver)(function_index) };
        if address.is_null() {
            return Err(format!(
                "standalone object cannot resolve native function {}",
                function.id.0
            )
            .into());
        }
        // SAFETY: every resolver result names a function lowered with
        // Runtime's exact NativeEntryPoint signature.
        let entrypoint = unsafe {
            std::mem::transmute::<*const std::ffi::c_void, runmat_runtime::native::NativeEntryPoint>(
                address,
            )
        };
        entrypoints.insert(function.id, entrypoint);
    }
    let executable = runmat_native_executor::NativeExecutable::linked(entrypoints)
        .map_err(|error| format!("standalone native entrypoints are invalid: {error}"))?;
    let options = runmat_native_executor::NativeExecutorOptions {
        selected_clock_invocation: clock_invocation,
        selected_local_invocation: selected_invocation,
        program_capture: Some(input.program.clone()),
        interpreter_resume_points: resume_points,
        ..runmat_native_executor::NativeExecutorOptions::default()
    };
    let liveness_admission = runmat_native_codegen::NativeLivenessAdmission::capture(&runtime)
        .map_err(|source| {
            runmat_runtime::context::host_allocation::host_allocation_runtime_error(
                Some(&runtime),
                "RunMat:NativeLivenessAdmission",
                "standalone native liveness metadata admission failed",
                source,
            )
        })?;
    let executor = Rc::new(
        match &linked_selected {
            Some(linked) => runmat_native_executor::NativeExecutor::bind_linked_selected(
                assembly.clone(),
                executable,
                options,
                &linked.product,
                linked.services.clone(),
                liveness_admission,
            ),
            None => runmat_native_executor::NativeExecutor::bind(
                assembly.clone(),
                executable,
                options,
                liveness_admission,
            ),
        }
        .map_err(|error| ProcessFailure::from(error.into_runtime_error()))?,
    );

    // Keep the original session owner through foreign teardown and terminal
    // rendering. ContextFuture performs the existing nonblocking final drain
    // after inner destruction on success, error, or abandonment.
    let started = std::time::Instant::now();
    let result = async_runtime.block_on(local_tasks.drive_until(runtime.clone().scope(async {
        let execution = if let Some(test) = mode.test() {
            test.invoke(
                Rc::clone(&executor),
                &program,
                requested_outputs,
                runtime.clone(),
            )
            .await
        } else if assembly.executable_identity.entrypoint_kind == ExecutableEntrypointKind::Script {
            let local_names = function
                .locals
                .iter()
                .filter_map(|local| local.binding.zip(local.name.clone()))
                .collect();
            match NativeWorkspaceInput::empty_for_context(&runtime, local_names) {
                Ok(workspace) => {
                    crate::program::invoke_workspace(
                        Rc::clone(&executor),
                        &program,
                        entrypoint,
                        workspace,
                        requested_outputs,
                        runtime.clone(),
                    )
                    .await
                }
                Err(error) => Err(*error),
            }
        } else {
            crate::program::invoke(
                Rc::clone(&executor),
                &program,
                entrypoint,
                Vec::new(),
                requested_outputs,
                runtime.clone(),
            )
            .await
        };
        // The native producer retains result roots through cleanup callbacks
        // and terminal rendering; no second result lease is needed here.
        // Cleanup callbacks may still call MEX or native foreign services.
        // Drain them before retiring those adapters, including on execution
        // failure. Teardown itself still runs after any cleanup error.
        let cleanup_lifecycle = match lifecycle_host.service_safepoint(&runtime) {
            Ok(()) => lifecycle_host
                .drain_gracefully(&runtime)
                .await
                .map_err(|error| format!("standalone cleanup callback failed: {error}")),
            Err(error) => Err(format!("standalone cleanup safepoint failed: {error}")),
        };
        let execution_failure = match execution {
            Ok(value) => {
                value.with_owned(|value| {
                    if !mode.is_test() && cleanup_lifecycle.is_ok() {
                        if let Some(expression) = value.expression {
                            crate::output::value(&expression);
                        } else {
                            for output in value.outputs {
                                crate::output::value(&output);
                            }
                        }
                    }
                });
                None
            }
            Err(error) => Some((
                target_test::classify_failure(&error),
                execution_diagnostic(error, &runtime),
            )),
        };
        // Retire caller-visible output ownership and admit any final callbacks
        // before foreign adapters are closed.
        let final_cleanup = lifecycle_host
            .drain_gracefully(&runtime)
            .await
            .map_err(|error| format!("standalone final cleanup failed: {error}"));
        let foreign_lifecycle = crate::foreign_retirement::retire(
            &runtime,
            &mut mex_guard,
            native_ffi.as_ref(),
            python.as_ref(),
            java.as_ref(),
        )
        .await;
        if let Err(error) = &cleanup_lifecycle {
            crate::output::error(&format!("standalone cleanup callback failed: {error}"));
        }
        if let Err(error) = &final_cleanup {
            crate::output::error(&format!("standalone final cleanup failed: {error}"));
        }
        if let Some(error) = foreign_lifecycle.error() {
            crate::output::error(&format!("standalone foreign shutdown failed: {error}"));
        }
        if mode.is_test() {
            let mut failures = Vec::new();
            if let Err(error) = cleanup_lifecycle {
                failures.push(error);
            }
            if let Err(error) = final_cleanup {
                failures.push(error);
            }
            if let Err(error) = foreign_lifecycle.into_result() {
                failures.push(format!("standalone foreign shutdown failed: {error}"));
            }
            if !failures.is_empty() {
                if let Some((_, diagnostic)) = execution_failure {
                    failures.insert(0, diagnostic);
                }
                return Ok(Some((
                    NativeTargetTestStatus::Failed(NativeTargetTestFailure::ProcessFailure),
                    failures.join("\n"),
                )));
            }
            return Ok(Some(match execution_failure {
                Some((failure, diagnostic)) => {
                    (NativeTargetTestStatus::Failed(failure), diagnostic)
                }
                None => (NativeTargetTestStatus::Passed, String::new()),
            }));
        }
        if let Some((_, diagnostic)) = execution_failure {
            return Err(ProcessFailure::Runtime(diagnostic));
        }
        cleanup_lifecycle.map_err(ProcessFailure::Runtime)?;
        final_cleanup.map_err(ProcessFailure::Runtime)?;
        foreign_lifecycle.into_result().map_err(|error| {
            ProcessFailure::Runtime(format!("standalone foreign shutdown failed: {error}"))
        })?;
        Ok(None)
    })));
    // A rejected graceful close is Abandoned, not a reason to close the
    // callback driver while worker requests still have origin authority.
    mex_guard.force_abandon_if_needed();
    lifecycle_host.force_close();
    let execution_result = result
        .map_err(|error| {
            ProcessFailure::Runtime(format!("standalone local task driver failed: {error}"))
        })
        .and_then(|result| result);
    #[cfg(feature = "source-evaluation")]
    let source_retirement = source_evaluation_owner
        .map(|owner| {
            owner
                .close()
                .map_err(|error| format!("standalone source evaluator retirement failed: {error}"))
        })
        .transpose();
    let completion = execution_result?;
    #[cfg(feature = "source-evaluation")]
    let mut completion = completion;
    #[cfg(feature = "source-evaluation")]
    if let Err(error) = source_retirement {
        if let Some((status, diagnostic)) = completion.as_mut() {
            *status = NativeTargetTestStatus::Failed(NativeTargetTestFailure::ProcessFailure);
            if !diagnostic.is_empty() {
                diagnostic.push('\n');
            }
            diagnostic.push_str(&error);
        } else {
            return Err(ProcessFailure::Runtime(error));
        }
    }
    if let Some(test) = mode.test() {
        let completion =
            completion.ok_or_else(|| "target-test completion is absent".to_string())?;
        let ledger = test_ledger
            .as_ref()
            .ok_or_else(|| "target-test managed ledger is absent".to_string())?;
        return test
            .result(completion, ledger, started)
            .map(Some)
            .map_err(ProcessFailure::from);
    }
    Ok(None)
}

fn execution_diagnostic(
    error: runmat_runtime::RuntimeError,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> String {
    let mut rendered = match error.identifier() {
        Some(identifier) => format!(
            "standalone native execution failed: {identifier}: {}",
            error.message
        ),
        None => format!("standalone native execution failed: {error}"),
    };
    if let Some(span) = error.span {
        rendered.push_str(&format!(
            " [source span {}..{}]",
            span.offset(),
            span.offset().saturating_add(span.len())
        ));
    }
    let frames = if error.context.call_stack.is_empty() {
        runmat_runtime::execution::render_source_call_stack_in_context(&error, runtime)
    } else {
        error.context.call_stack.clone()
    };
    for frame in frames {
        rendered.push_str("\n  at ");
        rendered.push_str(&frame);
    }
    let mut source = std::error::Error::source(&error);
    while let Some(cause) = source {
        rendered.push_str("\n  caused by: ");
        rendered.push_str(&cause.to_string());
        source = cause.source();
    }
    rendered
}

/// Initialize language policy from the verified program, independent of the
/// launching process's configuration or an embedding caller's active context.
fn standalone_runtime(
    program: &runmat_execution::ProgramRevision,
    selected: Option<&runmat_execution::host::NativeSelectedProductBinding>,
    local_inputs: Option<&runmat_execution::host::NativeLocalServiceInputs>,
) -> Result<runmat_runtime::context::RuntimeContext, String> {
    use runmat_execution::LanguageCompatibilityMode;
    use runmat_runtime::context::{RuntimeContext, RuntimeLanguageMode};

    let execution = Rc::new(runmat_runtime::execution::RuntimeExecutionService::new());
    // Prefer the current embedding/host authority. Otherwise install only a
    // lazy native adapter: plain programs (including untaken FEA branches)
    // never read credentials or contact Server. Each invocation gets fresh
    // selector capture rather than pinning a previous invocation's account.
    let fea_admission =
        match runmat_runtime::analysis::fea_run_admission::capture_fea_run_admission() {
            Ok(originating) => originating,
            Err(runmat_runtime::analysis::fea_run_admission::FeaRunDenied::SignInRequired) => {
                std::sync::Arc::new(runmat_fea_admission_native::NativeFeaAdmission::compiled(
                    None,
                ))
            }
            Err(_) => std::sync::Arc::new(
                runmat_fea_admission_native::NativeFeaAdmission::from_selection(
                    runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization::Invalid,
                ),
            ),
        };
    let runtime = if let Some(binding) = selected {
        let environment_read_selected = binding.services.iter().any(|service| {
            service.kind
                == runmat_execution::host::NativeDeploymentServiceKind::Effect(
                    runmat_types::EffectKind::EnvironmentRead,
                )
        });
        let hosted =
            binding.profile == runmat_execution::host::NativeDeploymentProfile::HostedNative;
        let hosted_clock = hosted
            && binding.services.iter().any(|service| {
                service.kind
                    == runmat_execution::host::NativeDeploymentServiceKind::Effect(
                        runmat_types::EffectKind::Clock,
                    )
            });
        RuntimeContext::selected_native_base(
            execution,
            local_inputs
                .and_then(|inputs| inputs.environment.as_ref())
                .map(|environment| &environment.variables),
            hosted && environment_read_selected,
            hosted_clock,
        )
        .map_err(|error| format!("selected native environment construction failed: {error}"))?
    } else {
        RuntimeContext::new(execution)
    }
    .with_program_revision(Some(program.clone()));
    let mode = match program.compatibility_mode() {
        LanguageCompatibilityMode::RunMat => RuntimeLanguageMode::RunMat,
        LanguageCompatibilityMode::Matlab => RuntimeLanguageMode::Matlab,
        LanguageCompatibilityMode::Strict => RuntimeLanguageMode::Strict,
    };
    runtime.set_language_mode(mode);
    runtime.set_runmat_extensions_enabled(matches!(mode, RuntimeLanguageMode::RunMat));
    let ports = runtime
        .service_ports()
        .clone()
        .with_sync_fea_run_admission(fea_admission);
    let runtime = runtime.with_service_ports(ports);
    Ok(crate::program::install_acceleration_services(runtime))
}

#[cfg(test)]
mod tests;
