use std::{collections::BTreeMap, rc::Rc};

use runmat_execution::ExecutableEntrypointKind;
use runmat_native_executor::execute::NativeWorkspaceInput;
use runmat_types::ProgramPointId;

use crate::{input::LinkedAotProcessInput, AotProcessInput};

#[path = "execute/linked_selected.rs"]
mod linked_selected;

pub fn execute(input: AotProcessInput) -> Result<(), String> {
    execute_with_linked_authority(input, false, std::ptr::null())
}

pub(crate) fn execute_linked(input: LinkedAotProcessInput) -> Result<(), String> {
    let (input, source_service_factory) = input.into_parts();
    execute_with_linked_authority(input, true, source_service_factory)
}

fn execute_with_linked_authority(
    input: AotProcessInput,
    verified_link: bool,
    source_service_factory: *const std::ffi::c_void,
) -> Result<(), String> {
    let _console_forwarder = crate::output::install_console_forwarder();
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
    let linked_selected =
        linked_selected::admit(&program, &assembly, verified_link, source_service_factory)?;
    if program.foreign_policy.is_none()
        && (!interop.foreign_types.is_empty()
            || !interop.adapters.is_empty()
            || !interop.adapter_contracts.is_empty())
    {
        return Err("standalone foreign program has no captured policy".into());
    }
    let mut native_sources = assembly
        .functions
        .iter()
        .map(|function| {
            (
                function.id,
                function.source,
                function.name.as_str(),
                function.source_role,
            )
        })
        .collect::<Vec<_>>();
    native_sources.sort_by_key(|function| function.0);
    if native_sources.len() != program.functions.len()
        || native_sources
            .iter()
            .zip(&program.functions)
            .any(|(native, manifest)| {
                native.0 != manifest.function
                    || native.1 != manifest.source
                    || native.2 != manifest.name
                    || native.3 != manifest.role
            })
    {
        return Err("standalone source/function inventory does not match its Native IR".into());
    }
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
    let entrypoint = program.entrypoint;
    let function = assembly
        .functions
        .iter()
        .find(|function| function.id == entrypoint)
        .ok_or_else(|| "standalone entrypoint is absent from Native IR".to_string())?;
    if !function.abi.fixed_inputs.is_empty() || function.abi.varargin.is_some() {
        return Err("standalone entrypoint requires unsupported command-line inputs".into());
    }
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
    let mut runtime = runmat_runtime::source_context::bind_portable_source_map_v2(
        standalone_runtime(&assembly.program),
        &program.source_map,
    )
    .map_err(|error| format!("standalone source binding failed: {error}"))?;
    let local_tasks = Rc::new(runmat_runtime::context::RuntimeLocalTaskSupervisor::new());
    let ports = runtime
        .service_ports()
        .clone()
        .with_local_tasks(local_tasks.submission_port(runtime.local_identity()))
        .with_call(Rc::new(runmat_runtime::context::RuntimeCallRouter));
    runtime = runtime.with_service_ports(ports);
    #[cfg(feature = "source-evaluation")]
    let (mut runtime, source_evaluation_owner) = if let Some(issuer) = linked_selected
        .as_ref()
        .and_then(|selected| selected.source_evaluation.clone())
    {
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
        // SAFETY: the generated resolver has the exact declared C ABI. It
        // returns only addresses of functions emitted from this same verified
        // assembly, or null when the object is inconsistent.
        let address = unsafe { (input.function_resolver)(function.id.0) };
        if address.is_null() {
            return Err(format!(
                "standalone object cannot resolve native function {}",
                function.id.0
            ));
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
        program_capture: Some(input.program.clone()),
        interpreter_resume_points: resume_points,
        ..runmat_native_executor::NativeExecutorOptions::default()
    };
    let executor = Rc::new(
        match &linked_selected {
            Some(linked) => runmat_native_executor::NativeExecutor::bind_linked_selected(
                assembly.clone(),
                executable,
                options,
                &linked.product,
                linked.source_evaluation.clone(),
            ),
            None => {
                runmat_native_executor::NativeExecutor::bind(assembly.clone(), executable, options)
            }
        }
        .map_err(|error| format!("standalone native host initialization failed: {error}"))?,
    );

    let async_runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("standalone async runtime initialization failed: {error}"))?;
    // Keep the original session owner through foreign teardown and terminal
    // rendering. ContextFuture performs the existing nonblocking final drain
    // after inner destruction on success, error, or abandonment.
    let result = async_runtime.block_on(local_tasks.drive_until(runtime.clone().scope(async {
        let execution =
            if assembly.executable_identity.entrypoint_kind == ExecutableEntrypointKind::Script {
                let local_names = function
                    .locals
                    .iter()
                    .filter_map(|local| local.binding.zip(local.name.clone()))
                    .collect();
                crate::program::invoke_workspace(
                    Rc::clone(&executor),
                    &program,
                    entrypoint,
                    NativeWorkspaceInput {
                        local_names,
                        ..NativeWorkspaceInput::default()
                    },
                    requested_outputs,
                    runtime.clone(),
                )
                .await
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
                    if cleanup_lifecycle.is_ok() {
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
            Err(error) => Some(execution_diagnostic(error, &runtime)),
        };
        // Retire caller-visible output ownership and admit any final callbacks
        // before foreign adapters are closed.
        let final_cleanup = lifecycle_host
            .drain_gracefully(&runtime)
            .await
            .map_err(|error| format!("standalone final cleanup failed: {error}"));
        let foreign_lifecycle = crate::foreign_retirement::retire(
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
        if let Some(error) = execution_failure {
            return Err(error);
        }
        cleanup_lifecycle?;
        final_cleanup?;
        foreign_lifecycle
            .into_result()
            .map_err(|error| format!("standalone foreign shutdown failed: {error}"))?;
        Ok(())
    })));
    // A rejected graceful close is Abandoned, not a reason to close the
    // callback driver while worker requests still have origin authority.
    mex_guard.force_abandon_if_needed();
    lifecycle_host.force_close();
    let execution_result = result
        .map_err(|error| format!("standalone local task driver failed: {error}"))
        .and_then(|result| result);
    #[cfg(feature = "source-evaluation")]
    let source_retirement = source_evaluation_owner
        .map(|owner| {
            owner
                .close()
                .map_err(|error| format!("standalone source evaluator retirement failed: {error}"))
        })
        .transpose();
    execution_result?;
    #[cfg(feature = "source-evaluation")]
    let _ = source_retirement?;
    Ok(())
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
) -> runmat_runtime::context::RuntimeContext {
    use runmat_execution::LanguageCompatibilityMode;
    use runmat_runtime::context::{RuntimeContext, RuntimeLanguageMode};

    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_program_revision(Some(program.clone()));
    let mode = match program.compatibility_mode() {
        LanguageCompatibilityMode::RunMat => RuntimeLanguageMode::RunMat,
        LanguageCompatibilityMode::Matlab => RuntimeLanguageMode::Matlab,
        LanguageCompatibilityMode::Strict => RuntimeLanguageMode::Strict,
    };
    runtime.set_language_mode(mode);
    runtime.set_runmat_extensions_enabled(matches!(mode, RuntimeLanguageMode::RunMat));
    crate::program::install_acceleration_services(runtime)
}

#[cfg(test)]
mod tests;
