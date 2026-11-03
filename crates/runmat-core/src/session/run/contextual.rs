//! Companion discovery retires before execution; publication retains the same session owners.
use super::*;

pub(super) struct Prepared<'a> {
    session: &'a mut RunMatSession,
    input: &'a str,
    previous_workspace_bindings: HashMap<String, crate::abi::WorkspaceBindingKey>,
    services: runmat_runtime::testing::TestServicesGuard,
}

pub(super) async fn prepare<'a>(
    session: &'a mut RunMatSession,
    input: &'a str,
) -> Result<Prepared<'a>, RunError> {
    let _test_services =
        runmat_runtime::testing::install_test_services(crate::testing::runtime_adapter::services(
            session.compat_mode,
            session.project_handoff.clone(),
            #[cfg(not(target_arch = "wasm32"))]
            session.local_tasks.child_scope_factory(),
            #[cfg(not(target_arch = "wasm32"))]
            Some(session.mex_runtime.native_lane_child_factory()),
        ));
    session.configure_dynamic_function_services();
    let companion = super::super::compile::companion_source_statements_for_input_async(
        session.current_source_fullpath_name(),
        session.compat_mode,
        session.project_handoff.as_ref(),
    )
    .await
    .map_err(|error| {
        RunError::Runtime(
            build_runtime_error(format!("project composition failed: {error}"))
                .with_source(error)
                .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
                .with_identifier("RunMat:ProjectComposition")
                .build(),
        )
    })?;
    session.pending_companion_source_discovery = Some(companion);
    session
        .sync_workspace_binding_metadata()
        .map_err(RunError::Runtime)?;
    let previous_workspace_bindings = session
        .logical_workspace_names()
        .map_err(RunError::Runtime)?
        .into_iter()
        .map(|name| {
            let key = session
                .admitted_workspace_binding_key(&name)
                .map_err(RunError::Runtime)?;
            Ok((name, key))
        })
        .collect::<std::result::Result<HashMap<_, _>, RunError>>()?;

    Ok(Prepared {
        session,
        input,
        previous_workspace_bindings,
        services: _test_services,
    })
}

pub(super) async fn execute(
    prepared: Prepared<'_>,
) -> Result<runmat_runtime::execution::RootedResult<crate::abi::ExecutionOutcome>, RunError> {
    // The original services guard outlives execution, roots, and ABI publication.
    let _services = prepared.services;
    let Prepared {
        session,
        input,
        previous_workspace_bindings,
        ..
    } = prepared;
    // Construct the existing execution allocation at its canonical admission
    // boundary, rather than in this caller's poll frame. The original runtime
    // quotes that concrete allocation and owns cancellation; no workspace is copied.
    let runtime = session.runtime_context.clone();
    let execution = runmat_runtime::execution::admit_execution_supervisor_with(&runtime, || {
        session.execute_internal(input, true)
    })?
    .await?;
    let workspace_names = execution
        .workspace_snapshot
        .values
        .iter()
        .map(|entry| entry.name.clone())
        .collect::<Vec<_>>();
    let workspace_full = execution.workspace_snapshot.full;
    let upserts = session.abi_workspace_upserts(workspace_names)?;
    let (binding_removals, removed_names) =
        session.abi_workspace_removals(previous_workspace_bindings, workspace_full)?;
    execution.outcome.try_map_owned(
        |mut outcome| {
            outcome.workspace_delta.upserts = upserts;
            outcome.workspace_delta.removals = binding_removals;
            if workspace_full && removed_names {
                outcome.effects.push(crate::abi::ObservedEffect::Workspace(
                    crate::abi::WorkspaceEffectKind::Clear,
                ));
            }
            Ok(outcome)
        },
        |error| RunError::Runtime(error.into_runtime_error()),
    )
}
