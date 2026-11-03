//! Retire each completed execution phase before polling the selected successor.
use super::*;
#[path = "phases/interpreted.rs"]
mod interpreted;
#[path = "phases/native.rs"]
mod native;
#[path = "phases/publication.rs"]
mod publication;
use runmat_runtime::workspace::WorkspaceProjectionPublication;
struct Active<'a> {
    session: &'a mut RunMatSession,
    state: State<'a>,
}
struct State<'a> {
    used_jit: bool,
    execution_completed: bool,
    result_value: Option<Value>,
    completed_vm_root: Option<runmat_vm::InterpreterOutcome>,
    #[cfg(not(target_arch = "wasm32"))]
    completed_native_root: Option<Box<crate::generic_native::RootedNativeExecution>>,
    #[cfg(not(target_arch = "wasm32"))]
    ready_route_selection: Option<crate::generic_native::placement::ReadyRouteSelection>,
    suppressed_value: Option<Value>,
    error: Option<RuntimeError>,
    workspace_updates: Vec<WorkspaceEntry>,
    workspace_snapshot_force_full: bool,
    vm_workspace_completion: Option<runmat_vm::WorkspaceExecutionCompletion>,
    #[cfg(not(target_arch = "wasm32"))]
    tiered_interactive: Option<(
        runmat_native_executor::execute::NativeWorkspaceInput,
        Option<runmat_jit::tiering::RepresentationProfile>,
    )>,
    #[cfg(not(target_arch = "wasm32"))]
    native_loop_backedges: std::collections::BTreeMap<runmat_types::ProgramPointId, u64>,
    is_expression_stmt: bool,
    is_semicolon_suppressed: bool,
    final_stmt_emit: FinalStmtEmitDisposition,
    // The selected child retires before this guard publishes its admitted
    // projection prefix; original command/profiling guards in Data retire last.
    workspace_publication: WorkspaceProjectionPublication,
    // Every result/workspace owner above retires before Data's original guards.
    data: command::Data<'a>,
}
pub(super) fn run<'a>(
    session: &'a mut RunMatSession,
    data: command::Data<'a>,
) -> impl std::future::Future<Output = std::result::Result<SessionExecution, RunError>> + 'a {
    runmat_runtime::execution::after_result(native::run(session, data), |active| {
        runmat_runtime::execution::after_result(interpreted::select(active), |active| {
            std::future::ready(publication::run(active))
        })
    })
}
