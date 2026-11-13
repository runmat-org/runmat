//! Foreign calls enter their original compiler owner while inheriting caller services.
use super::*;
use runmat_runtime::{context::RuntimeContext, user_functions};

impl SessionProgramService {
    pub(in crate::session) fn install_dependency_invoker(
        self: &Rc<Self>,
        unit: &ExecutableUnit,
        runtime: RuntimeContext,
    ) -> user_functions::ExternalFunctionInvokerGuard {
        let service = Rc::clone(self);
        let owners = unit.original_dependencies().clone();
        let current = Rc::new(unit.clone());
        let previous = user_functions::current_external_function_invoker();
        user_functions::install_local_external_function_invoker(Rc::new(move |caller, call| {
            let id = runmat_types::FunctionId(call.function);
            let owner = owners
                .get(&id)
                .cloned()
                .or_else(|| current.owns_callable(id).then(|| Rc::clone(&current)));
            let service = Rc::clone(&service);
            let previous = previous.clone();
            // The caller's context carries its live services; the installer's
            // context is only the fallback for unscoped dispatch.
            let runtime = caller
                .clone()
                .unwrap_or_else(|| runtime.clone())
                .inherit_same_session_environment();
            let admission_runtime = runtime.clone();
            user_functions::admit_user_function_with(Some(&admission_runtime), move || async move {
                let Some(owner) = owner else {
                    return match previous {
                        Some(previous) => previous(caller, call)?.await,
                        None => Err(program_error(
                            "external function has no original compiler owner",
                        )),
                    };
                };
                let function = owner
                    .functions()
                    .get(runmat_hir::FunctionId(call.function))
                    .ok_or_else(|| program_error("original function bytecode is absent"))?;
                if function.display_name != call.display_name {
                    return Err(program_error(
                        "external name differs from original function identity",
                    ));
                }
                let ports = runtime.service_ports().clone().without_workspace();
                let runtime = runtime
                    .with_service_ports(ports)
                    .with_program_revision(Some(owner.revision().program_revision.clone()));
                // Issue session-local source leases for the original owner's program.
                let runtime = runtime
                    .with_program_sources(owner.source_map().runtime_records())
                    .map_err(|error| program_error(error.to_string()))?;
                let nested_runtime = runtime.clone();
                runtime
                    .scope(async move {
                        let _program = service.enter(&owner);
                        let _dependencies =
                            service.install_dependency_invoker(&owner, nested_runtime.clone());
                        let _source =
                            runmat_runtime::source_context::replace_current_source_context(
                                Some(&owner.source().relative_path),
                                Some(&owner.source().text),
                            );
                        let _source_id = runmat_runtime::source_context::replace_current_source_id(
                            owner
                                .bytecode()
                                .source_id
                                .and_then(|source| source.resolve_in(&nested_runtime)),
                        );
                        runmat_vm::invoke_semantic_function_with_rooted_arguments_and_dynamic_admission_in_context(
                            call.function,
                            call.arguments,
                            call.requested_outputs,
                            owner.functions(),
                            nested_runtime,
                            None,
                            None,
                        )
                        .await
                    })
                    .await
            })
        }))
    }
}
