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
        user_functions::install_local_external_function_invoker(Rc::new(move |call| {
            let owner = u32::try_from(call.function).ok().and_then(|id| {
                let id = ProgramFunctionId(id);
                owners
                    .get(&id)
                    .cloned()
                    .or_else(|| current.owns_callable(id).then(|| Rc::clone(&current)))
            });
            let service = Rc::clone(&service);
            let previous = previous.clone();
            let runtime = runtime.clone();
            Box::pin(async move {
                let Some(owner) = owner else {
                    return match previous {
                        Some(previous) => previous(call).await,
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
                        let _catalog =
                            runmat_runtime::source_context::replace_source_catalog_with_fullpaths(
                                owner
                                    .source_map()
                                    .entries()
                                    .iter()
                                    .map(|entry| {
                                        (
                                            runmat_hir::SourceId(entry.source_id),
                                            entry.display_name.clone(),
                                            entry.full_path.clone(),
                                            entry.text.clone(),
                                        )
                                    })
                                    .collect(),
                            );
                        let _source_id = runmat_runtime::source_context::replace_current_source_id(
                            owner.bytecode().source_id,
                        );
                        runmat_vm::invoke_semantic_function_value_in_context(
                            call.function,
                            &call.arguments,
                            call.requested_outputs,
                            owner.functions(),
                            nested_runtime,
                        )
                        .await
                    })
                    .await
            })
        }))
    }
}
