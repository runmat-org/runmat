use std::rc::Rc;

use runmat_runtime::user_functions::DynamicFunctionLoadPhase;

pub struct MexRuntimeGuard {
    runtime: runmat_runtime::context::RuntimeContext,
    mex_runtime: Rc<runmat_runtime::foreign::MexRuntimeSession>,
    _lifecycle_supervisor: Rc<runmat_runtime::context::RuntimeLocalTaskSupervisor>,
    shutdown_complete: bool,
}

impl MexRuntimeGuard {
    pub fn force_abandon_if_needed(&mut self) {
        if !self.shutdown_complete {
            self.mex_runtime.force_abandon();
        }
    }

    pub async fn shutdown_gracefully_retirement(
        &mut self,
    ) -> runmat_runtime::foreign::MexSessionRetirement {
        let retirement = self
            .mex_runtime
            .shutdown_gracefully_retirement(self.runtime.clone())
            .await;
        self.shutdown_complete = matches!(
            &retirement,
            runmat_runtime::foreign::MexSessionRetirement::Retired(_)
        );
        retirement
    }
}

impl Drop for MexRuntimeGuard {
    fn drop(&mut self) {
        if self.shutdown_complete {
            return;
        }
        let abandoned = self.mex_runtime.has_unretired_work();
        self.force_abandon_if_needed();
        if abandoned {
            tracing::warn!(
                "abandoned MEX lifecycle during forced standalone disposal; mexAtExit completion is not guaranteed"
            );
        }
    }
}

pub fn install(
    runtime: &runmat_runtime::context::RuntimeContext,
    mex_runtime: Rc<runmat_runtime::foreign::MexRuntimeSession>,
    supervisor: Rc<runmat_runtime::context::RuntimeLocalTaskSupervisor>,
) -> Result<MexRuntimeGuard, String> {
    let tasks = runtime
        .service_ports()
        .local_tasks()
        .ok_or_else(|| "MEX admission requires a scoped native lifecycle host".to_string())?;
    let token = tasks
        .owner_token()
        .ok_or_else(|| "MEX admission requires a host-owned local task supervisor".to_string())?;
    if !token.same(&supervisor.owner_token()) {
        return Err("MEX lifecycle guard and runtime use different task supervisors".into());
    }
    runtime.set_native_name_observer(Some(mex_runtime.name_observer()));
    let load_runtime = Rc::clone(&mex_runtime);
    let guard_runtime = Rc::clone(&mex_runtime);
    let loader: Rc<runmat_runtime::user_functions::DynamicFunctionLoader> = Rc::new(
        move |runtime, name, arguments, requested_outputs, phase, _input_callsite| {
            let mex_runtime = Rc::clone(&load_runtime);
            runmat_runtime::execution::admit_dynamic_function_load_with(
                &runtime.clone(),
                move || async move {
                    if phase != DynamicFunctionLoadPhase::BeforeSemantic {
                        return std::ops::ControlFlow::Continue(arguments);
                    }
                    mex_runtime
                        .load_and_call(&name, arguments, requested_outputs, runtime)
                        .await
                },
            )
        },
    );
    runtime.set_dynamic_function_loader(Some(loader));

    let clearer: Rc<runmat_runtime::user_functions::DynamicFunctionClearer> =
        Rc::new(move |runtime, request| {
            let mex_runtime = Rc::clone(&mex_runtime);
            runmat_runtime::execution::admit_domain_continuation_with(
                &runtime.clone(),
                move || async move { mex_runtime.clear_gracefully(&request, runtime).await },
            )
        });
    runtime.set_dynamic_function_clearer(Some(clearer));
    Ok(MexRuntimeGuard {
        runtime: runtime.clone(),
        mex_runtime: guard_runtime,
        _lifecycle_supervisor: supervisor,
        shutdown_complete: false,
    })
}
