use std::rc::Rc;

use runmat_runtime::user_functions::DynamicFunctionLoadPhase;

pub struct MexRuntimeGuard {
    runtime: runmat_runtime::context::RuntimeContext,
    mex_runtime: Rc<runmat_runtime::foreign::MexRuntimeSession>,
    shutdown_complete: bool,
}

impl MexRuntimeGuard {
    pub async fn shutdown_gracefully(&mut self) -> Result<(), runmat_runtime::RuntimeError> {
        let result = self
            .mex_runtime
            .shutdown_gracefully(self.runtime.clone())
            .await;
        self.shutdown_complete = true;
        result
    }
}

impl Drop for MexRuntimeGuard {
    fn drop(&mut self) {
        if self.shutdown_complete {
            return;
        }
        if let Err(error) = self.mex_runtime.shutdown(self.runtime.clone()) {
            runmat_runtime::console::record_console_line(
                runmat_runtime::console::ConsoleStream::Stderr,
                format!("could not finish MEX lifecycle during standalone shutdown: {error}"),
            );
        }
    }
}

pub fn install(
    runtime: &runmat_runtime::context::RuntimeContext,
    mex_runtime: Rc<runmat_runtime::foreign::MexRuntimeSession>,
) -> MexRuntimeGuard {
    let load_runtime = Rc::clone(&mex_runtime);
    let guard_runtime = Rc::clone(&mex_runtime);
    let loader: Rc<runmat_runtime::user_functions::DynamicFunctionLoader> =
        Rc::new(move |runtime, name, arguments, requested_outputs, phase| {
            let mex_runtime = Rc::clone(&load_runtime);
            Box::pin(async move {
                if phase != DynamicFunctionLoadPhase::BeforeSemantic {
                    return None;
                }
                mex_runtime
                    .load_and_call(&name, arguments, requested_outputs, runtime)
                    .await
            })
        });
    runtime.set_dynamic_function_loader(Some(loader));

    let clearer: Rc<runmat_runtime::user_functions::DynamicFunctionClearer> =
        Rc::new(move |runtime, request| {
            let mex_runtime = Rc::clone(&mex_runtime);
            Box::pin(async move { mex_runtime.clear_gracefully(&request, runtime).await })
        });
    runtime.set_dynamic_function_clearer(Some(clearer));
    MexRuntimeGuard {
        runtime: runtime.clone(),
        mex_runtime: guard_runtime,
        shutdown_complete: false,
    }
}
