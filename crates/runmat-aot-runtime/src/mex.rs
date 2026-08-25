use std::rc::Rc;

use runmat_runtime::user_functions::DynamicFunctionLoadPhase;

pub struct MexRuntimeGuard {
    runtime: runmat_runtime::context::RuntimeContext,
    mex_runtime: Rc<runmat_runtime::foreign::MexRuntimeSession>,
}

impl Drop for MexRuntimeGuard {
    fn drop(&mut self) {
        if let Err(error) = self.mex_runtime.shutdown(self.runtime.clone()) {
            runmat_runtime::console::record_console_line(
                runmat_runtime::console::ConsoleStream::Stderr,
                format!("could not finish MEX lifecycle during standalone shutdown: {error}"),
            );
        }
    }
}

pub fn install(runtime: &runmat_runtime::context::RuntimeContext) -> MexRuntimeGuard {
    let mex_runtime = Rc::new(runmat_runtime::foreign::MexRuntimeSession::new());
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
        Rc::new(move |runtime, request| mex_runtime.clear(&request, runtime));
    runtime.set_dynamic_function_clearer(Some(clearer));
    MexRuntimeGuard {
        runtime: runtime.clone(),
        mex_runtime: guard_runtime,
    }
}
