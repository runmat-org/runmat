use runmat_filesystem::{
    js::JsFsProvider, DirectoryDialogRequest, DirectoryDialogSelection, OpenFileDialogRequest,
    OpenFileDialogSelection, SaveFileDialogRequest, SaveFileDialogSelection,
};
use runmat_runtime::context::{HostInteraction, RuntimeHostService, RuntimeServiceFuture};
use runmat_runtime::{interaction::InteractionResponse, RuntimeError};
use std::io;
use std::rc::Rc;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

pub(crate) fn install_js_fs_provider(
    bindings: &JsValue,
) -> Result<Rc<dyn RuntimeHostService>, JsValue> {
    let provider = Arc::new(JsFsProvider::new(bindings)?);
    runmat_filesystem::set_provider(provider.clone());
    Ok(Rc::new(JsDialogHost { provider }))
}

struct JsDialogHost {
    provider: Arc<JsFsProvider>,
}

impl RuntimeHostService for JsDialogHost {
    fn console(
        &self,
        stream: runmat_runtime::console::ConsoleStream,
        text: runmat_runtime::console::ConsoleText,
    ) {
        runmat_runtime::console::forward_recorded_console_output(stream, text);
    }

    fn warning(&self, _: runmat_runtime::warning_store::RuntimeWarning) {}

    fn interact(
        &self,
        request: HostInteraction,
    ) -> RuntimeServiceFuture<Result<InteractionResponse, RuntimeError>> {
        runmat_runtime::interaction::invoke_installed_async_handler(request).unwrap_or_else(|| {
            Box::pin(async {
                Err(runmat_runtime::build_runtime_error(
                    "The browser host does not provide interactive input",
                )
                .with_identifier("RunMat:host:InteractionUnavailable")
                .build())
            })
        })
    }

    fn select_file_open(
        &self,
        request: OpenFileDialogRequest,
    ) -> RuntimeServiceFuture<Result<Option<OpenFileDialogSelection>, RuntimeError>> {
        let provider = self.provider.clone();
        Box::pin(async move {
            provider
                .select_file_open(&request)
                .await
                .map_err(dialog_error)
        })
    }

    fn select_file_save(
        &self,
        request: SaveFileDialogRequest,
    ) -> RuntimeServiceFuture<Result<Option<SaveFileDialogSelection>, RuntimeError>> {
        let provider = self.provider.clone();
        Box::pin(async move {
            provider
                .select_file_save(&request)
                .await
                .map_err(dialog_error)
        })
    }

    fn select_directory(
        &self,
        _request: DirectoryDialogRequest,
    ) -> RuntimeServiceFuture<Result<Option<DirectoryDialogSelection>, RuntimeError>> {
        Box::pin(async { Ok(None) })
    }
}

fn dialog_error(source: io::Error) -> RuntimeError {
    runmat_runtime::build_runtime_error(source.to_string())
        .with_identifier("RunMat:host:DialogFailed")
        .with_source(source)
        .build()
}
