use std::rc::Rc;

use runmat_runtime::context::RuntimeHttpService;

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn default_http_service() -> anyhow::Result<Rc<dyn RuntimeHttpService>> {
    use std::sync::OnceLock;

    static DRIVER: OnceLock<Result<tokio::runtime::Runtime, String>> = OnceLock::new();
    let driver = DRIVER
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|error| anyhow::anyhow!("HTTP execution driver unavailable: {error}"))?;
    Ok(Rc::new(
        runmat_runtime::builtins::io::http::transport::NativeHttpService::new(
            driver.handle().clone(),
        ),
    ))
}

#[cfg(target_arch = "wasm32")]
pub(super) fn default_http_service() -> anyhow::Result<Rc<dyn RuntimeHttpService>> {
    Ok(Rc::new(
        runmat_runtime::builtins::io::http::transport::BrowserHttpService::new(),
    ))
}
