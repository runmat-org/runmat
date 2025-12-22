#![cfg(feature = "native")]

#[cfg(not(target_arch = "wasm32"))]
use runmat_runtime as _;

mod core;
mod native;
mod backend;

use native::RunMatLanguageServer;
use log::LevelFilter;
use tokio::io::{stdin, stdout};
use tower_lsp::{LspService, Server};

#[tokio::main]
async fn main() {
    init_logging();

    let stdin = stdin();
    let stdout = stdout();

    let (service, socket) = LspService::new(RunMatLanguageServer::new);
    Server::new(stdin, stdout, socket).serve(service).await;
}

fn init_logging() {
    if env_logger::builder().try_init().is_err() {
        // Logging already initialized.
    } else {
        log::set_max_level(LevelFilter::Info);
    }
}
