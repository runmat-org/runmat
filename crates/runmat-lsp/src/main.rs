#![cfg(feature = "native")]

#[cfg(not(target_arch = "wasm32"))]
use runmat_runtime as _;

use log::LevelFilter;
use runmat_lsp::native;
use tokio::io::{stdin, stdout};
use tower_lsp::Server;

#[tokio::main]
async fn main() {
    init_logging();

    let stdin = stdin();
    let stdout = stdout();

    let (service, socket) = native::service();
    Server::new(stdin, stdout, socket).serve(service).await;
}

fn init_logging() {
    if env_logger::builder().try_init().is_err() {
        // Logging already initialized.
    } else {
        log::set_max_level(LevelFilter::Info);
    }
}
