mod backend;

use backend::RunMatLanguageServer;
use log::LevelFilter;
use tokio::io::{stdin, stdout};
use tower_lsp::{LspService, Server};

#[tokio::main]
async fn main() {
    init_logging();

    let stdin = stdin();
    let stdout = stdout();

    let (service, socket) = LspService::new(|client| RunMatLanguageServer::new(client));
    Server::new(stdin, stdout, socket).serve(service).await;
}

fn init_logging() {
    if env_logger::builder().try_init().is_err() {
        // Logging already initialized.
    } else {
        log::set_max_level(LevelFilter::Info);
    }
}
