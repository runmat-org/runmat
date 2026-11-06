pub use crate::backend::RunMatLanguageServer;
use tower_lsp::{ClientSocket, LspService};

pub type RunMatLspService = LspService<RunMatLanguageServer>;

pub fn service() -> (RunMatLspService, ClientSocket) {
    LspService::build(RunMatLanguageServer::new)
        .custom_method("runmat/discoverTests", RunMatLanguageServer::discover_tests)
        .custom_method(
            "runmat/quickInformation",
            RunMatLanguageServer::quick_information,
        )
        .custom_method("runmat/semanticFacts", RunMatLanguageServer::semantic_facts)
        .custom_method("runmat/designValues", RunMatLanguageServer::design_values)
        .finish()
}
