mod model;
mod projection;

pub use model::{
    SemanticBindingFacts, SemanticBindingRegion, SemanticDocumentFacts, SemanticFactObservation,
    SemanticFunctionFacts, SemanticQuickInformation,
};
pub use projection::project_document_facts;
