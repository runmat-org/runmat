use super::{
    contract::*,
    documentation_support as docs,
    inference::{self, JavaForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, JavaForm::RemovePath)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(
    "javarmpath",
    docs::JavaDocumentationContent {
        summary: "Remove entries from the session's dynamic Java classpath.",
        description: "`javarmpath(path, ...)` removes text or cell-of-text entries from the current session's dynamic Java classpath.",
        keywords: &["javarmpath", "Java", "classpath", "dynamic classpath"],
        related: &["javaaddpath", "javaclasspath"],
        sections: &[BuiltinDocumentationSection { heading: "Session scope", paragraphs: &["Removal affects subsequent class loading, not the identity or defining class loader of objects already created. The operation has no result value."] }],
        examples: &[docs::example("remove-class-directory", "Remove a class directory", "path = pwd; javaaddpath(path); javarmpath(path);", "assert(~any(strcmp(javaclasspath('-dynamic'), path)));", false)],
        faqs: &[BuiltinDocumentationFaq { question: "Does removing a path destroy Java objects loaded from it?", answer: "No. Existing objects and their defining class loader remain session-owned; removal affects subsequent lookup." }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Session classpath mutation", location: "crates/runmat-runtime/tests/java_session.rs" }],
    },
);

pub const JAVA_REMOVE_PATH_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "javarmpath",
    DOCUMENTATION,
    &super::descriptor::JAVA_REMOVE_PATH_DESCRIPTOR,
    infer,
    JavaEntryBehavior {
        effects: CLASSPATH_EFFECTS,
        capabilities: FOREIGN_CAPABILITIES,
        purity: BuiltinPurity::Impure,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        placement: FOREIGN_PLACEMENT,
        link: FOREIGN_LINK,
        suppress_auto_output: true,
    },
);
