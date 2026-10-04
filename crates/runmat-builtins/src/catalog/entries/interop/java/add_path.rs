use super::{
    contract::*,
    documentation_support as docs,
    inference::{self, JavaForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, JavaForm::AddPath)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(
    "javaaddpath",
    "Add entries to the session's dynamic Java classpath.",
    "`javaaddpath(path, ...)` prepends entries in argument order. A final `'-end'` appends them instead. Entries may be text or cells of text.",
    &["javaaddpath", "Java", "classpath", "dynamic classpath"],
    &["javarmpath", "javaclasspath", "javaObject"],
    &[BuiltinDocumentationSection { heading: "Session class loader", paragraphs: &["The change belongs to the current Java session and may affect later class resolution. Existing Java objects retain their defining class loader and identity. The operation has no result value."] }],
    &[docs::example("prepend-current-directory", "Add a class directory", "javaaddpath(pwd);", "assert(any(strcmp(javaclasspath('-dynamic'), pwd)));", false)],
    &[BuiltinDocumentationFaq { question: "How can I append rather than prepend entries?", answer: "Pass `'-end'` as the final argument. The order of entries remains the same as the argument order." }],
    &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Session classpath mutation", location: "crates/runmat-runtime/tests/java_session.rs" }],
);

pub const JAVA_ADD_PATH_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "javaaddpath",
    DOCUMENTATION,
    &super::descriptor::JAVA_ADD_PATH_DESCRIPTOR,
    infer,
    CLASSPATH_EFFECTS,
    FOREIGN_CAPABILITIES,
    BuiltinPurity::Impure,
    BuiltinAsyncBehavior::MaySuspend,
    FOREIGN_PLACEMENT,
    FOREIGN_LINK,
    true,
);
