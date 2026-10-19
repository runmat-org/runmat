use super::{
    contract::*,
    documentation_support as docs,
    inference::{self, JavaForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, JavaForm::Classpath)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(
    "javaclasspath",
    docs::JavaDocumentationContent {
        summary: "Inspect or replace the session's Java classpath.",
        description: "`paths = javaclasspath()` returns the dynamic paths. `'-dynamic'`, `'-static'`, and `'-all'` select the returned layer. Path arguments replace the dynamic path; `'-v0'` and `'-v1'` are accepted compatibility forms with no result.",
        keywords: &["javaclasspath", "Java", "classpath", "dynamic classpath", "static classpath"],
        related: &["javaaddpath", "javarmpath", "jenv"],
        sections: &[BuiltinDocumentationSection { heading: "Queries and mutation", paragraphs: &["Query forms return a column cell array of character paths. Replacement affects the current session's dynamic class loader, while existing object handles retain their Java identity. The compatibility `-v0` and `-v1` forms do not return a value."] }],
        examples: &[docs::example("inspect-dynamic-path", "Inspect dynamic paths", "paths = javaclasspath('-dynamic');", "assert(iscell(paths));", false)],
        faqs: &[BuiltinDocumentationFaq { question: "Which classpath does the no-argument form return?", answer: "The current session's dynamic classpath. Use `'-static'` or `'-all'` for the other views." }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Classpath session behavior", location: "crates/runmat-runtime/tests/java_session.rs" }],
    },
);

pub const JAVA_CLASSPATH_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "javaclasspath",
    DOCUMENTATION,
    &super::descriptor::JAVA_CLASSPATH_DESCRIPTOR,
    infer,
    JavaEntryBehavior {
        effects: &[
            runmat_types::EffectKind::EnvironmentRead,
            runmat_types::EffectKind::EnvironmentWrite,
            runmat_types::EffectKind::FilesystemRead,
            runmat_types::EffectKind::MaySuspend,
            runmat_types::EffectKind::MayThrow,
        ],
        capabilities: FOREIGN_CAPABILITIES,
        purity: BuiltinPurity::Impure,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        placement: FOREIGN_PLACEMENT,
        link: FOREIGN_LINK,
        suppress_auto_output: false,
    },
);
