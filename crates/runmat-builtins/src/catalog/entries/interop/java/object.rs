use super::{
    contract::*,
    documentation_support as docs,
    inference::{self, JavaForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, JavaForm::Construct)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(
    "javaObject",
    docs::JavaDocumentationContent {
        summary: "Construct a session-owned Java object.",
        description: "`object = javaObject(className, arguments...)` resolves a Java constructor in the session class loader and converts arguments without routing fixed-width integers through double.",
        keywords: &["javaObject", "Java", "JVM", "constructor", "object"],
        related: &["javaMethod", "javaArray", "jenv"],
        sections: &[BuiltinDocumentationSection { heading: "Identity and lifetime", paragraphs: &["The returned object is a shared, process-affine foreign reference whose Java identity and defining class loader remain owned by the originating session. Reflection selects exact matches before safe widening and reports unresolved ambiguity."] }],
        examples: &[docs::example("construct-string-builder", "Construct a Java StringBuilder", "builder = javaObject('java.lang.StringBuilder');\njavaMethod('append', builder, 'RunMat');\ntext = javaMethod('toString', builder);", "assert(isjava(builder)); assert(strcmp(text, 'RunMat'));", false)],
        faqs: &[BuiltinDocumentationFaq { question: "Does this copy a Java object into a RunMat value?", answer: "No. Opaque Java objects retain JVM identity and a session-owned foreign handle; compatible primitive return values are converted when methods are called." }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Session object identity", location: "crates/runmat-runtime/tests/java_session.rs::java_objects_round_trip_through_runtime_owned_foreign_handles" }],
    },
);

pub const JAVA_OBJECT_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "javaObject",
    DOCUMENTATION,
    &super::descriptor::JAVA_OBJECT_DESCRIPTOR,
    infer,
    JavaEntryBehavior {
        effects: FOREIGN_EFFECTS,
        capabilities: FOREIGN_CAPABILITIES,
        purity: BuiltinPurity::Impure,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        placement: FOREIGN_PLACEMENT,
        link: FOREIGN_LINK,
        suppress_auto_output: false,
    },
);
