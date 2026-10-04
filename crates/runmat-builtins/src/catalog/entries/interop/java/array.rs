use super::{
    contract::*,
    documentation_support as docs,
    inference::{self, JavaForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, JavaForm::Array)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(
    "javaArray",
    docs::JavaDocumentationContent {
        summary: "Create a session-owned Java object array.",
        description: "`array = javaArray(className, dimensions...)` allocates an array through Java reflection after checking dimensions as nonnegative integers within Java's signed 32-bit dimension bound.",
        keywords: &["javaArray", "Java", "array", "dimensions", "JVM"],
        related: &["javaObject", "javaMethod", "isjava"],
        sections: &[BuiltinDocumentationSection { heading: "Storage and identity", paragraphs: &["The created multidimensional array remains a Java object instead of being copied to a RunMat tensor, so Java mutation and object identity are preserved. Dimensions may be exact fixed-width integers; oversized or negative dimensions fail before allocation."] }],
        examples: &[docs::example("string-array", "Create a Java string array", "items = javaArray('java.lang.String', 2, 3);", "assert(isjava(items));", false)],
        faqs: &[BuiltinDocumentationFaq { question: "Does javaArray return a RunMat numeric tensor?", answer: "No. It returns a session-owned Java array object; Java methods can mutate or inspect it through that identity." }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Wide integer and Java array boundary", location: "crates/runmat-runtime/tests/java_session.rs::wide_unsigned_values_and_java_arrays_cross_the_runtime_boundary_exactly" }],
    },
);

pub const JAVA_ARRAY_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "javaArray",
    DOCUMENTATION,
    &super::descriptor::JAVA_ARRAY_DESCRIPTOR,
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
