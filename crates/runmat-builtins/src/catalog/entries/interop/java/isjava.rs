use super::{
    contract::*,
    documentation_support as docs,
    inference::{self, JavaForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest, ExecutionStackRequirement};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, JavaForm::IsJava)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(
    "isjava",
    docs::JavaDocumentationContent {
        summary: "Test whether a value carries a Java foreign-reference identity.",
        description: "`tf = isjava(value)` checks the value's Java foreign family locally; it does not start a JVM or prove that a referenced Java session is still live.",
        keywords: &["isjava", "Java", "foreign reference", "identity"],
        related: &["javaObject", "javaArray", "usejava"],
        sections: &[BuiltinDocumentationSection { heading: "Local inspection", paragraphs: &["The result is a logical scalar. Ordinary RunMat values return false on native and browser hosts. A Java foreign reference reports true by family metadata; operations on a stale or foreign-session handle still require the adapter's lease validation."] }],
        examples: &[BuiltinExample { id: "ordinary-value", title: "Inspect an ordinary value", program: "tf = isjava(int32(7));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(~tf);" } }],
        faqs: &[BuiltinDocumentationFaq { question: "Does isjava validate that a Java handle is live?", answer: "No. It is a local type-family test; invoking a Java operation performs the authoritative session and handle validation." }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Portable non-Java value", location: "builtins::interop::java::tests::isjava_rejects_ordinary_runmat_values_without_host_access" }],
    },
);

pub const ISJAVA_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "isjava",
    DOCUMENTATION,
    &super::descriptor::ISJAVA_DESCRIPTOR,
    infer,
    JavaEntryBehavior {
        effects: &[],
        capabilities: &[],
        purity: BuiltinPurity::Pure,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        placement: BuiltinPlacementContract {
            portability: BuiltinPortability::NativeAndWasm,
            ..FOREIGN_PLACEMENT
        },
        link: BuiltinLinkContract {
            policy: BuiltinLinkPolicy::PortableRuntime,
            execution_stack: ExecutionStackRequirement::Any,
            ..FOREIGN_LINK
        },
        suppress_auto_output: false,
    },
);
