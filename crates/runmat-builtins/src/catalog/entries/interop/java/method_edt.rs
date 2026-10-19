use super::{
    contract::*,
    documentation_support as docs,
    inference::{self, JavaForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, JavaForm::MethodEdt)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(
    "javaMethodEDT",
    docs::JavaDocumentationContent {
        summary: "Invoke a Java method on the AWT event-dispatch thread.",
        description: "`result = javaMethodEDT(methodName, objectOrClass, arguments...)` synchronously invokes a Java instance or static method on the EDT.",
        keywords: &["javaMethodEDT", "Java", "AWT", "Swing", "Desktop"],
        related: &["javaObjectEDT", "javaMethod", "usejava"],
        sections: &[BuiltinDocumentationSection { heading: "Desktop authority", paragraphs: &["The session must advertise Desktop Java UI support. Foreign object identity and callback affinity remain tied to the originating session; a CLI or browser does not silently create a Java UI thread."] }],
        examples: &[docs::example("set-label-edt", "Update a Swing label on the EDT", "label = javaObjectEDT('javax.swing.JLabel', 'Ready');\njavaMethodEDT('setText', label, 'Running');\ntext = javaMethodEDT('getText', label);", "assert(strcmp(text, 'Running'));", true)],
        faqs: &[BuiltinDocumentationFaq { question: "Does this enqueue and return before the method runs?", answer: "No. The call synchronously completes on the EDT and returns the method result or error." }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "EDT method execution", location: "crates/runmat-runtime/tests/java_session.rs::java_edt_operations_require_desktop_authority_and_run_on_the_edt" }],
    },
);

pub const JAVA_METHOD_EDT_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "javaMethodEDT",
    DOCUMENTATION,
    &super::descriptor::JAVA_METHOD_EDT_DESCRIPTOR,
    infer,
    JavaEntryBehavior {
        effects: EDT_EFFECTS,
        capabilities: EDT_CAPABILITIES,
        purity: BuiltinPurity::Impure,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        placement: FOREIGN_PLACEMENT,
        link: FOREIGN_LINK,
        suppress_auto_output: false,
    },
);
