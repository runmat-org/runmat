use super::{
    contract::*,
    documentation_support as docs,
    inference::{self, JavaForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, JavaForm::ConstructEdt)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(
    "javaObjectEDT",
    docs::JavaDocumentationContent {
        summary: "Construct a Java object on the AWT event-dispatch thread.",
        description: "`object = javaObjectEDT(className, arguments...)` executes its constructor synchronously on the Java EDT and returns a session-owned object.",
        keywords: &["javaObjectEDT", "Java", "AWT", "Swing", "Desktop"],
        related: &["javaObject", "javaMethodEDT", "usejava"],
        sections: &[BuiltinDocumentationSection { heading: "Desktop authority", paragraphs: &["A Desktop host must advertise Java UI support; CLI, server, and browser sessions reject EDT execution rather than starting an implicit UI thread. Calls already on the EDT run directly, while other Desktop calls synchronously dispatch to it."] }],
        examples: &[docs::example("construct-label-edt", "Construct a Swing label", "label = javaObjectEDT('javax.swing.JLabel', 'Ready');", "assert(isjava(label));", true)],
        faqs: &[BuiltinDocumentationFaq { question: "Can a CLI session use Java EDT calls?", answer: "No. Use a Desktop host with Java UI support; otherwise RunMat:Java:EdtUnavailable is returned." }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "EDT authority and execution", location: "crates/runmat-runtime/tests/java_session.rs::java_edt_operations_require_desktop_authority_and_run_on_the_edt" }],
    },
);

pub const JAVA_OBJECT_EDT_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "javaObjectEDT",
    DOCUMENTATION,
    &super::descriptor::JAVA_OBJECT_EDT_DESCRIPTOR,
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
