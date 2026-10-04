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
    "Construct a Java object on the AWT event-dispatch thread.",
    "`object = javaObjectEDT(className, arguments...)` executes its constructor synchronously on the Java EDT and returns a session-owned object.",
    &["javaObjectEDT", "Java", "AWT", "Swing", "Desktop"],
    &["javaObject", "javaMethodEDT", "usejava"],
    &[BuiltinDocumentationSection { heading: "Desktop authority", paragraphs: &["A Desktop host must advertise Java UI support; CLI, server, and browser sessions reject EDT execution rather than starting an implicit UI thread. Calls already on the EDT run directly, while other Desktop calls synchronously dispatch to it."] }],
    &[docs::example("construct-label-edt", "Construct a Swing label", "label = javaObjectEDT('javax.swing.JLabel', 'Ready');", "assert(isjava(label));", true)],
    &[BuiltinDocumentationFaq { question: "Can a CLI session use Java EDT calls?", answer: "No. Use a Desktop host with Java UI support; otherwise RunMat:Java:EdtUnavailable is returned." }],
    &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "EDT authority and execution", location: "crates/runmat-runtime/tests/java_session.rs::java_edt_operations_require_desktop_authority_and_run_on_the_edt" }],
);

pub const JAVA_OBJECT_EDT_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "javaObjectEDT",
    DOCUMENTATION,
    &super::descriptor::JAVA_OBJECT_EDT_DESCRIPTOR,
    infer,
    EDT_EFFECTS,
    EDT_CAPABILITIES,
    BuiltinPurity::Impure,
    BuiltinAsyncBehavior::MaySuspend,
    FOREIGN_PLACEMENT,
    FOREIGN_LINK,
    false,
);
