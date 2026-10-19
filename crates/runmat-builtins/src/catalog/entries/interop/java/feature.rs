use super::{
    contract::*,
    documentation_support as docs,
    inference::{self, JavaForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, JavaForm::Feature)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(
    "usejava",
    docs::JavaDocumentationContent {
        summary: "Report availability of a Java runtime feature.",
        description: "`tf = usejava(feature)` accepts `jvm`, `desktop`, `awt`, and `swing` (case-insensitive). Browser hosts return false without starting Java.",
        keywords: &["usejava", "Java", "JVM", "desktop", "AWT", "Swing"],
        related: &["jenv", "javaObject", "javaObjectEDT"],
        sections: &[BuiltinDocumentationSection { heading: "Feature probes", paragraphs: &["`jvm` discovers a suitable installation, `desktop` reports the session host's Desktop authority, and `awt`/`swing` may initialize the JVM to query the Java graphics environment. Unsupported feature names produce a Java validation error on native hosts."] }],
        examples: &[docs::example("jvm-available", "Check native Java availability", "available = usejava('jvm');", "assert(available);", false)],
        faqs: &[BuiltinDocumentationFaq { question: "Does usejava start the JVM?", answer: "The jvm and desktop probes need not start it; awt and swing may start the JVM to inspect headless graphics support." }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Session feature capability", location: "crates/runmat-runtime/tests/java_session.rs::java_environment_configuration_and_desktop_capability_are_session_owned" }],
    },
);

pub const USEJAVA_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "usejava",
    DOCUMENTATION,
    &super::descriptor::USEJAVA_DESCRIPTOR,
    infer,
    JavaEntryBehavior {
        effects: &[
            EffectKind::EnvironmentRead,
            EffectKind::EnvironmentWrite,
            EffectKind::FilesystemRead,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: FOREIGN_CAPABILITIES,
        purity: BuiltinPurity::Impure,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        placement: BuiltinPlacementContract {
            portability: BuiltinPortability::NativeAndWasm,
            ..FOREIGN_PLACEMENT
        },
        link: FOREIGN_LINK,
        suppress_auto_output: false,
    },
);
