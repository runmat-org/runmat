use super::{
    contract::*,
    documentation_support as docs,
    inference::{self, JavaForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, JavaForm::Environment)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(
    "jenv",
    docs::JavaDocumentationContent {
        summary: "Inspect or configure the session Java environment.",
        description: "`environment = jenv()` returns Version, Home, Library, Status, and Configuration. `jenv(option, value)` accepts `Version` and `ExecutionMode`; configuration changes must precede JVM startup.",
        keywords: &["jenv", "Java", "JVM", "environment", "configuration"],
        related: &["usejava", "javaObject", "javaclasspath"],
        sections: &[BuiltinDocumentationSection { heading: "Configuration and lifetime", paragraphs: &["The status object reports the session's discovered Java installation and whether the JVM is loaded. `Version` selects a compatible installation or home, while `ExecutionMode` currently accepts only `InProcess`. Once Java is running, configuration changes are rejected instead of silently replacing live object handles."] }],
        examples: &[docs::example("inspect-java-environment", "Inspect the Java environment", "environment = jenv();", "assert(isprop(environment, 'Status'));", false)],
        faqs: &[BuiltinDocumentationFaq { question: "Can jenv change Java after the JVM starts?", answer: "No. Configuration changes are rejected after JVM startup so live Java references remain bound to the same session." }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Java environment and configuration", location: "crates/runmat-runtime/tests/java_session.rs::java_environment_configuration_and_desktop_capability_are_session_owned" }],
    },
);

pub const JENV_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "jenv",
    DOCUMENTATION,
    &super::descriptor::JENV_DESCRIPTOR,
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
