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
    "Inspect or configure the session Java environment.",
    "`environment = jenv()` returns Version, Home, Library, Status, and Configuration. `jenv(option, value)` accepts `Version` and `ExecutionMode`; configuration changes must precede JVM startup.",
    &["jenv", "Java", "JVM", "environment", "configuration"],
    &["usejava", "javaObject", "javaclasspath"],
    &[BuiltinDocumentationSection { heading: "Configuration and lifetime", paragraphs: &["The status object reports the session's discovered Java installation and whether the JVM is loaded. `Version` selects a compatible installation or home, while `ExecutionMode` currently accepts only `InProcess`. Once Java is running, configuration changes are rejected instead of silently replacing live object handles."] }],
    &[docs::example("inspect-java-environment", "Inspect the Java environment", "environment = jenv();", "assert(isprop(environment, 'Status'));", false)],
    &[BuiltinDocumentationFaq { question: "Can jenv change Java after the JVM starts?", answer: "No. Configuration changes are rejected after JVM startup so live Java references remain bound to the same session." }],
    &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Java environment and configuration", location: "crates/runmat-runtime/tests/java_session.rs::java_environment_configuration_and_desktop_capability_are_session_owned" }],
);

pub const JENV_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "jenv",
    DOCUMENTATION,
    &super::descriptor::JENV_DESCRIPTOR,
    infer,
    &[
        runmat_types::EffectKind::EnvironmentRead,
        runmat_types::EffectKind::EnvironmentWrite,
        runmat_types::EffectKind::FilesystemRead,
        runmat_types::EffectKind::MaySuspend,
        runmat_types::EffectKind::MayThrow,
    ],
    FOREIGN_CAPABILITIES,
    BuiltinPurity::Impure,
    BuiltinAsyncBehavior::MaySuspend,
    FOREIGN_PLACEMENT,
    FOREIGN_LINK,
    false,
);
