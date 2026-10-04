use super::{
    contract::*,
    documentation_support as docs,
    inference::{self, JavaForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, JavaForm::Method)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(
    "javaMethod",
    "Invoke a Java static or instance method.",
    "`result = javaMethod(methodName, objectOrClass, arguments...)` invokes an instance method for a Java foreign object or a static method for a class name.",
    &["javaMethod", "Java", "JVM", "method", "reflection"],
    &["javaObject", "javaMethodEDT", "javaArray"],
    &[BuiltinDocumentationSection { heading: "Return values and callbacks", paragraphs: &["Primitive and string results are converted to RunMat values; opaque objects retain session-owned Java identity. Functional-interface arguments may call back into the originating RunMat session on its permitted thread. Java exceptions preserve class, message, causes, and stack frames."] }],
    &[docs::example("static-max", "Call a static Java method", "largest = javaMethod('max', 'java.lang.Math', int32(4), int32(9));", "assert(largest == 9);", false)],
    &[BuiltinDocumentationFaq { question: "Can a Java method call a RunMat function handle?", answer: "Supported Java functional interfaces may synchronously call back into the originating RunMat session; callbacks from unrelated Java worker threads fail affinity validation." }],
    &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Runtime invocation and conversion", location: "crates/runmat-runtime/tests/java_session.rs::java_objects_round_trip_through_runtime_owned_foreign_handles" }],
);

pub const JAVA_METHOD_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "javaMethod",
    DOCUMENTATION,
    &super::descriptor::JAVA_METHOD_DESCRIPTOR,
    infer,
    FOREIGN_EFFECTS,
    FOREIGN_CAPABILITIES,
    BuiltinPurity::Impure,
    BuiltinAsyncBehavior::MaySuspend,
    FOREIGN_PLACEMENT,
    FOREIGN_LINK,
    false,
);
