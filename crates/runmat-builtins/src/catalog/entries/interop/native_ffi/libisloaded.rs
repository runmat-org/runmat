use super::contract::{FOREIGN_LINK, HOST_PLACEMENT};
use super::documentation_support::{IMPLEMENTATION, NATIVE_FIXTURE, NATIVE_HOST};
use super::inference_support::{infer_with, text_at};
use super::{LIBISLOADED_DESCRIPTOR, NATIVE_FFI_INTEGER_AUDIT};
use crate::*;
use runmat_types::*;

pub const LIBISLOADED_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "libisloaded",
    },
    category: "interop/native",
    documentation: DOCUMENTATION,
    descriptor: &LIBISLOADED_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::EnvironmentRead,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[
            CapabilityRequirement::ForeignRuntime,
            CapabilityRequirement::NativeCode,
        ],
    },
    placement: HOST_PLACEMENT,
    link: FOREIGN_LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: Some(&NATIVE_FFI_INTEGER_AUDIT),
    suppress_auto_output: false,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    text_at(request, 0, &mut diagnostics);
    infer_with(
        request,
        entry,
        vec![ValueFact::scalar(ValueKindFact::Logical)],
        diagnostics,
    )
}

const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("libisloaded"), slug: Some("libisloaded"),
    summary: "Test whether a native library alias is loaded in the current session.",
    description: "`libisloaded(alias)` returns a scalar logical. It reads only the current session's native adapter registry; it does not open a library or test whether a file exists on disk.",
    keywords: &["libisloaded", "shared library", "native", "alias"],
    related: &["loadlibrary", "unloadlibrary", "libfunctions", "calllib"],
    sections: &[BuiltinDocumentationSection { heading: "Session scope", paragraphs: &["The result refers to the exact session alias, not a globally loaded process library. Native shared libraries are unsupported in the browser."] }],
    examples: &[BuiltinExample {
        id: "loaded-state", title: "Observe a native alias before and after unload",
        program: "loadlibrary('fixture', 'fixture.h', 'Alias', 'fixture');\nassert(libisloaded('fixture'));\nunloadlibrary('fixture');\nloaded = libisloaded('fixture');\nassert(~loaded);",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeForeignRuntime,
        fixture: NATIVE_FIXTURE, requirements: NATIVE_HOST,
        verification: BuiltinExampleVerification::Assertions { source: "assert(~loaded);" },
    }],
    example_exemption: None, faqs: &[], links: &[], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Native session loaded state", location: "tests::native_ffi_session" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
