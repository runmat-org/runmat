use super::contract::{FOREIGN_LINK, HOST_PLACEMENT};
use super::descriptor::{scalar_text, ERRORS};
use super::documentation_support::{
    LOAD_IMPLEMENTATION as IMPLEMENTATION, NATIVE_FIXTURE, NATIVE_HOST,
};
use super::inference_support::{arity, infer_with, text_at};
use crate::*;
use runmat_types::*;

pub const LIBISLOADED_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "loaded = libisloaded(alias)",
        inputs: &[scalar_text(
            "alias",
            "Scalar text, number, or logical library alias.",
        )],
        outputs: &[BuiltinParamDescriptor {
            name: "loaded",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Scalar logical loaded-state result.",
        }],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
const INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "Scalar integer aliases are converted to text and the result is logical; no C ABI integer payload is accepted." };

pub const LIBISLOADED_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
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
            CapabilityRequirement::NativeCode,
            CapabilityRequirement::ForeignRuntime,
        ],
    },
    placement: HOST_PLACEMENT,
    link: FOREIGN_LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: Some(&INTEGER_AUDIT),
    suppress_auto_output: false,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 1, Some(1), &mut diagnostics);
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
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Native session loaded state", location: "native_ffi_session::library_and_pointer_state_are_owned_by_one_runtime_session" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
