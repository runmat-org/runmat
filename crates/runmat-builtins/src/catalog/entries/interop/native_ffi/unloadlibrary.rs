use super::contract::{FOREIGN_LINK, HOST_PLACEMENT};
use super::descriptor::{scalar_text, ERRORS};
use super::documentation_support::{
    LOAD_IMPLEMENTATION as IMPLEMENTATION, NATIVE_FIXTURE, NATIVE_HOST,
};
use super::inference_support::{arity, infer_with, text_at};
use crate::*;
use runmat_types::*;

pub const UNLOADLIBRARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "unloadlibrary(alias)",
        inputs: &[scalar_text(
            "alias",
            "Scalar text, number, or logical loaded library alias.",
        )],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
const INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Scalar integer aliases are converted to text; no C ABI integer payload is accepted.",
};

pub const UNLOADLIBRARY_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "unloadlibrary",
    },
    category: "interop/native",
    documentation: DOCUMENTATION,
    descriptor: &UNLOADLIBRARY_DESCRIPTOR,
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
            EffectKind::EnvironmentWrite,
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
    suppress_auto_output: true,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 1, Some(1), &mut diagnostics);
    text_at(request, 0, &mut diagnostics);
    infer_with(request, entry, vec![], diagnostics)
}

const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("unloadlibrary"), slug: Some("unloadlibrary"),
    summary: "Remove a named native library from the current session.",
    description: "`unloadlibrary(alias)` removes the loaded interface under that session alias. The call is a statement and returns no public value. An unknown alias produces `RunMat:Foreign:InvalidCall`.",
    keywords: &["unloadlibrary", "native", "shared library", "session"],
    related: &["loadlibrary", "libisloaded", "libfunctions", "calllib"],
    sections: &[BuiltinDocumentationSection { heading: "Ownership", paragraphs: &["The foreign adapter remains responsible for prepared symbols, native pointers, callbacks, and outstanding leases. Unloading the alias does not turn a borrowed native address into independently owned storage."] }],
    examples: &[BuiltinExample {
        id: "unload-fixture", title: "Unload a named native library",
        program: "loadlibrary('fixture', 'fixture.h', 'Alias', 'fixture');\nassert(libisloaded('fixture'));\nunloadlibrary('fixture');\nassert(~libisloaded('fixture'));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeForeignRuntime,
        fixture: NATIVE_FIXTURE,
        requirements: NATIVE_HOST,
        verification: BuiltinExampleVerification::Assertions { source: "assert(~libisloaded('fixture'));" },
    }],
    example_exemption: None,
    faqs: &[], links: &[], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Native session unload", location: "native_ffi_session::library_and_pointer_state_are_owned_by_one_runtime_session" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
