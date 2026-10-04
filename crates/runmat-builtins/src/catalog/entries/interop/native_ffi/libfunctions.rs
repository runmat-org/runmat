use super::contract::{FOREIGN_LINK, HOST_PLACEMENT};
use super::documentation_support::{IMPLEMENTATION, NATIVE_FIXTURE, NATIVE_HOST};
use super::inference_support::{arity, infer_with, string_cell_row, text_at};
use super::{LIBFUNCTIONS_DESCRIPTOR, NATIVE_FFI_INTEGER_AUDIT};
use crate::*;
use runmat_types::*;

pub const LIBFUNCTIONS_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "libfunctions",
    },
    category: "interop/native",
    documentation: DOCUMENTATION,
    descriptor: &LIBFUNCTIONS_DESCRIPTOR,
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
    integer_audit: Some(&NATIVE_FFI_INTEGER_AUDIT),
    suppress_auto_output: false,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 1, Some(2), &mut diagnostics);
    text_at(request, 0, &mut diagnostics);
    text_at(request, 1, &mut diagnostics);
    if let Some(option) = request.literals.literal_string_at(1) {
        if !option.eq_ignore_ascii_case("-full") {
            diagnostics.push(catalog::inference::argument_error(
                "RM-NATIVE-FFI-OPTION",
                "libfunctions supports only the -full option",
                1,
            ));
        }
    }
    let names = string_cell_row();
    infer_with(request, entry, vec![names], diagnostics)
}

const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("libfunctions"), slug: Some("libfunctions"),
    summary: "List functions in a loaded native library interface.",
    description: "`libfunctions(alias)` returns a row cell array of declared symbol names. `libfunctions(alias, '-full')` returns readable signatures from the prepared interface metadata. Unknown aliases and unsupported options raise a native FFI call error.",
    keywords: &["libfunctions", "shared library", "native", "symbols", "signatures"],
    related: &["loadlibrary", "calllib", "libisloaded", "unloadlibrary"],
    sections: &[BuiltinDocumentationSection { heading: "Prepared declarations", paragraphs: &["Only symbols admitted by the prepared C header/interface manifest are listed. The result is not a dynamic scan of every exported symbol in the library file."] }],
    examples: &[BuiltinExample {
        id: "list-fixture-symbol", title: "Inspect a declared C function",
        program: "loadlibrary('fixture', 'fixture.h', 'Alias', 'fixture');\nnames = libfunctions('fixture');\nassert(any(strcmp(names, 'fixture_add')));\nfull = libfunctions('fixture', '-full');\nassert(~isempty(full));\nunloadlibrary('fixture');",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeForeignRuntime,
        fixture: NATIVE_FIXTURE, requirements: NATIVE_HOST,
        verification: BuiltinExampleVerification::Assertions { source: "assert(any(strcmp(names, 'fixture_add'))); assert(~isempty(full));" },
    }],
    example_exemption: None, faqs: &[], links: &[], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Prepared symbol enumeration", location: "native_ffi_session::library_and_pointer_state_are_owned_by_one_runtime_session" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
