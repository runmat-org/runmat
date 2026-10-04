use super::contract::{NativeLoadOption, FOREIGN_LINK, HOST_PLACEMENT};
use super::documentation_support::{IMPLEMENTATION, NATIVE_FIXTURE, NATIVE_HOST};
use super::inference_support::{arity, infer_with, string_cell_row, text_at};
use super::{LOADLIBRARY_DESCRIPTOR, NATIVE_FFI_INTEGER_AUDIT};
use crate::*;
use runmat_types::*;

pub const LOADLIBRARY_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "loadlibrary",
    },
    category: "interop/native",
    documentation: DOCUMENTATION,
    descriptor: &LOADLIBRARY_DESCRIPTOR,
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
            EffectKind::EnvironmentWrite,
            EffectKind::FilesystemRead,
            EffectKind::FilesystemWrite,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[
            CapabilityRequirement::Filesystem,
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
    suppress_auto_output: true,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 1, None, &mut diagnostics);
    text_at(request, 0, &mut diagnostics);
    text_at(request, 1, &mut diagnostics);
    if request.arguments.len() > 2 && !(request.arguments.len() - 2).is_multiple_of(2) {
        diagnostics.push(catalog::inference::argument_error(
            "RM-NATIVE-FFI-OPTIONS",
            "loadlibrary options must be name-value pairs",
            2,
        ));
    }
    for index in (2..request.arguments.len()).step_by(2) {
        text_at(request, index, &mut diagnostics);
        text_at(request, index + 1, &mut diagnostics);
        if let Some(name) = request.literals.literal_string_at(index) {
            if NativeLoadOption::parse(name).is_none() {
                diagnostics.push(catalog::inference::argument_error(
                    "RM-NATIVE-FFI-OPTION",
                    "loadlibrary supports Alias, IncludePath, and AddHeader",
                    index,
                ));
            }
        }
    }
    let notfound = string_cell_row();
    let warnings = ValueFact::scalar(ValueKindFact::String);
    infer_with(request, entry, vec![notfound, warnings], diagnostics)
}

const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("loadlibrary"), slug: Some("loadlibrary"),
    summary: "Load a native C shared library and prepare its declared header interface.",
    description: "`loadlibrary(library, header)` prepares the header declarations and loads the matching shared library into the current session. With one argument, the header path is inferred by replacing the library extension with `.h`. `[notfound, warnings]` returns unprepared declarations and preparation warnings; a statement call suppresses the report.",
    keywords: &["loadlibrary", "shared library", "native", "C", "header", "FFI"],
    related: &["calllib", "libfunctions", "libisloaded", "unloadlibrary", "libpointer", "libstruct"],
    sections: &[
        BuiltinDocumentationSection { heading: "Options", paragraphs: &["`Alias` selects the session library name. `IncludePath` supplies additional header search paths; `AddHeader` includes another declaration header. Options are name-value pairs and are parsed before invoking the native adapter."] },
        BuiltinDocumentationSection { heading: "Native host", paragraphs: &["Requires a native host, a compatible C compiler/header frontend, the shared library, and the exact prepared interface artifact. Browser execution reports `RunMat:Foreign:UnsupportedOnWasm`; the process-local foreign adapter owns loaded symbols and pointer leases."] },
    ],
    examples: &[BuiltinExample {
        id: "load-fixture", title: "Load the prepared C fixture",
        program: "[notfound, warnings] = loadlibrary('fixture', 'fixture.h', 'Alias', 'fixture');\nassert(iscell(notfound));\nassert(ischar(warnings) || isstring(warnings));\nassert(libisloaded('fixture'));\nunloadlibrary('fixture');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeForeignRuntime,
        fixture: NATIVE_FIXTURE,
        requirements: NATIVE_HOST,
        verification: BuiltinExampleVerification::Assertions { source: "assert(iscell(notfound)); assert(libisloaded('fixture') == false);" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Is a header sufficient to load a library?", answer: "No. The matching native library and its declared interface must both be available; symbols are validated against the prepared manifest." }],
    links: &[], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Prepared native library loading", location: "tests::native_ffi_session" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
