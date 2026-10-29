//! Qualified native interface preparation uses the same canonical link owner
//! as the legacy C ABI frontdoors, not an unclassified external-name escape.
use super::contract::{NativeBuildOption, FOREIGN_LINK, HOST_PLACEMENT};
use super::descriptor::{scalar_text, ERRORS};
use super::documentation_support::{LOAD_IMPLEMENTATION, NATIVE_FIXTURE, NATIVE_HOST};
use super::inference_support::{arity, infer_with, text_at};
use crate::*;
use runmat_types::*;

pub const CLIBGEN_BUILD_INTERFACE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "interface = clibgen.buildInterface(header, 'Libraries', library, [name, value]...)",
        inputs: &[
            scalar_text("header", "C declaration header path."),
            scalar_text(
                "optionName",
                "Libraries, InterfaceName, IncludePath or IncludeDirectories.",
            ),
            scalar_text("optionValue", "One scalar text-coercible option value."),
            BuiltinParamDescriptor {
                name: "options",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Variadic,
                default: None,
                description: "Additional name-value pairs; include paths may repeat.",
            },
        ],
        outputs: &[BuiltinParamDescriptor {
            name: "interface",
            ty: BuiltinParamType::StringScalar,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Loaded session interface alias, not a MATLAB C++ definition object.",
        }],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};

const INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "Primitive integer scalars are converted to preparation text, not native ABI data or numeric computation.",
};

pub const CLIBGEN_BUILD_INTERFACE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "clibgen.buildInterface",
    },
    category: "interop/native",
    documentation: DOCUMENTATION,
    descriptor: &CLIBGEN_BUILD_INTERFACE_DESCRIPTOR,
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
    integer_audit: Some(&INTEGER_AUDIT),
    suppress_auto_output: true,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 3, None, &mut diagnostics);
    text_at(request, 0, &mut diagnostics);
    if request.arguments.len().is_multiple_of(2) {
        diagnostics.push(catalog::inference::argument_error(
            "RM-NATIVE-FFI-OPTIONS",
            "buildInterface requires a header and name-value pairs",
            1,
        ));
    }
    let mut libraries = false;
    let mut unknown_option = false;
    for index in (1..request.arguments.len()).step_by(2) {
        text_at(request, index, &mut diagnostics);
        text_at(request, index + 1, &mut diagnostics);
        match request.literals.literal_string_at(index) {
            Some(name) => match NativeBuildOption::parse(&name) {
                Some(NativeBuildOption::Libraries) => libraries = true,
                Some(_) => {}
                None => diagnostics.push(catalog::inference::argument_error(
                    "RM-NATIVE-FFI-OPTION",
                    "unsupported C ABI interface preparation option",
                    index,
                )),
            },
            None => unknown_option = true,
        }
    }
    if !libraries && !unknown_option {
        diagnostics.push(catalog::inference::argument_error(
            "RM-NATIVE-FFI-LIBRARY",
            "buildInterface requires the Libraries option",
            1,
        ));
    }
    infer_with(
        request,
        entry,
        vec![ValueFact::scalar(ValueKindFact::String)],
        diagnostics,
    )
}

const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("clibgen.buildInterface"), slug: Some("clibgen-buildinterface"),
    summary: "Prepare, publish and load a declared native C interface in this session.",
    description: "`clibgen.buildInterface(header, 'Libraries', library)` prepares a C ABI interface for one actual shared-library path, publishes its digest-bound adjacent `.runmat.json` manifest, and loads the session alias. `InterfaceName` overrides the default header stem; repeated `IncludePath` or `IncludeDirectories` supplies header search paths. The returned scalar string is that loaded alias. Calls through `clib.alias.symbol` use the same selected foreign-runtime host and exact prepared declarations.",
    keywords: &["clibgen", "buildInterface", "C", "header", "native", "FFI"],
    related: &["loadlibrary", "calllib", "libfunctions", "unloadlibrary"],
    sections: &[
        BuiltinDocumentationSection { heading: "Supported preparation", paragraphs: &["The header and each option name/value must be scalar text-coercible input. Libraries is required and names one actual shared-library file. Unsupported options, odd option tails and excessive outputs reject before invoking the adapter. An existing session alias cannot be replaced by this call."] },
        BuiltinDocumentationSection { heading: "Native execution and artifacts", paragraphs: &["A native host, C compiler/header frontend and the matching library are required. Preparation reads headers and library bytes and writes an adjacent canonical manifest. The selected unit admits the native-ffi adapter with Process execution-stack requirements; no ambient foreign port, numerical accelerator or distributed execution is implied. Browser execution refuses native preparation."] },
        BuiltinDocumentationSection { heading: "Current compatibility limits", paragraphs: &["This is the existing RunMat C ABI preparation path. It does not implement MATLAB's C++ interface-definition customization/build workflow, does not accept multiple libraries, and returns a loaded alias rather than a MATLAB definition object. General MATLAB clibgen parity is not claimed."] },
    ],
    examples: &[BuiltinExample {
        id: "build-c-interface", title: "Prepare and call a C interface",
        program: "prepared = dir('*.runmat.json');\nassert(numel(prepared) == 1);\nlibrary = extractBefore(prepared(1).name, '.runmat.json');\ninterface = clibgen.buildInterface('fixture.h', 'Libraries', library, 'InterfaceName', 'built_fixture');\nassert(strcmp(interface, 'built_fixture'));\ntotal = clib.built_fixture.fixture_add(int32(20), int32(22));\nassert(total == int32(42));\nunloadlibrary('built_fixture');",
        display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeForeignRuntime,
        fixture: NATIVE_FIXTURE, requirements: NATIVE_HOST,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(interface, 'built_fixture')); assert(total == int32(42)); assert(~libisloaded('built_fixture'));" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does this create a MATLAB C++ definition object?", answer: "No. RunMat currently prepares one declared C ABI library, publishes its exact interface artifact and returns the loaded session alias." }],
    links: &[], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: LOAD_IMPLEMENTATION,
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Qualified-only native preparation admission", location: "runmat-core/tests/native_ffi.rs::prepared_interface_is_installed_and_admitted_before_session_execution" }],
        notes: &["The example requires the typed native-FFI fixture preparation; its one adjacent manifest identifies the actual target-specific library filename without assuming a platform extension. A header alone is not an admitted native library."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
