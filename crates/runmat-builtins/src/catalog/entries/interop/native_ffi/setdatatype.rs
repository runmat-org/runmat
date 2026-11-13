use super::contract::{FOREIGN_LINK, HOST_PLACEMENT};
use super::descriptor::{scalar_text, ERRORS};
use super::documentation_support::{
    NATIVE_FIXTURE, NATIVE_HOST, POINTER_IMPLEMENTATION as IMPLEMENTATION,
};
use super::inference_support::{arity, infer_with, text_at};
use crate::*;
use runmat_types::*;

const POINTER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "pointer",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Opaque borrowed native pointer handle.",
};
const DIMENSIONS: BuiltinParamDescriptor = BuiltinParamDescriptor { name: "dimensions", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Variadic, default: None, description: "One or more positive, checked scalar integer values in exact integer or integral-double storage." };
pub const SETDATATYPE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "setdatatype(pointer, type, dimensions...)",
        inputs: &[
            POINTER,
            scalar_text(
                "type",
                "Scalar text, number, or logical normalized C pointer type.",
            ),
            DIMENSIONS,
        ],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
const INTEGER_INPUTS: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability { name: "dimensions", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The compatibility reference specifies double dimensions. The current RunMat runtime also accepts exact integer-class dimensions, checked positive and converted to usize." }];
const INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[BuiltinIntegerCapabilityDescriptor { form: "setdatatype(pointer, type, integer_dimensions...)", inputs: INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Dimension conversion and extent multiplication are checked before publishing the borrowed pointer view; this statement returns no public output." }];

pub const SETDATATYPE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "setdatatype",
    },
    category: "interop/native",
    documentation: DOCUMENTATION,
    descriptor: &SETDATATYPE_DESCRIPTOR,
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
    integer_capabilities: INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 3, None, &mut diagnostics);
    if request.arguments.first().is_some_and(|fact| {
        !matches!(
            fact.kind,
            ValueKindFact::Unknown | ValueKindFact::Foreign(_)
        )
    }) {
        diagnostics.push(catalog::inference::argument_error(
            "RM-NATIVE-FFI-POINTER",
            "setdatatype requires a native pointer handle",
            0,
        ));
    }
    text_at(request, 1, &mut diagnostics);
    for index in 2..request.arguments.len() {
        if !matches!(
            request.arguments[index].kind,
            ValueKindFact::Unknown | ValueKindFact::Numeric(_)
        ) {
            diagnostics.push(catalog::inference::argument_error(
                "RM-NATIVE-FFI-DIMENSION",
                "pointer dimensions must be positive integer scalars",
                index,
            ));
        }
    }
    infer_with(request, entry, vec![], diagnostics)
}

const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("setdatatype"), slug: Some("setdatatype"),
    summary: "Declare the pointee type and extent of an opaque native pointer.",
    description: "`setdatatype(pointer, type, dimensions...)` records a typed view of an opaque pointer returned by a native library. It does not allocate or copy the pointee. It applies only to an opaque borrowed pointer, not caller-owned `libpointer` storage, and requires at least one checked dimension.",
    keywords: &["setdatatype", "native", "pointer", "C", "extent"],
    related: &["libpointer", "calllib", "loadlibrary"],
    sections: &[BuiltinDocumentationSection { heading: "Borrowed memory", paragraphs: &["The foreign handle retains its original borrowed ownership, origin affinity, and session lifetime. Dimensions are validated with checked multiplication before the view is published; assigning a type does not authorize an unsafe copy from an unknown address.", "The compatibility reference specifies double-valued dimensions. This RunMat implementation also accepts exact integer-class scalar dimensions; that acceptance should not be assumed portable to MATLAB."] }],
    examples: &[BuiltinExample {
        id: "borrowed-int32", title: "Type an opaque pointer returned from C",
        program: "loadlibrary('fixture', 'fixture.h', 'Alias', 'fixture');\np = calllib('fixture', 'fixture_borrowed_value', int32(1));\nsetdatatype(p, 'int32Ptr', 1);\nassert(p.Value == int32(17));\nunloadlibrary('fixture');",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeForeignRuntime,
        fixture: NATIVE_FIXTURE, requirements: NATIVE_HOST,
        verification: BuiltinExampleVerification::Assertions { source: "assert(p.Value == int32(17));" },
    }],
    example_exemption: None, faqs: &[], links: &[BuiltinDocumentationLink { label: "MATLAB setdatatype input types", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/lib.pointer.setdatatype.html") }], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Borrowed pointer type and extent", location: "native_ffi_session::library_and_pointer_state_are_owned_by_one_runtime_session" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
