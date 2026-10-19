use super::*;
use crate::catalog::inference::{argument_error, finish_fixed};
use runmat_types::{CallInference, CallRequest, ShapeFact};

const OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Indexed duration or Format property; assignment returns the updated duration.",
}];
const fn index_input(name: &'static str, description: &'static str) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}
const SUBREF_INPUT: [BuiltinParamDescriptor; 2] = [
    index_input("obj", "Duration receiver."),
    index_input("S", "Standard substruct-compatible index path."),
];
const SUBASGN_INPUT: [BuiltinParamDescriptor; 3] = [
    index_input("obj", "Duration receiver."),
    index_input("S", "Standard substruct-compatible index path."),
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Assigned Format value.",
    },
];
pub const DURATION_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "out = duration.subsref(obj, S)",
        inputs: &SUBREF_INPUT,
        outputs: &OUT,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: errors::SUBREF_ERRORS,
};
pub const DURATION_SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "out = duration.subsasgn(obj, S, rhs)",
        inputs: &SUBASGN_INPUT,
        outputs: &OUT,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: errors::SUBSASGN_ERRORS,
};
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Duration indexing", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/duration/builtins.rs") };
const TESTS: &[BuiltinEvidenceReference] = &[BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Duration indexing and Format assignment",
    location: "crates/runmat-runtime/src/builtins/duration/tests.rs",
}];
const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [constructor::DURATION_GPU_INPUT_EXTENSION];
const SUBREF_DOC: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("duration.subsref"), slug: Some("duration.subsref"), summary: "Index a duration array or read its Format property.",
    description: "Parentheses indexing reads duration elements using the standard typed indexing path and preserves the duration class; member access reads Format. Brace indexing is unsupported. Index arrays are validated in their native class before any conversion.",
    keywords: &["duration", "subsref", "indexing", "Format"], related: &["duration"],
    sections: &[BuiltinDocumentationSection { heading: "Results and placement", paragraphs: &["The result of parentheses indexing is a host duration with the selected shape and the original Format. Accessing Format returns text. The selector may suspend if its numeric payload is resident, and compatibility admission precedes provider download; the duration payload itself is host-side."] }],
    examples: &[BuiltinExample { id: "linear-index", title: "Read one elapsed value", program: "d = hours([2 3]); one = d(2);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isduration(one)); assert(hours(one)==3);" } }],
    example_exemption: None, faqs: &[], links: &[BuiltinDocumentationLink { label: "duration", target: BuiltinDocumentationLinkTarget::Builtin("duration") }, SOURCE], media: &[], evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["Static result class is dynamic because the same method may select duration elements or return the Format text property."] }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
const SUBASGN_DOC: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("duration.subsasgn"), slug: Some("duration.subsasgn"), summary: "Assign the Format property of a duration array.",
    description: "Format property assignment updates rendering without changing elapsed day counts. Parentheses element assignment is not implemented in the current duration object path.",
    keywords: &["duration", "subsasgn", "Format"], related: &["duration"],
    sections: &[BuiltinDocumentationSection { heading: "Results and placement", paragraphs: &["The updated duration stays host-side. Invalid properties or a non-text Format value return an error; no GPU duration kernel is launched."] }],
    examples: &[BuiltinExample { id: "format", title: "Change duration rendering", program: "d = hours(2); d.Format = 'hh:mm'; n = hours(d);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isduration(d)); assert(strcmp(d.Format,'hh:mm')); assert(n==2);" } }],
    example_exemption: None, faqs: &[], links: &[BuiltinDocumentationLink { label: "duration", target: BuiltinDocumentationLinkTarget::Builtin("duration") }, SOURCE], media: &[], evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["Only Format member assignment is supported in this object path."] }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
fn infer_subref(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-DURATION-SUBSREF-ARITY",
            "subsref expects receiver and index path",
            0,
        ));
    }
    if let Some(receiver) = request.arguments.first() {
        if facts::duration_class(receiver) == Some(false) {
            diagnostics.push(argument_error(
                "RM-DURATION-SUBSREF-RECEIVER",
                "expected duration receiver",
                0,
            ));
        }
    }
    finish_fixed(
        entry,
        request,
        facts::unknown_host(ShapeFact::Unknown),
        diagnostics,
    )
}
fn infer_subsasgn(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 3 {
        diagnostics.push(argument_error(
            "RM-DURATION-SUBSASGN-ARITY",
            "subsasgn expects receiver, index path, and value",
            0,
        ));
    }
    let output = if let Some(receiver) = request.arguments.first() {
        if facts::duration_class(receiver) == Some(false) {
            diagnostics.push(argument_error(
                "RM-DURATION-SUBSASGN-RECEIVER",
                "expected duration receiver",
                0,
            ));
        }
        facts::duration(receiver.shape.clone())
    } else {
        facts::unknown_host(ShapeFact::Unknown)
    };
    finish_fixed(entry, request, output, diagnostics)
}
pub const DURATION_SUBSREF_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "duration.subsref",
    &DURATION_SUBSREF_DESCRIPTOR,
    SUBREF_DOC,
    infer_subref,
    true,
    &EXTENSIONS,
    &[],
);
pub const DURATION_SUBSASGN_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "duration.subsasgn",
    &DURATION_SUBSASGN_DESCRIPTOR,
    SUBASGN_DOC,
    infer_subsasgn,
    true,
    &EXTENSIONS,
    &[],
);
