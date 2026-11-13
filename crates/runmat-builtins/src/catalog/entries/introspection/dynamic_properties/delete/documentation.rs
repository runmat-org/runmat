use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.metadata.DynamicProperty.delete"),
    slug: Some("matlab.metadata.DynamicProperty.delete"),
    summary: "Remove a dynamic property using its metadata handle.",
    description: "`delete(prop)` removes the per-instance property represented by a live matlab.metadata.DynamicProperty metadata handle and invalidates that shared metadata object.",
    keywords: &["delete", "dynamicprops", "dynamic property", "matlab.metadata.DynamicProperty"],
    related: &["addprop", "findprop", "delete"],
    sections: &[
        BuiltinDocumentationSection { heading: "Mutation and aliases", paragraphs: &[
            "The input must be a valid dynamic-property metadata handle returned by addprop or findprop. Deletion removes the target object's definition and stored value. All metadata aliases observe invalidation because findprop reuses the canonical metadata object.",
            "Subsequent dot, getfield, or setfield access to the removed property fails as an undefined property. Deletion does not delete the owner object. Integer scalars/arrays and resident numeric handles are not metadata handles and reject without conversion or provider access.",
            "The qualified method has no output. It rejects a requested result before mutation, while a zero-output statement performs deletion. This is distinct from filesystem delete.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and limits", paragraphs: &[
            "This process-owned GC/object mutation is a host fusion boundary, not a GPU kernel. Metadata aliases are invalidated before the definition is removed, with metadata restored if the owner-removal step rejects; rooted owner and metadata allocations survive collection between those steps. A removed resident property value follows ordinary object ownership; delete does not independently gather or release a caller's borrowed numeric allocation.",
            "Full observable-property listener delivery and external MATLAB MAT-file dynamic-property persistence remain unsupported.",
        ] },
    ],
    examples: &[BuiltinExample { id: "remove", title: "Remove a dynamic property",
        program: "classdef Sensor < dynamicprops\nend\ns = new_object('Sensor'); p = addprop(s,'gain'); s.gain = 7; delete(p); q = findprop(s,'gain');",
        display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isempty(q)); assert(~isvalid(p));" },
    }],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Is this filesystem delete?", answer: "No. This qualified method removes a dynamic-property definition; filesystem deletion is the separate general delete builtin." },
        BuiltinDocumentationFaq { question: "Does it delete the owner object?", answer: "No. Only the selected property is removed." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "addprop", target: BuiltinDocumentationLinkTarget::Builtin("addprop") },
        BuiltinDocumentationLink { label: "findprop", target: BuiltinDocumentationLinkTarget::Builtin("findprop") },
        BuiltinDocumentationLink { label: "delete", target: BuiltinDocumentationLinkTarget::Builtin("delete") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime property deletion", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/dynamicprops/operations.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled property removal", location: "crates/runmat-vm/tests/functions.rs::dynamicprops_class_supports_addprop_dot_access_and_delete" }],
        notes: &["Composed deletion fault/cancellation and browser/worker qualification remain open."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
