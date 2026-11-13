use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("findprop"), slug: Some("findprop"),
    summary: "Find class or per-instance dynamic property metadata.",
    description: "`prop = findprop(obj,name)` returns class-defined property metadata, the canonical per-instance dynamic metadata handle, or empty double when no property exists.",
    keywords: &["findprop", "dynamicprops", "property metadata", "matlab.metadata.Property", "matlab.metadata.DynamicProperty"],
    related: &["addprop", "getfield", "setfield"],
    sections: &[
        BuiltinDocumentationSection { heading: "Results", paragraphs: &[
            "A valid handle-object dynamic property returns its shared matlab.metadata.DynamicProperty handle. Repeated findprop calls and the addprop result refer to the same metadata object; deleting one alias invalidates the others.",
            "A class-defined property returns a matlab.metadata.Property value object with Name, DefiningClass, GetAccess, SetAccess, and Dependent metadata. Missing names return an empty double array. A value-object dynamic definition is represented as metadata value, not a deletable handle.",
            "The receiver must be an object or live handle object. Invalid handles, numeric/logical/complex targets, all integer classes, and provider-resident numeric handles reject without conversion or provider access. Object arrays are not supported.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and limits", paragraphs: &[
            "findprop reads runtime class and object metadata without invoking user methods or numeric provider kernels. It is process-owned metadata, so worker transfer of an object value does not imply equivalent class registration or dynamic handle ownership.",
            "External MATLAB MAT-file class metadata is not decoded into dynamic-property metadata. At most one result is available; a zero-output query still validates its receiver and name.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "existing", title: "Find a dynamic property",
            program: "classdef Sensor < dynamicprops\nend\ns = new_object('Sensor'); p = addprop(s,'gain'); q = findprop(s,'gain'); name = q.Name;",
            display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(name,'gain')); assert(isvalid(q));" },
        },
        BuiltinExample { id: "missing", title: "Missing property returns empty",
            program: "classdef Sensor < dynamicprops\nend\ns = new_object('Sensor'); q = findprop(s,'missing');",
            display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isempty(q));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Can I delete class-defined metadata?", answer: "No. Only a live per-instance matlab.metadata.DynamicProperty metadata handle can be deleted." },
        BuiltinDocumentationFaq { question: "Does lookup invoke user methods?", answer: "No. It inspects class and object metadata." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "addprop", target: BuiltinDocumentationLinkTarget::Builtin("addprop") },
        BuiltinDocumentationLink { label: "getfield", target: BuiltinDocumentationLinkTarget::Builtin("getfield") },
        BuiltinDocumentationLink { label: "setfield", target: BuiltinDocumentationLinkTarget::Builtin("setfield") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime metadata lookup", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/dynamicprops/operations.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Alias invalidation and target classes", location: "builtins::introspection::dynamicprops::tests" }],
        notes: &["Composed native/browser and output-discard behavior remains to be qualified."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
