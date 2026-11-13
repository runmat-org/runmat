use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("addprop"), slug: Some("addprop"),
    summary: "Add a per-instance dynamic property to a dynamicprops handle object.",
    description: "`prop = addprop(obj,name)` creates a property on a live scalar handle object whose class inherits from dynamicprops and returns its matlab.metadata.DynamicProperty metadata handle.",
    keywords: &["addprop", "dynamicprops", "dynamic property", "handle", "matlab.metadata.DynamicProperty"],
    related: &["findprop", "matlab.metadata.DynamicProperty.delete", "getfield", "setfield"],
    sections: &[
        BuiltinDocumentationSection { heading: "Admission and mutation", paragraphs: &[
            "The receiver must be a valid dynamicprops handle object. The name must be a nonempty identifier and must not duplicate a class-defined or existing per-instance property. Value objects, plain handle classes, structs, numeric values, and resident numeric handles are not admitted. No initial property value is accepted.",
            "The property is immediately readable and writable through dot syntax, getfield, and setfield. Values later assigned to it retain ordinary object-property ownership, including exact integer and resident payloads; addprop itself does not inspect, convert, or gather those payloads.",
            "The metadata handle exposes Name, DefiningClass, GetAccess, SetAccess, Dependent, Hidden, Transient, NonCopyable, AbortSet, SetObservable, GetObservable, and Description. Supported writable metadata fields update the target's dynamic definition; Name and DefiningClass are read-only.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and limits", paragraphs: &[
            "Metadata allocation is admitted before property publication. The operation is host-side object mutation and a fusion boundary, not a numeric GPU kernel. A discarded result still creates the property. More than one requested output rejects before mutation.",
            "Observable get/set listener delivery and full external MATLAB MAT-file dynamic-property persistence are not implemented. RunMat object serialization retains supported property values but is not complete classdef metadata persistence. Worker transfer of a handle does not transfer this process-local object/GC ownership.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "add-value", title: "Add a property and assign a value",
            program: "classdef Sensor < dynamicprops\nend\ns = new_object('Sensor'); p = addprop(s,'gain'); s.gain = 7; value = s.gain;",
            display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(value == 7); assert(strcmp(p.Name,'gain'));" },
        },
        BuiltinExample { id: "metadata-hidden", title: "Set property metadata",
            program: "classdef Sensor < dynamicprops\nend\ns = new_object('Sensor'); p = addprop(s,'label'); p.Hidden = true; hidden = p.Hidden;",
            display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(hidden);" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does addprop work on value classes?", answer: "No. The receiver must be a live dynamicprops handle object." },
        BuiltinDocumentationFaq { question: "Are dynamic properties saved to MAT files?", answer: "Property values use the object serialization path, but complete external MATLAB dynamic-property metadata persistence is not implemented." },
        BuiltinDocumentationFaq { question: "Do observable metadata flags fire listeners?", answer: "The flags are stored and reported; full observable-property listener behavior is not implemented." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "findprop", target: BuiltinDocumentationLinkTarget::Builtin("findprop") },
        BuiltinDocumentationLink { label: "setfield", target: BuiltinDocumentationLinkTarget::Builtin("setfield") },
        BuiltinDocumentationLink { label: "getfield", target: BuiltinDocumentationLinkTarget::Builtin("getfield") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime dynamic-property owner", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/dynamicprops/operations.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Class dynamic property access and delete", location: "crates/runmat-vm/tests/functions.rs::dynamicprops_class_supports_addprop_dot_access_and_delete" }],
        notes: &["Composed native/browser, fault and cancellation gates remain open."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
