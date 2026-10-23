use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("metaclass"),
    slug: Some("metaclass"),
    summary: "Return metadata for the runtime class of a value.",
    description: "`metaclass(object)` returns RunMat's metadata reference for the runtime class of `object`. RunMat represents MATLAB `matlab.metadata.Class` values as `ClassRef` metadata tokens, the same representation produced by the `?ClassName` parser path.",
    keywords: &["metaclass",
        "meta.class",
        "classdef",
        "object",
        "introspection"],
    related: &["class",
        "superclasses"],
    sections: &[BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["Object and handle-object inputs return a metadata reference for their registered runtime class.",
                "Text inputs describe the runtime text class. A row character vector returns metadata for `char`; a string scalar or string array returns metadata for `string`.",
                "Class references are ordinary metadata objects to `metaclass` and return metadata for `meta.class`.",
                "Explicit gpuArray inputs return metadata for `gpuArray`; internal automatic residency returns metadata for the underlying host class without gathering provider buffers.",
                "Numeric, logical, cell, struct, function-handle, listener, and MException values return metadata for their RunMat runtime class.",
                "Every integer scalar and array reports its exact signed class and width (`int8` through `uint64`) from value or coherent resident metadata without reading or converting payload elements.",
                "RunMat does not yet materialize full MATLAB `matlab.metadata.Class` object properties such as `PropertyList`, `MethodList`, or `SuperclassList`; those remain part of the broader metadata-object work."]
        },
        BuiltinDocumentationSection {
            heading: "Execution and placement",
            paragraphs: &["Metadata-only builtin. Explicit gpuArray values expose gpuArray metadata; internal automatic residency exposes underlying host-class metadata. Neither path gathers provider buffers."]
        }
    ],
    examples: &[BuiltinExample {
            id: "example-1",
            title: "Get metadata for an object",
            program: "obj = Point();\nmc = metaclass(obj)",
            display_output: Some("mc = <class Point>"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "metaclass-1"
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[BuiltinFilesystemEntry::File {
                        relative_path: "Point.m",
                        content: BuiltinFixtureContent::Utf8("classdef Point\nmethods\nfunction obj = Point()\nend\nfunction y = step(obj)\ny = 1;\nend\nend\nend\n")
                    }
                ]
            }
            ),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(mc,'meta.class'));"
            }
        },
        BuiltinExample {
            id: "example-2",
            title: "Text inputs describe text classes",
            program: "mc = metaclass('Point')",
            display_output: Some("mc = <class char>"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(mc,'meta.class'));"
            }
        },
        BuiltinExample {
            id: "example-3",
            title: "Use the question-mark operator for class-name metadata",
            program: "mc = ?Point",
            display_output: Some("mc = <class Point>"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "metaclass-3"
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[BuiltinFilesystemEntry::File {
                        relative_path: "Point.m",
                        content: BuiltinFixtureContent::Utf8("classdef Point\nmethods\nfunction obj = Point()\nend\nfunction y = step(obj)\ny = 1;\nend\nend\nend\n")
                    }
                ]
            }
            ),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(mc,'meta.class'));"
            }
        }
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
            question: "Does `metaclass(\"Point\")` look up the Point class?",
            answer: "No. This follows MATLAB semantics: text passed to `metaclass` is a text value, so RunMat returns metadata for `string` or `char`. Use `?Point` or `classref(\"Point\")` for class-name metadata."
        },
        BuiltinDocumentationFaq {
            question: "Is the result a full MATLAB metadata object?",
            answer: "Not yet. RunMat returns its existing `ClassRef` metadata token. Static member and method dispatch already use this token, but full metadata property enumeration is still tracked by the object/type-introspection umbrella."
        }
    ],
    links: &[BuiltinDocumentationLink {
            label: "class",
            target: BuiltinDocumentationLinkTarget::Documentation("./class")
        },
        BuiltinDocumentationLink {
            label: "superclasses",
            target: BuiltinDocumentationLinkTarget::Documentation("./superclasses")
        }
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
                label: "`crates/runmat-runtime/src/builtins/introspection/metaclass.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/metaclass.rs")
            }
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Runtime semantic tests",
            location: "crates/runmat-runtime/src/builtins/introspection/metaclass.rs::tests",
        }],
        notes: &[]
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial)
}
;
