use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ismethod"),
    slug: Some("ismethod"),
    summary: "Return true when an object has a public method.",
    description: "`ismethod(obj, methodName)` returns true when `methodName` names a public method available on the runtime class of object instance `obj`. RunMat resolves inherited methods through its class registry and accepts method names as string scalars or row character vectors.",
    keywords: &["ismethod",
        "method",
        "classdef",
        "object",
        "handle",
        "inheritance"],
    related: &["class",
        "isa",
        "isobject"],
    sections: &[BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["Value-class and handle-class instances are inspected by runtime class name.",
                "Class-name text and class references are values rather than object instances, so they return `logical(0)`.",
                "Explicit gpuArray values use the documented `gpuArray` object-function surface without gathering provider buffers; internal automatic residency remains transparent and returns `logical(0)` for wrapper methods.",
                "Inherited public methods return `logical(1)`.",
                "Public static methods registered for the class return `logical(1)` because they are class methods visible through the class metadata.",
                "Private and protected methods return `logical(0)` for this public predicate.",
                "Missing class names, missing method names, empty method names, ordinary numeric data, structs, cells, and function handles return `logical(0)`.",
                "The method-name argument must be a string scalar or row character vector; nonscalar string arrays and nontext values raise `RunMat:IsMethodNameInvalid`.",
                "This builtin does not call user code, construct metaclass objects, or enumerate method lists."]
        },
        BuiltinDocumentationSection {
            heading: "Execution and placement",
            paragraphs: &["Metadata-only predicate. RunMat inspects host class registry metadata for object instances and explicit gpuArray values without gathering GPU buffers; automatic residency remains transparent."]
        },
        BuiltinDocumentationSection {
            heading: "Visibility limits",
            paragraphs: &["Public access and inheritance are checked through the runtime class registry. MATLAB’s additional nonhidden-method restriction is not yet represented by this query."]
        }
    ],
    examples: &[BuiltinExample {
            id: "example-1",
            title: "Check an object method",
            program: "obj = Point();\ntf = ismethod(obj, \"step\")",
            display_output: Some("tf = logical(1)"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "ismethod-1"
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
                source: "assert(islogical(tf) && isscalar(tf) && tf);"
            }
        },
        BuiltinExample {
            id: "example-2",
            title: "Check a missing method",
            program: "obj = Point();\ntf = ismethod(obj, \"notARealMethod\")",
            display_output: Some("tf = logical(0)"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "ismethod-2"
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
                source: "assert(islogical(tf) && ~tf);"
            }
        },
        BuiltinExample {
            id: "example-3",
            title: "Class-name text is not an object receiver",
            program: "tf = ismethod(\"MyClass\", \"step\")",
            display_output: Some("tf = logical(0)"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(islogical(tf) && ~tf);"
            }
        }
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
            question: "Does `ismethod` call the method?",
            answer: "No. It only checks RunMat class metadata."
        },
        BuiltinDocumentationFaq {
            question: "Does it report private methods?",
            answer: "RunMat returns false for private and protected registry entries. Method Hidden attributes are not currently represented in this runtime method-query surface."
        },
        BuiltinDocumentationFaq {
            question: "Can I pass a class name instead of an object?",
            answer: "No. `ismethod` inspects an object instance. Class-name text and class references return false."
        }
    ],
    links: &[BuiltinDocumentationLink {
            label: "class",
            target: BuiltinDocumentationLinkTarget::Documentation("./class")
        },
        BuiltinDocumentationLink {
            label: "isa",
            target: BuiltinDocumentationLinkTarget::Documentation("./isa")
        },
        BuiltinDocumentationLink {
            label: "isobject",
            target: BuiltinDocumentationLinkTarget::Documentation("./isobject")
        },
        BuiltinDocumentationLink {
            label: "MATLAB reference",
            target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/ismethod.html")
        }
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
                label: "`crates/runmat-runtime/src/builtins/introspection/ismethod.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/ismethod.rs")
            }
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Runtime semantic tests",
            location: "crates/runmat-runtime/src/builtins/introspection/ismethod.rs::tests",
        }],
        notes: &[]
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial)
}
;
