use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("new_object"),
    slug: Some("new_object"),
    summary: "Create a class instance initialized from registered property defaults.",
    description: "new_object(class_name) is a RunMat object-construction entry. It creates a scalar object from the selected class registry, including inherited nonstatic default properties, but does not call a user constructor method. A class with handle ancestry returns a handle instance; an unknown valid class name produces an empty value object.",
    keywords: &[
        "classdef",
        "object",
        "introspection"
    ],
    related: &[
        "isobject",
        "class"
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Inputs and outputs",
            paragraphs: &[
                "class_name accepts scalar string text or a character row. The existing runtime also formats scalar integers using exact decimal text, and scalar doubles and logicals using their legacy textual form. This formatting does not imply that the resulting text is a valid class identity. Numeric arrays and provider-resident values are not gathered.",
                "Zero or one output is supported; the output class follows class_name. The property set and handle/value semantics depend on the selected runtime registry and cannot be inferred from the name alone."
            ]
        },
        BuiltinDocumentationSection {
            heading: "Construction and errors",
            paragraphs: &[
                "Registered class defaults are traversed root-to-leaf; static properties are omitted. Default values retain their existing shared roots. Unknown valid class identities are permitted and have no declared properties.",
                "This helper initializes defaults only. It does not invoke class_name(args), evaluate arbitrary source, or establish a workspace variable by itself. Abstract classes reject with RunMat:AbstractMethodMissing. Malformed class names reject with RunMat:InvalidClassName; retained selected-owner mismatches and allocation failures preserve their existing diagnostics."
            ]
        },
        BuiltinDocumentationSection {
            heading: "Execution and resource limits",
            paragraphs: &[
                "The canonical binding retains its full BuiltinArguments owner and scopes the selected RuntimeContext around the existing class service. Registry reads and GC/default backing remain conservatively modeled as runtime-owned impure effects.",
                "Cancellation and stale allocation authority are checked before text projection, scope allocation, or object construction. Finite managed construction is refused with RunMat:Placement:InsufficientResourceEvidence until registry/default/handle backing has an exact physical bound. This is not a bounded-host success claim.",
                "Construction is host-only on native and WASM, a fusion boundary, and unsupported for distributed placement. There is no GPU provider or implicit gather."
            ]
        }
    ],
    examples: &[
        BuiltinExample {
            id: "empty-class",
            title: "Construct an empty instance for an unregistered class",
            program: "obj = new_object('ExampleEmpty');\nname = class(obj);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isobject(obj)); assert(strcmp(name, 'ExampleEmpty'));"
            }
        },
        BuiltinExample {
            id: "invalid-class",
            title: "Reject a malformed class identity",
            program: "obj = new_object('bad..Class');",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::ExpectedError {
                identifier: "RunMat:InvalidClassName"
            }
        }
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does new_object call my constructor method?",
            answer: "No. It initializes registered nonstatic property defaults; ordinary class construction is the route for invoking a user constructor."
        },
        BuiltinDocumentationFaq {
            question: "Is this the MATLAB constructor API?",
            answer: "No. This is a RunMat-native construction helper, not a claim of MATLAB new_object compatibility."
        }
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "isobject",
            target: BuiltinDocumentationLinkTarget::Builtin("isobject")
        },
        BuiltinDocumentationLink {
            label: "class",
            target: BuiltinDocumentationLinkTarget::Builtin("class")
        }
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime implementation",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/new_object.rs")
            }
        ],
        verification: &[],
        notes: &[
            "First canonical documentation for the existing public identity; no legacy JSON sidecar exists. These source-level standalone examples require central native/WASM qualification; no compiled, browser, or finite-budget receipt is claimed.",
            "RunMat-native API; no corresponding MATLAB public function is claimed. Remaining contract maturity is Incomplete and documentation status Partial."
        ]
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial)
};
