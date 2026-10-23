use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("getmethod"),
    slug: Some("getmethod"),
    summary: "Create a method-bound function handle from an object or class reference.",
    description: "getmethod(receiver, name) is a RunMat-only extension that creates a callable method handle. Object and handle-object receivers are captured for later instance dispatch. Class references select static method bindings when available, or retain a qualified named-function route. Creating a handle does not execute the method.",
    keywords: &[
        "classdef",
        "object",
        "introspection"
    ],
    related: &[
        "new_object",
        "str2func",
        "functions"
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Inputs, output and compatibility",
            paragraphs: &[
                "receiver must be a host object, handle object, or class reference. name must be scalar string text or a character row; surrounding whitespace is trimmed and the remaining name must be nonempty. Numeric and integer receivers or names reject without conversion, provider access, or gather.",
                "Zero or one output is supported. The output is a function handle; its eventual parameter/output facts, capabilities and resolved callable identity remain runtime-dependent. The getmethod-bound-method-handle extension is rejected when RunMat extensions are disabled, with RunMat:compatibility:GetmethodExtension."
            ]
        },
        BuiltinDocumentationSection {
            heading: "Binding and access",
            paragraphs: &[
                "The current class scope controls private and protected access. An inaccessible existing method rejects with RunMat:MethodPrivate. Public methods are accessible; protected access requires a compatible caller class or subclass.",
                "A known instance method preserves its registered semantic binding and captures the receiver. For an unknown instance member, the closure captures the receiver, trimmed method name and caller scope for the existing bound-method dispatcher; successful handle creation does not prove that later dispatch will succeed.",
                "A class reference with a registered static method binding returns a closure without receiver captures. Other class-reference requests use the existing qualified str2func route. A handle object capture refers to the same handle; a value-object capture follows existing value semantics."
            ]
        },
        BuiltinDocumentationSection {
            heading: "Execution and resource limits",
            paragraphs: &[
                "The contextual binding retains its original BuiltinArguments throughout construction and scopes the selected RuntimeContext over registry lookup, access checks and closure creation. No alternate registry or executor is introduced.",
                "Cancellation and revoked allocation authority are checked before cloning controls, context-scope allocation or creating captures. Closure metadata and capture backing have no exact finite-resource bound; managed invocations refuse with RunMat:Placement:InsufficientResourceEvidence rather than escaping the budget.",
                "Construction is native/WASM host-only and outside numeric fusion. No accelerator provider, distributed execution or implicit gather is available. Invoking the returned handle is a separate operation whose actual target may suspend, throw or perform arbitrary user effects."
            ]
        }
    ],
    examples: &[
        BuiltinExample {
            id: "deferred-instance-handle",
            title: "Create a deferred bound method handle",
            program: "obj = new_object('ExampleReceiver');\nf = getmethod(obj, 'compute');",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(f, 'function_handle'));"
            }
        },
        BuiltinExample {
            id: "reject-numeric",
            title: "Reject a numeric receiver",
            program: "f = getmethod(1, 'compute');",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::ExpectedError {
                identifier: "RunMat:GetMethodReceiverUnsupported"
            }
        }
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does a handle prove that the method exists?",
            answer: "No. Unknown instance methods retain deferred bound dispatch, so the error may arise when the handle is called."
        },
        BuiltinDocumentationFaq {
            question: "Does getmethod run the method?",
            answer: "No. It constructs a handle; invoke that handle separately to run the method."
        }
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "new_object",
            target: BuiltinDocumentationLinkTarget::Builtin("new_object")
        },
        BuiltinDocumentationLink {
            label: "str2func",
            target: BuiltinDocumentationLinkTarget::Builtin("str2func")
        },
        BuiltinDocumentationLink {
            label: "functions",
            target: BuiltinDocumentationLinkTarget::Builtin("functions")
        }
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime implementation",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/getmethod.rs")
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
