use crate::*;

pub(in super::super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("assignin"),
    slug: Some("assignin"),
    summary: "Assign an already evaluated value to a variable in the base or caller workspace.",
    description: "`assignin(workspace, name, value)` transfers the already evaluated value to a whole variable in the selected `base` or `caller` workspace, creating that binding when necessary. It has no public output and performs no numeric conversion.",
    keywords: &["assignin", "workspace", "base", "caller", "variable"],
    related: &["eval", "evalin", "isvarname", "namelengthmax"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Arguments and effects",
            paragraphs: &[
                "The workspace selector is a character row or scalar string naming `base` or `caller`. At the command prompt they address the same workspace; inside a function, `caller` addresses the calling function's workspace.",
                "The name is one valid variable identifier, not an indexed expression: it begins with an English letter, continues with English letters, digits, or underscores, is not a language keyword, and does not exceed `namelengthmax` characters.",
                "The value expression is evaluated in the scope containing the `assignin` call before the resulting scalar or array value is transferred to the selected workspace. A request for any public output raises RunMat:assignin:TooManyOutputs.",
                "Integer values retain exact class, shape, and authoritative storage; wide uint64 values are not routed through double.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU execution",
            paragraphs: &[
                "Assignment is structural. A resident value retains its provider-owned representation; `assignin` itself launches no numeric kernel and does not gather or re-upload that payload.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "base-integer-array",
            title: "Export an exact integer array to the base workspace",
            program: "base = uint64(9007199254740992); assignin('base', 'counts', base + uint64([1 2])); selected = evalin('base', 'counts');",
            display_output: Some("selected retains uint64 values above flintmax"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isa(selected, 'uint64')); assert(isequal(selected, base + uint64([1 2])));" },
        },
        BuiltinExample {
            id: "caller-workspace",
            title: "Assign in the calling workspace",
            program: "outer_x = 5; result = helper();\nfunction out = helper()\n    out = evalin('caller', 'outer_x + 2');\n    assignin('caller', 'caller_z', 11);\nend",
            display_output: Some("result = 7; caller_z = 11"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(result == 7); assert(evalin('base', 'caller_z') == 11);" },
        },
        BuiltinExample {
            id: "precomputed-expression",
            title: "Evaluate an expression before assigning its result",
            program: "publish_result([1 2 3]); selected = evalin('base', 'published_result');\nfunction publish_result(A)\n    result = sum(A);\n    assignin('base', 'published_result', result);\nend",
            display_output: Some("selected = 6"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(selected == 6);" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does `assignin` evaluate the value in the destination workspace?", answer: "No. Its value expression is evaluated in the calling scope, then the resulting value is transferred." },
        BuiltinDocumentationFaq { question: "Can the name contain indexing such as A(3)?", answer: "No. `assignin` writes a whole workspace variable; `evalin` can evaluate destination-workspace indexed source when necessary." },
        BuiltinDocumentationFaq { question: "Are integer arrays converted to double?", answer: "No. Exact integer class, shape, and payload are preserved, including uint64 values beyond flintmax." },
        BuiltinDocumentationFaq { question: "Does a resident GPU value move to the CPU?", answer: "No. The selected workspace retains the provider-owned value without a gather performed by `assignin`." },
        BuiltinDocumentationFaq { question: "What value does `assignin` return?", answer: "It has no public output; requesting one is an error." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "evalin", target: BuiltinDocumentationLinkTarget::Builtin("evalin") },
        BuiltinDocumentationLink { label: "isvarname", target: BuiltinDocumentationLinkTarget::Builtin("isvarname") },
        BuiltinDocumentationLink { label: "namelengthmax", target: BuiltinDocumentationLinkTarget::Builtin("namelengthmax") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Runtime workspace assignment binding", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/dynamic_workspace.rs") },
            BuiltinDocumentationLink { label: "VM workspace assignment dispatch", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-vm/src/call/builtins.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Base and caller workspace updates", location: "dynamic_workspace_evalin_base_and_assignin_update_base_workspace" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Exact integer-class transfer", location: "assignin_compiled_transfer_preserves_all_integer_classes_exactly" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
