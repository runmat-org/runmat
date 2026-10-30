use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Without a target length, `pad` extends each element to match the longest text in the input.",
            "Providing a nonnegative scalar target guarantees a minimum length; single, double, and all eight integer classes are accepted from authoritative native storage, and existing text that already meets or exceeds the target is returned unchanged.",
            "Direction keywords (`'left'`, `'right'`, `'both'`) are case-insensitive; `'right'` is the default. When an odd number of pad characters is required for `'both'`, the extra character is appended to the end.",
            "`padChar` must be a single character (string scalar or 1×1 char array). The default is a space.",
            "Character arrays remain rectangular. Each row is padded independently and then widened with spaces so the array keeps MATLAB’s column-major layout.",
            "Cell arrays preserve their structure. Elements must be string scalars or 1×N character vectors and are padded while keeping their original type.",
            "Missing strings (`string(missing)`) pass through unchanged. Zero-row character arrays, including those inside cells, retain their original owner and full shape even with a positive target. A one-row empty character vector is padded like any other row.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and effects",
        paragraphs: &[
            "Text inputs are host-only. Automatically placed numeric length controls gather transparently; explicit gpuArray length controls are unsupported.",
            "This pure host operation may throw and may suspend at admitted gathers. It performs no callbacks, workspace/environment mutation, filesystem operations, randomness, or Clock reads. There is no GPU kernel, fusion route, or distributed support. VM and native calls share the canonical portable Runtime binding; these declarations are not execution evidence.",
            "Text data is host-only. Automatically placed numeric length controls gather transparently; explicit gpuArray length controls reject before gather.",
            "`pad` executes on the CPU. Numeric resident values in the text role reject without provider access, while an admitted automatic length control gathers through its owning provider.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Current implementation frontier",
        paragraphs: &[
            "Character arrays with non-singleton dimensions beyond two, including those inside cells and delimiter controls, reject before row extraction. String and cell array geometry follows the per-entry behavior described here.",
            "Target lengths use exact native integer parsing. String and cell shapes are preserved, including N-D cells. Typed missing strings are distinct from literal text. General managed text allocation and cross-backend parity remain unqualified.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Pad Strings To A Common Width",
        program: "labels = [\"GPU\"; \"Accelerate\"; \"RunMat\"];\naligned = pad(labels)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(aligned,[\"GPU       \";\"Accelerate\";\"RunMat    \"]));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Pad Strings On The Left With Zeros",
        program: "ids = [\"42\"; \"7\"; \"512\"];\nzero_padded = pad(ids, 4, 'left', '0')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(zero_padded,[\"0042\";\"0007\";\"0512\"]));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Center Text With Both-Sided Padding",
        program: "titles = [\"core\"; \"planner\"];\ncentered = pad(titles, 10, 'both', '*')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(centered,[\"***core***\";\"*planner**\"]));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Pad Character Array Rows",
        program: "chars = char(\"GPU\", \"RunMat\");\nout = pad(chars, 8)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(ischar(out)); assert(isequal(size(out),[2 8])); assert(strcmp(out(1,:),'GPU     ')); assert(strcmp(out(2,:),'RunMat  '));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Pad A Cell Array Of Character Vectors",
        program: "C = {'solver', \"planner\", 'jit'};\ncell_out = pad(C, 'right', '.')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(cell_out,{'solver.',\"planner\",'jit....'}));",
        },
    },
    BuiltinExample {
        id: "example-6",
        title: "Leave Missing Strings Unchanged",
        program: "values = [\"RunMat\", string(missing), \"GPU\"];\nkept = pad(values, 8)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(kept(1),\"RunMat  \")); assert(ismissing(kept(2))); assert(isequal(kept(3),\"GPU     \"));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pad"),
    slug: Some("pad"),
    summary: "Pad text values to target lengths with MATLAB-compatible direction and fill-character options.",
    description: "`pad` extends string/char/cellstr text values to requested lengths by adding fill characters on left, right, or both sides, using MATLAB-compatible defaults and option parsing.",
    keywords: &["pad", "pad string", "left pad", "right pad", "center text", "character arrays"],
    related: &["strip", "strcat", "lower", "upper", "compose", "erase", "eraseBetween", "extractBetween", "join", "replace", "split", "strrep", "strtrim"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What inputs does `pad` accept?",
            answer: "String scalars, string arrays, character arrays, and cell arrays containing string scalars or character vectors. Other types raise MATLAB-compatible errors.",
        },
        BuiltinDocumentationFaq {
            question: "How are direction keywords interpreted?",
            answer: "`'left'`, `'right'`, and `'both'` are supported (case-insensitive). `'right'` is the default. With `'both'`, extra characters are added to the end when an odd number of padding characters is required.",
        },
        BuiltinDocumentationFaq {
            question: "Can I shorten text with `pad`?",
            answer: "No. When the existing text is already longer than the requested target length, it is returned unchanged.",
        },
        BuiltinDocumentationFaq {
            question: "What happens when I supply a custom padding character?",
            answer: "The character must be length one. RunMat repeats it as many times as needed in the specified direction.",
        },
        BuiltinDocumentationFaq {
            question: "Do missing strings get padded?",
            answer: "Typed missing strings (displayed as `<missing>`) are passed through untouched so downstream code that checks for missing values continues to work.",
        },
        BuiltinDocumentationFaq {
            question: "How are cell array elements returned?",
            answer: "Each cell retains its type: string scalars remain strings and character vectors remain 1×N character arrays after padding.",
        },
        BuiltinDocumentationFaq {
            question: "Does `pad` change the orientation of row or column string arrays?",
            answer: "No. The shape of the input array is preserved exactly; only element lengths change.",
        },
        BuiltinDocumentationFaq {
            question: "Will `pad` run on the GPU in the future?",
            answer: "The text operation is host-only. Automatically placed numeric length controls gather transparently, while explicit gpuArray length controls reject.",
        },
    ],
    links: &[
        BuiltinDocumentationLink { label: "strip", target: BuiltinDocumentationLinkTarget::Builtin("strip") },
        BuiltinDocumentationLink { label: "strcat", target: BuiltinDocumentationLinkTarget::Builtin("strcat") },
        BuiltinDocumentationLink { label: "lower", target: BuiltinDocumentationLinkTarget::Builtin("lower") },
        BuiltinDocumentationLink { label: "upper", target: BuiltinDocumentationLinkTarget::Builtin("upper") },
        BuiltinDocumentationLink { label: "compose", target: BuiltinDocumentationLinkTarget::Builtin("compose") },
        BuiltinDocumentationLink { label: "erase", target: BuiltinDocumentationLinkTarget::Builtin("erase") },
        BuiltinDocumentationLink { label: "eraseBetween", target: BuiltinDocumentationLinkTarget::Builtin("eraseBetween") },
        BuiltinDocumentationLink { label: "extractBetween", target: BuiltinDocumentationLinkTarget::Builtin("extractBetween") },
        BuiltinDocumentationLink { label: "join", target: BuiltinDocumentationLinkTarget::Builtin("join") },
        BuiltinDocumentationLink { label: "replace", target: BuiltinDocumentationLinkTarget::Builtin("replace") },
        BuiltinDocumentationLink { label: "split", target: BuiltinDocumentationLinkTarget::Builtin("split") },
        BuiltinDocumentationLink { label: "strrep", target: BuiltinDocumentationLinkTarget::Builtin("strrep") },
        BuiltinDocumentationLink { label: "strtrim", target: BuiltinDocumentationLinkTarget::Builtin("strtrim") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/strings/transform/pad.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/pad.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Runtime text semantics", location: "crates/runmat-runtime/src/builtins/strings/transform/pad.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed identity facts", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/pad/inference.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Original unit source reference (not execution evidence)", location: "builtins::strings::transform::pad::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Original integration source reference (not execution evidence)", location: "crates/runmat-vm/tests/integer_pattern_and_string_transform_semantics.rs" },
        ],
        notes: &["Original examples are preserved as standalone asserted programs except the impossible GPU text example, whose host equivalent is explicit. Historical display outputs were unverified presentation claims and are retired rather than invented as transcripts. Source-only migration requires centralized compiler, catalog, documentation, and backend qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
