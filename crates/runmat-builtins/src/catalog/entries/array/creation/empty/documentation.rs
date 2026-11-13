use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    example_exemption: None,
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("empty"),
    slug: Some("empty"),
    summary: "Create an empty array of a specified class.",
    description: concat!(
        "`ClassName.empty` creates a 0-by-0 empty array of the receiver class. ",
        "Size arguments can specify other empty shapes, but at least one ",
        "normalized dimension must be zero.",
    ),
    keywords: &[
        "empty",
        "ClassName.empty",
        "integer",
        "preallocate",
        "array",
    ],
    related: &[],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                concat!(
                    "RunMat registers class-qualified forms for `int8.empty`, ",
                    "`int16.empty`, `int32.empty`, `int64.empty`, `uint8.empty`, ",
                    "`uint16.empty`, `uint32.empty`, and `uint64.empty`.",
                ),
                concat!(
                    "All eight integer classes are accepted for scalar dimension controls ",
                    "and row size vectors. Values are decoded from authoritative integer ",
                    "storage without conversion through double.",
                ),
                concat!(
                    "Signed negative dimensions become zero. Nonnegative dimensions must be ",
                    "integer-valued and within the platform size range.",
                ),
                concat!(
                    "A single scalar dimension requests a square shape. A size-vector ",
                    "argument must be a row vector; column vectors reject.",
                ),
                concat!(
                    "Trailing singleton dimensions after dimension two are removed. At ",
                    "least one remaining dimension must be zero.",
                ),
                "Integer outputs contain exact empty `IntegerStorage` of the receiver class.",
                concat!(
                    "The global `empty(...)` shorthand and its trailing typename selector ",
                    "are RunMat-only extensions. MATLAB-compatible mode requires ",
                    "class-qualified syntax.",
                ),
                concat!(
                    "The former global `\"like\"` shorthand is not retained because it did ",
                    "not inherit output class and is not part of the documented ",
                    "class-static surface.",
                ),
                concat!(
                    "User-defined and general object-class empty arrays require broader ",
                    "class-array storage and static-method dispatch support.",
                ),
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU residency and provider behavior",
            paragraphs: &[
                concat!(
                    "Integer class-static forms construct host empty storage. Resident ",
                    "numeric size controls are accepted only in RunMat mode and reject ",
                    "before provider access when extensions are disabled.",
                ),
                concat!(
                    "Primitive integer ClassName.empty forms construct exact host integer ",
                    "storage. A resident size control is a compatibility-gated RunMat ",
                    "extension and is gathered only after validation and gating.",
                ),
            ],
        },
        BuiltinDocumentationSection {
            heading: "Syntax",
            paragraphs: &[
                concat!(
                    "A = ClassName.empty\nA = ClassName.empty(sz1, ..., szN)\nA = ",
                    "ClassName.empty(sizeVector)",
                ),
                "At least one normalized dimension must be zero.",
                "A size vector must be a row vector.",
                "Signed negative dimensions are treated as zero.",
                "Trailing singleton dimensions after dimension two are omitted.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "gpu-empty",
            title: "Create an explicitly GPU-resident empty array",
            program: "A = empty(0, 3, 'gpuArray');",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Wgpu,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isgpuarray(A)); H = gather(A); assert(isempty(H)); assert(isequal(size(H), [0 3]));",
            },
        },
        BuiltinExample {
            id: "example-1",
            title: "Create an empty unsigned integer array",
            program: "A = uint8.empty",
            display_output: Some("`A` is a 0-by-0 uint8 array with authoritative uint8 storage."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: concat!(
                    "assert(isempty(A)); assert(isequal(size(A), [0 0])); ",
                    "assert(strcmp(class(A), \"uint8\"));",
                ),
            },
        },
        BuiltinExample {
            id: "example-2",
            title: "Create an empty integer array with a nonzero second dimension",
            program: "A = int16.empty(int64(0), uint64(5))",
            display_output: Some("`A` is a 0-by-5 int16 array."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: concat!(
                    "assert(isempty(A)); assert(isequal(size(A), [0 5])); ",
                    "assert(strcmp(class(A), \"int16\"));",
                ),
            },
        },
        BuiltinExample {
            id: "example-3",
            title: "Use a row size vector",
            program: "A = uint64.empty(uint32([0 0 6]))",
            display_output: Some("`A` is a 0-by-0-by-6 uint64 array."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: concat!(
                    "assert(isempty(A)); assert(isequal(size(A), [0 0 6])); ",
                    "assert(strcmp(class(A), \"uint64\"));",
                ),
            },
        },
        BuiltinExample {
            id: "example-4",
            title: "RunMat global empty shorthand",
            program: "A = empty(0, 4, \"single\");",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: concat!(
                    "assert(isempty(A)); assert(isequal(size(A), [0 4])); ",
                    "assert(strcmp(class(A), \"single\"));",
                ),
            },
        },
    ],
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does the class come from a size argument?",
            answer: concat!(
                "No. In MATLAB-compatible syntax the receiver in `ClassName.empty` ",
                "determines the output class. Integer size arguments are structural ",
                "controls only.",
            ),
        },
        BuiltinDocumentationFaq {
            question: "Are wide uint64 dimensions converted through double?",
            answer: concat!(
                "No. Host integer controls are decoded exactly from integer storage and ",
                "checked against the platform size range.",
            ),
        },
        BuiltinDocumentationFaq {
            question: "Can empty create a nonempty array?",
            answer: concat!(
                "No. After signed-negative clamping and trailing-singleton ",
                "normalization, at least one dimension must be zero.",
            ),
        },
        BuiltinDocumentationFaq {
            question: "Can I call empty as a global function?",
            answer: concat!(
                "Only in RunMat mode. The global shorthand is explicitly ",
                "compatibility-gated; portable MATLAB-compatible code uses ",
                "`ClassName.empty`.",
            ),
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "MathWorks empty documentation",
            target: BuiltinDocumentationLinkTarget::External(
                "https://www.mathworks.com/help/matlab/ref/empty.html",
            ),
        },
        BuiltinDocumentationLink {
            label: "isempty",
            target: BuiltinDocumentationLinkTarget::Builtin("isempty"),
        },
        BuiltinDocumentationLink {
            label: "zeros",
            target: BuiltinDocumentationLinkTarget::Builtin("zeros"),
        },
        BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/array/creation/empty.rs`",
            target: BuiltinDocumentationLinkTarget::Source(concat!(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/s",
                "rc/builtins/array/creation/empty.rs",
            )),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Semantic and boundary tests",
            location: "builtins::array::creation::empty::tests",
        }],
        notes: &[concat!(
            "Legacy prose, syntax, examples, FAQs, and links are retained. Typed ",
            "contracts own signatures, integer capabilities, placement and fusion. ",
            "Feature and integration fields were null. Native/browser/hardware ",
            "evidence is recorded separately.",
        )],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
