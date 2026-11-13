use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,

 title: Some("imshow"),
 slug: Some("imshow"),
 summary: "Display images in MATLAB and RunMat.",
 description: concat!(
    "`imshow` displays grayscale, binary, and truecolor image data with MATLAB-style ",

    "defaults. File-input handling and display scaling behavior follow MATLAB ",

    "semantics.",

),

 keywords: &["imshow",
 "image display",
 "grayscale image",
 "binary image",
 "rgb image",
 "matlab imshow"],
 related: &["image",
 "imagesc",
 "colormap",
 "axis"],

 sections: &[BuiltinDocumentationSection { heading: "Behavior",
 paragraphs: &[concat!(
    "`imshow(I)` displays 2-D numeric data with grayscale limits `[0, 1]`, matching ",

    "MATLAB double-image defaults.",

),
 "`imshow(BW)` displays logical image data as black and white using `[0, 1]` limits.",
 "`imshow(RGB)` displays MxNx3 truecolor data directly.",
 concat!(
    "In `runmat` compatibility mode, `imshow(RGBA)` accepts supported numeric MxNx4 ",

    "data as a RunMat extension and displays its fourth channel as alpha.",

),
 "`imshow(I, [low high])` displays grayscale image data using an explicit display range.",
 "`imshow(I, [])` auto-scales grayscale image data from finite minimum to finite maximum.",
 "`imshow(filename)` reads and decodes an image file, then displays it as truecolor data."] },

BuiltinDocumentationSection { heading: "Rendering and residency",
 paragraphs: &[concat!(
    "The compatibility target documents GPU Array input for `imshow`. RunMat treats ",

    "`imshow` as a plotting sink rather than a precision-specialized compute kernel; ",

    "grayscale gpuArray inputs may use the image/surface rendering path, while ",

    "filename decoding and truecolor normalization run on the host.",

)] },

BuiltinDocumentationSection { heading: "Numeric provider precision frontier",
 paragraphs: &[concat!(
    "imshow declares no direct numeric provider precision list; host ",

    "generation/sampling or owner-aware gather and installed graphics rendering are ",

    "separate boundaries.",

)] },

BuiltinDocumentationSection { heading: "Coordinate broadcasting",
 paragraphs: &["none"] },

BuiltinDocumentationSection { heading: "Runtime contract and boundaries",
 paragraphs: &[concat!(
    "Catalog default bindings retain the actual RuntimeContext and original rooted ",

    "BuiltinArguments across parsing, provider readback, delegated calls and ",

    "completion. Value-selected axes, properties and output counts remain Runtime ",

    "decisions. Constructor results are host double graphics handles, not numerical ",

    "GPU arrays; hist instead returns double counts/centres, plotyy up to three ",

    "distinct handle outputs, and plotmatrix up to five.",

),
 concat!(
    "Rendering is a fusion boundary and requires the installed graphics host/device. ",

    "A direct compatible renderer route is not an elementwise or reduction provider ",

    "kernel. Resident buffers retain their owning provider and existing ",

    "export/error/completion protocol; an export, readback, allocation, cancellation ",

    "or emission failure is an error, not permission to replay an attempted ",

    "operation. Source data and renderer caches have distinct lifetimes. No new ",

    "distributed chart implementation is declared.",

),
 concat!(
    "The existing canonical CoordinateWork allocation policy refuses Managed calls ",

    "before unowned chart scratch, metadata, readback or scene mutation. It checks ",

    "captured cancellation generation and policy before publication and on ",

    "completion; ingress failures are Never replay. This is an explicit missing ",

    "finite owner, not an allocation estimate, algorithm completion or L25 closure. ",

    "Provider-private allocations, renderer lifetime accounting, synchronous long ",

    "geometry/copy cooperation, measured throughput and physical presentation remain ",

    "independent qualification gates.",

),
 concat!(
    "Native Single/Double/integer source ownership is distinguished from explicit ",

    "binary64/f32 graphics conversion. The renderer does not promise exact ",

    "wide-integer pixel positions. Logical, complex, sparse, scalar, vector, matrix, ",

    "empty and higher-rank forms follow the identity-specific parser rather than a ",

    "universal implicit cast. Sparse/complex rejection is not an integer ",

    "not-applicable audit. Full MATLAB properties, browser/native parity and ",

    "empty-shape conformance are not inferred from migration or historical test ",

    "references.",

)] },

BuiltinDocumentationSection { heading: "Supported representations and remaining forms",
 paragraphs: &[concat!(
    "Exactly one image/filename or image plus grayscale display range is accepted. ",

    "Scalars/logical grayscale normalize to a two-dimensional host tensor; numeric ",

    "grayscale uses class-derived limits (floating/logical 0..1, integer full class ",

    "range), [] selects finite bounds, and an explicit two-element finite increasing ",

    "vector supplies limits. Range controls reject truecolor and filename forms. ",

    "Single/Double/uint8/uint16 truecolor requires nonempty M-by-N-by-3 data; ",

    "supported M-by-N-by-4 is compatibility-gated. Logical truecolor and unsupported ",

    "integer truecolor classes reject.",

),
 concat!(
    "A filename is a string scalar, scalar string array or character row. Before the ",

    "first suspension, the existing filesystem APIs capture its provider, ",

    "provider-relative resolved path and resource lease. Every read and ",

    "uploaded-name directory search stays on that same owner, checks revocation ",

    "around awaits, and uses the supplied Runtime cancellation owner. Browser ",

    "basename search retains its bounded unique-match rule. Decoding uses the actual ",

    "installed image decoder and publishes RGBA doubles; no ambient provider ",

    "re-selection or synthetic image is used.",

),
 concat!(
    "Grayscale and truecolor rendering currently materialize host image/color grids ",

    "and retain CData; old claims of a grayscale zero-copy numeric kernel are not ",

    "made. The canonical union includes FilesystemRead/Filesystem for dynamic ",

    "filenames and removes them for statically proven numeric/logical forms. ",

    "Filename codec buffers and decoded-grid funding/cooperation are not a finite ",

    "implementation. The real PNG example uses the existing Filesystem+Browser ",

    "harness and property assertions, not a fabricated BrowserGraphics screenshot ",

    "witness; physical Figure/presentation acceptance remains separate.",

)] }],

 examples: &[BuiltinExample { id: "retained-1",
 title: "Display a simple grayscale image",
 program: "row = 100;\ncol = 100;\nrsm = ones(row, col);\nimshow(rsm);",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "witness=findobj(gca,'Type','image'); assert(numel(witness)>=1); ",

    "observed=get(witness(1),'CData'); expected=rsm; ",

    "assert(isequal(observed(:),expected(:))); ",

    "assert(isequal(size(get(witness(1),'CData')),[100 100]));",

) } },

BuiltinExample { id: "retained-2",
 title: "Auto-scale a matrix for display",
 program: "A = peaks(128);\nimshow(A, []);",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "witness=findobj(gca,'Type','image'); assert(numel(witness)>=1); ",

    "observed=get(witness(1),'CData'); expected=A; ",

    "assert(isequal(observed(:),expected(:))); ",

    "assert(isequal(size(get(witness(1),'CData')),[128 128]));",

) } },

BuiltinExample { id: "retained-3",
 title: "Display truecolor RGB data",
 program: "R = ones(64, 64);\nG = zeros(64, 64);\nB = zeros(64, 64);\nRGB = cat(3, R, G, B);\nimshow(RGB);",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "witness=findobj(gca,'Type','image'); assert(numel(witness)>=1); ",

    "observed=get(witness(1),'CData'); expected=RGB; ",

    "assert(isequal(observed(:),expected(:))); ",

    "assert(isequal(size(get(witness(1),'CData')),[64 64 3]));",

) } },

BuiltinExample { id: "retained-4",
 title: "Display an image file",
 program: "imshow(\"photo.png\");",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Browser,
 fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "imshow-photo-rgba" },
 root: BuiltinFilesystemRoot::IsolatedWorkspace,
 entries: &[BuiltinFilesystemEntry::File { relative_path: "photo.png",
 content: BuiltinFixtureContent::Bytes(&[137,
 80,
 78,
 71,
 13,
 10,
 26,
 10,
 0,
 0,
 0,
 13,
 73,
 72,
 68,
 82,
 0,
 0,
 0,
 2,
 0,
 0,
 0,
 1,
 8,
 6,
 0,
 0,
 0,
 244,
 34,
 127,
 138,
 0,
 0,
 0,
 14,
 73,
 68,
 65,
 84,
 120,
 156,
 99,
 248,
 207,
 192,
 240,
 31,
 4,
 1,
 16,
 248,
 3,
 253,
 78,
 149,
 193,
 111,
 0,
 0,
 0,
 0,
 73,
 69,
 78,
 68,
 174,
 66,
 96,
 130]) }] }),
 requirements: BuiltinExampleRequirements { runtime: &[BuiltinRuntimeCapability::PlotCore], ..BuiltinExampleRequirements::NONE },
 verification: BuiltinExampleVerification::Assertions { source: concat!(
    "witness=findobj(gca,'Type','image'); assert(numel(witness)>=1); ",

    "observed=get(witness(1),'CData'); assert(isequal(size(observed),[1 2 4])); ",

    "assert(observed(1,1,1)==1&&observed(1,1,2)==0&&observed(1,1,3)==0); ",

    "assert(observed(1,2,1)==0&&observed(1,2,2)==1&&observed(1,2,3)==0); ",

    "assert(all(observed(:,:,4)==1));",

) } }],

 example_exemption: None,

 faqs: &[BuiltinDocumentationFaq { question: "What display range does imshow use for doubles?",
 answer: concat!(
    "`imshow(I)` uses `[0, 1]` for floating-point grayscale data. Use `imshow(I, ",

    "[])` when you want RunMat to scale the visible range from the data minimum to ",

    "maximum.",

) },

BuiltinDocumentationFaq { question: "Does imshow support uint8 and uint16 image defaults?",
 answer: concat!(
    "Yes. Grayscale `uint8` data uses `[0, 255]`, grayscale `uint16` data uses `[0, ",

    "65535]`, and truecolor data of either class is normalized by the corresponding ",

    "full-scale value.",

) },

BuiltinDocumentationFaq { question: "How is imshow different from imagesc?",
 answer: concat!(
    "`imshow` is for image display and uses grayscale or truecolor image defaults. ",

    "`imagesc` is for scaled matrix visualization and maps data through the active ",

    "colormap.",

) }],

 links: &[BuiltinDocumentationLink { label: "image",
 target: BuiltinDocumentationLinkTarget::Builtin("image") },

BuiltinDocumentationLink { label: "imagesc",
 target: BuiltinDocumentationLinkTarget::Builtin("imagesc") },

BuiltinDocumentationLink { label: "colormap",
 target: BuiltinDocumentationLinkTarget::Builtin("colormap") },

BuiltinDocumentationLink { label: "axis",
 target: BuiltinDocumentationLinkTarget::Builtin("axis") }],

 media: &[],

 evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/plotting/ops/imshow.rs`",
 target: BuiltinDocumentationLinkTarget::Source(concat!(
    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtin",

    "s/plotting/ops/imshow.rs",

)) }],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
 label: "unit",
 location: "builtins::plotting::imshow::tests" }],
 notes: &[] },

 introduced: None,
 status: Some(BuiltinDocumentationStatus::Partial),

};
