use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("imhist"),
    slug: Some("imhist"),
    summary: "Compute or display grayscale and indexed-image histograms.",
    description: "`imhist` counts grayscale intensities or indexed-image colormap entries. Programmatic calls return double column vectors for counts and optional bin locations; host statement-form calls display a bar-chart histogram when presentation is available.",
    keywords: &["imhist", "image histogram", "grayscale image", "indexed image", "colormap", "matlab imhist"],
    related: &["image", "imshow", "imread", "histogram", "histcounts"],
    sections: &[
        BuiltinDocumentationSection { heading: "Forms and bins", paragraphs: &[
            "`imhist(I)` uses 256 bins for numeric grayscale data and two bins for logical data. `imhist(I,n)` uses a positive integer-valued scalar `n` up to one million bins. `[counts,binLocations]` returns the class-wide bin centers as a second double column vector.",
            "`imhist(X,map)` counts indexed values into one bin per row of a nonempty double Nx3 RGB colormap. Each colormap component must be finite and in [0,1]. Floating indexed values use one-based indices; uint8, uint16, and logical indexed values use zero-based indices. Nonfinite, fractional, and out-of-range indices reject.",
            "Floating grayscale images must be finite normalized single or double values in [0,1]. Grayscale int8, int16, int32, uint8, uint16, and uint32 use their full class ranges; int64 and uint64 image classes reject without lossy conversion. Counts and locations are double arrays.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and compatibility", paragraphs: &[
            "A native typed-integer scalar bin count is a RunMat-only extension. MATLAB-compatible mode accepts the documented positive integer-valued floating scalar form and rejects the extension before data access.",
            "Grayscale gpuArray input is downloaded non-destructively through its owning provider, counted on the host, and programmatic outputs are restored to the same owner when supported. Unsupported automatic restoration may retain the host result; operational provider failures remain errors. Indexed gpuArray input is unsupported. Histogram assembly terminates fusion chains; there is no histogram provider kernel.",
            "For host input, a zero-output statement call displays a bar chart when a graphics host is available; no-plot, headless, and non-main-thread presentation unavailability completes without a visible figure. Existing resident zero-output calls return host counts instead of plotting. That resident behavior remains distinct from programmatic restored outputs.",
            "The computation is a host materialization boundary, not an arbitrary CPU fallback for an existing provider histogram kernel. Download, output upload, graphics presentation, suspension, and error ordering use their existing shared owners. Graphics figures are local state, not portable values for worker or foreign-runtime transfer.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Count intensities in a byte image", program: "I = uint8([0 1 1; 2 2 2]);\n[counts, bins] = imhist(I);\n[peak, idx] = max(counts);\n[bins(idx), peak]", display_output: Some("ans =\n     2     3"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(counts),[256 1])); assert(isequal([bins(idx),peak],[2 3])); assert(sum(counts)==6);" } },
        BuiltinExample { id: "example-2", title: "Use fewer bins for a normalized image", program: "I = [0 0.25 0.5 1.0];\n[counts, bins] = imhist(I, 3);\ncounts", display_output: Some("ans =\n     1\n     2\n     1"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(counts,[1;2;1])); assert(isequal(bins,[0;0.5;1]));" } },
        BuiltinExample { id: "example-3", title: "Histogram an indexed image", program: "X = [1 2; 3 2];\nmap = [1 0 0; 0 1 0; 0 0 1];\n[counts, bins] = imhist(X, map);\ncounts", display_output: Some("ans =\n     1\n     2\n     1"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(counts,[1;2;1])); assert(isequal(bins,[1;2;3]));" } },
        BuiltinExample { id: "example-4", title: "Display a histogram", program: "I = uint8([0 64 128; 128 192 255]);\ncounts = imhist(I);\nimhist(I);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(sum(counts)==6); assert(counts(129)==2);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Why does imhist reject double values greater than 1?", answer: "Floating grayscale images represent normalized intensities in [0,1]. Use uint8 or uint16 for byte or word intensities." },
        BuiltinDocumentationFaq { question: "How are indexed image offsets handled?", answer: "Double and single indexed images use one-based indices. uint8, uint16, and logical indexed images use zero-based indices." },
        BuiltinDocumentationFaq { question: "Does statement-form imhist always render a plot?", answer: "Host calls render when a graphics backend and presentation context are available. Headless/no-plot presentation unavailability completes without a visible figure; existing resident zero-output calls return host counts." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "image", target: BuiltinDocumentationLinkTarget::Builtin("image") },
        BuiltinDocumentationLink { label: "imshow", target: BuiltinDocumentationLinkTarget::Builtin("imshow") },
        BuiltinDocumentationLink { label: "imread", target: BuiltinDocumentationLinkTarget::Builtin("imread") },
        BuiltinDocumentationLink { label: "histogram", target: BuiltinDocumentationLinkTarget::Builtin("histogram") },
        BuiltinDocumentationLink { label: "histcounts", target: BuiltinDocumentationLinkTarget::Builtin("histcounts") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Histogram admission and publication", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/imhist.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Image class, bin, indexed, resident, and output tests", location: "builtins::image::imhist::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Bytecode uint8 histogram workflow", location: "runmat_vm::bytecode::compile::tests::compile_interprets_imhist_uint8_workflow" },
        ],
        notes: &["Legacy docs/builtins/reference/imhist.json content was reconciled into this identity; there was no runtime builtins-json shadow at the cutover baseline. Actual browser graphics, native/JIT/AOT and real-provider performance remain integration proof requirements."],
    },
    introduced: None,
    status: None,
};
