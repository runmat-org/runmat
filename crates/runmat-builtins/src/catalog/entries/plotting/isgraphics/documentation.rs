use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("isgraphics"), slug: Some("isgraphics"),
    summary: "Test graphics-handle validity, optionally selecting a graphics type.",
    description: "`isgraphics` queries the live RunMat graphics registry and returns a host logical result. Exact graphics receiver objects are authenticated by their retained registry lease, not by their class-name text or a reusable numeric identifier. Numeric aliases use the existing graphics handle encoding; zero denotes the root. Invalid, retired, and non-graphics candidates return false. A second scalar text argument filters by the implemented graphics type.",
    keywords: &["isgraphics", "graphics handles", "validity mask", "plotting", "graphics receiver"],
    related: &["ishandle", "gobjects", "ancestor", "close"],
    sections: &[
        BuiltinDocumentationSection { heading: "Accepted forms and shapes", paragraphs: &[
            "isgraphics(h) accepts any candidate value. Host real numeric tensors, including native single and all integer storage classes, produce logical arrays with the original full shape, including rank-zero, rank-one and empty dimensions. Numeric scalar carriers return scalar logical.",
            "Other host values, including sparse matrices, logical arrays, complex values, strings, cells and non-graphics objects, produce scalar false rather than an elementwise mask. This is the bounded current RunMat representation, not a claim of complete MATLAB object-array parity.",
            "A valid graphics receiver object returns scalar true while its exact lease is registered; closing, clearing, or replacing its registry owner makes the old object false even if a numeric identifier is reused.",
            "isgraphics(h,type) accepts a scalar string, one-element string array or character row. Type names are trimmed and case-insensitive; an unknown or empty name matches nothing. Invalid selector representation is an argument error before predicate evaluation.",
            "Implemented selectors include root, figure, axes, ruler, xaxis, yaxis, legend, text, line, animatedline, scatter, bar, stem, errorbar, stairs, quiver, image, heatmap, histogram, histogram2, binscatter, area, surface, functionsurface, functioncontour, patch, contour, constantline, textscatter, wordcloud, colorbar and stackedplot. Both 2-D and 3-D lines/scatters share their typed family. Existing pie children match patch; the public pie selector remains unrecognized.",
        ] },
        BuiltinDocumentationSection { heading: "Integer aliases and compatibility", paragraphs: &[
            "Interpreting typed integers as encoded graphics aliases is a declared RunMat-only extension. The original native integer value is read without a floating mirror; only nonnegative integers exactly representable in the graphics encoding are candidates. Negative and nonrepresentable integers return false.",
            "With RunMat extensions disabled, integer scalars return scalar false and integer tensors return same-shaped false. No extension exception is raised by this predicate. Floating numeric aliases retain the existing RunMat registry semantics.",
        ] },
        BuiltinDocumentationSection { heading: "Placement, authority and remaining limits", paragraphs: &[
            "Resident tensors are not graphics objects: only their shape metadata is inspected, with no provider selection, download, kernel, input retirement or replay. The host result is a same-shaped false logical array, except a one-element resident input publishes scalar false. The operation is a fusion boundary, not an accelerated reduction.",
            "The Runtime binding retains the actual selected context and borrowed argument owner, validates output demand before work, and fences cancellation and finite-allocation authority through the existing graphics admission owner. Finite mask, selector and registry snapshot ownership remains incomplete: Managed calls fail closed rather than allocating unowned storage.",
            "The predicate observes mutable graphics state and therefore is not a pure constant-foldable query. It creates no figure or renderer by itself. Portable root/invalid examples need no display; examples that construct real charts require an actual graphics host. NativeAndWasm denotes source/link portability, not fresh browser-cooperation or desktop-presentation qualification.",
        ] },
    ],
    examples: &[
        BuiltinExample {
            id: "root-and-invalid-mask", title: "Test root and invalid numeric aliases without opening a figure", program: "tf = isgraphics([0 NaN -1]);",
            display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf,logical([1 0 0]))); assert(isequal(size(tf),[1 3]));" },
        },
        BuiltinExample {
            id: "empty-shape", title: "Retain empty dimensions in a numeric candidate array", program: "h = zeros(2,0,3); tf = isgraphics(h);",
            display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(tf)); assert(isequal(size(tf),[2 0 3])); assert(numel(tf)==0);" },
        },
        BuiltinExample {
            id: "live-and-retired-figure", title: "Recognize live graphics and reject retired aliases", program: "f = figure(); h = plot(1:3); live = isgraphics(h); close(f); retired = isgraphics(h); replacement = figure();",
            display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(live); assert(~retired); assert(isgraphics(replacement));" },
        },
        BuiltinExample {
            id: "native-integer-alias", title: "Read integer aliases exactly under RunMat extension policy", program: "h = int64([0 -1]); tf = isgraphics(h);",
            display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf,logical([1 0]))); assert(isa(h,'int64')); assert(islogical(tf));" },
        },
        BuiltinExample {
            id: "typed-graphics", title: "Match graphics kinds with scalar text", program: "f = figure(); h = plot(1:3); ax = gca; a = isgraphics(h,'line'); b = isgraphics(ax,'AXES'); c = isgraphics(h,'figure');",
            display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(a); assert(b); assert(~c); assert(isgraphics(f,'figure')); assert(~isgraphics(h,'unknown-type'));" },
        },
        BuiltinExample {
            id: "invalid-selector", title: "Reject a nontext type selector", program: "isgraphics(0,42);",
            display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::ExpectedError { identifier: "RunMat:isgraphics:InvalidArgument" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does this predicate create or revive graphics objects?", answer: "No. It observes the current registry. An invalid placeholder or retired receiver stays invalid; a newly created object must supply its own live authority." },
        BuiltinDocumentationFaq { question: "Are resident numeric values downloaded to inspect potential handles?", answer: "No. Resident inputs return a host false mask using only shape metadata; no acceleration provider is invoked." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "ishandle", target: BuiltinDocumentationLinkTarget::Builtin("ishandle") },
        BuiltinDocumentationLink { label: "gobjects", target: BuiltinDocumentationLinkTarget::Builtin("gobjects") },
        BuiltinDocumentationLink { label: "ancestor", target: BuiltinDocumentationLinkTarget::Builtin("ancestor") },
        BuiltinDocumentationLink { label: "close", target: BuiltinDocumentationLinkTarget::Builtin("close") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Runtime isgraphics owner", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/isgraphics.rs") },
            BuiltinDocumentationLink { label: "Exact graphics receiver registry authority", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/core/state/graphics_receiver.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Runtime isgraphics forms, shapes, errors and ownership", location: "builtins::plotting::isgraphics::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Canonical isgraphics typed inference", location: "catalog::entries::plotting::isgraphics::tests" },
        ],
        notes: &[
            "First-class canonical documentation for an existing previously undocumented Runtime identity; there was no JSON sidecar baseline to retire. Existing ancestor, copyobj and gobjects links now target this actual catalog identity.",
            "All examples have independent assertions or an exact expected error. Newly authored source fixtures and examples have not been executed in the isolated authoring worktree; no compiled exporter, provider or graphics-host qualification is claimed.",
            "Contract maturity remains Incomplete. Real graphics receiver matching is repaired through the existing exact lease/class owner; no class-name dispatch, new graphics registry or blanket allocation waiver is introduced.",
        ],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
