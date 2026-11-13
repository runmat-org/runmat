use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ishandle"), slug: Some("ishandle"),
    summary: "Test whether a value identifies a live graphics-registry handle.",
    description: "`ishandle` queries the live RunMat graphics registry and returns a host logical result. Exact graphics receiver objects are authenticated by their retained registry lease, not by their class-name text or a reusable numeric identifier. Numeric aliases use the existing graphics handle encoding; zero denotes the root. Invalid, retired, and non-graphics candidates return false.",
    keywords: &["ishandle", "graphics handles", "validity mask", "plotting", "graphics receiver"],
    related: &["isgraphics", "gobjects", "ancestor", "close"],
    sections: &[
        BuiltinDocumentationSection { heading: "Accepted forms and shapes", paragraphs: &[
            "ishandle(h) accepts any candidate value. Host real numeric tensors, including native single and all integer storage classes, produce logical arrays with the original full shape, including rank-zero, rank-one and empty dimensions. Numeric scalar carriers return scalar logical.",
            "Other host values, including sparse matrices, logical arrays, complex values, strings, cells and non-graphics objects, produce scalar false rather than an elementwise mask. This is the bounded current RunMat representation, not a claim of complete MATLAB object-array parity.",
            "A valid graphics receiver object returns scalar true while its exact lease is registered; closing, clearing, or replacing its registry owner makes the old object false even if a numeric identifier is reused.",
            "ishandle currently queries the graphics registry only. It is not a generic validity test for arbitrary user handle classes, parallel handles or other non-graphics objects; use the corresponding domain operation for those values.",
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
            id: "root-and-invalid-mask", title: "Test root and invalid numeric aliases without opening a figure", program: "tf = ishandle([0 NaN -1]);",
            display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf,logical([1 0 0]))); assert(isequal(size(tf),[1 3]));" },
        },
        BuiltinExample {
            id: "empty-shape", title: "Retain empty dimensions in a numeric candidate array", program: "h = zeros(2,0,3); tf = ishandle(h);",
            display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(tf)); assert(isequal(size(tf),[2 0 3])); assert(numel(tf)==0);" },
        },
        BuiltinExample {
            id: "live-and-retired-figure", title: "Recognize live graphics and reject retired aliases", program: "f = figure(); h = plot(1:3); live = ishandle(h); close(f); retired = ishandle(h); replacement = figure();",
            display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(live); assert(~retired); assert(ishandle(replacement));" },
        },
        BuiltinExample {
            id: "native-integer-alias", title: "Read integer aliases exactly under RunMat extension policy", program: "h = int64([0 -1]); tf = ishandle(h);",
            display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf,logical([1 0]))); assert(isa(h,'int64')); assert(islogical(tf));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does this predicate create or revive graphics objects?", answer: "No. It observes the current registry. An invalid placeholder or retired receiver stays invalid; a newly created object must supply its own live authority." },
        BuiltinDocumentationFaq { question: "Are resident numeric values downloaded to inspect potential handles?", answer: "No. Resident inputs return a host false mask using only shape metadata; no acceleration provider is invoked." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "isgraphics", target: BuiltinDocumentationLinkTarget::Builtin("isgraphics") },
        BuiltinDocumentationLink { label: "gobjects", target: BuiltinDocumentationLinkTarget::Builtin("gobjects") },
        BuiltinDocumentationLink { label: "ancestor", target: BuiltinDocumentationLinkTarget::Builtin("ancestor") },
        BuiltinDocumentationLink { label: "close", target: BuiltinDocumentationLinkTarget::Builtin("close") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Runtime ishandle owner", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/ishandle.rs") },
            BuiltinDocumentationLink { label: "Exact graphics receiver registry authority", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/core/state/graphics_receiver.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Runtime ishandle forms, shapes, errors and ownership", location: "builtins::plotting::ishandle::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Canonical ishandle typed inference", location: "catalog::entries::plotting::ishandle::tests" },
        ],
        notes: &[
            "First-class canonical documentation for an existing previously undocumented Runtime identity; there was no JSON sidecar baseline to retire. Existing ancestor, copyobj and gobjects links now target this actual catalog identity.",
            "All examples have independent assertions or an exact expected error. Newly authored source fixtures and examples have not been executed in the isolated authoring worktree; no compiled exporter, provider or graphics-host qualification is claimed.",
            "Contract maturity remains Incomplete. Real graphics receiver matching is repaired through the existing exact lease/class owner; no class-name dispatch, new graphics registry or blanket allocation waiver is introduced.",
        ],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
