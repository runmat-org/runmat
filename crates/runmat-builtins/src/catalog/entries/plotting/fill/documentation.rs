use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fill"),
    slug: Some("fill"),
    summary: "Create filled 2-D polygon patches.",
    description: concat!(
        "`fill(X,Y,C)` creates filled polygonal regions as Patch graphics. Integer ",
        "coordinate and CData classes remain authoritative on the Patch object; ",
        "conversion to renderer geometry is an explicit client-side boundary.",
    ),
    keywords: &["fill", "patch", "polygon", "2-D"],
    related: &["fill3", "patch", "createArray"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                concat!(
    "Equal-length row or column vector X and Y inputs create one region regardless ",
    "of orientation; matrix inputs create independently addressable regions from ",
    "columns.",
),
                concat!(
    "When either coordinate is a matrix, an equal-length coordinate vector matching ",
    "the matrix row count is shared across every matrix column; square matrices are ",
    "grouped by columns.",
),
                "Multiple X,Y,C groups create multiple Patch graphics.",
                "C accepts color text, RGB triples, or numeric indexed color data.",
                "XData, YData, and CData preserve the original numeric class and shape.",
                concat!(
    "A leading axes handle targets an existing axes; typed-integer numeric aliases ",
    "are available only when RunMat extensions are enabled.",
),
            ],
        },
        BuiltinDocumentationSection {
            heading: "Rendering and residency",
            paragraphs: &[concat!(
                "GPU-resident inputs gather exactly through their owning provider and Patch ",
                "construction executes on the client.",
            )],
        },
        BuiltinDocumentationSection {
            heading: "Numeric provider precision frontier",
            paragraphs: &[concat!(
                "fill declares no direct numeric provider precision list; host ",
                "generation/sampling or owner-aware gather and installed graphics rendering are ",
                "separate boundaries.",
            )],
        },
        BuiltinDocumentationSection {
            heading: "Coordinate broadcasting",
            paragraphs: &["none"],
        },
        BuiltinDocumentationSection {
            heading: "Runtime contract and boundaries",
            paragraphs: &[
                concat!(
                    "Graphics results are host double handles; fill/fill3 may return a row of ",
                    "handles for independently created polygon columns/groups. Sphere instead ",
                    "supports zero through three requested outputs: statement form plots, while ",
                    "output forms return host double coordinate grids. Handle aliases, ",
                    "value-selected options and exact graphics receiver roles are resolved by ",
                    "Runtime, not guessed from scalar facts.",
                ),
                concat!(
    "The existing numerical and geometry owners validate coordinates, integer ",
    "controls, shapes and properties. Single/double and all eight integer source ",
    "classes retain their native storage wherever the object has authoritative ",
    "source properties; client geometry and renderer buffers are deliberate floating ",
    "boundaries, not a promise of exact wide-integer pixel positions. There is no ",
    "sparse/complex plotting implementation or numeric broadcasting promise here. ",
    "Logical, scalar, vector, matrix, rank and empty admission follows each ",
    "documented owner below rather than a universal coercion.",
),
                concat!(
    "RuntimeOwned preparation retains the original BuiltinArguments through ",
    "suspension and completion. Eligible direct GPU routes require the live owning ",
    "provider, admitted precision/storage/shape, compatible exports and installed ",
    "plotting device. NoAttempt means pre-effect inapplicability, not permission to ",
    "swallow export, allocation, cancellation or emission errors. Existing provider ",
    "errors are propagated and ingress errors are terminal Never replay. ",
    "Source-property queries and figure persistence may require explicit owner-aware ",
    "readback; surface property snapshots and quiver3/patch construction are not ",
    "zero-copy routes.",
),
                concat!(
    "Canonical UserInterface capability and host-callback effects reflect scene ",
    "creation/presentation; function-sampling entries additionally declare Unknown ",
    "effects for arbitrary user callbacks and require the Process execution stack. ",
    "NativeAndWasm means the Runtime entry is portable, not that an absent graphics ",
    "host is emulated. Portable Runtime linkage introduces no separate native ",
    "numerical library artifact. Rendering is a fusion boundary and distributed ",
    "execution is unsupported; no numeric provider kernel or fusion primitive is ",
    "invented.",
),
                concat!(
    "Chart scratch, metadata, retained render state and the sphere coordinate ",
    "builder do not yet have complete finite host allocation ownership. The shared ",
    "canonical HostAllocationReceipt policy guard refuses Managed work before ",
    "extraction/gather/callbacks or scene mutation, preserves ordinary Unbounded ",
    "routes, and checks captured cancellation generation and policy before ",
    "publication. This is an explicit L25 gap, not allocation completion. Long ",
    "synchronous geometry/copy loops, renderer/provider-private allocation ",
    "accounting, full MATLAB property parity, physical performance and actual ",
    "example execution remain independently required.",
),
            ],
        },
        BuiltinDocumentationSection {
            heading: "Supported representations and remaining forms",
            paragraphs: &[
                concat!(
    "One or more X,Y,C groups create Patch objects; a leading axes target and ",
    "trailing shared Patch properties apply to every group. Equal-length row/column ",
    "vectors form one polygon; matrix inputs split by columns and a vector matching ",
    "the matrix row count is reused across columns. A single polygon returns one ",
    "handle, multiple groups/columns a row of handles. Colors can be text/RGB or ",
    "authoritative indexed CData.",
),
                concat!(
    "Coordinates/color data retain native class, exact values and original shape ",
    "independently of renderer geometry. Shared Patch grouping gathers GPU sources ",
    "through their owner and validates matching lengths/matrix geometry before ",
    "polygon construction. Integer numeric axes aliases are separately gated; they ",
    "are not ordinary documented axes objects. Complex/sparse input and full Patch ",
    "renderer colormap fidelity remain existing frontiers, not a new alternative ",
    "storage representation.",
),
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Create a filled triangle",
            program: "h = fill([0 1 0], [0 0 1], 'red');",
            display_output: Some("h is a Patch graphics handle"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: concat!(
    "assert(strcmp(get(h,'Type'),'patch')); observed = get(h, 'XData'); expected = ",
    "[0 1 0]; assert(isequal(observed(:), expected(:))); observed = get(h, 'YData'); ",
    "expected = [0 0 1]; assert(isequal(observed(:), expected(:))); ",
),
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Use exact integer coordinates and indexed color data",
            program: "h = fill(uint64([0 1 0]), uint64([0 0 1]), uint8(1));",
            display_output: Some("get(h,'XData') retains class uint64"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: concat!(
                    "assert(strcmp(class(get(h,'XData')),'uint64')); ",
                    "assert(strcmp(class(get(h,'CData')),'uint8')); observed = get(h, 'XData'); ",
                    "expected = uint64([0 1 0]); assert(isequal(observed(:), expected(:))); ",
                ),
            },
        },
        BuiltinExample {
            id: "retained-3",
            title: "Create one independently addressable Patch per matrix column",
            program: concat!(
    "X = uint64([0 2; 1 3; 0 2]);\nY = uint64([0 0; 0 0; 1 1]);\nC = uint16([1 4; 2 5; ",
    "3 6]);\nh = fill(X,Y,C);",
),
            display_output: Some(concat!(
                "h is a two-element Patch handle vector; each handle retains its column's typed ",
                "XData, YData, and CData",
            )),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: concat!(
    "assert(isequal(size(h),[1 2])); assert(strcmp(class(get(h(1),'XData')),'uint64')",
    "); assert(strcmp(class(get(h(2),'CData')),'uint16')); observed = get(h(1), ",
    "'XData'); expected = X(:,1); assert(isequal(observed(:), expected(:))); ",
    "observed = get(h(2), 'XData'); expected = X(:,2); assert(isequal(observed(:), ",
    "expected(:))); observed = get(h(2), 'CData'); expected = C(:,2); ",
    "assert(isequal(observed(:), expected(:))); ",
),
            },
        },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Is fill an array-construction function?",
        answer: concat!(
            "No. MATLAB-compatible fill creates Patch graphics. Use createArray to construct ",
            "constant-valued arrays.",
        ),
    }],
    links: &[
        BuiltinDocumentationLink {
            label: "fill3",
            target: BuiltinDocumentationLinkTarget::Builtin("fill3"),
        },
        BuiltinDocumentationLink {
            label: "patch",
            target: BuiltinDocumentationLinkTarget::Builtin("patch"),
        },
        BuiltinDocumentationLink {
            label: "createArray",
            target: BuiltinDocumentationLinkTarget::Builtin("createArray"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/plotting/ops/fill.rs`",
            target: BuiltinDocumentationLinkTarget::Source(concat!(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtin",
                "s/plotting/ops/fill.rs",
            )),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "unit",
            location: "builtins::plotting::fill::tests",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
