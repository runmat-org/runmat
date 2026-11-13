use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("interp2"),
    slug: Some("interp2"),
    summary: "Interpolate values on a two-dimensional grid.",
    description: "`interp2(X,Y,Z,Xq,Yq)` estimates surface values at query points. X and Y may be vector axes or meshgrid-style coordinate matrices. `interp2(Z,Xq,Yq)` uses implicit column and row axes.",
    keywords: &["interp2", "2-D interpolation", "bilinear", "nearest", "spline", "meshgrid"],
    related: &["interp1", "meshgrid", "peaks"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Grid and query geometry",
            paragraphs: &[
                "Z must be a dense 2-D matrix with at least two rows and columns. X and Y may be strictly increasing vector axes of matching length or full meshgrid-style matrices matching Z. The axes default to column indices for X and row indices for Y when omitted.",
                "Xq and Yq may be scalars or equally shaped query arrays. A scalar broadcasts over the other query array; a row Xq and column Yq form a query grid. The result follows the query geometry.",
                "The implemented methods are `linear` (bilinear interpolation), `nearest`, and tensor-product not-a-knot `spline`. Spline extrapolates by default; other methods return NaN out of range. Supply `extrap` or a scalar fill value to override. Cubic convolution and multidimensional makima are not aliases of spline and remain unsupported.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Classes and execution",
            paragraphs: &[
                "Linear and nearest retain their real double host result policy. Spline accepts real or complex single/double sample matrices, computes real and imaginary lanes independently, and returns the sample floating class. Typed integer grid, query, and fill inputs are RunMat-mode extensions checked for exact binary64 conversion. Logical numeric inputs are also gated as RunMat extensions.",
                "Device-resident inputs are gathered to the host reference path; this release does not provide an interp2 device kernel or resident output. The call is a fusion boundary. The GPU specification records accepted input precision, not an implemented kernel.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "implicit-axes",
            title: "Bilinear interpolation on implicit axes",
            program: "Z = [1 2; 3 4]; zq = interp2(Z,1.5,1.5)",
            display_output: Some("zq = 2.5"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(abs(zq-2.5)<1e-12); assert(strcmp(class(zq),'double'));" },
        },
        BuiltinExample {
            id: "meshgrid-axes",
            title: "Interpolate with explicit meshgrid axes",
            program: "[X,Y] = meshgrid([10 20],[100 200]); Z = [1 2; 3 4]; zq = interp2(X,Y,Z,15,150)",
            display_output: Some("zq = 2.5"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(abs(zq-2.5)<1e-12);" },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "interp1", target: BuiltinDocumentationLinkTarget::Builtin("interp1") },
        BuiltinDocumentationLink { label: "meshgrid", target: BuiltinDocumentationLinkTarget::Builtin("meshgrid") },
        BuiltinDocumentationLink { label: "peaks", target: BuiltinDocumentationLinkTarget::Builtin("peaks") },
        BuiltinDocumentationLink { label: "MATLAB interp2 reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/interp2.html") },
        BuiltinDocumentationLink { label: "Open an issue", target: BuiltinDocumentationLinkTarget::External("https://github.com/runmat-org/runmat/issues/new/choose") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "interp2 runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/interpolation/interp2.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "interp2 grid geometry, classes, and diagnostics", location: "builtins::math::interpolation::interp2::tests" }],
        notes: &["The sidecar's fusion, GPU, syntax, behaviors, examples, links, and source fields are retained in the catalog documentation and placement. No GPU kernel is claimed."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
