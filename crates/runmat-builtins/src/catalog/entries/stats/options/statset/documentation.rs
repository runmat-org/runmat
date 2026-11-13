use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("statset"), slug: Some("statset"),
    summary: "Create or update statistics options structures.",
    description: "`statset` constructs canonical options for fitting, clustering, resampling, and related statistics routines. A field can hold metadata or a callback handle; construction stores the handle without invoking it.",
    keywords: &["statset", "statistics options", "MaxIter", "TolFun", "TolX", "Display", "UseParallel"],
    related: &["statget", "kmeans", "tsne"],
    sections: &[
        BuiltinDocumentationSection { heading: "Forms and fields", paragraphs: &[
            "`statset()` creates twenty canonical fields initialized to empty values. `statset(statfun)` selects recognized function defaults. Name-value pairs can construct a new structure or override an existing one; `statset(oldopts,newopts)` copies nonempty fields from the second structure.",
            "Names are case-insensitive and accept unique leading prefixes. Fields are Display, MaxFunEvals, MaxIter, TolBnd, TolFun, TolTypeFun, TolX, TolTypeX, GradObj, Jacobian, DerivStep, FunValCheck, Robust, RobustWgtFun, WgtFun, Tune, UseParallel, UseSubstreams, Streams, and OutputFcn.",
            "Tolerance fields require finite positive scalars; iteration counts require positive integers. Enumerated text and logical fields are validated. Callback, stream, and robust-weight payloads are retained for the consuming routine; statset does not execute callbacks.",
        ]},
        BuiltinDocumentationSection { heading: "Typed values and execution", paragraphs: &[
            "MATLAB-compatible mode admits the documented single/double numeric and logical option surface. RunMat mode also admits all eight native integer classes as option values: counts are decoded exactly before bounds checks, boolean flags become logical, and DerivStep vectors retain typed storage.",
            "MATLAB-compatible mode rejects the native integer option extension before recursive gather. Invalid integer values in non-option roles retain the ordinary argument error. Scalar counts and tolerances normalize to ordinary option metadata; typed vector payloads such as DerivStep retain their storage.",
            "Statistics options are host metadata, not a GPU or fused operation. Admission precedes recursive materialization of admitted resident values through their owning provider. Stored callback or stream objects may impose their own affinity when a later routine uses them.",
        ]},
    ],
    examples: &[
        BuiltinExample { id: "custom", title: "Create custom statistics options", program: "opts = statset('TolX', 1e-8, 'Display', 'off')", display_output: Some("`opts` is a struct with canonical statistics option fields."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(statget(opts,'TolX'),1e-8)); assert(isequal(statget(opts,'Display'),'off'));" } },
        BuiltinExample { id: "defaults", title: "Override function defaults", program: "opts = statset('nbinfit');\nopts = statset(opts, 'TolX', 1e-8)", display_output: Some("`opts` keeps `nbinfit` defaults such as `MaxIter` and uses the new `TolX`."), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(statget(opts,'MaxIter'),200)); assert(isequal(statget(opts,'TolX'),1e-8));" } },
        BuiltinExample { id: "merge", title: "Merge nonempty fields", program: "base = statset('MaxIter', 20);\nupdate = statset('TolFun', 1e-7);\nopts = statset(base, update)", display_output: Some("`opts` contains both `MaxIter` and `TolFun`."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(statget(opts,'MaxIter'),20)); assert(isequal(statget(opts,'TolFun'),1e-7));" } },
    ],
    example_exemption: None, faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "statget", target: BuiltinDocumentationLinkTarget::Builtin("statget") },
        BuiltinDocumentationLink { label: "kmeans", target: BuiltinDocumentationLinkTarget::Builtin("kmeans") },
        BuiltinDocumentationLink { label: "tsne", target: BuiltinDocumentationLinkTarget::Builtin("tsne") },
        BuiltinDocumentationLink { label: "bootstrp", target: BuiltinDocumentationLinkTarget::Builtin("bootstrp") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Statistics options entrypoints", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/options.rs") },
            BuiltinDocumentationLink { label: "Statistics options schema and defaults", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/options/schema.rs") },
            BuiltinDocumentationLink { label: "Statistics option validation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/options/validation.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Statistics options forms and integer policy", location: "crates/runmat-runtime/src/builtins/stats/options/tests.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled statistics options calls", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
