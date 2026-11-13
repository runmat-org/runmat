use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("optimset"), slug: Some("optimset"),
    summary: "Create or update optimization options structures with MATLAB-compatible name/value handling.",
    description: "`optimset` builds option structs for optimization routines. It supports construction from name/value pairs and copying an existing struct before applying overrides.",
    keywords: &["optimset", "options", "TolX", "TolFun", "MaxIter", "Display", "SpecifyObjectiveGradient"],
    related: &["optimget", "fminsearch", "fminbnd", "fminunc", "fzero", "fsolve"],
    sections: &[
        BuiltinDocumentationSection { heading: "Names and forms", paragraphs: &[
            "`optimset()` returns the eight legacy fminsearch fields with numeric [] defaults; optimset('fminsearch') or optimset(@fminsearch) returns that solver's defaults. With no requested output, bare optimset lists these supported fields and value forms on the original invocation console route. Other solver-name default queries are currently unsupported. Name/value pairs create fields, and `optimset(oldopts,...)` copies existing fields before applying the pairs. `optimset(oldopts,newopts)` copies only nonempty new values; [] may stand for empty oldopts. Explicit name/value [] resets that field. Known names are case-insensitive: TolX, TolFun, MaxIter, MaxFunEvals, Display, Algorithm, and SpecifyObjectiveGradient. StepTolerance, FunctionTolerance, OptimalityTolerance, MaxFunctionEvaluations, and GradObj are accepted aliases. Unknown names are retained so a later solver can read them.",
            "Known legacy and modern parameter names accept case-insensitive unambiguous abbreviations through the same resolver used by optimget. These include Display, FunValCheck, MaxFunEvals, MaxIter, OutputFcn, PlotFcns, TolFun, TolX, Algorithm, SpecifyObjectiveGradient and TolCon/ConstraintTolerance. Equivalent canonical names shared by two option identities count once before ambiguity checks. Exact names and supported aliases take priority over abbreviations. OutputFcn and PlotFcns payloads are retained with their original roots, identity, closure captures and execution authority; they are not recursively gathered or inspected as numeric controls. Finite host-memory construction is unsupported and fails before metadata or provider transfer. Option consumers interpret known names case-insensitively. Constructing an options struct does not execute an objective callback or guarantee that every field is supported by every solver.",
        ]},
        BuiltinDocumentationSection { heading: "Typed and resident payloads", paragraphs: &[
            "Documented numeric option values use single or double. Native integer payloads are a gated RunMat extension: optimset preserves their exact class and bits inside the returned struct. A consuming solver may impose a narrower value contract.",
            "Options are host metadata, not an elementwise GPU operation. Numeric option payloads are prepared independently; OutputFcn and PlotFcns callbacks are passed through without inspecting or gathering captures. A payload without explicit gpuArray handles gathers automatically resident values recursively. If any explicit gpuArray handle occurs in a payload, preserving that entire payload is a separately gated RunMat extension: its nested automatic resident siblings also remain resident, including within cells, structs, and value-object arrays. The options struct itself is host-resident and marks a fusion boundary.",
        ]},
    ],
    examples: &[
        BuiltinExample {
            id: "fminbnd", title: "Set a tolerance for bounded minimization",
            program: "opts = optimset('TolX', 1e-10, 'Display', 'off');\nx = fminbnd(@cos, 0, pi, opts)",
            display_output: Some("x =\n    3.1416"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(opts),'struct')); assert(abs(opts.TolX-1e-10)<1e-18); assert(abs(x-pi)<1e-5);" },
        },
        BuiltinExample {
            id: "override", title: "Copy and override known fields",
            program: "base = optimset('MaxIter', 20);\nopts = optimset(base, 'GradObj', 'on')",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(opts.MaxIter==20); assert(strcmp(opts.SpecifyObjectiveGradient,'on'));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Are unknown option names rejected?", answer: "No. optimset retains them in the returned struct. A consuming solver decides whether it supports them." },
        BuiltinDocumentationFaq { question: "Are integer option payloads converted?", answer: "In RunMat mode, optimset retains native fixed-width integer payloads exactly. A consuming solver can normalize or reject a field according to its contract." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "fminbnd", target: BuiltinDocumentationLinkTarget::Builtin("fminbnd") },
        BuiltinDocumentationLink { label: "fminunc", target: BuiltinDocumentationLinkTarget::Builtin("fminunc") },
        BuiltinDocumentationLink { label: "fzero", target: BuiltinDocumentationLinkTarget::Builtin("fzero") },
        BuiltinDocumentationLink { label: "fsolve", target: BuiltinDocumentationLinkTarget::Builtin("fsolve") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/optimset.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "optimset builder and payload handling", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/optimset.rs") },
            BuiltinDocumentationLink { label: "Ordered name/value admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/optimset/admission.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Option fields, extensions, and resident payloads", location: "builtins::math::optim::optimset::tests" }],
        notes: &["Source tests and documentation declarations do not establish standalone example, provider, or cross-engine acceptance."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
