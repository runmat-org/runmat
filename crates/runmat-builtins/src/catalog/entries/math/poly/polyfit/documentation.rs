use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("polyfit"),
    slug: Some("polyfit"),
    summary: "Fit least-squares polynomial coefficients and return fit diagnostics.",
    description: "`polyfit(X,Y,n)` fits a degree-`n` polynomial using Householder QR. One or two outputs use the original X coordinates; requesting the third output fits in centered, scaled coordinates and returns that transformation as mu. Optional weights scale each residual.",
    keywords: &["polyfit", "polynomial", "least-squares", "Householder QR", "gpu"],
    related: &["polyval", "roots", "mldivide", "polyder", "polyint"],
    sections: &[
        BuiltinDocumentationSection { heading: "Samples and weighted fitting", paragraphs: &[
            "X must be real and Y may be real or complex. Both must be nonempty coefficient-style vectors with the same number of elements; row, column, and singleton-extended vector shapes are accepted. The current runtime does not flatten arbitrary matrices or N-D arrays with multiple nonsingleton dimensions.",
            "The optional fourth argument is a finite, real, nonnegative weight vector with one element per X/Y sample. It may differ from X in row/column orientation; the runtime checks element count rather than identical shape. Zero weights remove a sample's contribution to the weighted residual norm.",
            "Fixed-width integer X, Y, and weights are separately gated RunMat extensions and require exact binary64 representability. Degree is an exact, nonnegative structural integer control; logical scalar degree and logical sample/weight arrays are also admitted by the current runtime.",
        ] },
        BuiltinDocumentationSection { heading: "Outputs and evaluation", paragraphs: &[
            "The coefficient row p has n+1 entries in descending power order. With one or two outputs, p and S.R use the original X basis, so use `polyval(p,X)` or `polyval(p,X,S)`. Requesting `[p,S,mu]` changes p and S.R to the centered, scaled basis; mu records mean(X) and scale(X), and `polyval(p,X,[],mu)` or `polyval(p,X,S,mu)` applies it during evaluation. S also contains degrees of freedom df and residual norm normr.",
            "Host output storage is double or complex double. For requested outputs after p, S, and mu, RunMat supplies scalar zero padding. The fit uses numerical QR; rank-deficient or ill-conditioned data can affect accuracy.",
        ] },
        BuiltinDocumentationSection { heading: "Provider execution", paragraphs: &[
            "Compatible resident X/Y/weight inputs may use the owning provider's `polyfit` hook. The current WGPU provider gathers within this hook and runs the shared Householder QR solver; the returned p, S, and mu are host values. Unsupported provider routes gather through the exact owners and run the same host solver.",
            "The provider hook is real-data-only and requires compatible owners. It receives the requested coordinate basis explicitly, so provider and host fits return matching coefficient and QR bases. Complex Y, mixed locations, or incompatible inputs use the host route. Fitting is a fusion boundary; provider dispatch and host materialization still contribute to shared execution costs.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "line", title: "Fit a line through slightly noisy samples", program: "x = 0:5;\nnoise = 0.05*[1 -1 0 1 -1 0];\ny = 2.5*x + 1 + noise;\np = polyfit(x,y,1)", display_output: Some("p is close to [2.5 1]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(numel(p)==2); assert(max(abs(p-[2.5 1]))<0.1);" } },
        BuiltinExample { id: "quadratic-mu", title: "Reuse fit scaling when evaluating", program: "x = -3:3;\ny = x.^2-2*x+4;\n[p,S,mu] = polyfit(x,y,2);\nsmoothed = polyval(p,x,[],mu)", display_output: Some("smoothed matches y"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(smoothed-y))<1e-8); assert(numel(mu)==2);" } },
        BuiltinExample { id: "prediction", title: "Use S for prediction intervals", program: "t = linspace(0,2*pi,25);\ny = sin(t)+0.1*cos(3*t);\n[p,S,mu] = polyfit(t,y,3);\n[fitted,delta] = polyval(p,t,S,mu)", display_output: Some("delta contains one value per fitted sample"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(delta),size(t))); assert(isequal(size(fitted),size(t)));" } },
        BuiltinExample { id: "weighted", title: "Give central samples more weight", program: "x = linspace(-1,1,7);\ny = [1.1 0.4 0.2 0 0.1 0.5 1.4];\nw = [1 2 2 4 2 2 1];\np = polyfit(x,y,2,w)", display_output: Some("p is a three-coefficient row vector"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(p),[1 3])); assert(all(isfinite(p)));" } },
        BuiltinExample { id: "gpu", title: "Fit resident samples and receive host coefficients", program: "x = gpuArray(linspace(-2,2,50));\ny = (x-0.5).^3;\np = polyfit(x,y,3)", display_output: Some("p is host-resident and close to [1 -1.5 0.75 -0.125]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(~isa(p,'gpuArray')); assert(max(abs(p-[1 -1.5 0.75 -0.125]))<1e-6);" } },
        BuiltinExample { id: "complex", title: "Fit complex observations", program: "x = 0:4;\ny = exp(1i*x);\np = polyfit(x,y,2)", display_output: Some("p contains complex coefficients"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(numel(p)==3); assert(any(abs(imag(p))>1e-8));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "How do I choose the degree?", answer: "Supply a nonnegative integer. Degree relative to sample count controls conditioning; validate a chosen model against your data." },
        BuiltinDocumentationFaq { question: "What do S and mu contain?", answer: "S has R, df, and normr; mu contains centering mean and scale. Pass both to supported polyval prediction-interval forms." },
        BuiltinDocumentationFaq { question: "How are weights used?", answer: "Each nonnegative weight multiplies a squared residual. A zero weight removes that observation from the weighted fit." },
        BuiltinDocumentationFaq { question: "Are outputs GPU-resident?", answer: "No. The current provider hook and host fallback both produce host p, S, and mu values." },
        BuiltinDocumentationFaq { question: "What if all sample locations coincide?", answer: "A positive-degree fit is rank-deficient with coincident sample locations; a stable unique coefficient vector cannot be inferred from those samples." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "polyval", target: BuiltinDocumentationLinkTarget::Builtin("polyval") },
        BuiltinDocumentationLink { label: "roots", target: BuiltinDocumentationLinkTarget::Builtin("roots") },
        BuiltinDocumentationLink { label: "mldivide", target: BuiltinDocumentationLinkTarget::Builtin("mldivide") },
        BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
        BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
        BuiltinDocumentationLink { label: "polyder", target: BuiltinDocumentationLinkTarget::Builtin("polyder") },
        BuiltinDocumentationLink { label: "polyint", target: BuiltinDocumentationLinkTarget::Builtin("polyint") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/poly/polyfit.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Householder QR and provider adapter", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/poly/polyfit.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Weighted/complex fitting, degree, class, and result fields", location: "builtins::math::poly::polyfit::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU provider hook parity", location: "builtins::math::poly::polyfit::tests::polyfit_wgpu_matches_cpu" },
        ],
        notes: &["Reconciles legacy claims about arbitrary N-D flattening, weight shape equality, and provider output location. No standalone example acceptance or provider performance result is claimed by these source references."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
