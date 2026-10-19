use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("polyval"),
    slug: Some("polyval"),
    summary: "Evaluate a polynomial at numeric points and optionally compute prediction intervals.",
    description: "`polyval(p,x)` evaluates descending-power coefficients by Horner's method at every element of x. Polyfit statistics S and scaling values mu enable centering-aware evaluation and prediction intervals.",
    keywords: &["polyval", "polynomial", "polyfit", "prediction interval", "gpu"],
    related: &["polyfit", "conv", "deconv", "polyder", "polyint", "roots"],
    sections: &[
        BuiltinDocumentationSection { heading: "Inputs and shape", paragraphs: &[
            "Coefficients may be a scalar, empty vector, or any vector shape with at most one nonsingleton dimension. Numeric or logical points may be scalar, vector, matrix, or N-D; y follows x's shape. Empty points return empty output; empty coefficients represent the zero polynomial.",
            "Logical values convert to double. Typed fixed-width integer coefficients, points, and numeric fit metadata are independent RunMat-mode extensions, checked for exact binary64 conversion before evaluation. Floating complex coefficients and points use complex host arithmetic.",
            "`polyval(p,x,[],mu)` evaluates at `(x-mu(1))/mu(2)`. The runtime uses the first two mu elements; both must be finite and the scale nonzero. A complex mu is admitted only when those elements have zero imaginary components.",
        ] },
        BuiltinDocumentationSection { heading: "Prediction intervals", paragraphs: &[
            "`[y,delta] = polyval(p,x,S)` and the four-input form require nonempty polyfit statistics S with fields R, df, and normr. R must be square with one row per coefficient; df and normr are real scalars. Without S, requesting delta is an argument error, not a zero interval.",
            "The interval uses the fit's QR factor in the same coordinate basis as p. For `[p,S] = polyfit(...)`, pass S without mu; for `[p,S,mu] = polyfit(...)`, pass both S and mu. If df is nonpositive or normr is zero, delta is zero. Further requested output slots are padded with scalar zero values.",
        ] },
        BuiltinDocumentationSection { heading: "Provider execution", paragraphs: &[
            "A compatible real workload with resident coefficient or point input may use the owning provider's Horner hook. The current adapter first converts operands to host numerical vectors, then uploads provider-precision buffers for the hook; this is not zero-copy resident evaluation. Complex and prediction-interval routes execute on the host.",
            "Real host results may be uploaded to preserve requested residency, while unsupported complex output remains host-resident. Provider output shape, storage, owner, and non-aliasing are validated before publication. Evaluation is a fusion boundary.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "scalar", title: "Evaluate at one point", program: "p = [2 -3 5];\ny = polyval(p,4)", display_output: Some("y = 25"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(y==25);" } },
        BuiltinExample { id: "vector", title: "Evaluate across a vector", program: "p = [1 0 -2 1];\nx = linspace(-2,2,5);\ny = polyval(p,x)", display_output: Some("y = [-3 2 1 0 5]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(y-[-3 2 1 0 5]))<1e-12);" } },
        BuiltinExample { id: "matrix", title: "Preserve a matrix of query points", program: "[X,Y] = meshgrid(-1:1);\nZ = polyval([1 -3 2],X+Y)", display_output: Some("Z = [12 6 2; 6 2 0; 2 0 0]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(Z,[12 6 2;6 2 0;2 0 0]));" } },
        BuiltinExample { id: "scaling", title: "Reuse polyfit centering and scaling", program: "x = -2:2;\ny0 = sin(x)+0.05*cos(3*x);\n[p,S,mu] = polyfit(x,y0,3);\ny = polyval(p,x,[],mu)", display_output: Some("y is a smoothed fit with x's shape"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(y),size(x))); assert(all(isfinite(y)));" } },
        BuiltinExample { id: "delta", title: "Compute an interval from fit statistics", program: "x = (0:10)';\n[p,S,mu] = polyfit(x,exp(x/10),3);\n[y,delta] = polyval(p,5,S,mu)", display_output: Some("y is the fitted value; delta is its interval estimate"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isscalar(y)); assert(isscalar(delta)); assert(delta>=0);" } },
        BuiltinExample { id: "complex", title: "Evaluate with complex coefficients and points", program: "p = [1+2i -3 4i];\nz = [-1+1i 0 1-2i];\ny = polyval(p,z)", display_output: Some("y contains complex values"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(y),size(z))); assert(max(abs(y-((1+2i)*z.^2-3*z+4i)))<1e-10);" } },
        BuiltinExample { id: "gpu", title: "Evaluate a real polynomial with resident points", program: "x = gpuArray(linspace(-1,1,2048));\np = [1 0 1];\ny = polyval(p,x);\nvalues = gather(y)", display_output: Some("values = x.^2 + 1 on the host"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(y,'gpuArray')); assert(max(abs(values-(linspace(-1,1,2048).^2+1)))<1e-5);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Do coefficients need to be a row?", answer: "No. A row, column, or singleton-extended vector is accepted. The result follows the shape of x." },
        BuiltinDocumentationFaq { question: "Which point shapes are supported?", answer: "Scalar, vector, matrix, and N-D numeric or logical point arrays are supported, including complex floating data." },
        BuiltinDocumentationFaq { question: "How does mu affect evaluation?", answer: "The first two mu values center and scale each point before Horner evaluation: `(x-mu(1))/mu(2)`. The scale must be finite and nonzero." },
        BuiltinDocumentationFaq { question: "Why does delta require S?", answer: "The interval uses polyfit's QR factor R, degrees of freedom df, and residual norm normr; without these statistics it cannot be computed." },
        BuiltinDocumentationFaq { question: "When is delta zero?", answer: "When df is nonpositive or normr is zero, the current implementation returns a zero interval." },
        BuiltinDocumentationFaq { question: "Can the result stay on a GPU?", answer: "Supported real provider output stays resident. Complex output uses host arithmetic and may remain on the host." },
        BuiltinDocumentationFaq { question: "Are sparse points supported?", answer: "The current direct implementation accepts dense numeric and logical values, including supported provider tensors; sparse arrays are not a direct input form." },
        BuiltinDocumentationFaq { question: "How is the polynomial evaluated?", answer: "Horner's method nests multiplication and addition instead of computing each power separately." },
        BuiltinDocumentationFaq { question: "How do I use polyval with polyfit?", answer: "Fit `[p,S,mu] = polyfit(x,y,n)`, then evaluate `polyval(p,xfine,[],mu)` or request `[yf,delta] = polyval(p,xfine,S,mu)` for intervals." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "polyfit", target: BuiltinDocumentationLinkTarget::Builtin("polyfit") },
        BuiltinDocumentationLink { label: "conv", target: BuiltinDocumentationLinkTarget::Builtin("conv") },
        BuiltinDocumentationLink { label: "deconv", target: BuiltinDocumentationLinkTarget::Builtin("deconv") },
        BuiltinDocumentationLink { label: "polyder", target: BuiltinDocumentationLinkTarget::Builtin("polyder") },
        BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
        BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
        BuiltinDocumentationLink { label: "polyint", target: BuiltinDocumentationLinkTarget::Builtin("polyint") },
        BuiltinDocumentationLink { label: "roots", target: BuiltinDocumentationLinkTarget::Builtin("roots") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/poly/polyval.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Polynomial evaluation runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/poly/polyval.rs") },
            BuiltinDocumentationLink { label: "Resident provider adapter", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/poly/polyval/resident.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, N-D, complex, fit options, and intervals", location: "builtins::math::poly::polyval::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Resident output and provider validation", location: "builtins::math::poly::polyval::resident::tests" },
        ],
        notes: &["Legacy seven examples and useful FAQ content retained with deterministic assertions. The old scalar example claimed 21 for 2*4^2-3*4+5; actual result is 25. The unsupported two-output/no-S signature and invalid gpuArray.linspace syntax were corrected. Standalone acceptance and hardware performance remain open."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
