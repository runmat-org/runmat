use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("deconv"),
    slug: Some("deconv"),
    summary: "Compute one-dimensional deconvolution in MATLAB and RunMat.",
    description: "`deconv(b,a)` performs one-dimensional polynomial long division in descending coefficient order. It returns quotient `q` and optionally remainder `r`. Real and complex single/double coefficients are supported; RunMat mode also admits logical coefficients and integers that are exactly representable as double.",
    keywords: &["deconv", "deconvolution", "polynomial division", "signal", "gpu"],
    related: &["conv", "conv2", "fft", "ifft", "gpuArray", "gather", "filter"],
    sections: &[
        BuiltinDocumentationSection { heading: "Polynomial division", paragraphs: &[
            "Inputs are scalars, rows, or columns; genuine matrices and N-D arrays are rejected. Only exact zero coefficients count as leading zeros. The remainder keeps the numerator's full length and orientation, making aligned reconstruction with `conv(a,q)+r` straightforward.",
            "If the numerator degree is lower than the denominator degree, `q` is zero and `r` equals the numerator. Empty or all-zero denominators are invalid. The supported public form is classic vector polynomial division; newer shape, least-squares Method, and RegularizationFactor forms remain unsupported.",
        ] },
        BuiltinDocumentationSection { heading: "Classes and execution", paragraphs: &[
            "Single coefficients select single output when every coefficient input is floating-point. A logical or typed-integer input selects double on both host and resident paths. Explicit complex coefficient input remains complex.",
            "Integer and logical inputs are RunMat extensions, rejected in MATLAB compatibility mode before data access. Each integer coefficient is checked for exact binary64 representation before division; inexact wide integer values are rejected.",
            "Resident input coefficients gather through their owning provider for host long division. Only requested output values are restored, and each restored handle must preserve the numerical class, complexity, shape, device, and owner. There is no native polynomial-division provider hook or fusion path at present.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Recover a polynomial factor", program: "p = conv([1 2], [1 -3 2]);   % Multiply two polynomials\n[q, r] = deconv(p, [1 2])   % Divide by the first factor", display_output: Some("q = [1 -3 2];\nr = [0 0 0 0]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(q,[1 -3 2])); assert(isequal(r,[0 0 0 0]));" } },
        BuiltinExample { id: "example-2", title: "Find a nonzero remainder", program: "[q, r] = deconv([1 4 7], [1 2])", display_output: Some("q = [1 2];\nr = [0 0 3]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(q,[1 2])); assert(isequal(r,[0 0 3]));" } },
        BuiltinExample { id: "example-3", title: "Divide by a longer sequence", program: "[q, r] = deconv([3 5], [1 0 2])", display_output: Some("q = 0;\nr = [3 5]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(q,0)); assert(isequal(r,[3 5]));" } },
        BuiltinExample { id: "example-4", title: "Retain leading zero coefficients", program: "b = [0 0 1 2];\na = [0 1 1];\n[q, r] = deconv(b, a)", display_output: Some("q = [1];\nr = [0 0 0 1]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(q,1)); assert(isequal(r,[0 0 0 1]));" } },
        BuiltinExample { id: "example-5", title: "Divide complex polynomials", program: "b = [1+2i 3-4i 2];\na = [1-i 2+i];\n[q, r] = deconv(b, a)", display_output: Some("q = [-0.5+1.5i 6-0.5i];\nr = [0 0 -10.5-5i]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(q),[1 2])); assert(isequal(size(r),size(b))); assert(max(abs(q-[-0.5+1.5i 6-0.5i]))<1e-12); assert(max(abs(r-[0 0 -10.5-5i]))<1e-12); assert(max(abs((conv(a,q)+r)-b))<1e-10);" } },
        BuiltinExample { id: "example-6", title: "Divide resident coefficient arrays", program: "b = gpuArray([1 3 3 1]);\na = gpuArray([1 1]);\n[q, r] = deconv(b, a); % Quotient stays on the GPU when possible\nr = gather(r);         % Explicitly gather the remainder if needed on the host", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(gather(q),[1 2 1])); assert(isequal(r,[0 0 0 0]));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What if the denominator is empty or all zero?", answer: "`deconv` raises an error; the denominator must contain a nonzero coefficient." },
        BuiltinDocumentationFaq { question: "Can I pass logical or integer vectors?", answer: "Yes, in RunMat mode only. Logical values and exactly binary64-representable integers select double output; MATLAB compatibility mode rejects these extensions before data access." },
        BuiltinDocumentationFaq { question: "Why does the remainder contain leading zeros?", answer: "It keeps the full numerator length so aligned polynomial reconstruction remains unambiguous." },
        BuiltinDocumentationFaq { question: "Does `deconv` support column vectors?", answer: "Yes. The numerator determines the orientation of quotient and remainder." },
        BuiltinDocumentationFaq { question: "How do I reconstruct the numerator?", answer: "Use `conv(a,q)+r` with coefficient alignment (shorter vectors padded on the left)." },
        BuiltinDocumentationFaq { question: "Does `deconv` perform FFT deconvolution?", answer: "No. It performs polynomial long division; FFT-based deconvolution can be assembled with `fft`, division, and `ifft`." },
        BuiltinDocumentationFaq { question: "What about ill-conditioned coefficients?", answer: "Long division can amplify rounding error. Scale coefficients or use a higher-precision method when the polynomial is ill-conditioned." },
        BuiltinDocumentationFaq { question: "Can I request both outputs?", answer: "Yes. `[q,r] = deconv(b,a)` returns both; a single-output call returns only the quotient." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "conv", target: BuiltinDocumentationLinkTarget::Builtin("conv") },
        BuiltinDocumentationLink { label: "conv2", target: BuiltinDocumentationLinkTarget::Builtin("conv2") },
        BuiltinDocumentationLink { label: "fft", target: BuiltinDocumentationLinkTarget::Builtin("fft") },
        BuiltinDocumentationLink { label: "ifft", target: BuiltinDocumentationLinkTarget::Builtin("ifft") },
        BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
        BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
        BuiltinDocumentationLink { label: "filter", target: BuiltinDocumentationLinkTarget::Builtin("filter") },
        BuiltinDocumentationLink { label: "Open a ticket", target: BuiltinDocumentationLinkTarget::External("https://github.com/runmat-org/runmat/issues/new/choose") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Division, requested outputs, and restoration", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/deconv.rs") },
            BuiltinDocumentationLink { label: "Shared resident convolution contracts", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/common/convolution.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Polynomial forms, class, extension, and resident ownership tests", location: "builtins::math::signal::deconv::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Resident quotient and remainder round trip", location: "builtins::math::signal::deconv::tests::deconv_gpu_roundtrip" },
        ],
        notes: &[],
    },
    introduced: None,
    status: None,
};
