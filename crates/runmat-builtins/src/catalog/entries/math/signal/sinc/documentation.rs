use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("sinc"),
    slug: Some("sinc"),
    summary: "Compute the normalized sinc function `sin(pi*x)/(pi*x)` element-wise.",
    description: "`y = sinc(x)` evaluates the normalized sinc function using MATLAB's definition `sin(pi*x)/(pi*x)`, with the removable singularity handled as `sinc(0) = 1`.",
    keywords: &["sinc", "normalized sinc", "signal processing", "elementwise", "complex"],
    related: &["sin", "fft", "filter", "conv", "gpuArray", "gather"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Values and classes",
            paragraphs: &[
                "Real and complex scalar and array inputs retain their shape. Single or double floating inputs retain their precision, including on supported GPU execution and restoration paths.",
                "Exactly zero real input returns one. Finite, nonzero integer-valued real input returns exact zero. Complex input uses `sin(pi*z)/(pi*z)`, with `sinc(0 + 0i) = 1 + 0i`.",
                "Single and double real or complex values are the MATLAB-compatible forms. RunMat mode also admits typed integers and logical values, producing double output. Full-width integer inputs use the exact zero/nonzero identity without a binary64 conversion requirement.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU execution",
            paragraphs: &[
                "Real and complex-interleaved GPU inputs can remain resident when their owning provider supports `unary_sinc`. A typed unsupported hook permits host evaluation and restoration through the same provider. Operational failures do not trigger host retry.",
                "The real elementwise fusion expression guards zero and integer-valued inputs. Provider support and fusion eligibility depend on the admitted class, storage, device, and expression route; neither is implied solely by a GPU handle.",
            ],
        },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Evaluate sinc at zero", program: "y = sinc(0)", display_output: Some("y = 1"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(y == 1);" } },
        BuiltinExample { id: "example-2", title: "Evaluate sinc at integer inputs", program: "x = 1:5;\ny = sinc(x)", display_output: Some("y = [0 0 0 0 0]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(y,[0 0 0 0 0]));" } },
        BuiltinExample { id: "example-3", title: "Evaluate normalized sinc values", program: "x = [-1 0 1 0.5];\ny = sinc(x)", display_output: Some("y = [0 1 0 0.6366]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(y(1:3),[0 1 0])); assert(abs(y(4)-2/pi) < 1e-12);" } },
        BuiltinExample { id: "example-4", title: "Preserve matrix shape", program: "A = [0 0.5; 1.5 2];\nY = sinc(A)", display_output: Some("Y =\n    1.0000    0.6366\n   -0.2122         0"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(Y),[2 2])); assert(Y(1,1)==1 && Y(2,2)==0); assert(abs(Y(1,2)-2/pi)<1e-12); assert(abs(Y(2,1)+2/(3*pi))<1e-12);" } },
        BuiltinExample { id: "example-5", title: "Evaluate sinc for a complex value", program: "z = sinc(0.5 + 0.25i)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "expected = cosh(pi/4)*(8-4i)/(5*pi); assert(~isreal(z)); assert(abs(z-expected)<1e-12);" } },
        BuiltinExample { id: "example-6", title: "Use gpuArray input", program: "g = gpuArray([0 0.5 1]);\ny = sinc(g);\nresult = gather(y)", display_output: Some("result = [1 0.6366 0]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(y,'gpuArray')); assert(isequal(size(result),[1 3])); assert(isequal(result([1 3]),[1 0])); assert(abs(result(2)-2/pi)<1e-12);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Which sinc definition does RunMat use?", answer: "RunMat uses MATLAB's normalized definition, `sin(pi*x)/(pi*x)`, with the removable singularity at zero defined as `1`." },
        BuiltinDocumentationFaq { question: "Why does `sinc(1:5)` return exact zeros?", answer: "The implementation detects finite nonzero integer-valued real inputs before evaluating the trigonometric expression. This avoids floating-point residuals from `sin(pi*x)` and matches MATLAB's practical `sinpi` behaviour." },
        BuiltinDocumentationFaq { question: "Does `sinc` support complex values?", answer: "Yes. Complex scalars and complex tensors use the analytic extension `sin(pi*z)/(pi*z)` and preserve tensor shape." },
        BuiltinDocumentationFaq { question: "Will `sinc(gpuArray(...))` stay on the GPU?", answer: "Yes for real and complex gpuArray inputs when the active provider implements `unary_sinc`. Providers without that hook fall back to host evaluation so zero and integer cases still use the exact MATLAB-compatible path." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "sin", target: BuiltinDocumentationLinkTarget::Builtin("sin") },
        BuiltinDocumentationLink { label: "fft", target: BuiltinDocumentationLinkTarget::Builtin("fft") },
        BuiltinDocumentationLink { label: "filter", target: BuiltinDocumentationLinkTarget::Builtin("filter") },
        BuiltinDocumentationLink { label: "conv", target: BuiltinDocumentationLinkTarget::Builtin("conv") },
        BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
        BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/sinc.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Sinc host, provider, and fusion implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/sinc.rs") }, BuiltinDocumentationLink { label: "Typed, cooperative real host waveform mapping", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/common/waveform_map.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Host and provider-backed sinc behavior", location: "builtins::math::signal::sinc::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Resident unary_sinc provider route", location: "builtins::math::signal::sinc::tests::sinc_gpu_input_stays_resident_when_provider_supports_sinc" }],
        notes: &[],
    },
    introduced: None,
    status: None,
};
