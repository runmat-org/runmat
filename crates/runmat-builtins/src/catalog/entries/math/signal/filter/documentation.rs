use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("filter"),
    slug: Some("filter"),
    summary: "Apply a causal FIR or IIR digital filter along a selected array dimension.",
    description: "`filter(b, a, x)` applies the difference equation specified by numerator coefficients `b` and denominator coefficients `a`. It operates along the first nonsingleton dimension unless `dim` is supplied. The optional `zi` supplies initial state; a second requested output returns final state for a later chunk.",
    keywords: &["filter", "IIR", "FIR", "difference equation", "initial conditions", "GPU"],
    related: &["conv", "deconv", "conv2", "fft"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Coefficients, dimensions, and state",
            paragraphs: &[
                "`b` and `a` must be nonempty vectors and `a(1)` must be nonzero. RunMat divides both coefficient vectors by `a(1)` before filtering.",
                "A matrix is filtered column-wise by default. `filter(b,a,x,zi,dim)` selects another dimension; a dimension beyond `ndims(x)` is a length-one axis.",
                "`zi` has the same shape as the returned `zf`: filter order first, followed by the dimensions of `x` not being filtered. `[]` selects zero initial state. For vector input, `zf` is a column vector.",
                "For empty `x`, the output signal is empty and final state equals the supplied or zero initial state. Complex coefficients, signals, and states are supported.",
                "The default call returns `y`; `[y,zf] = filter(...)` also returns the final state. Pass `zf` as `zi` to continue filtering the next chunk.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Classes and execution",
            paragraphs: &[
                "Single signal inputs produce single outputs; double, integer, and logical signals produce double outputs. Floating filter arithmetic follows the runtime's coefficient and signal conversion rules; wide-integer rounding at this numerical boundary is not an exact-integer computation.",
                "A real resident signal can use its owning provider's `iir_filter` operation when the coefficient, state, dimension, and physical-class requirements are met. Complex forms and unsupported provider operations use the host reference path. Operational provider errors are returned with their cause; they are not retried as host computation.",
                "The operation is a fusion boundary. Runtime placement and provider capabilities determine whether the result stays resident; GPU support does not imply that every input form is device-resident.",
            ],
        },
    ],
    examples: &[
        BuiltinExample { id: "fir-average", title: "Smooth a signal with an FIR average", program: "b = ones(1,3)/3; a = 1; x = [1 5 2 0 3]; y = filter(b,a,x)", display_output: Some("y =\n    0.3333    2.0000    2.6667    2.3333    1.6667"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(y-[1/3 2 8/3 7/3 5/3])) < 1e-12);" } },
        BuiltinExample { id: "iir-impulse", title: "Apply a first-order IIR low-pass filter", program: "alpha = 0.8; b = 1-alpha; a = [1 -alpha]; imp = [1 zeros(1,4)]; y = filter(b,a,imp)", display_output: Some("y =\n    0.2000    0.1600    0.1280    0.1024    0.0819"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(y-0.2*0.8.^(0:4))) < 1e-12);" } },
        BuiltinExample { id: "continuation", title: "Continue filtering from a saved state", program: "b = ones(1,3)/3; a = 1; x1 = [1 5 2]; [y1,zf] = filter(b,a,x1); x2 = [0 3]; [y2,zf2] = filter(b,a,x2,zf); y1, zf, y2, zf2", display_output: Some("y1 = [0.3333 2.0000 2.6667]\nzf = [2.3333; 0.6667]\ny2 = [2.3333 1.6667]\nzf2 = [1.0000; 1.0000]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(y1-[1/3 2 8/3])) < 1e-12); assert(isequal(size(zf),[2 1])); assert(max(abs(zf-[7/3;2/3])) < 1e-12); assert(max(abs([y1 y2]-filter(b,a,[x1 x2]))) < 1e-12); assert(max(abs(y2-[7/3 5/3])) < 1e-12); assert(isequal(size(zf2),[2 1])); assert(max(abs(zf2-[1;1])) < 1e-12);" } },
        BuiltinExample { id: "dimension-two", title: "Filter across matrix rows", program: "X = [1 2 3 4; 0 1 0 1]; b = [1 -1]; a = 1; Y = filter(b,a,X,[],2)", display_output: Some("Y =\n     1     1     1     1\n     0     1    -1     1"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(Y,[1 1 1 1;0 1 -1 1]));" } },
        BuiltinExample { id: "complex", title: "Filter complex-valued data", program: "b = [1 1i]; a = 1; t = 0:3; x = exp(1i*pi/4*t); y = filter(b,a,x)", display_output: Some("y =\n   1.0000 + 0.0000i   0.7071 + 1.7071i  -0.7071 + 1.7071i  -1.7071 + 0.7071i"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(y(1)-x(1)) < 1e-12); assert(max(abs(y(2:end)-(x(2:end)+1i*x(1:end-1)))) < 1e-12);" } },
        BuiltinExample { id: "resident", title: "Filter a real resident signal", program: "g = gpuArray([1 5 2 0 3]); b = ones(1,3)/3; a = 1; [y_gpu,zf_gpu] = filter(b,a,g); y = gather(y_gpu)", display_output: Some("y =\n    0.3333    2.0000    2.6667    2.3333    1.6667"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(y-[1/3 2 8/3 7/3 5/3])) < 1e-12); zf = gather(zf_gpu); assert(isequal(size(zf),[2 1])); assert(max(abs(zf-[1;1])) < 1e-12);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What if `a(1)` is not one?", answer: "Both coefficient vectors are normalized by the nonzero leading denominator coefficient." },
        BuiltinDocumentationFaq { question: "How large must `zi` be?", answer: "It matches `zf`: filter order first, then the dimensions not being filtered. Use `[]` for zero state." },
        BuiltinDocumentationFaq { question: "Can `dim` exceed `ndims(x)`?", answer: "Yes. A missing higher axis is treated as length one." },
        BuiltinDocumentationFaq { question: "Does the filter support complex inputs?", answer: "Yes. Complex arithmetic uses the host reference route when the resident operation lacks that form." },
        BuiltinDocumentationFaq { question: "Must I initialize GPU filtering?", answer: "No. RunMat selects the owning provider when its real-valued `iir_filter` form is available; unsupported forms follow the host path." },
        BuiltinDocumentationFaq { question: "What happens for an empty signal?", answer: "The signal output is empty and final state retains the initial or zero state." },
        BuiltinDocumentationFaq { question: "Can I pass integer or logical samples?", answer: "Yes. They enter floating filter arithmetic and produce double output; this conversion is not exact wide-integer arithmetic." },
        BuiltinDocumentationFaq { question: "How do I filter a stream in chunks?", answer: "Request `[y,zf]` for each chunk and pass that `zf` as the next call's `zi`." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "conv", target: BuiltinDocumentationLinkTarget::Builtin("conv") },
        BuiltinDocumentationLink { label: "deconv", target: BuiltinDocumentationLinkTarget::Builtin("deconv") },
        BuiltinDocumentationLink { label: "fft", target: BuiltinDocumentationLinkTarget::Builtin("fft") },
        BuiltinDocumentationLink { label: "conv2", target: BuiltinDocumentationLinkTarget::Builtin("conv2") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/filter.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/filter.rs") },
            BuiltinDocumentationLink { label: "Host filter", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/filter/host.rs") },
            BuiltinDocumentationLink { label: "Provider filter", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/filter/provider.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Filter semantics and provider admission", location: "builtins::math::signal::filter::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Resident filter comparison", location: "builtins::math::signal::filter::tests::provider::filter_gpu_matches_cpu" },
        ],
        notes: &["The resident example requires an available WGPU device. Filtering in chunks retains a column-vector state for a vector signal."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
