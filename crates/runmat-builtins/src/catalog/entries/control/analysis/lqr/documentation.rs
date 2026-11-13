use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/lqr.rs") };
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("lqr"), slug: Some("lqr"),
    summary: "Compute a linear-quadratic state-feedback gain.",
    description: "`lqr(A,B,Q,R[,N])` solves the continuous-time Riccati equation for the state-feedback law `u = -K*x`. `lqr(sys,Q,R[,N])` reads A, B, and sample time from a state-space `ss` model, choosing a continuous or discrete Riccati solve. One, two, or three outputs return K, then S, then the closed-loop poles e.",
    keywords: &["lqr", "linear quadratic regulator", "state feedback", "riccati", "control system", "ss"],
    related: &["ss", "eig", "pole"],
    sections: &[
        BuiltinDocumentationSection { heading: "Inputs and outputs", paragraphs: &[
            "A is n-by-n square, B is n-by-m with m at least one, Q is n-by-n symmetric, R is m-by-m symmetric positive definite, and optional N is n-by-m. All values must be finite and real. The block weighting matrix [Q N; N' R] must be positive semidefinite. The solver rejects a model for which it cannot find a finite stabilizing solution.",
            "K is an m-by-n real double matrix, S is an n-by-n real double matrix, and e is an n-by-1 real or complex double pole column. The pole vector is checked for stability on every request, even if not returned. Matrix-input forms always use continuous time; ss object forms use Ts=0 for continuous time and Ts>0 for discrete time. More than three outputs raise an error. A zero-output request still validates and solves the problem, then returns an empty output list.",
            "RunMat accepts all eight fixed-width integer classes for A, B, Q, R, and N only in RunMat compatibility mode. The values convert to double at the host Riccati input boundary, so integers beyond binary64 exact precision can round. Public MATLAB lqr documentation specifies matrix dimensions and weighting conditions but does not document fixed-width integer matrix support; this is not presented as verified MATLAB parity.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
            "Continuous and discrete standard problems use a QR-deflated extended Riccati pencil and ordered generalized Schur decomposition. Both paths check finite solution and gain values, the original equation residual, and closed-loop pole stability before constructing output arrays; an unresolved or nonstabilizing subspace returns a solver error.",
            "lqr gathers valid resident numeric matrices through their owner, then solves on the host. It has no provider-level GPU Riccati kernel and is a fusion boundary; all outputs are host-resident. State-space form reads only Ts, A, and B from the model, plus supplied weights. Invalid output count and strict-mode integer admission reject before any new provider gather.",
            "The call may suspend during gathers and may throw for invalid dimensions, unsupported data, or solver failure. It has no workspace, filesystem, network, callback, user-interface, randomness, or time effect. Large dense solves consume host CPU and memory; the general scheduler, not this builtin, owns placement timing.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "double-integrator", title: "Continuous-time double-integrator regulator", program: "A = [0 1; 0 0]; B = [0; 1]; [K, S, e] = lqr(A, B, eye(2), 1); fprintf('K is approximately [%g %.4f]\\n', K(1), K(2));", display_output: Some("K is approximately [1 1.7321]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(K - [1 sqrt(3)])) < 1e-4); assert(isequal(size(S), [2 2])); assert(all(real(e) < 0));" } },
        BuiltinExample { id: "state-space", title: "Continuous state-space model", program: "sys = ss([0 1; 0 0], [0; 1], [1 0], 0); K = lqr(sys, eye(2), 1);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(K - [1 sqrt(3)])) < 1e-4);" } },
        BuiltinExample { id: "discrete-state-space", title: "Discrete state-space model", program: "sysd = ss([1 0.1; 0 1], [0.005; 0.1], [1 0], 0, 0.1); [K, S, e] = lqr(sysd, eye(2), 1);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(K), [1 2])); assert(isequal(size(S), [2 2])); assert(all(abs(e) < 1));" } },
        BuiltinExample { id: "cross-weight", title: "Cross-weight matrix", program: "A = [0 1; 0 0]; B = [0; 1]; N = [0.25; 0]; K = lqr(A, B, diag([2 3]), 2, N);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(K), [1 2])); assert(all(isfinite(K)));" } },
        BuiltinExample { id: "integer-extension", title: "RunMat integer matrix extension", program: "A = int16([0 1; 0 0]); B = int16([0; 1]); K = lqr(A, B, eye(2), 1);", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(K), [1 2])); assert(strcmp(class(K), 'double'));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Can I supply the cross-weight matrix?", answer: "Yes. N is the fifth matrix argument or fourth ss-form argument." },
        BuiltinDocumentationFaq { question: "Does lqr run on the GPU?", answer: "No. Supported resident inputs gather and the dense Riccati solve runs on host CPU." },
        BuiltinDocumentationFaq { question: "Why can the pole output be complex?", answer: "A real closed-loop matrix can have conjugate complex eigenvalues." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "ss", target: BuiltinDocumentationLinkTarget::Builtin("ss") },
        BuiltinDocumentationLink { label: "eig", target: BuiltinDocumentationLinkTarget::Builtin("eig") },
        BuiltinDocumentationLink { label: "pole", target: BuiltinDocumentationLinkTarget::Builtin("pole") },
        BuiltinDocumentationLink { label: "MathWorks lqr reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/control/ref/lti.lqr.html") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Matrix, ss, integer, output count and error cases", location: "crates/runmat-runtime/src/builtins/control/lqr.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM lqr workflow", location: "crates/runmat-vm/tests/control.rs" },
    ], notes: &["Riccati algorithm/solver limits and unverified integer MATLAB parity leave maturity partial. Native/browser/real-provider qualification remains integrated work."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
