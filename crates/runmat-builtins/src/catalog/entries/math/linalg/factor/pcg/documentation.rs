use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Matrices and callable operators",
        paragraphs: &[
            "`pcg(A,b)` uses conjugate gradients when `A` is a square dense or sparse CSC matrix with native single or double real or complex coefficients and `b` is a matching column vector. Dense storage is borrowed in column-major order; sparse storage is multiplied directly without densifying A.",
            "`A` can also be a function handle implementing the matrix-vector product. Each call receives an n-by-1 vector and must return an n-by-1 vector with the same single/double precision and real/complex domain as `b`. The solver invokes the callback asynchronously and keeps its input and result rooted across suspension.",
            "`tol`, `maxit`, and `x0` occupy the usual optional positions. Empty placeholders select the defaults. The default tolerance is 1e-6 for double and 1e-3 for single; default `maxit` is `min(size(A,1),20)`. The initial guess defaults to zero.",
            "Up to five outputs are `[x,flag,relres,iter,resvec]`. `resvec` starts with `norm(b-A*x0)`. A nonconverged result returns the iterate with the lowest computed residual norm.",
            "`M1` and `M2` can be function handles implementing inverse actions, or matching dense/CSC matrix preconditioners. The solver applies M1 first and M2 second; empty placeholders omit either action. Triangular matrices remain borrowed; a general matrix is factored once and reused, not multiplied into the residual. General sparse preconditioners support finite managed allocation with individually admitted sparse fill growth and reusable solve workspace.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Current boundaries",
        paragraphs: &[
            "Matrix coefficients, vectors, and triangular factors must have matching native floating classes. Mixed coefficient/vector classes, gpuArray inputs, and distributed inputs are not silently converted or transferred. Callable outputs are checked on every application rather than inferred from the first result.",
            "Matrix and callable solves support finite managed host allocation for up to five outputs, subject to the selected callback's own admission. Solver scratch and output backing are admitted before iteration. Each callback receives an independently claimed vector, so retaining an input cannot alias or retain mutable recurrence scratch. Callback captures and source authority remain owned by the execution context.",
            "Callback vector and result copies, solution publication and residual-history copies check execution authority in bounded chunks and use the context's CPU cooperation budget. A suspended output constructor owns its exact backing until completion or drop. A receipt-less solve or factor cannot continue if its allocation scope becomes finite; it must fail rather than publish unclaimed storage.",
            "The implemented flags are 0 (converged), 1 (iteration limit), 2 (a numerical preconditioner failure or non-finite inverse-action result), 3 (stagnation), and 4 (breakdown or nonpositive curvature). Flag 2 retains the best iterate, its relative residual and the residual history computed before the failed action. Callback language errors, invalid classes/shapes, allocation denial, cancellation and authority changes remain errors; the solver does not retry a completed callback or turn those failures into a numerical flag.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "pcg-sparse-spd",
    title: "Solve a sparse positive-definite system",
    program: "A = sparse([4 1; 1 3]); b = [1; 2]; [x,flag,relres] = pcg(A,b);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(flag == 0); assert(relres < 1e-6); assert(norm(A*x-b) < 1e-6);",
    },
}, BuiltinExample {
    id: "pcg-callable-inverse-action",
    title: "Use a matrix-vector callback and a diagonal inverse action",
    program: "A = [4 1; 1 3]; b = [1; 2]; Afun = @(v) A*v; Mfun = @(v) [v(1)/4; v(2)/3]; [x,flag,relres] = pcg(Afun,b,1e-8,20,Mfun);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(flag == 0); assert(relres < 1e-8); assert(norm(A*x-b) < 1e-8);",
    },
}];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pcg"),
    slug: Some("pcg"),
    summary: "Solve a positive-definite linear system with conjugate gradients.",
    description: "Solve dense, sparse CSC, or callable systems with a shared precision-preserving host recurrence, optionally using reusable matrix or callable inverse-action preconditioners. The full compatibility contract remains incomplete.",
    keywords: &["pcg", "dense", "sparse", "conjugate gradients", "iterative solver"],
    related: &["sparse", "mldivide", "eigs"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink {
        label: "sparse",
        target: BuiltinDocumentationLinkTarget::Builtin("sparse"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "typed dense and CSC PCG",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/pcg/mod.rs",
            ),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "real and Hermitian complex CSC convergence",
            location: "builtins::math::linalg::factor::pcg::tests::pcg_csc_real_and_hermitian_complex_converge_without_dense_coefficients",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
