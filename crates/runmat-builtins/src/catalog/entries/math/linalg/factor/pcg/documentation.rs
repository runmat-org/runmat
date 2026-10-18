use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Matrices and callable operators",
        paragraphs: &[
            "`pcg(A,b)` uses conjugate gradients when `A` is a square dense or sparse CSC matrix with native single or double real or complex coefficients and `b` is a matching column vector. Dense storage is borrowed in column-major order; sparse storage is multiplied directly without densifying A.",
            "`A` can also be a function handle implementing the matrix-vector product. Each call receives an n-by-1 vector and must return an n-by-1 vector with the same single/double precision and real/complex domain as `b`. The solver invokes the callback asynchronously and keeps its input and result rooted across suspension.",
            "`tol`, `maxit`, and `x0` occupy the usual optional positions. Empty placeholders select the defaults. The default tolerance is 1e-6 for double and 1e-3 for single; default `maxit` is `min(size(A,1),20)`. The initial guess defaults to zero.",
            "Up to five outputs are `[x,flag,relres,iter,resvec]`. `resvec` starts with `norm(b-A*x0)`. A nonconverged result returns the iterate with the lowest computed residual norm.",
            "`M1` and `M2` can be function handles implementing their respective inverse actions, or validated triangular dense/CSC factors. The solver applies M1 first and M2 second; an empty placeholder omits that action. A matrix factor is solved as a triangular system, not multiplied into the residual. General nontriangular matrix preconditioners remain unsupported.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Current boundaries",
        paragraphs: &[
            "Matrix coefficients, vectors, and triangular factors must have matching native floating classes. Mixed coefficient/vector classes, gpuArray inputs, and distributed inputs are not silently converted or transferred. Callable outputs are checked on every application rather than inferred from the first result.",
            "For matrix-only solves, a finite managed host operation currently admits the single-result tensor, including a one-output assignment. Callable forms still require unbounded host allocation policy until reusable callable metadata has retained exact ownership; managed calls reject those forms before invoking user code. Two or more requested outputs remain unavailable under managed policy until the shared `OutputList` representation gains retained exact metadata ownership; explicitly unbounded calls support up to five outputs.",
            "The compatibility contract is incomplete. The implemented flags are 0 (converged), 1 (iteration limit), 3 (stagnation), and 4 (breakdown or nonpositive curvature). The complete MATLAB preconditioner-failure flag contract remains an implementation obligation; callback errors and invalid factors are not relabeled as successful solves.",
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
    description: "Solve dense, sparse CSC, or callable systems with a shared precision-preserving host recurrence, optionally using callable inverse actions or triangular preconditioners. General matrix preconditioners and the full compatibility contract remain incomplete.",
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
