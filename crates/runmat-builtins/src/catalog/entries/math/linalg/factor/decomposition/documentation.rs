use crate::*;

pub(super) const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Construction and factor selection",
        paragraphs: &[
            "Coefficient matrices can be full or sparse, real or complex, single or double. Trailing singleton dimensions are accepted; non-singleton dimensions beyond the second are rejected. Empty matrices retain their exact shape.",
            "Supported types are 'auto', 'lu', 'qr', 'cod', 'chol', 'triangular' and 'diagonal'. Automatic selection uses diagonal or triangular structure when available, LU for other square matrices and QR for rectangular matrices. Other type names are rejected rather than silently mapped to a different algorithm.",
            "LU, Cholesky, triangular and diagonal objects require square matrices. A diagonal or unqualified triangular request validates that structure. For triangular and Cholesky requests, an optional 'upper' or 'lower' flag selects one triangle and ignores the other. Cholesky constructs the corresponding symmetric or Hermitian matrix and requires positive definiteness.",
            "Explicit LU, Cholesky, QR and COD constructors compute and retain their numerical factors once. Dense matrices use compact native dense factors; sparse matrices retain sparse factors and ordering maps without densifying the coefficients. Diagonal and triangular objects borrow their original storage and use substitution without creating an unnecessary LU factorization.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Solves, rank and warnings",
        paragraphs: &[
            "Use dA \\ b for a left solve and b / dA for a right solve. A QR object returns a basic least-squares solution. A COD object instead returns a minimum-norm least-squares solution, including wide and rank-deficient systems. QR does not support solving through dA' or using b / dA; choose COD for those forms. Singular explicit LU retains its selected LU arithmetic and does not silently switch to a minimum-norm solver.",
            "RankTolerance is an absolute threshold on the magnitudes of the pivoted R diagonal. It determines the retained rank for QR, COD and automatically selected rectangular QR; other types preserve but ignore it. It must be a nonnegative real scalar. A different tolerance can change the low-rank approximation and the returned solution.",
            "CheckCondition defaults to true. Set it to false to suppress the implemented singular-pivot and rank-deficiency warnings; it does not change the selected algorithm, rank tolerance or numerical result. RunMat does not yet implement a general near-singular condition estimator for this object. MATLAB documents logical controls; RunMat also retains its existing real-numeric truth-value coercion, with typed integer controls separately gated as an extension.",
            "Solutions are dense arrays. Single precision is retained if the coefficient factor or right-hand side is single; otherwise floating solutions use double. Complex coefficients, right-hand sides or scale factors produce complex solutions. A real factor can solve a complex right-hand side without discarding its imaginary part.",
            "A nonscalar floating coefficient matrix requires a floating right-hand side. The real scalar double-coefficient/integer-right-hand-side form retains the existing integer division rules; an integer coefficient matrix admitted by RunMat extensions instead uses the documented checked floating calculation boundary.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Properties and value semantics",
        paragraphs: &[
            "Public properties include MatrixSize, Type, CheckCondition, Datatype, IsConjugateTransposed, IsReal, IsSparse and ScaleFactor. RankTolerance and Triangle are present when supplied. The original coefficient payload and numerical factor arrays are not publicly readable.",
            "MatrixSize is a two-element double row vector. Exact storage dimensions remain authoritative; this display property is not used to reconstruct a shape. Datatype records the original coefficient class, including admitted integer or logical extensions.",
            "Unary plus preserves the object. Negation, scalar multiplication and scalar division create new metadata sharing the same immutable factors. Conjugate transpose toggles the selected orientation and conjugates the scale without forming a transposed coefficient array or refactoring it. Neither operation mutates the original object. QR solve restrictions still apply after transposition.",
            "Element-wise left division (ldivide or .\\) is not a decomposition-object method. The registry does not expose decomposition.ldivide.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "RunMat extensions and GPU inputs",
        paragraphs: &[
            "With RunMat extensions enabled, coefficient arrays may use any of the eight integer classes or logical storage. Their original class and values remain in the object. Strict compatibility mode rejects extension-only inputs before a provider gather.",
            "LU, QR and COD convert nonfloating coefficients once at factor construction. Every integer real and imaginary component must be exactly representable in binary64; conversion failure is reported instead of rounding a wide value silently. Cholesky keeps its additional rejection of paired complex-integer storage. Structural diagonal and triangular objects can retain wide integers exactly at construction, but still require a checked floating boundary when solving.",
            "Typed integer RankTolerance, integer CheckCondition and real or paired-complex integer scale controls have independent extension checks. Scale and tolerance components must be exactly representable at their floating metadata boundary.",
            "Explicit gpuArray coefficients, operands and controls use the declared RunMat extension gates. Admitted resident inputs gather through their provider owner. Factor construction and retained-factor solves execute on the host. A solve with a resident right-hand side restores the result through that operand's provider, preserving its device, output class and placement intent. It does not refactor the matrix through a generic GPU solver or claim a resident GPU factorization.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution, ownership and persistence",
        paragraphs: &[
            "Each operation returns one value. Discarding it still validates the call; requesting extra outputs fails before provider access. Runtime services own gathering, cooperative work, CPU jobs, rooting and output publication. A later solve uses its current caller's authority and fresh scratch/output owners while sharing immutable factor payloads.",
            "Structural operations do not duplicate the coefficient or factor arrays. Right and adjoint solves read the selected orientation without first creating a whole transposed right-hand-side copy. Factor payloads retain their physical allocation owners until the last alias is retired.",
            "Native, JIT, AOT and browser execution use the same typed Runtime bindings and numerical contracts. Placement uses the existing host work and provider-transfer boundaries; no separate decomposition timer, cost table or dispatch threshold is introduced.",
            "Durable preferences preserve retained factors and their rank/permutation metadata. Selected readback validates and imports them under the current Runtime context rather than refactoring on every solve. MAT persistence stores the coefficient/options recipe and reconstructs it once during selected import. ExecutionV2 worker transport currently rejects decomposition objects; a generic object record is not an alternative transport for native factor state.",
        ],
    },
];

pub(super) const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Can I read the LU or QR arrays from the object?",
        answer: "No. Use lu, qr or chol when you need explicit factors. A decomposition object retains private immutable factors for reuse by its solve methods.",
    },
    BuiltinDocumentationFaq {
        question: "When should I choose COD instead of QR?",
        answer: "Choose COD when a wide or rank-deficient problem needs the minimum-norm answer, or when you need a right or conjugate-transposed solve. QR provides a basic least-squares answer and rejects those solve orientations.",
    },
    BuiltinDocumentationFaq {
        question: "Does sparse construction create a dense coefficient copy?",
        answer: "No. Sparse input uses sparse factor storage. Factor fill can still require more space than the original matrix; allocation and growth are admitted by the existing numerical owners.",
    },
    BuiltinDocumentationFaq {
        question: "Which broader forms remain unsupported?",
        answer: "LDL, banded, Hessenberg and permuted-triangular decomposition types are not implemented. General near-singular condition estimation and ExecutionV2 transport also remain outside the supported object surface. Unsupported forms return an error rather than a different factorization.",
    },
];

pub(super) const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "mldivide", target: BuiltinDocumentationLinkTarget::Builtin("mldivide") },
    BuiltinDocumentationLink { label: "mrdivide", target: BuiltinDocumentationLinkTarget::Builtin("mrdivide") },
    BuiltinDocumentationLink { label: "lu", target: BuiltinDocumentationLinkTarget::Builtin("lu") },
    BuiltinDocumentationLink { label: "qr", target: BuiltinDocumentationLinkTarget::Builtin("qr") },
    BuiltinDocumentationLink { label: "chol", target: BuiltinDocumentationLinkTarget::Builtin("chol") },
    BuiltinDocumentationLink { label: "svd", target: BuiltinDocumentationLinkTarget::Builtin("svd") },
    BuiltinDocumentationLink {
        label: "MATLAB decomposition reference",
        target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/decomposition.html"),
    },
    BuiltinDocumentationLink {
        label: "Runtime implementation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/decomposition.rs"),
    },
];

pub(super) const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Decomposition semantics, native factors, ownership and extension gates",
        location: "builtins::math::linalg::factor::decomposition::tests",
    }],
    notes: &[
        "Opaque object facts do not prove the class or shape of every solve output. Native factors, selected-context persistence, browser and hardware execution require their actual matching qualification receipts; documentation is not execution evidence.",
    ],
};
