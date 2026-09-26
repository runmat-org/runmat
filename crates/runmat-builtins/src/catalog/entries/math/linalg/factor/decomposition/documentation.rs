use crate::*;
pub(super) const SECTIONS: &[BuiltinDocumentationSection] = &[
 BuiltinDocumentationSection { heading: "Behavior",
 paragraphs: &["The MATLAB-compatible surface accepts full real or complex coefficient matrices of class `single` or `double`. With RunMat extensions enabled, all eight integer coefficient classes and logical coefficients are accepted and retained exactly in the object; strict compatibility mode rejects those forms before gathering a resident value. Sparse inputs remain rejected until RunMat has a sparse solve path behind the decomposition object.",
"Supported type strings are `'auto'`, `'lu'`, `'qr'`, `'chol'`, `'triangular'`, and `'diagonal'`. The default `'auto'` selects `diagonal`, `triangular`, `lu`, or `qr` based on the matrix shape and structure.",
"Square-only types reject rectangular matrices. The `diagonal` and `triangular` types validate the corresponding structure. The optional `'upper'` or `'lower'` triangular flag is supported for `decomposition(A, 'triangular', flag)` and ignores the opposite triangle.",
"Name-value options include `CheckCondition` and `RankTolerance`. MATLAB-compatible `CheckCondition` values are logical; accepting an integer scalar is a separately gated RunMat extension. `CheckCondition` is exposed as an object property, while `RankTolerance` is accepted and stored for QR compatibility metadata.",
"Typed-integer `RankTolerance` and real or paired-complex typed-integer scale factors are independently gated RunMat extensions. Each authoritative integer component must be exactly representable before crossing into binary64 scale or tolerance metadata.",
"Object properties include `MatrixSize`, `Type`, `CheckCondition`, `Datatype`, `IsConjugateTransposed`, `IsReal`, `IsSparse`, and `ScaleFactor`. Internal coefficient storage is not publicly readable.",
"Unary plus returns the same decomposition. Unary minus, scalar multiplication, and scalar division return a new decomposition with an adjusted scale factor. GPU scalar operands are gated before provider access because the public decomposition surface does not document GPU Arrays.",
"Conjugate transpose (`dA'`) toggles `IsConjugateTransposed`, conjugates the scale factor, and solves against the conjugate-transposed coefficient matrix. Gated integer and logical objects retain exact coefficient storage through this structural operation.",
"Left division (`dA \\ b`) and right division (`b / dA`) support the same real, complex, square, rectangular, and least-squares behavior as RunMat's `mldivide` and `mrdivide` builtins. Integer and logical coefficients are promoted explicitly to binary64 only at this solve boundary, so their solutions are double or complex double.",
"Element-wise left division (`ldivide` or `.\\`) is not a decomposition-object method, and `decomposition.ldivide` is absent from the builtin registry."] },

 BuiltinDocumentationSection { heading: "GPU behavior",
 paragraphs: &["The compatibility target does not document GPU Array support for `decomposition`. RunMat retains gpuArray arguments behind its explicit extension mode and gathers them to host storage because the object owns normalized matrix metadata.",
"Solves through `dA \\ b` and `b / dA` call the existing `mldivide` and `mrdivide` implementations with the effective coefficient matrix. If a provider accepts that solve, residency follows the same provider behavior as direct matrix division; otherwise RunMat uses the host fallback."] },

 BuiltinDocumentationSection { heading: "Execution and ownership",
 paragraphs: &["All identities return one value. Discarding the output does not skip validation; extra requested outputs fail before provider access. Construction defaults to auto type, CheckCondition=true and no RankTolerance. All arrays with more than two dimensions and sparse coefficients are rejected; empty matrices retain shape.",
 "Objects have value semantics and own host coefficient storage; structural operations return new metadata without mutating the input. There are no user callbacks or filesystem, network, clock, RNG, or workspace effects. Compatibility mode and optional provider services are ambient dependencies. GPU controls are gated and gathered through their owner; solves retain their shared solver provider route. Nested property indexing uses the canonical object indexing protocol.",
 "The portable typed runtime binding is the native/JIT/AOT semantic slow path. Provider gather and solve may suspend. Execution stack, rooting, output adoption and continuation are managed by shared runtime services. Direct distributed values are unsupported; ordinary workers may use transferable value objects and numeric operands, subject to canonical value-transfer and provider-affinity admission. Provider failures are not an invitation to replay side effects.",
 "Timing and placement use shared gather and solver operation boundaries; no decomposition-specific threshold, scheduler, timer or cost table is introduced. Coefficient shape, storage and precision, effective transpose/scale, solve direction and RHS shape are the semantic workload inputs. Native, browser, provider ownership and timing conformance need integrated execution evidence."] }
];
pub(super) const FAQS: &[BuiltinDocumentationFaq] = &[BuiltinDocumentationFaq { question: "Does `decomposition` expose the raw LU or QR factors?",
 answer: "No. MATLAB's `decomposition` object is an opaque solve object, and RunMat follows that surface. Use `lu`, `qr`, `chol`, or `svd` when you need explicit factors." },

BuiltinDocumentationFaq { question: "Does RunMat cache native factor arrays internally?",
 answer: "The current implementation stores the normalized coefficient matrix plus decomposition metadata and delegates each solve to the shared matrix-division solvers. This preserves MATLAB solve semantics across square, rectangular, and complex cases; dedicated cached factor storage can be added behind the same object surface later." },

BuiltinDocumentationFaq { question: "Which object operations are supported?",
 answer: "RunMat supports property access, unary plus/minus, conjugate transpose, scalar multiplication, scalar division, left division, and right division for decomposition objects." },

BuiltinDocumentationFaq { question: "How are integer and logical coefficient matrices handled?",
 answer: "They are RunMat-only extensions. With extensions enabled, construction and structural transpose preserve exact host storage for all eight integer classes and logical arrays. A solve promotes the effective coefficient matrix to double at the solve boundary and reports a double or complex-double solution. With extensions disabled, construction and later object methods reject the extended object with a stable compatibility identifier." },

BuiltinDocumentationFaq { question: "Are condition warnings emitted?",
 answer: "No. The `CheckCondition` property is parsed and preserved, but RunMat does not yet emit MATLAB-compatible condition warnings during decomposition construction or solve." },

BuiltinDocumentationFaq { question: "Are sparse decomposition objects supported?",
 answer: "Not yet. RunMat rejects sparse coefficient matrices instead of silently densifying them, because large sparse matrices need a dedicated sparse solve path to be safe and MATLAB-compatible." },

BuiltinDocumentationFaq { question: "Which broader decomposition features are still incomplete?",
 answer: "The current object does not implement sparse decomposition, cached native factor storage, condition warnings, algorithmic use of RankTolerance, or the additional banded, Hessenberg, LDL, COD, and permuted-triangular forms. Unsupported type strings are rejected explicitly." }];
pub(super) const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink { label: "mldivide",
 target: BuiltinDocumentationLinkTarget::Builtin("mldivide") },

BuiltinDocumentationLink { label: "mrdivide",
 target: BuiltinDocumentationLinkTarget::Builtin("mrdivide") },

BuiltinDocumentationLink { label: "lu",
 target: BuiltinDocumentationLinkTarget::Builtin("lu") },

BuiltinDocumentationLink { label: "qr",
 target: BuiltinDocumentationLinkTarget::Builtin("qr") },

BuiltinDocumentationLink { label: "chol",
 target: BuiltinDocumentationLinkTarget::Builtin("chol") },

BuiltinDocumentationLink { label: "svd",
 target: BuiltinDocumentationLinkTarget::Builtin("svd") },
 BuiltinDocumentationLink { label: "crates/runmat-runtime/src/builtins/math/linalg/factor/decomposition.rs",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/decomposition.rs") }];
pub(super) const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence { implementation: &[],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
 label: "Decomposition semantic, integer and gate tests",
 location: "builtins::math::linalg::factor::decomposition::tests" }],
 notes: &["Legacy title/category/keywords/summary/description/behaviors/examples/FAQ/links/source and GPU behavior preserved; examples now carry numerical oracles. Legacy null requires_feature is no requirement. GPU precisions and fusion flags are represented by typed optional-provider/boundary declarations and prose; no direct fusion is claimed. No legacy media or integration test fields existed. DynamicByDesign reflects opaque coefficient metadata and property/solve output-dependent facts, not a claim of platform qualification."] };
