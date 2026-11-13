use crate::*;
pub(super) const DOCUMENTATION:BuiltinDocumentation=BuiltinDocumentation {
    authority:BuiltinDocumentationAuthority::Catalog,title:Some("pca"),slug:Some("pca"),summary:"Principal components of real observation matrices.",
    description:"Compute principal directions, scores, component variances and Hotelling statistics through singular value decomposition.",
    keywords:&["pca","principal components","SVD","dimensionality reduction","variance"],related:&["svd","cov","mean"],
    sections:&[
        BuiltinDocumentationSection{heading:"Data and computation",paragraphs:&[
            "Rows of X are observations and columns are variables. Real single and double matrices are supported. The default centers each variable, removes incomplete rows, and uses the existing SVD factorization, not an independently approximated covariance matrix. Loadings are ordered by descending singular values; the largest-magnitude loading in each returned component has positive sign.",
            "All six output arrays retain X's floating class. The host factorization and statistical workspaces use binary64, with one final conversion for single outputs. Integer, logical, complex, sparse, distributed, and explicit gpuArray observation arrays are unsupported; these are not silently converted or advertised as additional overloads.",
            "Rows='complete' omits rows containing NaN, then reinserts NaNs at those observation positions in score and tsquared. Centering means use all available non-NaN observations in each original column, even in rows excluded because another column is missing. Rows='all' rejects NaN observations. Retained infinities, or infinities contributing to centering means, are invalid. Positive degrees of freedom are required after omission: at least two retained rows when centered, or one when uncentered.",
        ]},
        BuiltinDocumentationSection{heading:"Options and six outputs",paragraphs:&[
            "Supported case-insensitive name/value options are Algorithm='svd', Centered, Economy, NumComponents, and Rows='complete' or 'all'. Centered and Economy accept logical scalars or on/off text. NumComponents is a positive single/double scalar integer no greater than the number of variables. Later repeated options replace earlier values. Unsupported options and values raise errors; weights, VariableWeights, eig/als algorithms, pairwise covariance and iterative missing-data estimation are not accepted and ignored.",
            "[coeff,score,latent,tsquared,explained,mu] returns principal directions, observation scores, component variances, full-space Hotelling T-squared, percentage variance explained, and variable means. Centered='off' returns a zero mean row and uses the retained row count as degrees of freedom; centered analysis uses one fewer.",
            "Economy defaults to true and limits variances and components to min(degrees of freedom,variables). Economy=false retains all variance entries and orthonormal loading directions, with zero variance/score padding beyond the degrees-of-freedom bound. NumComponents limits coeff/score, but does not truncate latent, explained, or T-squared over all degrees-of-freedom-admissible PCs. With Economy=false, a requested component count at least as large as the degrees of freedom returns the full variable-space basis. With incomplete rows, all-available column means can make the retained matrix's raw rank exceed that degrees-of-freedom bound; the bound is still applied and padded directions are not claimed to be its covariance nullspace.",
            "T-squared excludes numerical null directions using the shared rank/pinv SVD tolerance; no covariance eigenvalues are clamped to positive values. Explained percentages use normalized singular values, independent of overflow/underflow in displayed variances. An all-zero variance spectrum has NaN explained percentages and zero full-space T-squared; degenerate eigenspaces may have different, equally valid orthonormal bases.",
        ]},
        BuiltinDocumentationSection{heading:"Residency, cancellation and resources",paragraphs:&[
            "Only authenticated automatic residency can use the context-owned host gather fallback, including numeric controls. Explicit or unproven device intent is rejected before any argument transfer; use gather explicitly for host analysis. There is no provider-native PCA kernel or fusion producer.",
            "Finite host-memory requests are rejected before gather or SVD workspace allocation because the factorization and output workspaces have no complete allocation certificate. Input/output scans cooperate with the original execution context. The existing dense SVD kernel is synchronous between cancellation checkpoints and is not claimed to be hard-preemptible. Original gather, factorization and typed cancellation errors are not replayed or relabeled. Unrequested public output arrays are not constructed.",
        ]},
    ],
    examples:&[
        BuiltinExample{id:"orthogonal-variables",title:"Analyze two orthogonal variables",program:"X = [-1 -1;-1 1;1 -1;1 1]; [C,S,L,T,E,M] = pca(X);",display_output:None,
            compatibility:BuiltinExampleCompatibility::Matlab,harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{source:"assert(norm(C'*C-eye(2))<1e-10); assert(norm(S*C'+M-X)<1e-10); assert(norm(L-[4/3;4/3])<1e-10); assert(norm(E-[50;50])<1e-10); assert(norm(T-1.5*ones(4,1))<1e-10);"}},
        BuiltinExample{id:"reduced-scores",title:"Retain one score direction",program:"X = [-2 -1;0 2;2 -1]; [C,S,L,T,E,M] = pca(X,'NumComponents',1);",display_output:None,
            compatibility:BuiltinExampleCompatibility::Matlab,harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{source:"assert(isequal(size(C),[2 1])); assert(isequal(size(S),[3 1])); assert(isequal(size(L),[2 1])); assert(abs(sum(E)-100)<1e-10); assert(norm(T-(4/3)*ones(3,1))<1e-10); assert(norm(M)<1e-10);"}},
    ],example_exemption:None,faqs:&[],links:&[
        BuiltinDocumentationLink{label:"MATLAB pca",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/stats/pca.html")},
        BuiltinDocumentationLink{label:"Implementation",target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/pca.rs")},
    ],media:&[],evidence:BuiltinDocumentationEvidence{implementation:&[],verification:&[BuiltinEvidenceReference{kind:BuiltinEvidenceKind::UnitTest,label:"Analytic PCA reconstruction, full-space statistics and boundaries",location:"builtins::stats::summary::pca::tests"}],notes:&[]},
    introduced:None,status:Some(BuiltinDocumentationStatus::Partial),
};
