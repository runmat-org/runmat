use crate::*;
pub(super) const DOCUMENTATION:BuiltinDocumentation=BuiltinDocumentation {
    authority:BuiltinDocumentationAuthority::Catalog,title:Some("optimget"),slug:Some("optimget"),
    summary:"Read an optimization option or return a default when it is unset.",
    description:"optimget(options,parameter) returns a stored option value. optimget(options,parameter,default) substitutes the supplied default for an absent or numeric-empty setting; without a supplied default, an unset option returns [].",
    keywords:&["optimget","optimset","options","default","optimization"],related:&["optimset","optimoptions","fminsearch","quadprog"],
    sections:&[
        BuiltinDocumentationSection{heading:"Names and defaults",paragraphs:&[
            "Parameter names ignore case and accept enough leading characters to identify one supported parameter uniquely. Exact names and modern aliases resolve before abbreviations. Shared canonical names such as OutputFcn and TolX count once even when both legacy and modern option identities recognize them. Ambiguous abbreviations, unknown parameter names, non-text names and non-scalar option containers raise an error.",
            "The known parameter namespace is the same typed namespace consumed by optimset: legacy fminsearch controls and supported modern controls. Examples include Display, FunValCheck, MaxFunEvals, MaxIter, OutputFcn, PlotFcns, TolFun, TolX, Algorithm, SpecifyObjectiveGradient and ConstraintTolerance/TolCon where available. Supported aliases identify their canonical option field; unique abbreviations apply to canonical parameter names, not arbitrary extension keys.",
            "For an unprofiled options struct, abbreviations use the legacy and shared option namespace; solver-specific global-optimization controls do not make ordinary legacy names ambiguous. A single recognized scalar Solver field additionally selects that solver's published option names, such as patternsearch MaxIterations and ga MaxGenerations. Exact known aliases remain available. A solver profile is never inferred from the other stored fields or values.",
            "Use a scalar options struct, or numeric [] for no stored settings. A missing field or numeric [] setting returns the supplied default unchanged, or [] without a default. Zero, false and empty text are stored values, not numeric-empty default markers. This is not an arbitrary struct-field accessor; custom extension fields retained by optimset do not create new recognized parameter names. Duplicate fields that identify the same canonical parameter reject rather than selecting an arbitrary value."]},
        BuiltinDocumentationSection{heading:"Value ownership and execution",paragraphs:&[
            "The chosen setting or default keeps its original value class, shape, resident-array carrier and callable/capture identity. optimget does not evaluate defaults, invoke callbacks, gather resident data, round native integers or validate a stored value against a particular solver. Solver admission still decides whether that option/value is supported.",
            "No device numerical kernel or elementwise fusion is used; this is a synchronous metadata query with canonical result-root ownership. Finite host-memory execution is currently unsupported and refused at ingress before text normalization or result-root metadata allocation. No finite-memory certificate is implied by returning an existing setting."]},
    ],
    examples:&[
        BuiltinExample{id:"stored-option",title:"Read a setting with a unique abbreviation",program:"options=optimset('Display','iter','TolX',1e-7);\nmode=optimget(options,'dIsP');\ntol=optimget(options,'TolX')",display_output:None,
            compatibility:BuiltinExampleCompatibility::RunMat,harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{source:"assert(strcmp(mode,'iter')); assert(tol==1e-7);"}},
        BuiltinExample{id:"unset-default",title:"Supply a default for an unset parameter",program:"options=optimset();\nmode=optimget(options,'Display','final');\nlimit=optimget(options,'MaxIterations',300)",display_output:None,
            compatibility:BuiltinExampleCompatibility::RunMat,harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{source:"assert(strcmp(mode,'final')); assert(limit==300); assert(isempty(optimget(options,'TolX')));"}},
    ],example_exemption:None,faqs:&[],
    links:&[BuiltinDocumentationLink{label:"MATLAB optimget",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/optimget.html")},BuiltinDocumentationLink{label:"optimset",target:BuiltinDocumentationLinkTarget::Builtin("optimset")}],media:&[],
    evidence:BuiltinDocumentationEvidence{implementation:&[BuiltinDocumentationLink{label:"Runtime query",target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/optimget.rs")}],verification:&[BuiltinEvidenceReference{kind:BuiltinEvidenceKind::UnitTest,label:"Parameter resolution and value preservation",location:"builtins::math::optim::optimget::tests"}],notes:&[]},
    introduced:None,status:Some(BuiltinDocumentationStatus::Partial),
};
