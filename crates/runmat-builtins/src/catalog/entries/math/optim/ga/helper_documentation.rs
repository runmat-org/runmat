use crate::*;
pub(super) const SELECTION_SECTIONS:&[BuiltinDocumentationSection]=&[
    BuiltinDocumentationSection {heading:"Parent selection",paragraphs:&[
        "parents=selectionFcn(expectation,nParents,options) returns a one-based double row of selected population indices. expectation is a full real finite nonnegative vector; positive nParents requires some positive weight. nParents is a nonnegative structural scalar. options is a scalar structure or [] and supplies no active numeric fields to these standalone predefined functions. A zero parent count returns an empty row without random draws.",
        "selectionstochunif uses equally spaced CDF positions with one uniform origin. selectionroulette draws independently from the weighted CDF. selectionremainder deterministically assigns floor(expectation) copies, then samples the remaining fractional weights; its expectations must sum to nParents. selectionuniform samples member indices uniformly. selectiontournament draws size players per parent and chooses the largest scaled expectation; size defaults to four and must be >=2. Ties keep the first drawn player. Its optional fourth argument supplies size.",
        "These are callable predefined functions, including through function handles and operator cells. They reuse ga's numerical selection kernels and the selected session RNG, not another random stream or fitness evaluation. Only needed expectation/count/size carriers are admitted. Native numeric/single extensions and original automatic-residency materialization follow the declared ga numeric/control policy; explicit or unproven residency requires caller gather. Resource or cancellation failures propagate without replay. Finite host-memory policies refuse before random work.",
    ]},
];
pub(super) const SCALING_SECTIONS:&[BuiltinDocumentationSection]=&[
    BuiltinDocumentationSection {heading:"Offspring expectations",paragraphs:&[
        "expectation=fitscalingFcn(scores,nParents) returns a host double column with one nonnegative expectation per full real score. The sum is nParents. nParents is a nonnegative exact structural scalar; positive nParents needs a nonempty score vector. These numerical helpers neither call the objective nor consume RNG. The same kernels are used by ga's named scaling options.",
        "fitscalingrank weights sorted finite scores by 1/sqrt(rank); undefined real scores rank last with stable member tie order. fitscalingprop is proportional to finite nonnegative raw scores, using uniform weights when all scores are zero. fitscalingtop gives equal weight to the selected best entries; quantity is a fraction [0,1] or an integer count, default0.4, supplied as the third argument. Positive nParents requires at least one selected member. fitscalingshiftlinear uses nonnegative affine weights with requested best/mean ratio rate>=1, default2; ties cap the attainable ratio and equal scores receive equal expectations.",
        "Only active score/count/parameter values are read; supplied options or arbitrary extra arguments are not accepted. Exact native numeric/single promotion and automatic-only residency materialization use the declared ga extensions and original context. Explicit/unproven device arrays require caller gather. Finite host-memory policies refuse before workspace construction; original cancellation/allocation/provider errors remain errors rather than substituted rank values.",
    ]},
];
pub(super) const fn doc(
    name: &'static str,
    summary: &'static str,
    sections: &'static [BuiltinDocumentationSection],
    examples: &'static [BuiltinExample],
) -> BuiltinDocumentation {
    BuiltinDocumentation {authority:BuiltinDocumentationAuthority::Catalog,title:Some(name),slug:Some(name),summary,description:summary,
        keywords:&["genetic algorithm","selection","fitness scaling"],related:&["ga","optimoptions"],sections,
        examples,example_exemption:None,faqs:&[],media:&[],
        links:&[BuiltinDocumentationLink{label:"MathWorks genetic operator contracts",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/gads/genetic-algorithm-options.html")}],
        evidence:BuiltinDocumentationEvidence{
            implementation:&[BuiltinDocumentationLink{label:"Genetic solver and predefined helper modules",target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/ga.rs")}],
            verification:&[BuiltinEvidenceReference{kind:BuiltinEvidenceKind::UnitTest,label:"Selection expectations and numerical operator controls",location:"builtins::math::optim::ga::operators"}],notes:&[]},
        introduced:None,status:Some(BuiltinDocumentationStatus::Stable)}
}
pub(super) const fn example(program: &'static str, assertions: &'static str) -> BuiltinExample {
    BuiltinExample {
        id: "operator-result",
        title: "Use the predefined genetic operator",
        program,
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: assertions },
    }
}
