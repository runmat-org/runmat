use crate::*;
const SECTIONS:&[BuiltinDocumentationSection]=&[
    BuiltinDocumentationSection{heading:"Callback contract",paragraphs:&[
        "These are real predefined optimization plot callbacks with signature stop=helper(x,optimValues,state). optimplotx shows the current coordinates as bars; optimplotfunccount and optimplotfval show scalar histories against iteration. A successful call returns logical false. Custom PlotFcns may return true to stop a solver; these predefined visualizers do not manufacture a numerical stopping condition.",
        "state is init, iter or done. init resets only this helper's chart on the current axes, iter updates its payload, and done retains the final chart. Repeated calls for the same last iteration replace that scalar point rather than duplicating it. The helper kinds have separate receiver-backed slots per axes. Solver PlotFcns layouts assign one axes per callback; standalone calls use the current axes.",
        "optimValues must be a scalar struct with a nonnegative, platform-representable scalar floating iteration. optimplotfunccount additionally reads nonnegative integer funccount; optimplotfval reads scalar floating fval, or [] before an objective value is available. NaN/Inf fval values are retained as plotting data rather than clamped or converted into solver success. Additional fields are not inspected or downloaded. Only optimplotx reads x: it plots finite real single/double coordinates in column-major variable order. An empty x means no point is available yet. The history helpers do not convert or copy x.",
    ]},
    BuiltinDocumentationSection{heading:"Constrained progress",paragraphs:&[
        "optimplotconstrviolation, optimplotstepsize and optimplotfirstorderopt read their corresponding scalar optimValues fields. They do not read x or unrelated fields.",
        "optimplotfvalconstr reads fval and constrviolation. Its default feasibility tolerance is 1e-6. An explicit finite nonnegative constraintTolerance field selects a different nonlinear tolerance and permits signed raw constraint maxima. The chart retains separately colored feasible/infeasible histories and the best finite feasible value.",
        "When fval is [], the constrained chart uses constrviolation as its measured value. An optional nonnegative integer feasibleCount then supplies the feasible-solutions title. Empty metrics initialize axes without adding a zero sample; later missing metrics retain preceding chart data. Empty initialization retires only this helper's payloads, preserving unrelated charts on the same axes. All histories remain ordinary receiver-backed scene payloads.",
    ]},
    BuiltinDocumentationSection{heading:"Graphics and execution",paragraphs:&[
        "Charts are ordinary line/bar scene payloads with exact figure, axes and child receivers. Histories are not retained in a solver store or global function-name map. Axes clearing drops their slots and payloads; closing a figure retires its receivers. A solver's retained layout never silently recreates a retired target.",
        "Needed floating operands may be transparently gathered only from authenticated automatic residency through the original execution context. Explicit/unproven device intent refuses before any needed transfer; call gather explicitly for host data. Unused fields and captures have no readback side effect. Native integer/logical/complex plot data is not implicitly accepted as floating callback data.",
        "Finite host-memory execution refuses before transfers or scene work because progress history, labels and graphics scenes have no complete allocation certificate. CPU scans cooperate with the original invocation; typed cancellation and graphics/provider failures propagate without retry or fake stop values. plot-core and UI capability are required.",
        "RunMat uses its normal plot renderer and cancellation controls. It does not provide MATLAB-specific optimization Pause/Stop widgets or claim identical plot-window layout. Original execution cancellation remains cancellation, distinct from a custom plot callback's logical true stop result.",
    ]},
];
const LINKS:&[BuiltinDocumentationLink]=&[
    BuiltinDocumentationLink{label:"MATLAB optimization plot functions",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/math/plot-functions.html")},
    BuiltinDocumentationLink{label:"Runtime implementation",target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/optimplot.rs")},
];
const fn doc(
    name: &'static str,
    summary: &'static str,
    examples: &'static [BuiltinExample],
) -> BuiltinDocumentation {
    BuiltinDocumentation {
        authority: BuiltinDocumentationAuthority::Catalog,
        title: Some(name),
        slug: Some(name),
        summary,
        description: summary,
        keywords: &["optimization", "plot callback", "PlotFcns", "progress"],
        related: &["fminsearch", "optimset"],
        sections: SECTIONS,
        examples,
        example_exemption: None,
        faqs: &[],
        links: LINKS,
        media: &[],
        evidence: BuiltinDocumentationEvidence {
            implementation: &[],
            verification: &[BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Progress payloads, exact lifecycle and callback boundaries",
                location: "builtins::plotting::optimplot::tests",
            }],
            notes: &[],
        },
        introduced: None,
        status: Some(BuiltinDocumentationStatus::Partial),
    }
}
pub(super) const X:BuiltinDocumentation=doc("optimplotx","Plot the current optimization point as variable bars.",&[
    BuiltinExample{id:"init-iter-done",title:"Update the current-point chart",program:"v.iteration=0; v.funccount=1; v.fval=4; stop=optimplotx([2;-1],v,'init'); v.iteration=1; optimplotx([1;0],v,'iter'); optimplotx([1;0],v,'done');",display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,harness:BuiltinExampleHarness::BrowserGraphics,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,verification:BuiltinExampleVerification::Assertions{source:"assert(islogical(stop) && ~stop);"}},
]);
pub(super) const COUNT:BuiltinDocumentation=doc("optimplotfunccount","Plot cumulative objective evaluations against iteration.",&[
    BuiltinExample{id:"init-iter-done",title:"Update evaluation-count history",program:"v.iteration=0; v.funccount=1; stop=optimplotfunccount([],v,'init'); v.iteration=1; v.funccount=4; optimplotfunccount([],v,'iter'); optimplotfunccount([],v,'done');",display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,harness:BuiltinExampleHarness::BrowserGraphics,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,verification:BuiltinExampleVerification::Assertions{source:"assert(islogical(stop) && ~stop);"}},
]);
pub(super) const VALUE:BuiltinDocumentation=doc("optimplotfval","Plot the objective function value against iteration.",&[
    BuiltinExample{id:"init-iter-done",title:"Update objective-value history",program:"v.iteration=0; v.fval=4; stop=optimplotfval([],v,'init'); v.iteration=1; v.fval=1; optimplotfval([],v,'iter'); optimplotfval([],v,'done');",display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,harness:BuiltinExampleHarness::BrowserGraphics,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,verification:BuiltinExampleVerification::Assertions{source:"assert(islogical(stop) && ~stop);"}},
]);

pub(super) const CONSTRAINT:BuiltinDocumentation=doc(
    "optimplotconstrviolation","Plot constrained optimization progress.",&[
        BuiltinExample{
            id:"init-iter-done",title:"Update a constrained progress chart",
            program:"v.iteration=0; v.constrviolation=2; v.constrviolation=0; stop=optimplotconstrviolation([],v,'init'); v.iteration=1; v.constrviolation=1; optimplotconstrviolation([],v,'iter'); optimplotconstrviolation([],v,'done');",
            display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,
            harness:BuiltinExampleHarness::BrowserGraphics,fixture:BuiltinExampleFixture::None,
            requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{source:"assert(islogical(stop) && ~stop);"},
        },
    ]);

pub(super) const STEP:BuiltinDocumentation=doc(
    "optimplotstepsize","Plot constrained optimization progress.",&[
        BuiltinExample{
            id:"init-iter-done",title:"Update a constrained progress chart",
            program:"v.iteration=0; v.stepsize=2; v.constrviolation=0; stop=optimplotstepsize([],v,'init'); v.iteration=1; v.stepsize=1; optimplotstepsize([],v,'iter'); optimplotstepsize([],v,'done');",
            display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,
            harness:BuiltinExampleHarness::BrowserGraphics,fixture:BuiltinExampleFixture::None,
            requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{source:"assert(islogical(stop) && ~stop);"},
        },
    ]);

pub(super) const FIRSTORDER:BuiltinDocumentation=doc(
    "optimplotfirstorderopt","Plot constrained optimization progress.",&[
        BuiltinExample{
            id:"init-iter-done",title:"Update a constrained progress chart",
            program:"v.iteration=0; v.firstorderopt=2; v.constrviolation=0; stop=optimplotfirstorderopt([],v,'init'); v.iteration=1; v.firstorderopt=1; optimplotfirstorderopt([],v,'iter'); optimplotfirstorderopt([],v,'done');",
            display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,
            harness:BuiltinExampleHarness::BrowserGraphics,fixture:BuiltinExampleFixture::None,
            requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{source:"assert(islogical(stop) && ~stop);"},
        },
    ]);

pub(super) const FEASIBLE:BuiltinDocumentation=doc(
    "optimplotfvalconstr","Plot constrained optimization progress.",&[
        BuiltinExample{
            id:"init-iter-done",title:"Update a constrained progress chart",
            program:"v.iteration=0; v.fval=2; v.constrviolation=0; stop=optimplotfvalconstr([],v,'init'); v.iteration=1; v.fval=1; optimplotfvalconstr([],v,'iter'); optimplotfvalconstr([],v,'done');",
            display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,
            harness:BuiltinExampleHarness::BrowserGraphics,fixture:BuiltinExampleFixture::None,
            requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{source:"assert(islogical(stop) && ~stop);"},
        },
    ]);
