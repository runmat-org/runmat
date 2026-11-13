use crate::*;
const SAVE_SECTIONS:&[BuiltinDocumentationSection]=&[
    BuiltinDocumentationSection{heading:"Saved event solutions",paragraphs:&[
        "Use savemilpsolutions as intlinprog's OutputFcn, by function handle or its predefined name. The callback signature is stop=savemilpsolutions(x,optimValues,state). A successful call returns logical false. Standalone calls record the supplied events; they do not test feasibility against an optimization problem.",
        "init resets xIntSol and fIntSol in the selected base workspace. Each nonempty iter point is appended as a column of xIntSol; its optimValues.fval is appended to the corresponding position in the row vector fIntSol. Empty iter points and done do not append or reread an old solution. Standalone calls use the same base variables. An inconsistent existing pair or changed decision dimension is an error; call init to start another history.",
        "intlinprog supplies nonempty x only for newly publishable incumbent improvements. numfeaspoints in its observer payload counts improving solutions, while the solver's output/MaxFeasiblePoints retain the separate total accepted-candidate count. Observer fval corresponds to the actual published binary64 point. Exact rational witnesses and gap/pruning bounds remain separate solver authority.",
        "The execution session must provide a base workspace. Assignments use that workspace, including variables bound as globals. Both replacement histories are prepared before assignment. If a workspace assignment fails, completed assignments remain and the error propagates; the helper does not repeat the writes.",
    ]},
    BuiltinDocumentationSection{heading:"Data and resources",paragraphs:&[
        "Point/objective data must be real single/double values. Event points are finite decision columns; objective values are recorded as supplied. Only x and fval are read for nonempty iter events. Automatically placed GPU data can be gathered by the session. Explicit GPU data, or data whose placement origin is unknown, must be gathered before calling the helper.",
        "The helper cannot run under a finite host-memory limit because its solution history can grow. This restriction is checked before workspace changes. Cancellation and errors propagate without repeating completed updates.",
    ]},
];
const PLOT_SECTIONS:&[BuiltinDocumentationSection]=&[
    BuiltinDocumentationSection{heading:"Primal and dual histories",paragraphs:&[
        "optimplotmilp is an intlinprog PlotFcn with signature stop=optimplotmilp(x,optimValues,state). It plots optimValues.fval as Primal Bound and optimValues.dualbound as Dual Bound against optimValues.time in seconds. It does not read or download x, relativegap, counters or unused fields. time must be finite and nonnegative; nonfinite bounds appear as gaps in the lines.",
        "init resets the two line histories, iter updates them, and done leaves the final chart. A repeated last time replaces its values instead of duplicating a point. The lines retain their distinct labels and colors. Clearing the axes removes these histories; closing the figure retires its target.",
        "intlinprog assigns one axes target per PlotFcn, including custom callbacks in a cell array. Each callback uses its assigned target even if another callback changes the current axes. Standalone calls use the current axes. Closing a solver's target causes an error rather than silently creating a replacement.",
        "Successful predefined helpers return logical false. Custom output/plot callbacks may return logical true to stop the solver. RunMat retains ordinary plot controls but does not provide MATLAB-specific optimization Pause/Stop widgets or claim identical plot-window layout; original execution cancellation remains a typed cancellation error.",
    ]},
    BuiltinDocumentationSection{heading:"Execution boundary",paragraphs:&[
        "Bound/time fields must be real single/double scalars. Automatically placed GPU data can be gathered by the session; explicit GPU data, or data whose placement origin is unknown, must be gathered first. Graphics support is required. The helper cannot run under a finite host-memory limit because its plot history can grow. Cancellation and presentation errors propagate without repeating completed updates.",
    ]},
];
const LINKS:&[BuiltinDocumentationLink]=&[
    BuiltinDocumentationLink{label:"MILP output and plot callback contract",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/optim/ug/intlinprog-output-functions-and-plot-functions.html")},
    BuiltinDocumentationLink{label:"Runtime helpers",target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/milp_observers.rs")},
];
const fn doc(
    name: &'static str,
    summary: &'static str,
    sections: &'static [BuiltinDocumentationSection],
    examples: &'static [BuiltinExample],
) -> BuiltinDocumentation {
    BuiltinDocumentation {
        authority: BuiltinDocumentationAuthority::Catalog,
        title: Some(name),
        slug: Some(name),
        summary,
        description: summary,
        keywords: &["MILP", "intlinprog", "observer", "OutputFcn", "PlotFcn"],
        related: &["intlinprog", "optimoptions"],
        sections,
        examples,
        example_exemption: None,
        faqs: &[],
        links: LINKS,
        media: &[],
        evidence: BuiltinDocumentationEvidence {
            implementation: &[],
            verification: &[BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label:
                    "MILP event data, selected workspace, scene lifecycle and stop/error controls",
                location: "builtins::math::optim::milp_observers::tests",
            }],
            notes: &[],
        },
        introduced: None,
        status: Some(BuiltinDocumentationStatus::Partial),
    }
}
pub(super) const SAVE:BuiltinDocumentation=doc("savemilpsolutions","Save newly found MILP event solutions in the base workspace.",SAVE_SECTIONS,&[
    BuiltinExample{id:"event-history",title:"Record two explicit event points",program:"v.fval=4; stop=savemilpsolutions([],v,'init'); savemilpsolutions([1;3],v,'iter'); v.fval=3; savemilpsolutions([1;2],v,'iter'); savemilpsolutions([],v,'done');",display_output:None,compatibility:BuiltinExampleCompatibility::RunMat,harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,verification:BuiltinExampleVerification::Assertions{source:"assert(islogical(stop) && ~stop); assert(isequal(xIntSol,[1 1;3 2])); assert(isequal(fIntSol,[4 3]));"}},
    BuiltinExample{id:"solver-output-function",title:"Collect actual binary solver improvements",program:"options=optimoptions('intlinprog','Display','off','OutputFcn','savemilpsolutions'); [x,fval,flag]=intlinprog([-1;-1],1:2,[2 2],3,[],[],[0;0],[1;1],[],options);",display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,verification:BuiltinExampleVerification::Assertions{source:"assert(flag==1); assert(size(xIntSol,1)==2); assert(size(xIntSol,2)>=1); assert(isequal(xIntSol(:,end),x)); assert(fIntSol(end)==fval);"}},
]);
pub(super) const PLOT:BuiltinDocumentation=doc("optimplotmilp","Plot actual primal and dual objective bounds at MILP events.",PLOT_SECTIONS,&[
    BuiltinExample{id:"bound-history",title:"Update the two objective-bound lines",program:"v.time=0; v.fval=10; v.dualbound=0; stop=optimplotmilp([],v,'init'); v.time=1; v.fval=6; v.dualbound=4; optimplotmilp([],v,'iter'); optimplotmilp([],v,'done');",display_output:None,compatibility:BuiltinExampleCompatibility::RunMat,harness:BuiltinExampleHarness::BrowserGraphics,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,verification:BuiltinExampleVerification::Assertions{source:"assert(islogical(stop) && ~stop);"}},
    BuiltinExample{id:"solver-plot-function",title:"Plot bounds from a real binary solve",program:"options=optimoptions('intlinprog','Display','off','PlotFcn','optimplotmilp'); [x,fval,flag]=intlinprog([-1;-1],1:2,[2 2],3,[],[],[0;0],[1;1],[],options);",display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,harness:BuiltinExampleHarness::BrowserGraphics,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,verification:BuiltinExampleVerification::Assertions{source:"assert(flag==1); assert(all(x==round(x))); assert(fval==-1);"}},
]);
