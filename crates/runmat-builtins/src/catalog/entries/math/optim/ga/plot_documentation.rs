use crate::*;
pub(super) const SECTIONS:&[BuiltinDocumentationSection]=&[
    BuiltinDocumentationSection{heading:"Callback and graphics lifecycle",paragraphs:&[
        "state=gaplotFcn(options,state,flag) updates the current assigned axes and returns state unchanged, including user fields and StopFlag. options is a scalar structure or []; flag is init,iter,interrupt,done. These predefined charts update at outer-generation init,iter,done events; interrupt returns the state without plotting unavailable inner-subproblem fields. Custom plot callbacks can consume interrupt events separately.",
        "Each PlotFcn in ga receives an exact assigned axes in a shared figure. History lives in ordinary line/bar scene payloads, not the solver or a global callback table. Updating the same generation replaces its old sample. Clearing or closing targets follows the graphics lifecycle; a retired assigned receiver is not silently replaced. RunMat does not provide MATLAB-specific Pause/Stop widgets; custom callbacks can set StopFlag, while session cancellation remains a separate operation.",
    ]},
    BuiltinDocumentationSection{heading:"Charts and state fields",paragraphs:&[
        "gaplotbestf plots the minimum and mean finite-real Score. gaplotrange plots minimum, maximum and mean. gaplotscores displays each Score, with undefined values as gaps. gaplotscorediversity counts finite-real scores in ceil(sqrt(valid count)) equal-width bins; constant samples use one bin. gaplotbestindiv displays the numeric Population row with the best finite-real Score. gaplotdistance plots the average Euclidean distance over distinct numeric population pairs. Opaque custom genomes need a custom coordinate plot for these two numeric-population charts.",
        "gaplotexpectation displays state.Expectation when available. gaplotselection counts one-based state.Selection entries, using Score's row count without reading its payload. gaplotgenealogy draws actual one-based GeneticParents edges, with NaN marking absent parents, and uses TotalInnerGenerations when positive to label completed variation steps. Those two RunMat state fields are supplied by ga; standalone callers must provide the parentage they wish to display. An observer-created row without parentage appears as a new node rather than an invented edge.",
        "gaplotmaxconstr requires auglag and displays max(0,NonlinIneq,abs(NonlinEq)) over the supplied constraint matrices. gaplotstopping displays FunctionTolerance progress over Best history, generation and stall-generation fractions, and the selected MaxTime fraction. Its state.StartTime is the session's existing tic token; finite MaxTime reads that selected Clock, not a second wall clock. The bars describe supplied state and are not independent feasibility or convergence certificates.",
    ]},
    BuiltinDocumentationSection{heading:"Data and services",paragraphs:&[
        "Only the selected chart's active fields and controls are admitted before transfers. Exact native numeric data and single promotion use the declared GA extensions; exact structural controls have their independent gate. Authenticated automatic device placement can materialize on the host. Explicit or unproven device inputs require caller gather. Unused callback captures and user fields are never recursively downloaded.",
        "Plots require plot-core and the selected runtime's UI/resource permissions. Finite host-memory policies refuse before history or scene mutation. Cancellation, provider, allocation and presentation failures remain errors and do not replay the plot. Missing Expectation displays an unavailable gap; a wholly undefined score histogram displays a labelled zero finite-score count. Without a finite winning score, best-individual coordinates are gaps rather than a retained old winner. All metric reductions use supplied measurements; they do not call the objective or mutate optimizer options.",
    ]},
];
pub(super) const fn example(program: &'static str) -> BuiltinExample {
    BuiltinExample {
        id: "update-genetic-chart",
        title: "Plot a completed genetic state",
        program,
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(returned,state));",
        },
    }
}
pub(super) const fn doc(
    name: &'static str,
    summary: &'static str,
    examples: &'static [BuiltinExample],
) -> BuiltinDocumentation {
    super::helper_documentation::doc(name, summary, SECTIONS, examples)
}
