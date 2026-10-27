use crate::*;
macro_rules! doc {
    ($constant:ident,$name:literal,$summary:literal,$detail:literal,$program:literal,$assertion:literal) => {
        pub(super) const $constant: BuiltinDocumentation = BuiltinDocumentation {
            authority:BuiltinDocumentationAuthority::Catalog,title:Some($name),slug:Some($name),
            summary:$summary,description:$summary,keywords:&["patternsearch","optimization","progress","plot"],
            related:&["patternsearch","optimoptions"],
            sections:&[BuiltinDocumentationSection{heading:"Usage",paragraphs:&[
                $detail,
                "Use stop = helper(optimvalues,flag), where flag is 'init', 'iter', or 'done'. The helper updates its selected axes and returns logical false. Select it with patternsearch PlotFcn; multiple helpers share the solver's subplot layout. Standalone calls use the current graphics receiver. The function never evaluates the objective or constraints.",
                "Metadata must be a scalar structure with a finite nonnegative integral double iteration. Required metrics are full real host double data; resident arrays and opaque fields are not converted. Selected plotting requires plot-core and the UI service. Finite host-memory execution is unsupported before scene work.",
            ]}],
            examples:&[BuiltinExample{id:$name,title:$summary,program:$program,display_output:None,
                compatibility:BuiltinExampleCompatibility::Matlab,harness:BuiltinExampleHarness::BrowserGraphics,
                fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,
                verification:BuiltinExampleVerification::Assertions{source:$assertion}}],
            example_exemption:None,faqs:&[],
            links:&[BuiltinDocumentationLink{label:"MathWorks Pattern Search Plot Options",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/gads/pattern-search-options.html")}],
            media:&[],evidence:BuiltinDocumentationEvidence{
                implementation:&[BuiltinDocumentationLink{label:"Pattern progress chart helpers",target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/patternsearch/helpers.rs")}],
                verification:&[BuiltinEvidenceReference{kind:BuiltinEvidenceKind::UnitTest,label:"Concrete chart metrics and selected receiver updates",location:"builtins::math::optim::patternsearch::tests"}],notes:&[],
            },introduced:None,status:Some(BuiltinDocumentationStatus::Experimental),
        };
    };
}
doc!(BESTF,"psplotbestf","Plot the current Pattern search objective.",
    "optimvalues.fval is a finite real scalar. Each iteration adds the measured current objective to a line chart; no global optimum or monotone original objective is assumed for nonlinear augmented-Lagrangian steps.",
    "v=struct('iteration',0,'fval',4); stop=psplotbestf(v,'init');\nv.iteration=1; v.fval=1; psplotbestf(v,'iter'); psplotbestf(v,'done');",
    "assert(islogical(stop) && ~stop);");
doc!(COUNT,"psplotfuncount","Plot actual Pattern objective evaluation counts.",
    "optimvalues.funccount is a finite nonnegative integral scalar. Cache hits and observer calls do not add objective evaluations. The chart records this supplied count at each iteration.",
    "v=struct('iteration',0,'funccount',1); stop=psplotfuncount(v,'init');\nv.iteration=1; v.funccount=5; psplotfuncount(v,'iter'); psplotfuncount(v,'done');",
    "assert(islogical(stop) && ~stop);");
doc!(MESH,"psplotmeshsize","Plot the current Pattern mesh size.",
    "optimvalues.meshsize is a finite nonnegative scalar. The line chart displays the represented mesh scale after search/poll transitions.",
    "v=struct('iteration',0,'meshsize',1); stop=psplotmeshsize(v,'init');\nv.iteration=1; v.meshsize=.5; psplotmeshsize(v,'iter'); psplotmeshsize(v,'done');",
    "assert(islogical(stop) && ~stop);");
doc!(POINT,"psplotbestx","Plot the coordinates of the current Pattern point.",
    "optimvalues.x contains finite full real double coordinates. A bar chart shows the current coordinate values; it is updated in place on the selected axes rather than plotted as a scalar objective history.",
    "v=struct('iteration',0,'x',[2;-1]); stop=psplotbestx(v,'init');\nv.iteration=1; v.x=[1;0]; psplotbestx(v,'iter'); psplotbestx(v,'done');",
    "assert(islogical(stop) && ~stop);");
doc!(CONSTRAINT,"psplotmaxconstr","Plot maximum nonlinear constraint violation.",
    "optimvalues.nonlinineq and nonlineq are optional finite real double arrays. The metric is max(0,max(nonlinineq),max(abs(nonlineq))); absent/empty nonlinear constraints give zero. No user constraint callback is invoked.",
    "v=struct('iteration',0,'nonlinineq',[-1;.25],'nonlineq',-.5); stop=psplotmaxconstr(v,'init');\nv.iteration=1; v.nonlinineq=[-1;0]; v.nonlineq=0; psplotmaxconstr(v,'iter'); psplotmaxconstr(v,'done');",
    "assert(islogical(stop) && ~stop);");
