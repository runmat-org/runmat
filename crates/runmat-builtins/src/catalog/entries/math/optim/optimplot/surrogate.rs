//! Phase-aware measured-value chart; its renderer is the shared scene owner.
use super::*;
const fn param(name: &'static str, description: &'static str) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        description,
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
    }
}
pub const SURROGATEOPTPLOT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.SURROGATEOPTPLOT.INVALID_ARGUMENT",
        identifier: Some("RunMat:surrogateoptplot:InvalidArgument"),
        when: "Callback metadata, scalar classes, state, counts or selected residency are invalid.",
        message: "surrogateoptplot: invalid callback argument",
    };
pub const SURROGATEOPTPLOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "stop = surrogateoptplot(x,optimValues,state)",
        inputs: &[param("x", "Ignored; plotted values come from measured optimValues metadata."),
            param("optimValues", "Best/incumbent/current values, phase flag, actual function count and reset metadata."),
            param("state", "init, iter or done.")],
        outputs: &[BuiltinParamDescriptor { name: "stop", description: "Logical false after successful scene publication.",
            ty: BuiltinParamType::Any, arity: BuiltinParamArity::Optional, default: None }],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[SURROGATEOPTPLOT_ERROR_INVALID_ARGUMENT],
};
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("surrogateoptplot"), slug: Some("surrogateoptplot"),
    summary: "Plot surrogate optimization samples, best/incumbent values and resets.",
    description: "Display the measured values and algorithm phases of surrogate optimization.",
    keywords: &["surrogate optimization", "PlotFcn", "progress", "nonlinear feasibility"],
    related: &["surrogateopt", "optimplotfvalconstr"],
    sections: &[
        BuiltinDocumentationSection { heading: "Callback", paragraphs: &[
            "stop = surrogateoptplot(x,optimValues,state) returns logical false. state is init, iter or done. x is not read. Function counts, not model-probe counts, form the horizontal axis. Repeated updates at the same count replace their last sample.",
            "optimValues is a scalar struct containing funccount, fval, constrviolation, incumbentFval, incumbentConstrviolation, currentFval, currentConstrviolation, currentFlag, surrogateReset and surrogateResetCount. Value fields are real floating scalars or []. Counts are nonnegative platform-representable integers. surrogateReset is a logical scalar; a true reset has positive reset count.",
            "currentFlag is initial, random or adaptive when a current measurement is available. Initial samples use diamonds, random samples triangles, adaptive samples filled circles, best values circles and incumbents crosses. Feasible and infeasible samples have separate colors. Surrogate resets are actual vertical reference-line annotations, independent of objective-value scale.",
        ] },
        BuiltinDocumentationSection { heading: "Missing values and feasibility", paragraphs: &[
            "Empty initialization creates axes without a synthetic objective sample. NaN measurements do not produce a finite sample. For an absent objective, the corresponding raw maximum nonlinear constraint value is plotted instead. Signed constraint maxima are not clamped to zero.",
            "The nonlinear feasibility tolerance is 1e-3 unless optimValues.constraintTolerance supplies a finite nonnegative selected tolerance. The title reports the available best, incumbent and current measured values. Init retires only this callback's series and reset annotations; unrelated charts on the same axes are preserved.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and graphics", paragraphs: &[
            "The callback uses ordinary receiver-backed scene payloads and the standard renderer. It does not retain a separate solver history. Figure/axes/child retirement is checked against the original selected graphics authority. Closing the retained target does not create a replacement figure.",
            "Only consumed floating metadata may be gathered from authenticated automatic residency through the original context. Explicit or unproven device metadata requires an explicit gather. Unused x, Ineq arrays, captures and unrelated fields are not inspected or downloaded. Finite host-memory execution refuses before transfer or scene work because chart history and scene allocation have no complete bound.",
            "plot-core and UI capability are required. Cancellation, provider and graphics failures propagate without replay. RunMat provides its normal rendering and execution cancellation controls, not MATLAB-specific optimization Pause/Stop widgets or an identical plot-window layout.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "measured-phases", title: "Plot measured random and adaptive samples",
        program: "v.funccount=0; v.fval=[]; v.constrviolation=[]; v.incumbentFval=[]; v.incumbentConstrviolation=[]; v.currentFval=[]; v.currentConstrviolation=[]; v.currentFlag=[]; v.surrogateReset=false; v.surrogateResetCount=0; stop=surrogateoptplot([],v,'init');\nv.funccount=1; v.fval=2; v.constrviolation=-0.1; v.incumbentFval=2; v.incumbentConstrviolation=-0.1; v.currentFval=2; v.currentConstrviolation=-0.1; v.currentFlag='random'; surrogateoptplot([],v,'iter');\nv.funccount=2; v.currentFval=3; v.currentConstrviolation=0.2; v.currentFlag='adaptive'; v.surrogateReset=true; v.surrogateResetCount=1; surrogateoptplot([],v,'iter'); v.surrogateReset=false; stop=surrogateoptplot([],v,'done');",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(stop) && ~stop);" },
    }],
    example_exemption: None, faqs: &[], media: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB surrogateoptplot", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/gads/interpret-surrogateoptplot.html") }],
    evidence: BuiltinDocumentationEvidence { implementation: &[], verification: &[], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
pub const SURROGATEOPTPLOT_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    OptimPlotKind::SurrogateProgress,
    &SURROGATEOPTPLOT_DESCRIPTOR,
    DOCUMENTATION,
);
