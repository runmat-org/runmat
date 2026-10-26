use crate::*;
pub(super) const SECTIONS:&[BuiltinDocumentationSection]=&[
    BuiltinDocumentationSection{heading:"Standalone creation",paragraphs:&[
        "Population=creationFcn(nvars,problem,options) returns PopulationSize-by-nvars host doubles without calling problem.FitnessFcn. The R2026a operator context uses FitnessFcn (fitnessfcn is a legacy alias), Aineq,Bineq,Aeq,Beq,lb,ub,nonlcon,intcon; nvars is the explicit genome-length argument and options is separate. A complete InitialPopulationMatrix is retained; a partial prefix is preserved while missing rows are filled. PopulationSize,PopulationType,InitialPopulationMatrix and ConstraintTolerance are active; InitialPopulationRange is also active except for linear-feasible creation.",
        "gacreationuniform samples the adjusted finite initial range, including exact bitstring draws when PopulationType='bitstring'. It does not enforce linear or integer membership; the invoking ga performs its later admission. Bounds shape the sampling interval rather than authorizing invalid supplied rows. gacreationlinearfeasible uses the retained original linear/bound model and verified feasible witnesses; it ignores InitialPopulationRange. gacreationuniformint uses uniform representable-integer creation followed by original domain admission. gacreationsobol uses the pinned Gray-code/Joe-Kuo sequence and original linear/bound/integer admission. Custom PopulationType requires a user-defined creator.",
        "gacreationnonlinearfeasible also accepts unique SolverOpts,UseParallel,NumStartPts name/value pairs. SolverOpts uses the supported fmincon options with default sqp/off, without a user objective: the feasibility objective is the mathematical constant zero and only real constraint callbacks are counted. Default starts equals the number of missing rows. Additional starts cycle missing rows and keep the best observed constraint candidate per row; this is not a feasibility guarantee. UseParallel='on' requires real selected-callable workers, never a serial substitute. Parent GA timing is checked at wave boundaries rather than converted into a new worker physical deadline.",
        "Only needed creation controls and original geometry are prepared; fitness scores, observer/hybrid fields, unused callbacks and inactive nonlinear constraints are not recursively gathered. Named parameter errors retain their GA/fmincon owner identifiers. Native numeric/single/control extensions use the selected compatibility policy; authenticated automatic residency may materialize, but no explicit-GPU GA fallback is added. Finite host-memory policies refuse before random or constrained work. Original cancellation and provider errors are not converted to infeasible points. A standalone feasible creator that cannot complete a verified linear/integer population returns an error rather than a falsely feasible matrix.",
    ]},
];
pub(super) const fn doc(
    name: &'static str,
    summary: &'static str,
    examples: &'static [BuiltinExample],
) -> BuiltinDocumentation {
    BuiltinDocumentation{authority:BuiltinDocumentationAuthority::Catalog,title:Some(name),slug:Some(name),summary,description:summary,
        keywords:&["genetic algorithm","initial population","feasibility"],related:&["ga","optimoptions","fmincon","intlinprog"],
        sections:SECTIONS,examples,example_exemption:None,faqs:&[],media:&[],
        links:&[BuiltinDocumentationLink{label:"MathWorks population creation contracts",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/gads/genetic-algorithm-options.html")}],
        evidence:BuiltinDocumentationEvidence{implementation:&[BuiltinDocumentationLink{label:"Predefined creator entry points",target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/ga/helpers/creation.rs")}],
            verification:&[BuiltinEvidenceReference{kind:BuiltinEvidenceKind::UnitTest,label:"Supplied generation backing and original-domain creation",location:"builtins::math::optim::ga::creation::tests"}],notes:&[]},
        introduced:None,status:Some(BuiltinDocumentationStatus::Partial)}
}
