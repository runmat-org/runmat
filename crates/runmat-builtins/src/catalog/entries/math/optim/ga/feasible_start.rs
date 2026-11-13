//! The registered numerical worker has the same authored contract as callers.
use super::super::nonlinear_facts as facts;
use crate::*;
use runmat_types::*;
const fn p(name: &'static str, description: &'static str) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        description,
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
    }
}
pub const GA_FEASIBLE_START_DESCRIPTOR:BuiltinDescriptor=BuiltinDescriptor{
    signatures:&[BuiltinSignatureDescriptor{label:"result=__runmat_ga_feasible_start(x,geometry,nonlcon,SolverOpts)",inputs:&[
        p("x","Nonempty finite real start vector."),p("geometry","Closed scalar structure with A,b,Aeq,beq,lb,ub original linear roles."),p("nonlcon","Selected nonlinear callback returning c,ceq."),p("SolverOpts","Supported constrained-feasibility options; objective work is an internal constant mode.")],outputs:&[p("result","Closed result with double x/violation, exact uint64 counters/extents and logical flags.")]}],
    output_mode:BuiltinOutputMode::Fixed,completion_policy:BuiltinCompletionPolicy::HiddenInternal,errors:&[super::GA_ERROR_INVALID_ARGUMENT],
};
fn infer(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if r.arguments.len() != 4 {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM-CATALOG-GA-START-ARITY",
            "expected x,geometry,nonlcon,SolverOpts",
            0,
        ));
    }
    for index in [1] {
        if let Some(value) = r.arguments.get(index) {
            if !matches!(
                value.kind,
                ValueKindFact::Unknown | ValueKindFact::Struct(_)
            ) || value.shape.element_count().is_some_and(|n| n != 1)
            {
                diagnostics.push(crate::catalog::inference::argument_error(
                    "RM-CATALOG-GA-START-CONTEXT",
                    "geometry and SolverOpts must be scalar structures",
                    index,
                ));
            }
        }
    }
    facts::options(r.arguments.get(3), 3, true, &mut diagnostics);
    let mut fields = std::collections::BTreeMap::new();
    fields.insert(
        "x".into(),
        facts::matrix(
            Some(1),
            r.arguments.first().and_then(|x| x.shape.element_count()),
        ),
    );
    fields.insert("violation".into(), facts::scalar());
    for name in ["calls", "points", "inequalities", "equalities"] {
        fields.insert(
            name.into(),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            })),
        );
    }
    for name in ["timedOut", "hasExtents"] {
        fields.insert(name.into(), ValueFact::scalar(ValueKindFact::Logical));
    }
    let output = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)));
    facts::finish(r, e, vec![output], diagnostics, None)
}
pub const GA_FEASIBLE_START_CATALOG_ENTRY: BuiltinCatalogEntry = {
    let documentation=super::helper_documentation::doc("__runmat_ga_feasible_start","Run one selected nonlinear feasible-start transaction.",&[
        BuiltinDocumentationSection{heading:"Worker transaction",paragraphs:&[
            "This internal registered callable runs one constrained start with a constant internal objective. It never calls a fabricated fitness function or counts synthetic objective evaluations. The supplied nonlcon and optional solver observers retain their normal semantic invocation, original cancellation and resource controls. x must be nonempty and finite; geometry has exactly A,b,Aeq,beq,lb,ub, with dimensionally consistent numeric roles.",
            "The result contains a host-double row x, maximum observed nonlinear violation (NaN when undefined), exact uint64 calls/points and inequality/equality counts, plus logical timedOut/hasExtents. A returned candidate is not an independent feasibility certificate: the invoking creation owner admits it against its original geometry and compares actual measured constraints. Parallel parents check their own selected deadline at wave boundaries; workers do not replace a logical parent deadline with a physical timeout.",
            "The selected runtime must supply the actual executable source for nonlcon, Clock and the constrained solver's service permissions. Active input and option class/intent gates precede provider transfers. Finite host-memory policies refuse before numerical work; errors and cancellation remain original failures rather than replaying a start.",
        ]}],&[BuiltinExample{compatibility:BuiltinExampleCompatibility::RunMat,..super::helper_documentation::example(
        "g=struct('A',[],'b',[],'Aeq',[],'beq',[],'lb',0,'ub',0); o=optimoptions('fmincon','Algorithm','sqp','Display','off'); r=feval('__runmat_ga_feasible_start',0,g,@(x)deal([],x(1)),o);",
        "assert(isequal(r.x,0)); assert(r.violation==0); assert(r.calls>=1 && r.points>=1); assert(r.hasExtents && ~r.timedOut); assert(r.inequalities==0 && r.equalities==1);")}]);
    let mut entry = super::helper_entries::creator(
        "__runmat_ga_feasible_start",
        &GA_FEASIBLE_START_DESCRIPTOR,
        documentation,
        infer,
        true,
        true,
    );
    entry.contract.compatibility = BuiltinCompatibility::RunMat;
    entry
};
