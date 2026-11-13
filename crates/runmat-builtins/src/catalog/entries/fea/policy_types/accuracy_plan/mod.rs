//! Catalog identities of the `fea.AccuracyPlan` typed policy class.

use crate::*;

mod baseline;
mod new_method;
#[cfg(test)]
mod tests;

pub(super) const PLAN_STUDY: &str = concat!(
    "Attach the plan to a procedure step with the seventh argument of ",
    "`fea.ProcedureStep.configure(step, independentDomain, activations, stateInputs, ",
    "stateOutputs, observations, accuracy, restart)`. Procedure steps created by step ",
    "constructors such as `fea.StructuralStep.static` already carry ",
    "`fea.AccuracyPlan.baseline` with the identifier `<stepId>.accuracy` ",
    "until `configure` replaces it. A plan is validated with its step: observation ",
    "targets resolve against that step's observations and resource ceilings against their",
    " declared costs.",
);
pub(super) const PLAN_FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "Can an accuracy plan publish a result that misses its tolerances?",
    answer: concat!(
        "No. A satisfied plan publishes. Missing evidence publishes as indeterminate only when ",
        "the plan's unavailable-evidence policy is `markIndeterminate`. An unsatisfied plan ",
        "always blocks publication.",
    ),
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &new_method::FEA_ACCURACY_PLAN_NEW_CATALOG_ENTRY,
    &baseline::FEA_ACCURACY_PLAN_BASELINE_CATALOG_ENTRY,
];
