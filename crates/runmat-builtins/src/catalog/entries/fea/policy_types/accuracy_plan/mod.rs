//! Catalog identities of the `fea.AccuracyPlan` typed policy class.

use crate::*;

mod experimental_baseline;
mod new_method;
#[cfg(test)]
mod tests;

pub(super) const PLAN_STUDY: &str = concat!(
    "Attach the plan to a procedure step with the seventh argument of ",
    "`fea.ProcedureStep.configure(step, independentDomain, activations, stateInputs, ",
    "stateOutputs, observations, accuracy, restart)`. Procedure steps created by step ",
    "constructors such as `fea.StructuralStep.static` already carry ",
    "`fea.AccuracyPlan.experimentalBaseline` with the identifier `<stepId>.accuracy` ",
    "until `configure` replaces it. A plan is validated with its step: observation ",
    "targets resolve against that step's observations and resource ceilings against their",
    " declared costs.",
);
pub(super) const PLAN_FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "Does an accuracy plan make an experimental formulation production-ready?",
    answer: concat!(
        "No. Numerical acceptance, formulation maturity (`fea.FormulationRequirement`), and ",
        "publication status are independent decisions. A plan can block a result; it cannot ",
        "promote one.",
    ),
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &new_method::FEA_ACCURACY_PLAN_NEW_CATALOG_ENTRY,
    &experimental_baseline::FEA_ACCURACY_PLAN_EXPERIMENTAL_BASELINE_CATALOG_ENTRY,
];
