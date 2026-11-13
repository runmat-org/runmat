//! Catalog identities of `fea.AccuracyTarget`: what an accuracy criterion measures.

use crate::*;

mod execution_equivalence;
mod global_equilibrium;
mod interface_closure;
mod numerical_result;
mod observation;
mod quantity;
#[cfg(test)]
mod tests;

pub(super) const TARGET_ROLE: &str = concat!(
    "An accuracy target names what one `fea.AccuracyCriterion` measures. The target is ",
    "immutable data: constructing it never runs a solver or reads results. RunMat checks ",
    "that the target is meaningful for its procedure step, and that its error ",
    "contribution and tolerance are compatible with it, when the study is validated ",
    "(`fea.validate`, `fea.plan`, or `fea.run`).",
);
pub(super) const TARGET_STUDY: &str = concat!(
    "Pass the target as the second argument of `fea.AccuracyCriterion.new`. Collect ",
    "criteria in `fea.AccuracyPlan.new`, then attach the plan to a step with ",
    "`fea.ProcedureStep.configure(step, domain, activations, stateInputs, stateOutputs, ",
    "observations, accuracy, restart)`.",
);
pub(super) const TARGET_FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "Does a target change what the solver computes?",
    answer: concat!(
        "No. A target only says which result a criterion assesses. Observations decide what ",
        "is computed and retained; the accuracy plan decides what evidence must accompany it.",
    ),
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &observation::FEA_ACCURACY_TARGET_OBSERVATION_CATALOG_ENTRY,
    &quantity::FEA_ACCURACY_TARGET_QUANTITY_CATALOG_ENTRY,
    &numerical_result::FEA_ACCURACY_TARGET_NUMERICAL_RESULT_CATALOG_ENTRY,
    &global_equilibrium::FEA_ACCURACY_TARGET_GLOBAL_EQUILIBRIUM_CATALOG_ENTRY,
    &interface_closure::FEA_ACCURACY_TARGET_INTERFACE_CLOSURE_CATALOG_ENTRY,
    &execution_equivalence::FEA_ACCURACY_TARGET_EXECUTION_EQUIVALENCE_CATALOG_ENTRY,
];
