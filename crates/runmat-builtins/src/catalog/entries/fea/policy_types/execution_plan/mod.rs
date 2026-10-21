//! Catalog identities of the `fea.ExecutionPlan` typed policy class.

use crate::*;

mod defaults;
mod new_method;
#[cfg(test)]
mod tests;

pub(super) const EXECUTION_ROLE: &str = concat!(
    "Execution values bind procedure steps to how they run: the requested compute ",
    "backend, the formulation maturity the caller will accept, and the solver options for",
    " each step. They never change the physics, the sequence order, or an accuracy ",
    "criterion, and they contain no host paths, device handles, or worker identities. ",
    "Placement on local, GPU, remote, or cluster workers is selected later by the shared ",
    "RunMat execution layer.",
);
pub(super) const EXECUTION_STUDY: &str = concat!(
    "Pass an `fea.ExecutionPlan` to `fea.study(id, geometry, model, \"Execution\", plan)`. ",
    "Without that option, `fea.study` builds the default plan for the model's sequence ",
    "with the CPU backend.",
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &new_method::FEA_EXECUTION_PLAN_NEW_CATALOG_ENTRY,
    &defaults::FEA_EXECUTION_PLAN_DEFAULTS_CATALOG_ENTRY,
];
