mod descriptor;
mod documentation;
mod entry;
mod inference;
mod helper_descriptors;
mod helper_entries;
mod helper_inference;
mod helper_documentation;
mod creator_entries;
mod creator_inference;
mod creator_documentation;
mod crossover_descriptors;
mod crossover_entries;
mod crossover_inference;
mod mutation_descriptors;
mod mutation_entries;
mod mutation_inference;
mod plot_descriptor;
mod plot_documentation;
mod plot_entries;
mod plot_inference;
mod feasible_start;
#[cfg(test)]
mod tests;
pub use creator_entries::*;
pub use crossover_descriptors::*;
pub use crossover_entries::*;
pub use descriptor::*;
pub use entry::*;
pub use feasible_start::*;
pub use helper_descriptors::*;
pub use helper_entries::*;
pub use mutation_descriptors::*;
pub use mutation_entries::*;
pub use plot_descriptor::*;
pub use plot_entries::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &GA_CATALOG_ENTRY,
    &GA_FEASIBLE_START_CATALOG_ENTRY,
    &SELECTIONSTOCHUNIF_CATALOG_ENTRY,
    &SELECTIONREMAINDER_CATALOG_ENTRY,
    &SELECTIONUNIFORM_CATALOG_ENTRY,
    &SELECTIONROULETTE_CATALOG_ENTRY,
    &SELECTIONTOURNAMENT_CATALOG_ENTRY,
    &FITSCALINGRANK_CATALOG_ENTRY,
    &FITSCALINGPROP_CATALOG_ENTRY,
    &FITSCALINGTOP_CATALOG_ENTRY,
    &FITSCALINGSHIFTLINEAR_CATALOG_ENTRY,
    &GACREATIONUNIFORM_CATALOG_ENTRY,
    &GACREATIONLINEARFEASIBLE_CATALOG_ENTRY,
    &GACREATIONNONLINEARFEASIBLE_CATALOG_ENTRY,
    &GACREATIONUNIFORMINT_CATALOG_ENTRY,
    &GACREATIONSOBOL_CATALOG_ENTRY,
    &CROSSOVERSCATTERED_CATALOG_ENTRY,
    &CROSSOVERSINGLEPOINT_CATALOG_ENTRY,
    &CROSSOVERTWOPOINT_CATALOG_ENTRY,
    &CROSSOVERINTERMEDIATE_CATALOG_ENTRY,
    &CROSSOVERARITHMETIC_CATALOG_ENTRY,
    &CROSSOVERHEURISTIC_CATALOG_ENTRY,
    &CROSSOVERLAPLACE_CATALOG_ENTRY,
    &MUTATIONGAUSSIAN_CATALOG_ENTRY,
    &MUTATIONUNIFORM_CATALOG_ENTRY,
    &MUTATIONPOWER_CATALOG_ENTRY,
    &MUTATIONADAPTFEASIBLE_CATALOG_ENTRY,
    &MUTATIONPOSITIVEBASIS_CATALOG_ENTRY,
    &GAPLOTBESTF_CATALOG_ENTRY,
    &GAPLOTBESTINDIV_CATALOG_ENTRY,
    &GAPLOTDISTANCE_CATALOG_ENTRY,
    &GAPLOTEXPECTATION_CATALOG_ENTRY,
    &GAPLOTGENEALOGY_CATALOG_ENTRY,
    &GAPLOTMAXCONSTR_CATALOG_ENTRY,
    &GAPLOTRANGE_CATALOG_ENTRY,
    &GAPLOTSCOREDIVERSITY_CATALOG_ENTRY,
    &GAPLOTSCORES_CATALOG_ENTRY,
    &GAPLOTSELECTION_CATALOG_ENTRY,
    &GAPLOTSTOPPING_CATALOG_ENTRY,
];
