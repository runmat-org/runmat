//! `fea.StructuralField.*` selectors: one pure constructor per structural
//! result quantity. Generated from a quantity table; prose is per quantity.

use crate::fea_contract::doc_links;
use crate::*;

pub(super) const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Why use a selector instead of a field-id string?", answer: "Typed selectors are checked when you write them, so a misspelled quantity fails immediately instead of at result lookup. `fea.field` does not accept text field ids." },
    BuiltinDocumentationFaq { question: "What units are structural fields in?", answer: "SI: meters, radians, pascals, newtons, newton-meters, and joules, regardless of the units the geometry was authored in." },
];
pub(super) const LINKS: &[BuiltinDocumentationLink] =
    &[doc_links::OUTPUTS_AND_FIELDS, doc_links::STRUCTURAL_PHYSICS];
pub(super) const IMPLEMENTATION: &[BuiltinDocumentationLink] =
    &[doc_links::FEA_FIELD_SELECTORS_SOURCE];
pub(super) const VERIFICATION: &[BuiltinEvidenceReference] =
    &[doc_links::STRUCTURAL_SELECTORS_TEST];

mod beam_axial_force;
mod beam_bending_moment;
mod beam_bending_stress;
mod beam_shear_force;
mod beam_torsion_moment;
mod beam_torsion_stress;
mod constraint_reaction_force;
mod constraint_reaction_moment_about_global_origin;
mod displacement;
mod equation_scale;
mod integration_point_strain;
mod integration_point_stress;
mod nodal_von_mises;
mod reaction_force;
mod reaction_moment;
mod residual_norm;
mod rotation;
mod shell_bending_moment;
mod shell_membrane_force;
mod shell_transverse_shear;
mod shell_von_mises;
mod strain;
mod strain_energy_density;
mod stress;
mod total_strain_energy;
mod von_mises;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &displacement::ENTRY,
    &rotation::ENTRY,
    &von_mises::ENTRY,
    &nodal_von_mises::ENTRY,
    &strain::ENTRY,
    &stress::ENTRY,
    &integration_point_strain::ENTRY,
    &integration_point_stress::ENTRY,
    &reaction_force::ENTRY,
    &reaction_moment::ENTRY,
    &constraint_reaction_force::ENTRY,
    &constraint_reaction_moment_about_global_origin::ENTRY,
    &strain_energy_density::ENTRY,
    &total_strain_energy::ENTRY,
    &residual_norm::ENTRY,
    &equation_scale::ENTRY,
    &beam_axial_force::ENTRY,
    &beam_shear_force::ENTRY,
    &beam_torsion_moment::ENTRY,
    &beam_bending_moment::ENTRY,
    &beam_bending_stress::ENTRY,
    &beam_torsion_stress::ENTRY,
    &shell_membrane_force::ENTRY,
    &shell_bending_moment::ENTRY,
    &shell_transverse_shear::ENTRY,
    &shell_von_mises::ENTRY,
];
