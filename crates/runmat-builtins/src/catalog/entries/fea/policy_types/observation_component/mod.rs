//! Catalog identities of the `fea.ObservationComponent` typed policy class.

use crate::*;

mod all;
mod deviatoric;
mod hydrostatic;
mod magnitude;
mod maximum_principal;
mod minimum_principal;
mod normal;
mod tangent1;
mod tangent2;
mod trace;
mod von_mises;
mod x;
mod y;
mod z;
#[cfg(test)]
mod tests;

pub(super) const ROLE: &str = concat!(
    "A component projects an observation's quantity to the part that is reported. Which ",
    "components are valid depends on the quantity's shape: scalars and section resultants",
    " accept only `all`; vectors accept `all`, `x`, `y`, `z`, `normal`, `tangent1`, ",
    "`tangent2`, and `magnitude`; symmetric tensors accept `all`, normal and tangent ",
    "components, `trace`, `hydrostatic`, `deviatoric`, `vonMises`, and the principal ",
    "values. Other combinations fail study validation.",
);
pub(super) const STUDY: &str = concat!(
    "Pass the component as the seventh argument of an observation constructor, with the ",
    "coordinate frame as the eighth. `fea.Observation.field`, `history`, `probe`, and ",
    "`reduction` take `(id, quantity, target, sourceLocation, outputLocation, recovery, ",
    "component, frame, sign, complexRepresentation, sampling, reduction, storage, ",
    "purpose, cost)`.",
);
pub(super) const FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "Are x, y, and z always global axes?",
    answer: concat!(
        "They are the axes of the observation's coordinate frame, which is the global frame ",
        "when you pass `fea.CoordinateFrame.globalFrame()`.",
    ),
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &all::FEA_OBSERVATION_COMPONENT_ALL_CATALOG_ENTRY,
    &x::FEA_OBSERVATION_COMPONENT_X_CATALOG_ENTRY,
    &y::FEA_OBSERVATION_COMPONENT_Y_CATALOG_ENTRY,
    &z::FEA_OBSERVATION_COMPONENT_Z_CATALOG_ENTRY,
    &normal::FEA_OBSERVATION_COMPONENT_NORMAL_CATALOG_ENTRY,
    &tangent1::FEA_OBSERVATION_COMPONENT_TANGENT1_CATALOG_ENTRY,
    &tangent2::FEA_OBSERVATION_COMPONENT_TANGENT2_CATALOG_ENTRY,
    &trace::FEA_OBSERVATION_COMPONENT_TRACE_CATALOG_ENTRY,
    &hydrostatic::FEA_OBSERVATION_COMPONENT_HYDROSTATIC_CATALOG_ENTRY,
    &deviatoric::FEA_OBSERVATION_COMPONENT_DEVIATORIC_CATALOG_ENTRY,
    &von_mises::FEA_OBSERVATION_COMPONENT_VON_MISES_CATALOG_ENTRY,
    &maximum_principal::FEA_OBSERVATION_COMPONENT_MAXIMUM_PRINCIPAL_CATALOG_ENTRY,
    &minimum_principal::FEA_OBSERVATION_COMPONENT_MINIMUM_PRINCIPAL_CATALOG_ENTRY,
    &magnitude::FEA_OBSERVATION_COMPONENT_MAGNITUDE_CATALOG_ENTRY,
];
