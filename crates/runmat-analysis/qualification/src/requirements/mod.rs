//! Explicit domain inventories; registration is separate from satisfaction.
mod acoustics;
mod acoustics_scope;
mod coupling_scope;
mod electromagnetics_scope;
mod flow;
mod meshing;
mod public_surface;
mod scale;
mod structural_scope;
mod thermal_scope;
mod adaptation;
mod authoring;
mod coupling;
mod dynamics;
mod electromagnetics;
mod fluid;
mod nonlinear_contact;
mod release;
mod shared;
mod structural;
mod thermal;
pub(crate) const ALL: &[&[crate::requirement::Requirement]] = &[
    authoring::REQUIREMENTS,
    shared::REQUIREMENTS,
    structural::REQUIREMENTS,
    dynamics::REQUIREMENTS,
    thermal::REQUIREMENTS,
    nonlinear_contact::REQUIREMENTS,
    electromagnetics::REQUIREMENTS,
    acoustics::REQUIREMENTS,
    fluid::REQUIREMENTS,
    coupling::REQUIREMENTS,
    adaptation::REQUIREMENTS,
    release::REQUIREMENTS,
    acoustics_scope::REQUIREMENTS,
    coupling_scope::REQUIREMENTS,
    electromagnetics_scope::REQUIREMENTS,
    flow::REQUIREMENTS,
    meshing::REQUIREMENTS,
    public_surface::REQUIREMENTS,
    scale::REQUIREMENTS,
    structural_scope::REQUIREMENTS,
    thermal_scope::REQUIREMENTS,
];
