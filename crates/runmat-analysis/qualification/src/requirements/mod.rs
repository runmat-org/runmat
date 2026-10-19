//! Explicit domain inventories; registration is separate from satisfaction.
mod acoustics;
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
];
