//! Compilation of authored property assignments onto canonical formulation elements.

mod compile;
mod contract;
mod identity;
mod preparation;
mod realize;
mod selection;

pub(crate) use preparation::PreparedProperties;

pub use compile::{
    compile_property_map, compile_solid_property_map, compile_structural_property_map,
    compile_thermal_property_map, PropertyCompilationError,
};
pub use contract::{
    CompiledBeamProperty, CompiledElementMaterial, CompiledLaminatePly, CompiledPropertyAudit,
    CompiledPropertyAuditOwner, CompiledPropertyMap, CompiledPropertyMapIdentity,
    CompiledShellProperty, CompiledShellPropertyOwner,
};
pub use realize::{realize_formulation_properties, PropertyRealizationError};

#[cfg(test)]
mod realize_tests;
#[cfg(test)]
mod tests;
