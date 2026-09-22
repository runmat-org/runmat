//! Compilation of authored property assignments onto canonical formulation elements.

mod compile;
mod contract;
mod realize;

pub use compile::{compile_property_map, compile_solid_property_map, PropertyCompilationError};
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
