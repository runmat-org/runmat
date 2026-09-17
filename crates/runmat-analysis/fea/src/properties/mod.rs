//! Compilation of authored property assignments onto canonical formulation elements.

mod compile;
mod contract;

pub use compile::{compile_property_map, compile_solid_property_map, PropertyCompilationError};
pub use contract::{
    CompiledElementMaterial, CompiledExplicitElementMaterial, CompiledPropertyAudit,
    CompiledPropertyMap, CompiledPropertyMapIdentity,
};

#[cfg(test)]
mod tests;
