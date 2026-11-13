use std::collections::BTreeMap;

use runmat_analysis_core::StructuralModel;

use crate::{
    assembly::dofs::StructuralDofLayout,
    physics::structural::topology::node_space::CanonicalStructuralNodeSpace,
    properties::CompiledPropertyMap,
};

pub(super) struct ExplicitLoadDomain<'a> {
    pub structural: &'a StructuralModel,
    pub properties: &'a CompiledPropertyMap,
    pub topology: &'a CanonicalStructuralNodeSpace,
    pub authored_node_indices: &'a BTreeMap<u32, usize>,
    pub layout: &'a StructuralDofLayout,
    pub coordinates_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Position, &'a [[f64; 3]]>,
}
