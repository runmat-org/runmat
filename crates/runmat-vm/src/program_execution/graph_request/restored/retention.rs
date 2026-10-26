//! Exact unnamed views for escaped handles and later capture operations.

use super::*;

impl RestoredSelectedGraph {
    /// Retain original unit-qualified immutable views, never merge functions,
    /// numeric namespaces or names. A later get_owned still requires the exact
    /// live definition lease, and unused source views prune by existing rules.
    pub fn retain_definitions_into(
        &self,
        destination: &mut crate::FunctionRegistry,
    ) -> Result<(), RuntimeError> {
        let mut views = self
            .code
            .units
            .iter()
            .map(|unit| unit.registry().clone())
            .collect::<Vec<_>>();
        for (owner, product) in &self.code.class_products {
            let binding = self
                .code
                .bindings
                .get(owner)
                .ok_or_else(|| rejected("restored constructor binding is absent"))?;
            let view = views
                .get_mut(usize::from(owner.unit.0))
                .ok_or_else(|| rejected("restored constructor unit is absent"))?;
            view.attach_selected_class_product(
                runmat_hir::FunctionId(binding.function),
                &binding.authority,
                Rc::clone(product),
            )
            .map_err(rejected)?;
        }
        for (owner, binding) in &self.code.bindings {
            self.code.unit(*owner).map_err(rejected)?;
            let view = views
                .get(usize::from(owner.unit.0))
                .ok_or_else(|| rejected("restored definition unit is absent"))?;
            destination
                .retain_selected_site_view(view, binding)
                .map_err(rejected)?;
        }
        Ok(())
    }

    pub fn canonical_units(&self) -> impl Iterator<Item = Result<&Arc<[u8]>, RuntimeError>> {
        self.code.units.iter().map(|unit| {
            unit.canonical_v3()
                .ok_or_else(|| rejected("restored canonical V3 unit is absent"))
        })
    }

    pub fn retained_sources(
        &self,
    ) -> Result<Vec<Rc<crate::RetainedExecutableSourceOwner>>, RuntimeError> {
        self.code
            .units
            .iter()
            .map(|unit| {
                let bytes = unit
                    .canonical_v3()
                    .ok_or_else(|| rejected("restored canonical V3 unit is absent"))?;
                let source = unit
                    .selected_source_owner()
                    .ok_or_else(|| rejected("restored source owner is absent"))?;
                crate::RetainedExecutableSourceOwner::from_selected_product(
                    Arc::clone(bytes),
                    source.clone(),
                    unit.registry(),
                )
                .map(Rc::new)
                .map_err(rejected)
            })
            .collect()
    }
}
