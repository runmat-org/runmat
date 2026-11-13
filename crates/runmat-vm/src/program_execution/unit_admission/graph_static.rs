//! Attach exact cold Graph metadata to both executable body views.
use super::RuntimeBoundExecutableUnit;
use crate::{SelectedExecutableProduct, SelectedStaticCapture, SelectedStaticSiteOwner};
use std::{rc::Rc, sync::Arc};

impl RuntimeBoundExecutableUnit {
    pub(crate) fn attach_graph_product(
        &mut self,
        receipt: Option<Rc<crate::SelectedUnitInteropReceipt>>,
    ) -> Result<Rc<SelectedExecutableProduct>, String> {
        let product = SelectedExecutableProduct::admit_graph(
            Arc::clone(
                self.canonical_v3
                    .as_ref()
                    .ok_or("graph unit has no canonical product")?,
            ),
            self.selected_source_owner
                .clone()
                .ok_or("graph unit has no issued source owner")?,
            receipt,
        )?;
        let ids = self.registry.functions.keys().copied().collect::<Vec<_>>();
        self.registry
            .attach_current_selected_executable_product(ids, Rc::clone(&product))
            .map_err(|e| e.to_string())?;
        for body in self.registry.functions.values_mut() {
            body.selected_interop_receipt = product.interop_receipt().cloned();
        }
        self.sync_graph_bodies()?;
        Ok(product)
    }

    pub(crate) fn attach_graph_capture(
        &mut self,
        capture: &Rc<SelectedStaticCapture>,
    ) -> Result<(), String> {
        let ids = self.registry.functions.keys().copied().collect::<Vec<_>>();
        self.registry
            .attach_current_selected_static_capture(ids, capture)?;
        self.sync_graph_bodies()
    }

    pub(crate) fn retain_graph_views(
        &mut self,
        views: &[crate::FunctionRegistry],
        sites: &[(usize, SelectedStaticSiteOwner)],
    ) -> Result<(), String> {
        for (unit, owner) in sites {
            self.registry.retain_graph_static_site_view(
                views
                    .get(*unit)
                    .ok_or("graph static snapshot unit is absent")?,
                owner,
            )?;
        }
        self.bytecode.function_registry = self.registry.clone();
        Ok(())
    }

    fn sync_graph_bodies(&mut self) -> Result<(), String> {
        for (id, selected) in &self.registry.functions {
            if let Some(bound) = self.bytecode.bound_functions.get_mut(id) {
                if bound.binding_origin != selected.binding_origin
                    || bound.source_id != selected.source_id
                    || !Arc::ptr_eq(&bound.definition_owner, &selected.definition_owner)
                {
                    return Err("graph bytecode differs from its exact registry definition".into());
                }
                bound.selected_executable_product = selected.selected_executable_product.clone();
                bound.selected_static_capture = selected.selected_static_capture.clone();
                bound.selected_interop_receipt = selected.selected_interop_receipt.clone();
            }
        }
        self.bytecode.function_registry = self.registry.clone();
        Ok(())
    }
}
