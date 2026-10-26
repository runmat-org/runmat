//! Cold Graph authority reuses immutable product captures and unnamed source
//! views. Wire unit registries are never merged or published by numeric ID.

use super::{
    SelectedExecutableCode, SelectedExecutableProduct, SelectedStaticCapture,
    SelectedStaticProductDependency, SelectedStaticSiteOwner,
};
use crate::bytecode::program::{definition_dependencies, FunctionRegistry, SelectedSiteView};
use runmat_hir::FunctionId;
use std::{rc::Rc, sync::Arc};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SelectedForeignBasis {
    Local(Option<runmat_execution::Digest>),
    /// The exact archive receipt is held on the product. This is not a native
    /// prepared-host reuse key and must never be exposed as one.
    ArchiveAdmitted,
}

impl SelectedExecutableProduct {
    pub(crate) fn admit_graph(
        bytes: Arc<[u8]>,
        owner: runmat_runtime::source_context::SelectedProgramSourceOwner,
        receipt: Option<Rc<crate::SelectedUnitInteropReceipt>>,
    ) -> Result<Rc<Self>, String> {
        let code = SelectedExecutableCode::admit(bytes)?;
        if !owner.is_live() || &code.source_map != owner.portable_map() {
            return Err("selected graph product differs from its issued source map".into());
        }
        if code.requires_interop() && receipt.is_none() {
            return Err("selected graph foreign product has no checked archive receipt".into());
        }
        if receipt
            .as_ref()
            .is_some_and(|r| !r.matches_selected_unit(code.canonical_v3(), &owner))
        {
            return Err("selected graph archive differs from its issued code/source".into());
        }
        let foreign_basis = if code.requires_interop() {
            SelectedForeignBasis::ArchiveAdmitted
        } else {
            SelectedForeignBasis::Local(None)
        };
        Ok(Rc::new(Self {
            code,
            source_owner: owner,
            foreign_basis,
            interop_receipt: receipt,
            dynamic_admission: None,
        }))
    }
}

impl SelectedStaticSiteOwner {
    /// Capture the real admitted target definition and local owners without
    /// minting a temporary callable. The supplied capture is already complete
    /// and checked; it supplies instruction-site proof during bootstrap.
    pub(crate) fn from_graph_target(
        edge: SelectedStaticProductDependency,
        registry: &FunctionRegistry,
        capture: &SelectedStaticCapture,
    ) -> Result<Self, String> {
        if !Rc::ptr_eq(edge.target(), capture.product()) {
            return Err("graph target capture belongs to another product".into());
        }
        let target_origin = edge.import().target;
        let mut bodies = registry
            .selected_execution_bodies()
            .into_iter()
            .filter(|body| body.binding_origin == Some(target_origin));
        let body = bodies.next().ok_or("graph static target body is absent")?;
        if bodies.next().is_some()
            || body
                .selected_executable_product
                .as_ref()
                .is_none_or(|p| !Rc::ptr_eq(p, edge.target()))
            || body
                .selected_source_owner
                .as_ref()
                .is_none_or(|s| !s.same_issuance(edge.target().source_owner()))
            || body
                .source_id
                .and_then(super::super::BytecodeSourceRef::runtime)
                .is_none_or(|s| !edge.target().source_owner().owns_source(s))
        {
            return Err("graph static target differs from exact product/source".into());
        }
        let dependencies =
            definition_dependencies::owners_with_capture(registry, body.function, Some(capture))
                .map_err(|error| error.to_string())?;
        let mut root = false;
        let mut retained = Vec::new();
        for (function, owner) in dependencies {
            if function == body.function && Arc::ptr_eq(&owner, &body.definition_owner) {
                root = true;
            } else {
                retained.push((function.0, owner));
            }
        }
        if !root {
            return Err("graph static target lacks its original definition owner".into());
        }
        Self::new(
            edge,
            body.function,
            Arc::clone(&body.definition_owner),
            retained,
        )
        .map_err(str::to_owned)
    }
}

impl FunctionRegistry {
    /// An unnamed immutable code view, admitted by the actual site owner, not
    /// a synthetic counted binding. Snapshots have captures but no backviews:
    /// cyclic imports therefore cannot form an Rc ownership cycle.
    pub(crate) fn retain_graph_static_site_view(
        &mut self,
        candidate: &Self,
        owner: &SelectedStaticSiteOwner,
    ) -> Result<(), &'static str> {
        if !candidate.selected_site_views.is_empty() || candidate.static_overlay.is_some() {
            return Err("graph static snapshot contains invocation backviews");
        }
        let body = candidate
            .get_current(owner.function())
            .ok_or("graph static target is absent from its exact snapshot")?;
        let product = owner.edge().target();
        if !Arc::ptr_eq(&body.definition_owner, owner.definition())
            || body.binding_origin != Some(owner.edge().import().target)
            || body
                .selected_executable_product
                .as_ref()
                .is_none_or(|p| !Rc::ptr_eq(p, product))
            || body
                .selected_source_owner
                .as_ref()
                .is_none_or(|s| !s.is_live() || !s.same_issuance(product.source_owner()))
            || body
                .selected_static_capture
                .as_ref()
                .is_none_or(|capture| !Rc::ptr_eq(capture.product(), product))
        {
            return Err("graph static snapshot differs from its exact target owner");
        }
        for (function, definition) in owner.dependencies.iter() {
            if candidate
                .get_current(FunctionId(*function))
                .is_none_or(|b| !Arc::ptr_eq(&b.definition_owner, definition))
            {
                return Err("graph static snapshot lost a local definition owner");
            }
        }
        if self
            .get_current(owner.function())
            .is_some_and(|b| Arc::ptr_eq(&b.definition_owner, owner.definition()))
        {
            return Ok(());
        }
        if self.selected_site_views.iter().any(|view| {
            Rc::ptr_eq(&view.registry.data, &candidate.data)
                && Arc::ptr_eq(&view.root_owner, owner.definition())
        }) {
            return Ok(());
        }
        self.selected_site_views.push(SelectedSiteView {
            registry: candidate.current_view(),
            root_owner: Arc::clone(owner.definition()),
        });
        Ok(())
    }
}
