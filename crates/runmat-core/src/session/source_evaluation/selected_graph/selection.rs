use runmat_execution::FunctionBindingOrigin;
use runmat_hir::FunctionId;
use runmat_runtime::execution::value_codec::ValueCodecError;
use runmat_runtime::source_context::SelectedProgramSourceOwner;
use runmat_value::{BoundFunctionAuthority, BoundFunctionBinding, SelectedObjectClassFacts};
use runmat_vm::{BytecodeSourceRef, FunctionRegistry};
use std::rc::Rc;
use std::sync::Arc;

#[path = "selection/units.rs"]
mod units;
pub(super) use units::SelectedGraphUnits;

pub(super) struct SelectedLocalBody {
    pub(super) view: FunctionRegistry,
    pub(super) function: FunctionId,
    pub(super) source_owner: SelectedProgramSourceOwner,
    class_product: Option<Rc<runmat_vm::SelectedClassProduct>>,
}

pub(super) struct SelectedDeclarationBody {
    pub(super) class: runmat_types::ClassIdentity,
    pub(super) bytes: Arc<[u8]>,
    pub(super) source_owner: SelectedProgramSourceOwner,
    pub(super) interop_receipt: Option<std::rc::Rc<runmat_vm::SelectedUnitInteropReceipt>>,
    pub(super) static_targets: Vec<runmat_vm::SelectedStaticProductTarget>,
}

pub(super) struct SelectedDependencyBody {
    pub(super) origin: FunctionBindingOrigin,
    pub(super) source_owner: SelectedProgramSourceOwner,
    pub(super) fingerprint: String,
    pub(super) product: Option<Rc<runmat_vm::SelectedExecutableProduct>>,
    pub(super) static_capture: Option<Rc<runmat_vm::SelectedStaticCapture>>,
    pub(super) interop_receipt: Option<std::rc::Rc<runmat_vm::SelectedUnitInteropReceipt>>,
}

impl SelectedLocalBody {
    pub(super) fn selected_receiver_facts(
        &self,
        authority: &BoundFunctionAuthority,
    ) -> Result<Option<Rc<SelectedObjectClassFacts>>, ValueCodecError> {
        let Some(product) = self.class_product.as_ref() else {
            return Ok(None);
        };
        let lease = authority
            .local_lease(self.function.0)
            .ok_or_else(|| invalid("selected constructor lacks its exact definition lease"))?;
        let body = self
            .view
            .get_owned(self.function, lease)
            .ok_or_else(|| invalid("selected constructor body is absent"))?;
        let Some(owner) = body.class_method_owner.as_ref() else {
            return Ok(None);
        };
        if owner.role != runmat_types::ClassMethodRole::Constructor {
            return Ok(None);
        }
        if owner.is_static
            || product.class() != &owner.declaring_class
            || body
                .selected_source_owner
                .as_ref()
                .is_none_or(|issued| !issued.same_issuance(&self.source_owner))
        {
            return Err(invalid(
                "selected constructor differs from its class product",
            ));
        }
        let binding = BoundFunctionBinding {
            function: self.function.0,
            authority: authority.clone(),
        };
        product
            .receiver_facts_with_binding(&self.view, &binding)
            .map(Some)
            .map_err(|error| invalid(error.to_string()))
    }

    pub(super) fn declaration_dependencies(&self) -> Vec<SelectedDeclarationBody> {
        self.class_product
            .iter()
            .flat_map(|product| product.declaration_products())
            .map(|declaration| SelectedDeclarationBody {
                class: declaration.class().clone(),
                bytes: Arc::clone(declaration.canonical_bytes()),
                source_owner: declaration.source_owner().clone(),
                interop_receipt: declaration.interop_receipt().cloned(),
                static_targets: declaration.static_targets().to_vec(),
            })
            .collect()
    }

    /// The static overlay is the authority for every reachable dependency.
    /// Numeric IDs only index leases in that overlay; they never select a
    /// newly published body with the same number.
    pub(super) fn dependencies(
        &self,
        authority: &BoundFunctionAuthority,
    ) -> Result<Vec<SelectedDependencyBody>, ValueCodecError> {
        let owners = self
            .view
            .definition_dependency_owners(self.function)
            .ok_or_else(|| invalid("callable dependency graph is incomplete"))?;
        let imports = owners
            .iter()
            .map(|(function, owner)| {
                let lease = authority
                    .local_lease(function.0)
                    .filter(|lease| lease.matches(owner))
                    .ok_or_else(|| invalid("dependency lacks its exact fingerprint lease"))?;
                let body = self.view.get_owned(*function, lease).ok_or_else(|| {
                    invalid("dependency lease does not select its fingerprint body")
                })?;
                Ok((
                    *function,
                    body.selected_executable_product
                        .as_ref()
                        .map(|product| product.fingerprint_imports())
                        .unwrap_or_default(),
                ))
            })
            .collect::<Result<Vec<_>, ValueCodecError>>()?;
        let inputs = imports
            .iter()
            .map(|(function, imports)| (*function, imports.as_slice()))
            .collect::<Vec<_>>();
        let mut fingerprints = self
            .view
            .selected_runtime_body_fingerprints_by_product(&inputs)
            .map_err(|reason| {
                invalid(format!(
                    "selected dependency fingerprints are invalid: {reason}"
                ))
            })?;
        owners
            .into_iter()
            .map(|(function, _)| {
                let lease = authority
                    .local_lease(function.0)
                    .ok_or_else(|| invalid("dependency lacks exact definition lease"))?;
                let body = self
                    .view
                    .get_owned(function, lease)
                    .ok_or_else(|| invalid("dependency lease does not select its body"))?;
                let origin = body
                    .binding_origin
                    .ok_or_else(|| invalid("dependency has no portable origin"))?;
                let source_owner = body
                    .selected_source_owner
                    .as_ref()
                    .ok_or_else(|| invalid("dependency has no selected source owner"))?;
                if !source_owner.is_live() {
                    return Err(invalid("dependency source owner has retired"));
                }
                let source = body
                    .source_id
                    .and_then(BytecodeSourceRef::runtime)
                    .ok_or_else(|| invalid("dependency has no issued source"))?;
                if source_owner.source_ordinal_for(source).is_none() {
                    return Err(invalid("dependency source differs from its owner"));
                }
                let (checked_origin, fingerprint) = fingerprints
                    .remove(&function)
                    .ok_or_else(|| invalid("selected dependency fingerprint is absent"))?;
                if checked_origin != origin {
                    return Err(invalid(
                        "dependency fingerprint changes its portable origin",
                    ));
                }
                let interop_receipt = match body.selected_executable_product.as_ref() {
                    Some(product) => {
                        if !product.source_owner().same_issuance(source_owner) {
                            return Err(invalid(
                                "dependency product differs from its selected source issuance",
                            ));
                        }
                        let selected = product.interop_receipt().cloned();
                        if let Some(legacy) = body.selected_interop_receipt.as_ref() {
                            if selected
                                .as_ref()
                                .is_none_or(|receipt| !receipt.same_selected_admission(legacy))
                            {
                                return Err(invalid(
                                    "dependency has conflicting selected foreign receipts",
                                ));
                            }
                        }
                        selected
                    }
                    None => body.selected_interop_receipt.clone(),
                };
                if let Some(capture) = body.selected_static_capture.as_ref() {
                    if body
                        .selected_executable_product
                        .as_ref()
                        .is_none_or(|product| !Rc::ptr_eq(product, capture.product()))
                    {
                        return Err(invalid(
                            "dependency static capture differs from its product",
                        ));
                    }
                } else if body
                    .selected_executable_product
                    .as_ref()
                    .is_some_and(|product| {
                        !product.static_imports().is_empty()
                            || !product.linked_native_imports().is_empty()
                    })
                {
                    return Err(invalid("dependency lacks its sealed static capture"));
                }
                Ok(SelectedDependencyBody {
                    origin,
                    source_owner: source_owner.clone(),
                    fingerprint,
                    product: body.selected_executable_product.clone(),
                    static_capture: body.selected_static_capture.clone(),
                    interop_receipt,
                })
            })
            .collect()
    }
}

pub(super) fn invalid(message: impl Into<String>) -> ValueCodecError {
    ValueCodecError::Invalid {
        path: "$".into(),
        message: message.into(),
    }
}

/// Resolve a rooted callable only through its issued definition lease. The
/// overlay includes every exact static dependency even if publication has
/// removed or reused their session-local numeric IDs.
pub(super) fn select_local_body(
    registry: &FunctionRegistry,
    origin: FunctionBindingOrigin,
    authority: &BoundFunctionAuthority,
) -> Result<SelectedLocalBody, ValueCodecError> {
    let lease = authority
        .local_definition()
        .ok_or_else(|| invalid("callable has no local definition lease"))?;
    let function = FunctionId(lease.function());
    let view = registry
        .owned_view(function, authority)
        .ok_or_else(|| invalid("callable definition graph is not retained"))?;
    let body = view
        .get_owned(function, lease)
        .ok_or_else(|| invalid("callable lease does not select its body"))?;
    if body.binding_origin != Some(origin) {
        return Err(invalid("callable origin differs from its selected body"));
    }
    let source_owner = body
        .selected_source_owner
        .as_ref()
        .ok_or_else(|| invalid("selected callable has no source owner"))?;
    if !source_owner.is_live() {
        return Err(invalid("selected callable source owner has retired"));
    }
    let source = body
        .source_id
        .and_then(BytecodeSourceRef::runtime)
        .ok_or_else(|| invalid("selected callable has no issued source"))?;
    if source_owner.source_ordinal_for(source).is_none() {
        return Err(invalid("selected callable source differs from its owner"));
    }
    let source_owner = source_owner.clone();
    let class_product = body.selected_class_product.clone();
    Ok(SelectedLocalBody {
        view,
        function,
        source_owner,
        class_product,
    })
}
