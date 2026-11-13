//! Restore a closed original-product DAG before exposing executable authority.
use super::{decode, ExecutableUnit, Kind};
use runmat_execution::{ExecutableUnitEnvelope, FunctionBindingOrigin};
use runmat_vm::FunctionRegistry;
use std::{collections::HashMap, rc::Rc};

pub(super) type OriginalOwners = HashMap<FunctionBindingOrigin, Rc<ExecutableUnit>>;

/// A parsed compiler inventory is not VM execution evidence. Inherited bodies
/// must first match already admitted original products; only the current body
/// slice is then passed to the unchanged VM admission boundary.
struct PendingProduct {
    envelope: ExecutableUnitEnvelope,
    registry: FunctionRegistry,
    foreign: Vec<FunctionBindingOrigin>,
}

impl PendingProduct {
    fn new(envelope: ExecutableUnitEnvelope) -> Result<Self, String> {
        let registry: FunctionRegistry = decode(&envelope, Kind::FunctionRegistry)?;
        let program = envelope
            .manifest
            .identity
            .program
            .identity_digest()
            .map_err(|error| error.to_string())?;
        let mut foreign = registry
            .functions
            .values()
            .map(|body| {
                body.binding_origin
                    .ok_or("compiler function projection lacks its original binding origin")
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|origin| origin.program != program)
            .collect::<Vec<_>>();
        if let Some(section) = envelope
            .manifest
            .optional_sections
            .iter()
            .find(|section| section.name == runmat_execution::executable::STATIC_IMPORT_SECTION)
        {
            foreign.extend(
                runmat_execution::executable::decode_static_imports(section)
                    .map_err(|error| error.to_string())?
                    .into_iter()
                    .map(|import| import.target),
            );
        }
        Ok(Self {
            envelope,
            registry,
            foreign,
        })
    }

    fn admit(self, owners: &OriginalOwners) -> Result<ExecutableUnit, String> {
        ExecutableUnit::decode_local_product(&self.envelope, self.registry, owners)
    }
}

pub(super) fn restore(envelope: &ExecutableUnitEnvelope) -> Result<ExecutableUnit, String> {
    let mut pending = envelope
        .dependency_products
        .iter()
        .map(|product| {
            product.validate().map_err(|e| e.to_string())?;
            PendingProduct::new(
                ExecutableUnitEnvelope::new(product.manifest.clone(), product.payloads.clone())
                    .map_err(|e| e.to_string())?,
            )
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut owners = OriginalOwners::new();
    while !pending.is_empty() {
        let ready = pending
            .iter()
            .position(|product| {
                product
                    .foreign
                    .iter()
                    .all(|origin| owners.contains_key(origin))
            })
            .ok_or("dependency inventory is incomplete or cyclic")?;
        let unit = Rc::new(pending.remove(ready).admit(&owners)?);
        for body in unit.functions().functions.values() {
            if !unit.owns_callable(body.function) {
                continue;
            }
            let origin = body
                .binding_origin
                .ok_or("original compiler body has no binding origin")?;
            if owners.insert(origin, Rc::clone(&unit)).is_some() {
                return Err("dependency inventory has ambiguous original owners".into());
            }
        }
    }
    PendingProduct::new(envelope.clone())?.admit(&owners)
}
