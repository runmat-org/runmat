//! Restore a closed flat inventory before exposing any executable unit.
use super::ExecutableUnit;
use runmat_execution::{ExecutableDependencyProduct, ExecutableUnitEnvelope};
use runmat_types::FunctionId;
use std::{collections::HashMap, rc::Rc};

pub(super) fn restore(
    root: ExecutableUnit,
    products: &[ExecutableDependencyProduct],
) -> Result<ExecutableUnit, String> {
    let mut pending = products
        .iter()
        .map(|product| {
            product.validate().map_err(|e| e.to_string())?;
            let envelope =
                ExecutableUnitEnvelope::new(product.manifest.clone(), product.payloads.clone())
                    .map_err(|e| e.to_string())?;
            ExecutableUnit::decode_local_product(&envelope)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut owners: HashMap<FunctionId, Rc<ExecutableUnit>> = HashMap::new();
    while !pending.is_empty() {
        let ready = pending
            .iter()
            .position(|unit| {
                unit.foreign_functions()
                    .is_ok_and(|ids| ids.iter().all(|id| owners.contains_key(id)))
            })
            .ok_or("dependency inventory is incomplete or cyclic")?;
        let unit = Rc::new(pending.remove(ready).bind_original_dependencies(&owners)?);
        for analysis in &unit.analysis().functions {
            let Some(runmat_types::CallableIdentity::BoundFunction(function)) =
                &analysis.callable.identity
            else {
                continue;
            };
            if unit.owns_callable(*function) && owners.insert(*function, Rc::clone(&unit)).is_some()
            {
                return Err("dependency inventory has ambiguous original owners".into());
            }
        }
    }
    root.bind_original_dependencies(&owners)
}
