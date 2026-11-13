//! Seal wire-qualified static imports before any callable lease is issued.
use super::RuntimeBoundExecutableUnit;
use crate::{
    FunctionRegistry, SelectedExecutableProduct, SelectedStaticCapture,
    SelectedStaticProductDependency, SelectedStaticSiteOwner, SelectedUnitInteropReceipt,
};
use runmat_execution::{
    executable::ExecutableStaticCaller,
    value::{GraphStaticCaller, GraphStaticDependency},
};
use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

struct Edge {
    caller: usize,
    target: usize,
    selected: SelectedStaticProductDependency,
}

fn caller_unit(caller: GraphStaticCaller) -> usize {
    usize::from(match caller {
        GraphStaticCaller::Script(unit) => unit.0,
        GraphStaticCaller::Function(owner) => owner.unit.0,
    })
}

pub(super) fn attach(
    units: &mut [RuntimeBoundExecutableUnit],
    receipts: &[Option<Rc<SelectedUnitInteropReceipt>>],
    requested: &[GraphStaticDependency],
) -> Result<(), String> {
    if units.len() != receipts.len() {
        return Err("graph product receipt count differs from wire unit count".into());
    }
    // The Graph protocol has no linked-native code-lease handoff. Preserve the
    // existing source selection refusal rather than reconstruct a native target.
    if units
        .iter()
        .any(|unit| !unit.linked_native_imports().is_empty())
    {
        return Err("selected Graph transfer has no linked-native code lease handoff".into());
    }
    let products = units
        .iter_mut()
        .zip(receipts)
        .map(|(unit, receipt)| unit.attach_graph_product(receipt.clone()))
        .collect::<Result<Vec<Rc<SelectedExecutableProduct>>, _>>()?;
    let mut sealed = HashMap::new();
    for (index, unit) in units.iter().enumerate() {
        for import in unit.static_imports() {
            if sealed
                .insert((index, import.caller, import.site), *import)
                .is_some()
            {
                return Err("selected unit repeats a sealed static site".into());
            }
        }
    }
    let mut observed = HashSet::new();
    let mut edges = Vec::with_capacity(requested.len());
    for edge in requested {
        let caller = caller_unit(edge.caller);
        let target = usize::from(edge.target.unit.0);
        let local_caller = match edge.caller {
            GraphStaticCaller::Script(_) => ExecutableStaticCaller::Script,
            GraphStaticCaller::Function(owner) => ExecutableStaticCaller::Function(owner.origin),
        };
        let key = (caller, local_caller, edge.site);
        let import = *sealed
            .get(&key)
            .ok_or("graph static edge has no sealed caller site")?;
        let target_product = products
            .get(target)
            .ok_or("graph static target unit is absent")?;
        if import.target != edge.target.origin
            || import.target_body != edge.target_body_digest
            || target_product.full_digest() != edge.target_unit_digest
            || !observed.insert(key)
        {
            return Err("graph static edge differs from its exact sealed target".into());
        }
        let selected = SelectedStaticProductDependency::new(
            Rc::clone(
                products
                    .get(caller)
                    .ok_or("graph static caller unit is absent")?,
            ),
            import,
            Rc::clone(target_product),
        )?;
        edges.push(Edge {
            caller,
            target,
            selected,
        });
    }
    if observed.len() != sealed.len() {
        return Err("selected graph lacks an exact static target for a sealed site".into());
    }
    // Each root gets exactly its reachable graph. Product nodes never retain
    // this vector; recursive unit imports therefore have no ownership backedge.
    let mut reachable = Vec::with_capacity(units.len());
    let mut captures = Vec::with_capacity(units.len());
    for (root, product) in products.iter().enumerate() {
        let mut visited = HashSet::new();
        let mut pending = vec![root];
        while let Some(unit) = pending.pop() {
            if visited.insert(unit) {
                pending.extend(edges.iter().filter(|e| e.caller == unit).map(|e| e.target));
            }
        }
        let indices = edges
            .iter()
            .enumerate()
            .filter_map(|(index, e)| visited.contains(&e.caller).then_some(index))
            .collect::<Vec<_>>();
        let capture = SelectedStaticCapture::admit_with_linked(
            Rc::clone(product),
            indices.iter().map(|i| edges[*i].selected.clone()).collect(),
            Vec::new(),
        )?;
        reachable.push(indices);
        captures.push(capture);
    }
    let owners = edges
        .iter()
        .map(|edge| {
            SelectedStaticSiteOwner::from_graph_target(
                edge.selected.clone(),
                units[edge.target].registry(),
                &captures[edge.target],
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let complete = captures
        .into_iter()
        .zip(&reachable)
        .map(|(capture, indices)| {
            capture
                .with_site_owners(indices.iter().map(|i| owners[*i].clone()).collect())
                .map_err(str::to_owned)
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (unit, capture) in units.iter_mut().zip(&complete) {
        unit.attach_graph_capture(capture)?;
    }
    // Snapshots contain final captures but no source backviews. Later COW
    // additions retain these snapshots only, never one another's mutable graph.
    let snapshots = units
        .iter()
        .map(|u| u.registry().clone())
        .collect::<Vec<FunctionRegistry>>();
    for (unit, indices) in units.iter_mut().zip(reachable) {
        let sites = indices
            .into_iter()
            .map(|i| (edges[i].target, owners[i].clone()))
            .collect::<Vec<_>>();
        unit.retain_graph_views(&snapshots, &sites)?;
    }
    Ok(())
}
