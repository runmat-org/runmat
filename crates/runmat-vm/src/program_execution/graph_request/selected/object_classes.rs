use super::*;
use runmat_execution::value::GraphNode;
use runmat_types::ClassMethodOwner;

/// Admit receiver facts for all executable constructors in the selected
/// bundle. A wire object may claim only one of these exact unit-qualified
/// constructor bindings; method bodies are indexed by typed owner once, not
/// resolved by class or method text during invocation.
pub(super) fn admit(
    graph: &runmat_execution::value::PayloadGraph,
    units: &[RuntimeBoundExecutableUnit],
    receipts: &[Option<Rc<crate::SelectedUnitInteropReceipt>>],
    bindings: &HashMap<GraphBoundOwner, BoundFunctionBinding>,
    class_method_targets: &HashMap<(GraphBoundOwner, ClassMethodOwner), GraphBoundOwner>,
    class_products: &mut HashMap<GraphBoundOwner, Rc<crate::SelectedClassProduct>>,
) -> Result<HashMap<GraphBoundOwner, Rc<runmat_value::SelectedObjectClassFacts>>, String> {
    let mut selected_owners = graph
        .nodes
        .iter()
        .filter_map(|node| match node {
            GraphNode::SelectedObject(object) => Some(object.constructor),
            _ => None,
        })
        .collect::<HashSet<_>>();
    let mut method_index = HashMap::<ClassMethodOwner, Vec<(usize, GraphBoundOwner)>>::new();
    for (position, unit) in units.iter().enumerate() {
        let unit_id = GraphProgramUnitId(
            u16::try_from(position).map_err(|_| "selected class unit overflows")?,
        );
        for body in unit.registry().selected_execution_bodies() {
            let Some(method) = body.class_method_owner.as_ref() else {
                continue;
            };
            let origin = body
                .binding_origin
                .ok_or("selected class method lacks a portable origin")?;
            let owner = GraphBoundOwner {
                unit: unit_id,
                origin,
            };
            let binding = bindings
                .get(&owner)
                .ok_or("selected class method has no admitted binding")?;
            if !std::ptr::eq(selected_bound_body(unit.registry(), binding)?, body)
                || body.selected_source_owner.as_ref().is_none_or(|issued| {
                    unit.selected_source_owner()
                        .is_none_or(|selected| !issued.same_issuance(selected))
                })
            {
                return Err("selected class method binding changed its admitted body".into());
            }
            if method.role == runmat_types::ClassMethodRole::Constructor && !method.is_static {
                selected_owners.insert(owner);
                if let std::collections::hash_map::Entry::Vacant(entry) =
                    class_products.entry(owner)
                {
                    let mut positions = vec![position];
                    positions.extend(class_method_targets.iter().filter_map(
                        |((caller, _), target)| {
                            (*caller == owner).then_some(usize::from(target.unit.0))
                        },
                    ));
                    positions.sort_unstable();
                    positions.dedup();
                    let selected_units = positions
                        .iter()
                        .map(|position| {
                            units
                                .get(*position)
                                .ok_or("selected class method unit is absent")
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let selected_receipts = positions
                        .iter()
                        .map(|position| {
                            receipts
                                .get(*position)
                                .cloned()
                                .ok_or("selected class unit receipt is absent")
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let product = crate::SelectedClassProduct::from_bound_units_with_receipts(
                        &selected_units,
                        &selected_receipts,
                        &method.declaring_class,
                    )
                    .map_err(|error| error.to_string())?;
                    entry.insert(Rc::new(product));
                }
            } else if method.role == runmat_types::ClassMethodRole::Ordinary {
                method_index
                    .entry(method.clone())
                    .or_default()
                    .push((position, owner));
            }
        }
    }
    let mut object_classes = HashMap::new();
    let mut consumed_method_edges = HashSet::new();
    for owner in selected_owners {
        let product = class_products
            .get(&owner)
            .ok_or("selected object lacks its exact class product")?;
        let projection = product
            .invocation_projection()
            .map_err(|error| error.to_string())?;
        let declarations = projection
            .receiver_declarations(product.class())
            .map_err(|error| error.to_string())?;
        let mut methods = HashMap::with_capacity(declarations.owner_methods().len());
        for (method_owner, declaration) in declarations.owner_methods() {
            let admitted_binding = if let Some(target) =
                class_method_targets.get(&(owner, declaration.owner.clone()))
            {
                let selected = bindings
                    .get(target)
                    .ok_or("selected class method binding is absent")?;
                if !method_index
                    .get(&declaration.owner)
                    .is_some_and(|candidates| {
                        candidates.iter().any(|(_, candidate)| candidate == target)
                    })
                {
                    return Err("selected class method edge has no admitted owner body".into());
                }
                consumed_method_edges.insert((owner, declaration.owner.clone()));
                Some(selected.clone())
            } else {
                None
            };
            if !declaration.is_abstract
                && !declaration.owner.is_static
                && admitted_binding.is_none()
            {
                return Err("selected ordinary class method lacks an admitted body".into());
            }
            methods.insert(method_owner.clone(), admitted_binding);
        }
        let constructor = bindings
            .get(&owner)
            .ok_or("selected object constructor binding is absent")?
            .clone();
        product.seal_receiver_declarations(&constructor, std::sync::Arc::clone(&declarations))?;
        let facts = runmat_value::SelectedObjectClassFacts::new(declarations, constructor, methods)
            .ok_or("selected object class facts differ from admitted product")?;
        object_classes.insert(owner, Rc::new(facts));
    }
    if consumed_method_edges.len() != class_method_targets.len() {
        return Err("selected class method edge is outside its constructor declaration".into());
    }
    Ok(object_classes)
}
