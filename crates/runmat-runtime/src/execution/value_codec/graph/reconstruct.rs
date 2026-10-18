use std::collections::HashMap;

use runmat_execution::value::{
    CallableForm, GraphBoundOwner, GraphCallableOwner, GraphLexicalLayout, GraphNode, GraphNodeRef,
    GraphRegisteredKind, GraphSelectedObjectKind, PayloadGraph, ValueLimits,
};
use runmat_execution::ExecutableSourceMapV2;
use runmat_gc::ExplicitRoot;
use runmat_value::{
    BoundFunctionBinding, CellArray, Closure, LexicalEnvironment, ObjectInstance, StructValue,
    Value,
};

use super::{invalid, validate_bound_callable_proofs, GraphCallableProof, ValueCodecError};

pub(super) fn decode(
    graph: &PayloadGraph,
    runtime: &crate::context::RuntimeContext,
    source_map: &ExecutableSourceMapV2,
    source_issuer: Option<&super::GraphSourceIssuer>,
    layouts: &HashMap<GraphBoundOwner, GraphLexicalLayout>,
    proofs: &HashMap<GraphBoundOwner, GraphCallableProof>,
    bindings: &HashMap<GraphBoundOwner, BoundFunctionBinding>,
    selected_classes: &HashMap<
        GraphBoundOwner,
        std::rc::Rc<runmat_value::SelectedObjectClassFacts>,
    >,
    limits: ValueLimits,
) -> Result<crate::execution::RootedValues, ValueCodecError> {
    if let Some(issuer) = source_issuer {
        issuer.ensure_live()?;
    }
    let value_order = graph
        .validate_for_program_and_value_order(
            limits,
            Some(source_map),
            layouts,
            source_issuer.map_or(1, super::GraphSourceIssuer::selected_unit_count),
        )
        .map_err(|error| invalid(error.to_string()))?;
    validate_bound_callable_proofs(graph, proofs)?;
    for node in &graph.nodes {
        if let GraphNode::SelectedObject(object) = node {
            let binding = bindings
                .get(&object.constructor)
                .ok_or_else(|| invalid("selected object constructor has no admitted definition"))?;
            let facts = selected_classes.get(&object.constructor).ok_or_else(|| {
                invalid("selected object constructor has no admitted class product")
            })?;
            if facts.class() != &object.class
                || facts.constructor() != binding
                || facts.is_handle_class() != matches!(object.kind, GraphSelectedObjectKind::Handle)
                || binding.authority.export_origin(&runtime.session_lease())
                    != Some(object.constructor.origin)
            {
                return Err(invalid(
                    "selected object differs from its admitted constructor/class",
                ));
            }
        }
        let GraphNode::Callable(callable) = node else {
            continue;
        };
        let GraphCallableOwner::Bound(owner) = callable.owner else {
            continue;
        };
        let binding = bindings
            .get(&owner)
            .ok_or_else(|| invalid("bound unit owner is absent from selected definitions"))?;
        if binding.authority.export_origin(&runtime.session_lease()) != Some(owner.origin)
            || binding
                .authority
                .local_definition()
                .is_some_and(|lease| lease.function() != binding.function)
        {
            return Err(invalid(
                "bound binding differs from its selected function origin",
            ));
        }
    }
    let remaining_uses = value_use_counts(graph)?;
    let mut decoder = Decoder {
        graph,
        bindings,
        proofs,
        selected_classes,
        limits,
        source_issuer,
        environments: vec![None; graph.nodes.len()],
        environment_roots: Vec::new(),
        values: (0..graph.nodes.len()).map(|_| None).collect(),
        selected_handles: vec![None; graph.nodes.len()],
        selected_handle_roots: Vec::new(),
        selected_metadata: vec![None; graph.nodes.len()],
        selected_metadata_roots: Vec::new(),
        remaining_uses,
        visiting: vec![false; graph.nodes.len()],
    };
    decoder.allocate_selected_handle_shells()?;
    decoder.allocate_selected_metadata_shells()?;
    for (index, node) in graph.nodes.iter().enumerate() {
        if matches!(node, GraphNode::Activation(_)) {
            decoder.activation(index)?;
        }
    }
    // Allocate values bottom-up. `value` below only sees already-rooted
    // children, so even the portable maximum depth never becomes Rust stack
    // depth. Activation edges are separate: their shells were rooted above.
    for reference in value_order {
        decoder.build_value(reference, 0)?;
    }
    decoder.fill_selected_handle_shells()?;
    decoder.fill_selected_metadata_shells()?;
    for root in &graph.roots {
        decoder.build_value(*root, 0)?;
    }
    for (index, node) in graph.nodes.iter().enumerate() {
        let GraphNode::Activation(activation) = node else {
            continue;
        };
        let environment = decoder.environments[index]
            .as_ref()
            .ok_or_else(|| invalid("allocated activation is missing"))?
            .clone();
        for binding in &activation.bindings {
            if let Some(reference) = binding.value {
                let mut leases = Vec::new();
                let value = decoder.value(reference, 0, &mut leases)?;
                crate::call::lexical::write_owned_binding(&environment, binding.binding, value)
                    .map_err(|error| invalid(error.to_string()))?;
                drop(leases);
            }
        }
    }
    let mut roots = crate::execution::RootedValuesBuilder::with_capacity(graph.roots.len());
    for root in &graph.roots {
        let mut leases = Vec::new();
        let value = decoder.value(*root, 0, &mut leases)?;
        roots
            .push(value)
            .map_err(|error| invalid(error.to_string()))?;
        drop(leases);
    }
    let rooted = roots.finish().map_err(|error| invalid(error.to_string()))?;
    drop(decoder);
    Ok(rooted)
}

struct Decoder<'a> {
    graph: &'a PayloadGraph,
    bindings: &'a HashMap<GraphBoundOwner, BoundFunctionBinding>,
    proofs: &'a HashMap<GraphBoundOwner, GraphCallableProof>,
    selected_classes:
        &'a HashMap<GraphBoundOwner, std::rc::Rc<runmat_value::SelectedObjectClassFacts>>,
    limits: ValueLimits,
    source_issuer: Option<&'a super::GraphSourceIssuer>,
    environments: Vec<Option<LexicalEnvironment>>,
    environment_roots: Vec<ExplicitRoot>,
    values: Vec<Option<crate::execution::RootedValue>>,
    selected_handles: Vec<Option<runmat_gc::GcHandle>>,
    selected_handle_roots: Vec<ExplicitRoot>,
    selected_metadata: Vec<Option<runmat_gc::GcHandle>>,
    selected_metadata_roots: Vec<ExplicitRoot>,
    remaining_uses: Vec<usize>,
    visiting: Vec<bool>,
}

fn value_use_counts(graph: &PayloadGraph) -> Result<Vec<usize>, ValueCodecError> {
    let mut counts = vec![0usize; graph.nodes.len()];
    let mut count = |reference: GraphNodeRef| -> Result<(), ValueCodecError> {
        let use_count = &mut counts[reference.0 as usize];
        *use_count = use_count
            .checked_add(1)
            .ok_or_else(|| invalid("value reference count overflows"))?;
        Ok(())
    };
    for root in &graph.roots {
        count(*root)?;
    }
    for node in &graph.nodes {
        match node {
            GraphNode::Leaf { .. } => {}
            GraphNode::Cell { values, .. } => {
                for value in values {
                    count(*value)?;
                }
            }
            GraphNode::Struct { fields } => {
                for field in fields {
                    count(field.value)?;
                }
            }
            GraphNode::Registered(value) => {
                for field in &value.fields {
                    count(field.value)?;
                }
            }
            GraphNode::SelectedObject(value) => {
                for field in &value.fields {
                    count(field.value)?;
                }
            }
            GraphNode::SelectedObjectArray(value) => {
                for reference in value.values.iter().chain(&value.empty_provenance) {
                    count(*reference)?;
                }
            }
            GraphNode::SelectedDynamicPropertyMetadata(_) => {}
            GraphNode::Callable(value) => {
                for capture in &value.captures {
                    count(*capture)?;
                }
            }
            GraphNode::Activation(value) => {
                for binding in &value.bindings {
                    if let Some(reference) = binding.value {
                        count(reference)?;
                    }
                }
            }
        }
    }
    Ok(counts)
}

fn selected_dynamic_properties(
    selected: &runmat_execution::value::GraphSelectedObject,
) -> Result<Option<HashMap<String, runmat_value::DynamicPropertyDef>>, ValueCodecError> {
    if selected.dynamic_properties.is_empty() {
        return Ok(None);
    }
    let mut dynamic = HashMap::with_capacity(selected.dynamic_properties.len());
    for property in &selected.dynamic_properties {
        if property.name != property.definition_name {
            return Err(invalid("selected dynamic property names disagree"));
        }
        if dynamic
            .insert(
                property.name.clone(),
                runmat_value::DynamicPropertyDef {
                    name: property.definition_name.clone(),
                    defining_class: property.defining_class.clone(),
                    metadata_handle: None,
                    get_access: property.get_access,
                    set_access: property.set_access,
                    dependent: property.dependent,
                    hidden: property.hidden,
                    transient: property.transient,
                    non_copyable: property.non_copyable,
                    abort_set: property.abort_set,
                    set_observable: property.set_observable,
                    get_observable: property.get_observable,
                    description: property.description.clone(),
                },
            )
            .is_some()
        {
            return Err(invalid("selected dynamic property repeats"));
        }
    }
    Ok(Some(dynamic))
}

impl Decoder<'_> {
    fn allocate_selected_metadata_shells(&mut self) -> Result<(), ValueCodecError> {
        for (index, node) in self.graph.nodes.iter().enumerate() {
            if !matches!(node, GraphNode::SelectedDynamicPropertyMetadata(_)) {
                continue;
            }
            let class = crate::builtins::introspection::dynamicprops::DYNAMIC_PROPERTY_CLASS;
            let root = runmat_gc::gc_allocate_rooted(Value::Object(ObjectInstance::new(class)))
                .map_err(|error| invalid(error.to_string()))?;
            let value = Value::HandleObject(runmat_value::HandleRef {
                class_name: class.into(),
                target: root.handle(),
                valid: true,
            });
            self.selected_metadata[index] = Some(root.handle());
            self.values[index] = Some(
                crate::execution::RootedValue::new(value)
                    .map_err(|error| invalid(error.to_string()))?,
            );
            self.selected_metadata_roots.push(root);
        }
        Ok(())
    }

    fn allocate_selected_handle_shells(&mut self) -> Result<(), ValueCodecError> {
        for (index, node) in self.graph.nodes.iter().enumerate() {
            let GraphNode::SelectedObject(selected) = node else {
                continue;
            };
            if selected.kind != GraphSelectedObjectKind::Handle {
                continue;
            }
            let facts = std::rc::Rc::clone(
                self.selected_classes
                    .get(&selected.constructor)
                    .ok_or_else(|| invalid("selected handle class product is absent"))?,
            );
            let object = ObjectInstance::new(selected.class.clone())
                .with_selected_class(facts)
                .ok_or_else(|| invalid("selected handle class product differs"))?;
            let root = runmat_gc::gc_allocate_rooted(Value::Object(object))
                .map_err(|error| invalid(error.to_string()))?;
            let value = Value::HandleObject(runmat_value::HandleRef {
                class_name: selected.class.clone(),
                target: root.handle(),
                valid: true,
            });
            self.selected_handles[index] = Some(root.handle());
            self.values[index] = Some(
                crate::execution::RootedValue::new(value)
                    .map_err(|error| invalid(error.to_string()))?,
            );
            self.selected_handle_roots.push(root);
        }
        Ok(())
    }

    fn fill_selected_handle_shells(&mut self) -> Result<(), ValueCodecError> {
        for index in 0..self.graph.nodes.len() {
            let GraphNode::SelectedObject(selected) = &self.graph.nodes[index] else {
                continue;
            };
            if selected.kind != GraphSelectedObjectKind::Handle {
                continue;
            }
            let handle = self.selected_handles[index]
                .ok_or_else(|| invalid("selected handle shell is absent"))?;
            let mut fields = HashMap::with_capacity(selected.fields.len());
            let mut leases = Vec::new();
            for field in &selected.fields {
                if fields
                    .insert(field.name.clone(), self.value(field.value, 1, &mut leases)?)
                    .is_some()
                {
                    return Err(invalid("selected handle field repeats"));
                }
            }
            let mut dynamic = selected_dynamic_properties(selected)?;
            for reference in &selected.dynamic_metadata {
                let GraphNode::SelectedDynamicPropertyMetadata(metadata) =
                    &self.graph.nodes[reference.0 as usize]
                else {
                    return Err(invalid("selected metadata changed after validation"));
                };
                let metadata_handle = self.selected_metadata[reference.0 as usize]
                    .ok_or_else(|| invalid("selected metadata shell is absent"))?;
                let definition = dynamic
                    .as_mut()
                    .and_then(|dynamic| dynamic.get_mut(&metadata.property))
                    .ok_or_else(|| invalid("selected metadata property is absent"))?;
                definition.metadata_handle = Some(metadata_handle);
                runmat_gc::gc_record_handle_write(
                    &handle,
                    &Value::HandleObject(runmat_value::HandleRef {
                        class_name:
                            crate::builtins::introspection::dynamicprops::DYNAMIC_PROPERTY_CLASS
                                .into(),
                        target: metadata_handle,
                        valid: true,
                    }),
                );
            }
            runmat_gc::gc_with_value_mut(&handle, |stored| match stored {
                Value::Object(object) if object.class_name == selected.class => {
                    object.properties = fields;
                    object.dynamic_properties = dynamic.map(Box::new);
                    Ok(())
                }
                _ => Err(invalid("selected handle shell changed before field fill")),
            })
            .map_err(|error| invalid(error.to_string()))??;
            drop(leases);
        }
        Ok(())
    }

    fn fill_selected_metadata_shells(&mut self) -> Result<(), ValueCodecError> {
        for (index, node) in self.graph.nodes.iter().enumerate() {
            let GraphNode::SelectedDynamicPropertyMetadata(metadata) = node else {
                continue;
            };
            let metadata_handle = self.selected_metadata[index]
                .ok_or_else(|| invalid("selected metadata shell is absent"))?;
            let receiver = self.selected_handles[metadata.receiver.0 as usize]
                .ok_or_else(|| invalid("selected metadata receiver shell is absent"))?;
            let receiver_class = match &self.graph.nodes[metadata.receiver.0 as usize] {
                GraphNode::SelectedObject(object) => object.class.clone(),
                _ => {
                    return Err(invalid(
                        "selected metadata receiver changed after validation",
                    ))
                }
            };
            let definition = runmat_gc::gc_with_value(&receiver, |stored| {
                let Value::Object(object) = stored else {
                    return None;
                };
                object.dynamic_property(&metadata.property).cloned()
            })
            .map_err(|error| invalid(error.to_string()))?
            .ok_or_else(|| invalid("selected metadata definition is absent"))?;
            if definition.metadata_handle != Some(metadata_handle) {
                return Err(invalid("selected metadata differs from its receiver"));
            }
            let object = crate::builtins::introspection::dynamicprops::selected_metadata_object(
                &definition,
                runmat_value::HandleRef {
                    class_name: receiver_class,
                    target: receiver,
                    valid: true,
                },
            );
            runmat_gc::gc_record_handle_write(
                &metadata_handle,
                &Value::HandleObject(runmat_value::HandleRef {
                    class_name: object.class_name.clone(),
                    target: receiver,
                    valid: true,
                }),
            );
            runmat_gc::gc_with_value_mut(&metadata_handle, |stored| match stored {
                Value::Object(existing)
                    if existing.class_name
                        == crate::builtins::introspection::dynamicprops::DYNAMIC_PROPERTY_CLASS
                            .owned() =>
                {
                    *existing = object;
                    Ok(())
                }
                _ => Err(invalid("selected metadata shell changed before fill")),
            })
            .map_err(|error| invalid(error.to_string()))??;
        }
        Ok(())
    }

    fn child_depth(&self, depth: u16) -> Result<u16, ValueCodecError> {
        depth
            .checked_add(1)
            .filter(|next| *next <= self.limits.max_depth)
            .ok_or_else(|| invalid("value construction exceeds its depth limit"))
    }

    fn activation(&mut self, index: usize) -> Result<LexicalEnvironment, ValueCodecError> {
        if let Some(environment) = &self.environments[index] {
            return Ok(environment.clone());
        }
        let mut path = Vec::new();
        let mut cursor = Some(index);
        while let Some(current) = cursor {
            if self.environments[current].is_some() {
                break;
            }
            let GraphNode::Activation(activation) = &self.graph.nodes[current] else {
                return Err(invalid("activation reference has the wrong node kind"));
            };
            path.push(current);
            cursor = activation.parent.map(|parent| parent.0 as usize);
        }
        for current in path.into_iter().rev() {
            let GraphNode::Activation(activation) = &self.graph.nodes[current] else {
                unreachable!("validated activation path")
            };
            let parent = activation
                .parent
                .map(|reference| {
                    self.environments[reference.0 as usize]
                        .as_ref()
                        .cloned()
                        .ok_or_else(|| invalid("activation parent was not allocated"))
                })
                .transpose()?;
            let (environment, roots) = crate::call::lexical::allocate_environment(parent)
                .map_err(|error| invalid(error.to_string()))?;
            self.environment_roots.extend(roots);
            for binding in &activation.bindings {
                if !crate::call::lexical::reserve_owned_binding(&environment, binding.binding)
                    .map_err(|error| invalid(error.to_string()))?
                {
                    return Err(invalid("duplicate reconstructed lexical binding"));
                }
            }
            self.environments[current] = Some(environment);
        }
        self.environments[index]
            .as_ref()
            .cloned()
            .ok_or_else(|| invalid("activation was not allocated"))
    }

    fn value(
        &mut self,
        reference: GraphNodeRef,
        depth: u16,
        leases: &mut Vec<crate::execution::RootedValueLease>,
    ) -> Result<Value, ValueCodecError> {
        let index = reference.0 as usize;
        if self.remaining_uses[index] == 0 {
            return Err(invalid("reconstructed value was consumed too many times"));
        }
        self.build_value(reference, depth)?;
        let use_count = &mut self.remaining_uses[index];
        *use_count -= 1;
        if *use_count == 0 {
            let rooted = self.values[index]
                .take()
                .ok_or_else(|| invalid("reconstructed value is missing"))?;
            let (value, lease) = rooted.into_value_and_lease();
            leases.push(lease);
            Ok(value)
        } else {
            self.values[index]
                .as_ref()
                .map(|rooted| rooted.value().clone())
                .ok_or_else(|| invalid("reconstructed value is missing"))
        }
    }

    fn build_value(&mut self, reference: GraphNodeRef, depth: u16) -> Result<(), ValueCodecError> {
        if depth > self.limits.max_depth {
            return Err(invalid("value construction exceeds its depth limit"));
        }
        let index = reference.0 as usize;
        // A selected handle or metadata shell may already have been consumed
        // by its final incoming value edge before the topological construction
        // walk reaches this node. Its separately rooted GC shell remains the
        // one exact object to fill; never try to construct a second value.
        if self.values[index].is_some()
            || self.selected_handles[index].is_some()
            || self.selected_metadata[index].is_some()
        {
            return Ok(());
        }
        if std::mem::replace(&mut self.visiting[index], true) {
            return Err(invalid("value-only cycle cannot be reconstructed"));
        }
        if let GraphNode::Leaf { value } = &self.graph.nodes[index] {
            let rooted = match self.source_issuer {
                Some(issuer) => super::super::decode::decode_inline_value_owned_with_frame_sources(
                    value,
                    self.limits,
                    issuer,
                ),
                None => {
                    super::super::decode::decode_inline_value_owned_with_limits(value, self.limits)
                }
            }
            .map_err(|error| invalid(error.to_string()))?;
            self.values[index] = Some(rooted);
            self.visiting[index] = false;
            return Ok(());
        }
        let graph = self.graph;
        let node = &graph.nodes[index];
        let child_depth = self.child_depth(depth).ok();
        let mut child_leases = Vec::new();
        let value = match node {
            GraphNode::Leaf { .. } => unreachable!("leaf returned with its root"),
            GraphNode::Cell { shape, values } => {
                let data = values
                    .iter()
                    .map(|value| {
                        self.value(
                            *value,
                            child_depth.ok_or_else(|| {
                                invalid("value construction exceeds its depth limit")
                            })?,
                            &mut child_leases,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let shape = shape
                    .iter()
                    .map(|extent| {
                        usize::try_from(*extent)
                            .map_err(|_| invalid("cell shape is not host-addressable"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Value::Cell(
                    CellArray::new_with_shape(data, shape)
                        .map_err(|error| invalid(error.to_string()))?,
                )
            }
            GraphNode::Struct { fields } => {
                let mut output = StructValue::new();
                for field in fields {
                    output.insert(
                        field.name.clone(),
                        self.value(
                            field.value,
                            child_depth.ok_or_else(|| {
                                invalid("value construction exceeds its depth limit")
                            })?,
                            &mut child_leases,
                        )?,
                    );
                }
                Value::Struct(output)
            }
            GraphNode::Registered(registered) => {
                if registered.kind != GraphRegisteredKind::ImmutableValueClass {
                    return Err(invalid("registered graph class has no runtime codec"));
                }
                let class = runmat_types::ClassIdentity::new(&registered.type_identity)
                    .map_err(|error| invalid(error.to_string()))?;
                let mut object = ObjectInstance::new(class);
                for field in &registered.fields {
                    object.properties.insert(
                        field.name.clone(),
                        self.value(
                            field.value,
                            child_depth.ok_or_else(|| {
                                invalid("value construction exceeds its depth limit")
                            })?,
                            &mut child_leases,
                        )?,
                    );
                }
                crate::parallel::codistributor::validate_definition(&object)
                    .map_err(|error| invalid(error.to_string()))?;
                Value::Object(object)
            }
            GraphNode::SelectedObject(selected) => {
                if selected.kind != GraphSelectedObjectKind::Value {
                    return Err(invalid("selected handle was not preallocated"));
                }
                let facts = std::rc::Rc::clone(
                    self.selected_classes
                        .get(&selected.constructor)
                        .ok_or_else(|| invalid("selected object class product is absent"))?,
                );
                let mut object = ObjectInstance::new(selected.class.clone())
                    .with_selected_class(facts)
                    .ok_or_else(|| invalid("selected object class product differs"))?;
                for field in &selected.fields {
                    object.properties.insert(
                        field.name.clone(),
                        self.value(
                            field.value,
                            child_depth.ok_or_else(|| {
                                invalid("value construction exceeds its depth limit")
                            })?,
                            &mut child_leases,
                        )?,
                    );
                }
                object.dynamic_properties = selected_dynamic_properties(selected)?.map(Box::new);
                Value::Object(object)
            }
            GraphNode::SelectedObjectArray(selected) => {
                let data = selected
                    .values
                    .iter()
                    .map(|reference| {
                        self.value(
                            *reference,
                            child_depth.ok_or_else(|| {
                                invalid("value construction exceeds its depth limit")
                            })?,
                            &mut child_leases,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let shape = selected
                    .shape
                    .iter()
                    .map(|extent| {
                        usize::try_from(*extent)
                            .map_err(|_| invalid("object-array shape is not host-addressable"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut array = runmat_value::ObjectArray::new(selected.class.clone(), data, shape)
                    .map_err(|error| invalid(error.to_string()))?;
                if array.is_empty() {
                    let mut provenance = Vec::with_capacity(selected.empty_provenance.len());
                    for reference in &selected.empty_provenance {
                        let GraphNode::Callable(anchor) = &self.graph.nodes[reference.0 as usize]
                        else {
                            return Err(invalid("selected array anchor changed after validation"));
                        };
                        let GraphCallableOwner::Bound(owner) = anchor.owner else {
                            return Err(invalid("selected array anchor has no selected owner"));
                        };
                        let facts = std::rc::Rc::clone(
                            self.selected_classes.get(&owner).ok_or_else(|| {
                                invalid("selected array provenance lacks an admitted class product")
                            })?,
                        );
                        if facts.class() != &selected.class
                            || facts.is_handle_class()
                                != (selected.kind == GraphSelectedObjectKind::Handle)
                        {
                            return Err(invalid("selected array provenance changes class or kind"));
                        }
                        let value = self.value(
                            *reference,
                            child_depth.ok_or_else(|| {
                                invalid("value construction exceeds its depth limit")
                            })?,
                            &mut child_leases,
                        )?;
                        let Value::BoundFunctionHandle(handle) = value else {
                            return Err(invalid("selected array anchor is not a bound handle"));
                        };
                        if handle.function != facts.constructor().function
                            || handle.authority != facts.constructor().authority
                        {
                            return Err(invalid("selected array anchor differs from its class"));
                        }
                        provenance.push(facts);
                    }
                    array = array
                        .with_selected_provenance(provenance)
                        .map_err(|error| invalid(error.to_string()))?;
                    if selected.includes_unselected {
                        array = array.with_unselected_provenance();
                    }
                }
                Value::ObjectArray(array)
            }
            GraphNode::SelectedDynamicPropertyMetadata(_) => {
                return Err(invalid("selected metadata was not preallocated"));
            }
            GraphNode::Callable(callable) => {
                let source_file = match callable.owner {
                    GraphCallableOwner::Bound(origin) => self
                        .proofs
                        .get(&origin)
                        .ok_or_else(|| invalid("bound callable has no selected source proof"))?
                        .issued_source_file
                        .clone(),
                    _ => callable.source_file.clone(),
                };
                let captures = callable
                    .captures
                    .iter()
                    .map(|child| {
                        self.value(
                            *child,
                            child_depth.ok_or_else(|| {
                                invalid("value construction exceeds its depth limit")
                            })?,
                            &mut child_leases,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let binding = match callable.owner {
                    GraphCallableOwner::Bound(origin) => {
                        Some(self.bindings.get(&origin).cloned().ok_or_else(|| {
                            invalid("bound unit owner is absent from selected definitions")
                        })?)
                    }
                    _ => None,
                };
                let environment = callable
                    .activation
                    .map(|reference| {
                        self.environments[reference.0 as usize]
                            .as_ref()
                            .cloned()
                            .ok_or_else(|| invalid("callable activation was not allocated"))
                    })
                    .transpose()?;
                match callable.form {
                    CallableForm::Closure => {
                        Value::Closure(Closure::from_data(runmat_value::ClosureData {
                            function_name: callable.qualified_name.clone(),
                            binding,
                            source_role: callable.source_role,
                            source_file: source_file.clone().map(String::into_boxed_str),
                            captures,
                            lexical_environment: environment,
                        }))
                    }
                    CallableForm::Handle => match callable.owner {
                        GraphCallableOwner::Workspace => {
                            Value::function_handle(callable.qualified_name.clone())
                        }
                        GraphCallableOwner::External => {
                            Value::external_function_handle(callable.qualified_name.clone())
                        }
                        GraphCallableOwner::Method => {
                            Value::method_function_handle(callable.qualified_name.clone())
                        }
                        GraphCallableOwner::Bound(_) => {
                            let binding =
                                binding.ok_or_else(|| invalid("bound handle lacks admission"))?;
                            Value::bound_function_handle(
                                callable.qualified_name.clone(),
                                binding.function,
                                binding.authority,
                                callable.source_role,
                                source_file,
                            )
                        }
                    },
                }
            }
            GraphNode::Activation(_) => return Err(invalid("activation is not a value")),
        };
        let rooted = crate::execution::RootedValue::new(value)
            .map_err(|error| invalid(error.to_string()))?;
        self.values[index] = Some(rooted);
        drop(child_leases);
        self.visiting[index] = false;
        Ok(())
    }
}
