//! Incremental reader of the existing canonical sessionless GraphNode JSON.
//! The actual graph is the sole reconstructed tree; source/static/class and
//! value-limit admission remain with the original artifact/VM codec owners.
use super::*;
use runmat_execution::value::*;
mod budget;
mod exception;
mod leaves;

struct Budget {
    limits: ValueLimits,
    usage: ValueLimitBudget,
}
impl Reader {
    pub(crate) async fn graph(
        &mut self,
        limits: ValueLimits,
        c: &mut Cooperation<'_>,
    ) -> BuiltinResult<PayloadGraph> {
        let mut budget = Budget {
            limits,
            usage: ValueLimitBudget::new(limits),
        };
        self.token(b'{')?;
        self.key("roots", true)?;
        let roots = budget.references(self, c).await?;
        self.key("nodes", false)?;
        self.token(b'[')?;
        let mut nodes = Vec::new();
        while !self.is(b']') {
            budget.nodes(1)?;
            if !nodes.is_empty() {
                self.token(b',')?;
            }
            nodes
                .try_reserve(1)
                .map_err(|source| with_source("selected graph node allocation failed", source))?;
            nodes.push(node(self, &mut budget, c).await?);
        }
        self.token(b']')?;
        self.token(b'}')?;
        Ok(PayloadGraph { roots, nodes })
    }
}

async fn node(
    reader: &mut Reader,
    budget: &mut Budget,
    c: &mut Cooperation<'_>,
) -> BuiltinResult<GraphNode> {
    reader.token(b'{')?;
    reader.key("node", true)?;
    let kind = reader.string_limited(64, c).await?;
    let value = match kind.as_str() {
        "leaf" => {
            reader.key("value", false)?;
            GraphNode::Leaf {
                value: leaves::payload(reader, budget, c).await?,
            }
        }
        "cell" => {
            reader.key("shape", false)?;
            let shape = budget.shape(reader, c).await?;
            reader.key("values", false)?;
            let values = budget.references(reader, c).await?;
            GraphNode::Cell { shape, values }
        }
        "struct" => {
            reader.key("fields", false)?;
            GraphNode::Struct {
                fields: fields(reader, budget, c).await?,
            }
        }
        "registered" => {
            reader.key("kind", false)?;
            let kind = reader.metadata(64, c).await?;
            reader.key("type_identity", false)?;
            let type_identity = budget.text(reader, c).await?;
            reader.key("schema_version", false)?;
            let schema_version = reader.metadata(10, c).await?;
            reader.key("fields", false)?;
            let fields = fields(reader, budget, c).await?;
            GraphNode::Registered(GraphRegistered {
                kind,
                type_identity,
                schema_version,
                fields,
            })
        }
        "selected_object" => {
            budget.backing(40)?;
            reader.key("constructor", false)?;
            let constructor = reader.metadata(256, c).await?;
            reader.key("class", false)?;
            let class = class(reader, budget, c).await?;
            reader.key("kind", false)?;
            let kind = reader.metadata(32, c).await?;
            reader.key("fields", false)?;
            let fields = fields(reader, budget, c).await?;
            reader.key("dynamic_properties", false)?;
            let dynamic_properties = dynamics(reader, budget, c).await?;
            reader.key("dynamic_metadata", false)?;
            let dynamic_metadata = budget.references(reader, c).await?;
            GraphNode::SelectedObject(GraphSelectedObject {
                constructor,
                class,
                kind,
                fields,
                dynamic_properties,
                dynamic_metadata,
            })
        }
        "selected_object_array" => {
            reader.key("class", false)?;
            let class = class(reader, budget, c).await?;
            reader.key("kind", false)?;
            let kind = reader.metadata(32, c).await?;
            reader.key("shape", false)?;
            let shape = budget.shape(reader, c).await?;
            reader.key("values", false)?;
            let values = budget.references(reader, c).await?;
            reader.key("empty_provenance", false)?;
            let empty_provenance = budget.references(reader, c).await?;
            reader.key("includes_unselected", false)?;
            let includes_unselected = reader.metadata(5, c).await?;
            GraphNode::SelectedObjectArray(GraphSelectedObjectArray {
                class,
                kind,
                shape,
                values,
                empty_provenance,
                includes_unselected,
            })
        }
        "selected_dynamic_property_metadata" => {
            budget.elements(1)?;
            budget.backing(12)?;
            reader.key("receiver", false)?;
            let receiver = reader.metadata(20, c).await?;
            reader.key("property", false)?;
            let property = budget.text(reader, c).await?;
            GraphNode::SelectedDynamicPropertyMetadata(GraphSelectedDynamicPropertyMetadata {
                receiver,
                property,
            })
        }
        "callable" => {
            reader.key("owner", false)?;
            let owner: GraphCallableOwner = reader.metadata(384, c).await?;
            budget.backing(if matches!(owner, GraphCallableOwner::Bound(_)) {
                39
            } else {
                1
            })?;
            reader.key("qualified_name", false)?;
            let qualified_name = budget.text(reader, c).await?;
            reader.key("callable_digest", false)?;
            let callable_digest = reader.metadata(128, c).await?;
            reader.key("form", false)?;
            let form = reader.metadata(32, c).await?;
            reader.key("source_role", false)?;
            let source_role = reader.metadata(64, c).await?;
            reader.key("source_file", false)?;
            let source_file = budget.optional_text(reader, c).await?;
            reader.key("captures", false)?;
            let captures = budget.references(reader, c).await?;
            reader.key("activation", false)?;
            let activation: Option<GraphNodeRef> = reader.metadata(20, c).await?;
            if activation.is_some() {
                budget.backing(4)?;
            }
            GraphNode::Callable(GraphCallable {
                owner,
                qualified_name,
                callable_digest,
                form,
                source_role,
                source_file,
                captures,
                activation,
            })
        }
        "activation" => {
            budget.backing(36)?;
            reader.key("owner", false)?;
            let owner = reader.metadata(256, c).await?;
            reader.key("parent", false)?;
            let parent: Option<GraphNodeRef> = reader.metadata(20, c).await?;
            if parent.is_some() {
                budget.backing(4)?;
            }
            reader.key("bindings", false)?;
            reader.token(b'[')?;
            let mut bindings = Vec::new();
            while !reader.is(b']') {
                budget.elements(1)?;
                budget.backing(8)?;
                if !bindings.is_empty() {
                    reader.token(b',')?;
                }
                let binding: GraphBinding = reader.metadata(128, c).await?;
                if binding.value.is_some() {
                    budget.backing(4)?;
                }
                bindings.try_reserve(1).map_err(|source| {
                    with_source("selected activation allocation failed", source)
                })?;
                bindings.push(binding);
            }
            reader.token(b']')?;
            GraphNode::Activation(GraphActivation {
                owner,
                parent,
                bindings,
            })
        }
        _ => return Err(error("unknown selected graph node kind")),
    };
    reader.token(b'}')?;
    Ok(value)
}

async fn fields(
    reader: &mut Reader,
    budget: &mut Budget,
    c: &mut Cooperation<'_>,
) -> BuiltinResult<Vec<GraphField>> {
    reader.token(b'[')?;
    let mut fields = Vec::new();
    while !reader.is(b']') {
        if fields.len() as u64 >= u64::from(budget.limits.max_fields) {
            return Err(error("selected graph field extent exceeded"));
        }
        budget.elements(1)?;
        budget.backing(4)?;
        if !fields.is_empty() {
            reader.token(b',')?;
        }
        reader.token(b'{')?;
        reader.key("name", true)?;
        let name = budget.text(reader, c).await?;
        reader.key("value", false)?;
        let value = reader.metadata(20, c).await?;
        reader.token(b'}')?;
        fields
            .try_reserve(1)
            .map_err(|source| with_source("selected graph field allocation failed", source))?;
        fields.push(GraphField { name, value });
    }
    reader.token(b']')?;
    Ok(fields)
}
async fn class(
    reader: &mut Reader,
    budget: &mut Budget,
    c: &mut Cooperation<'_>,
) -> BuiltinResult<runmat_types::ClassIdentity> {
    runmat_types::ClassIdentity::new(budget.text(reader, c).await?)
        .map_err(|source| with_source("invalid selected graph class identity", source))
}
async fn dynamics(
    reader: &mut Reader,
    budget: &mut Budget,
    c: &mut Cooperation<'_>,
) -> BuiltinResult<Vec<DurableDynamicProperty>> {
    reader.token(b'[')?;
    let mut values = Vec::new();
    while !reader.is(b']') {
        if values.len() as u64 >= u64::from(budget.limits.max_fields) {
            return Err(error("selected dynamic-property extent exceeded"));
        }
        budget.elements(1)?;
        if !values.is_empty() {
            reader.token(b',')?;
        }
        reader.token(b'{')?;
        reader.key("name", true)?;
        let name = budget.text(reader, c).await?;
        reader.key("definition_name", false)?;
        let definition_name = budget.text(reader, c).await?;
        reader.key("defining_class", false)?;
        let defining_class = class(reader, budget, c).await?;
        reader.key("get_access", false)?;
        let get_access = reader.metadata(32, c).await?;
        reader.key("set_access", false)?;
        let set_access = reader.metadata(32, c).await?;
        reader.key("dependent", false)?;
        let dependent = reader.metadata(5, c).await?;
        reader.key("hidden", false)?;
        let hidden = reader.metadata(5, c).await?;
        reader.key("transient", false)?;
        let transient = reader.metadata(5, c).await?;
        reader.key("non_copyable", false)?;
        let non_copyable = reader.metadata(5, c).await?;
        reader.key("abort_set", false)?;
        let abort_set = reader.metadata(5, c).await?;
        reader.key("set_observable", false)?;
        let set_observable = reader.metadata(5, c).await?;
        reader.key("get_observable", false)?;
        let get_observable = reader.metadata(5, c).await?;
        reader.key("description", false)?;
        let description = budget.text(reader, c).await?;
        reader.token(b'}')?;
        values
            .try_reserve(1)
            .map_err(|source| with_source("selected dynamic-property allocation failed", source))?;
        values.push(DurableDynamicProperty {
            name,
            definition_name,
            defining_class,
            get_access,
            set_access,
            dependent,
            hidden,
            transient,
            non_copyable,
            abort_set,
            set_observable,
            get_observable,
            description,
        });
    }
    reader.token(b']')?;
    Ok(values)
}
