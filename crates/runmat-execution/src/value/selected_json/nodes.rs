//! Canonical GraphNode field order; every variable-size collection/text is a
//! continuation, while fixed-size typed identities use their existing serde.
use super::*;
impl<'a> SelectedGraphJsonCursor<'a> {
    pub(super) fn node(&mut self, value: &'a GraphNode) -> Result<(), SelectedGraphJsonError> {
        match value {
            GraphNode::Leaf { value } => self.object([
                ("node", Task::Text("leaf")),
                ("value", Task::Payload(value)),
            ]),
            GraphNode::Cell { shape, values } => self.object([
                ("node", Task::Text("cell")),
                ("shape", Task::Array(Array::U64(shape))),
                ("values", Task::Array(Array::Refs(values))),
            ]),
            GraphNode::Struct { fields } => self.object([
                ("node", Task::Text("struct")),
                ("fields", Task::Array(Array::Fields(fields))),
            ]),
            GraphNode::Registered(value) => self.object([
                ("node", Task::Text("registered")),
                ("kind", Task::Atom(&value.kind)),
                ("type_identity", Task::Text(&value.type_identity)),
                ("schema_version", Task::Atom(&value.schema_version)),
                ("fields", Task::Array(Array::Fields(&value.fields))),
            ]),
            GraphNode::SelectedObject(value) => self.object([
                ("node", Task::Text("selected_object")),
                ("constructor", Task::Atom(&value.constructor)),
                ("class", Task::Text(value.class.display_name())),
                ("kind", Task::Atom(&value.kind)),
                ("fields", Task::Array(Array::Fields(&value.fields))),
                (
                    "dynamic_properties",
                    Task::Array(Array::Dynamic(&value.dynamic_properties)),
                ),
                (
                    "dynamic_metadata",
                    Task::Array(Array::Refs(&value.dynamic_metadata)),
                ),
            ]),
            GraphNode::SelectedObjectArray(value) => self.object([
                ("node", Task::Text("selected_object_array")),
                ("class", Task::Text(value.class.display_name())),
                ("kind", Task::Atom(&value.kind)),
                ("shape", Task::Array(Array::U64(&value.shape))),
                ("values", Task::Array(Array::Refs(&value.values))),
                (
                    "empty_provenance",
                    Task::Array(Array::Refs(&value.empty_provenance)),
                ),
                (
                    "includes_unselected",
                    Task::Atom(&value.includes_unselected),
                ),
            ]),
            GraphNode::SelectedDynamicPropertyMetadata(value) => self.object([
                ("node", Task::Text("selected_dynamic_property_metadata")),
                ("receiver", Task::Atom(&value.receiver)),
                ("property", Task::Text(&value.property)),
            ]),
            GraphNode::Callable(value) => self.object([
                ("node", Task::Text("callable")),
                ("owner", Task::Atom(&value.owner)),
                ("qualified_name", Task::Text(&value.qualified_name)),
                ("callable_digest", Task::Atom(&value.callable_digest)),
                ("form", Task::Atom(&value.form)),
                ("source_role", Task::Atom(&value.source_role)),
                (
                    "source_file",
                    Task::OptionalText(value.source_file.as_deref()),
                ),
                ("captures", Task::Array(Array::Refs(&value.captures))),
                ("activation", Task::Atom(&value.activation)),
            ]),
            GraphNode::Activation(value) => self.object([
                ("node", Task::Text("activation")),
                ("owner", Task::Atom(&value.owner)),
                ("parent", Task::Atom(&value.parent)),
                ("bindings", Task::Array(Array::Bindings(&value.bindings))),
            ]),
        }
    }
    pub(super) fn dynamic(
        &mut self,
        value: &'a DurableDynamicProperty,
    ) -> Result<(), SelectedGraphJsonError> {
        self.object([
            ("name", Task::Text(&value.name)),
            ("definition_name", Task::Text(&value.definition_name)),
            (
                "defining_class",
                Task::Text(value.defining_class.display_name()),
            ),
            ("get_access", Task::Atom(&value.get_access)),
            ("set_access", Task::Atom(&value.set_access)),
            ("dependent", Task::Atom(&value.dependent)),
            ("hidden", Task::Atom(&value.hidden)),
            ("transient", Task::Atom(&value.transient)),
            ("non_copyable", Task::Atom(&value.non_copyable)),
            ("abort_set", Task::Atom(&value.abort_set)),
            ("set_observable", Task::Atom(&value.set_observable)),
            ("get_observable", Task::Atom(&value.get_observable)),
            ("description", Task::Text(&value.description)),
        ])
    }
}
