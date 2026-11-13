//! Fixed public options are staged on the stack and published once. Setting
//! an option never mutates a published Struct or builds an intermediate map.
use super::*;

pub(super) struct Options {
    fields: [(WebOptionField, Value); 11],
}
impl Options {
    pub(super) fn new() -> Self {
        Self {
            fields: [
                (WebOptionField::ContentType, Value::from("auto")),
                (WebOptionField::Timeout, Value::Num(DEFAULT_TIMEOUT_SECONDS)),
                (
                    WebOptionField::HeaderFields,
                    Value::Struct(StructValue::new()),
                ),
                (WebOptionField::UserAgent, Value::from("")),
                (WebOptionField::Username, Value::from("")),
                (WebOptionField::Password, Value::from("")),
                (WebOptionField::KeyName, Value::from("")),
                (WebOptionField::KeyValue, Value::from("")),
                (WebOptionField::RequestMethod, Value::from("auto")),
                (WebOptionField::MediaType, Value::from("auto")),
                (
                    WebOptionField::QueryParameters,
                    Value::Struct(StructValue::new()),
                ),
            ],
        }
    }
    pub(super) fn set(&mut self, field: WebOptionField, value: Value) {
        let (_, slot) = self
            .fields
            .iter_mut()
            .find(|(key, _)| *key == field)
            .expect("every public option has a fixed slot");
        *slot = value;
    }
    pub(super) fn finish(
        self,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<crate::execution::RootedValue> {
        crate::execution::structure::construct_struct_from_fields_in_optional_context(
            runtime,
            self.fields
                .iter()
                .map(|(key, value)| (key.canonical_name(), value)),
        )
    }
}
