//! Ordered normalized scalar metadata, before its one final Struct publication.
use super::*;

/// Validated controls before contextual text publication. Preflight can use
/// the same decoding without allocating a discarded output generation.
pub(super) enum PreparedOptionValue {
    Number(f64),
    Integer(runmat_value::IntValue),
    Display(OptimDisplay),
    Algorithm(OptimAlgorithm),
    Logical(bool),
}

impl PreparedOptionValue {
    pub(super) fn into_value(
        self,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<Value> {
        match self {
            Self::Number(value) => Ok(Value::Num(value)),
            Self::Integer(value) => Ok(Value::Int(value)),
            Self::Display(value) => text(runtime, value.name()),
            Self::Algorithm(value) => text(runtime, value.name()),
            Self::Logical(value) => Ok(Value::Bool(value)),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum OptionKey {
    Solver,
    Field(OptimOptionField),
}

impl OptionKey {
    fn name(self) -> &'static str {
        match self {
            Self::Solver => "Solver",
            Self::Field(field) => field.name(),
        }
    }
}

pub(super) struct OptionState {
    entries: [Option<(OptionKey, Value)>; 8],
}

impl OptionState {
    pub(super) fn new() -> Self {
        Self {
            entries: std::array::from_fn(|_| None),
        }
    }

    pub(super) fn set(&mut self, field: OptimOptionField, value: Value) {
        self.set_key(OptionKey::Field(field), value);
    }

    pub(super) fn set_solver(&mut self, value: Value) {
        self.set_key(OptionKey::Solver, value);
    }

    fn set_key(&mut self, key: OptionKey, value: Value) {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.as_ref().is_some_and(|(previous, _)| *previous == key))
            .or_else(|| self.entries.iter().position(Option::is_none))
            .expect("seven typed option fields and one solver fit exactly");
        self.entries[index] = Some((key, value));
    }

    pub(super) fn get(&self, field: OptimOptionField) -> Option<&Value> {
        self.fields()
            .find_map(|(key, value)| (key == OptionKey::Field(field)).then_some(value))
    }

    pub(super) fn fields(&self) -> impl Iterator<Item = (OptionKey, &Value)> + Clone {
        self.entries
            .iter()
            .filter_map(|entry| entry.as_ref().map(|(key, value)| (*key, value)))
    }

    pub(super) fn finish(
        &self,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<crate::execution::RootedValue> {
        crate::execution::structure::construct_struct_from_fields_in_optional_context(
            runtime,
            self.fields().map(|(key, value)| (key.name(), value)),
        )
    }
}

pub(super) fn text(
    runtime: Option<&crate::context::RuntimeContext>,
    value: &str,
) -> BuiltinResult<Value> {
    crate::call::argument_text::copy_str(value, runtime).map(Value::String)
}
