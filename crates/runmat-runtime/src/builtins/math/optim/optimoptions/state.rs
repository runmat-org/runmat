//! Ordered normalized scalar metadata, before its one final Struct publication.
use super::*;

/// Validated controls before contextual text publication. Preflight can use
/// the same decoding without allocating a discarded output generation.
pub(super) enum PreparedOptionValue<'a> {
    Number(f64),
    Integer(runmat_value::IntValue),
    Display(OptimDisplay),
    Algorithm(OptimAlgorithm),
    Logical(bool),
    Callback(&'a Value),
    Hessian(runmat_builtins::OptimHessianSource),
    Subproblem(runmat_builtins::OptimSubproblemAlgorithm),
    Difference(runmat_builtins::OptimDifferenceType),
    Merit(runmat_builtins::OptimMeritFunction),
}

impl PreparedOptionValue<'_> {
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
            Self::Callback(value) => Ok(value.clone()),
            Self::Hessian(runmat_builtins::OptimHessianSource::Objective) => {
                text(runtime, "objective")
            }
            Self::Hessian(runmat_builtins::OptimHessianSource::FiniteDifference) => {
                super::super::common::output_callback::empty(runtime)
            }
            Self::Subproblem(value) => text(runtime, value.name()),
            Self::Difference(value) => text(runtime, value.name()),
            Self::Merit(value) => text(runtime, value.name()),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum OptionKey {
    Solver,
    Field(OptimOptionField),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum OptionOrigin {
    Default,
    Imported,
    Explicit,
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
    entries: [Option<(OptionKey, Value, OptionOrigin)>; OptimOptionField::ALL.len() + 1],
}

impl OptionState {
    pub(super) fn new() -> Self {
        Self {
            entries: std::array::from_fn(|_| None),
        }
    }

    pub(super) fn set(&mut self, field: OptimOptionField, value: Value) {
        self.set_with_origin(field, value, OptionOrigin::Default);
    }

    pub(super) fn set_with_origin(
        &mut self,
        field: OptimOptionField,
        value: Value,
        origin: OptionOrigin,
    ) {
        self.set_key(OptionKey::Field(field), value, origin);
    }
    pub(super) fn origin(&self, field: OptimOptionField) -> Option<OptionOrigin> {
        self.entries
            .iter()
            .flatten()
            .find_map(|(key, _, origin)| (*key == OptionKey::Field(field)).then_some(*origin))
    }
    pub(super) fn remove(&mut self, field: OptimOptionField) {
        for entry in &mut self.entries {
            if entry
                .as_ref()
                .is_some_and(|(key, _, _)| *key == OptionKey::Field(field))
            {
                *entry = None;
                return;
            }
        }
    }

    pub(super) fn set_solver(&mut self, value: Value) {
        self.set_key(OptionKey::Solver, value, OptionOrigin::Default);
    }

    fn set_key(&mut self, key: OptionKey, value: Value, origin: OptionOrigin) {
        let index = self
            .entries
            .iter()
            .position(|entry| {
                entry
                    .as_ref()
                    .is_some_and(|(previous, _, _)| *previous == key)
            })
            .or_else(|| self.entries.iter().position(Option::is_none))
            .expect("typed option fields and one solver fit exactly");
        self.entries[index] = Some((key, value, origin));
    }

    pub(super) fn get(&self, field: OptimOptionField) -> Option<&Value> {
        self.fields()
            .find_map(|(key, value)| (key == OptionKey::Field(field)).then_some(value))
    }

    pub(super) fn fields(&self) -> impl Iterator<Item = (OptionKey, &Value)> + Clone {
        self.entries
            .iter()
            .filter_map(|entry| entry.as_ref().map(|(key, value, _)| (*key, value)))
    }

    pub(super) fn finish(
        &self,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<crate::execution::RootedValue> {
        let solver = self
            .fields()
            .find_map(|(key, value)| {
                (key == OptionKey::Solver)
                    .then(|| {
                        crate::builtins::common::validation::value_text_scalar_string(value)
                            .and_then(|name| Solver::parse(&name))
                    })
                    .flatten()
            })
            .unwrap_or(Solver::Generic);
        crate::execution::structure::construct_struct_from_fields_in_optional_context(
            runtime,
            self.fields().map(|(key, value)| {
                (
                    match key {
                        OptionKey::Solver => key.name(),
                        OptionKey::Field(field) => field.name_for_solver(solver),
                    },
                    value,
                )
            }),
        )
    }
}

pub(super) fn text(
    runtime: Option<&crate::context::RuntimeContext>,
    value: &str,
) -> BuiltinResult<Value> {
    crate::call::argument_text::copy_str(value, runtime).map(|text| Value::String(text.into()))
}
