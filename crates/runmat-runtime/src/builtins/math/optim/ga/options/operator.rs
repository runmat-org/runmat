//! Named built-in selectors and user-selected semantic calls are distinct.
//! A handle is never reclassified by its display name or function spelling.
use super::super::{invalid, ScalarCooperation, POLICY};
use super::Owner;
use crate::BuiltinResult;
use runmat_value::Value;

pub(in super::super) enum Operator<K> {
    Builtin {
        kind: K,
        specification: Value,
        owner: Owner,
    },
    Callback {
        specification: Value,
        selected: Option<Box<crate::call::descriptor::RepeatableCallable>>,
        owner: Owner,
    },
}
impl<K: Copy> Operator<K> {
    pub fn default(kind: K) -> Self {
        Self::Builtin {
            kind,
            specification: super::super::inputs::raw::empty(),
            owner: Owner::Solver,
        }
    }
    pub fn parse(value: &Value, parse: fn(&str) -> Option<K>) -> BuiltinResult<Self> {
        Self::parse_for(Owner::Solver, value, parse)
    }
    pub fn parse_for(
        owner: Owner,
        value: &Value,
        parse: fn(&str) -> Option<K>,
    ) -> BuiltinResult<Self> {
        let target = if let Value::Cell(values) = value {
            values
                .data()
                .first()
                .ok_or_else(|| owner.invalid("operator cell must contain a function"))?
        } else {
            value
        };
        if let Some(name) = crate::builtins::common::validation::value_text_scalar_string(target) {
            let kind = parse(&name)
                .ok_or_else(|| owner.invalid("unrecognized built-in genetic operator"))?;
            Ok(Self::Builtin {
                kind,
                specification: value.clone(),
                owner,
            })
        } else if super::super::inputs::raw::callable(target) {
            Ok(Self::Callback {
                specification: value.clone(),
                selected: None,
                owner,
            })
        } else {
            Err(owner
                .invalid("operator must be a built-in name, function handle, or parameter cell"))
        }
    }
    pub fn kind(&self) -> Option<K> {
        match self {
            Self::Builtin { kind, .. } => Some(*kind),
            Self::Callback { .. } => None,
        }
    }
    pub fn public_value(&self, name: fn(K) -> &'static str) -> Value {
        match self {
            Self::Builtin {
                kind,
                specification,
                ..
            } if super::super::inputs::raw::unspecified(specification) => Value::from(name(*kind)),
            Self::Builtin { specification, .. } | Self::Callback { specification, .. } => {
                specification.clone()
            }
        }
    }
    pub fn extra(&self) -> &[Value] {
        let value = match self {
            Self::Builtin { specification, .. } | Self::Callback { specification, .. } => {
                specification
            }
        };
        match value {
            Value::Cell(values) => &values.data()[1..],
            _ => &[],
        }
    }
    pub fn custom(&self) -> Option<&Value> {
        let Self::Callback { specification, .. } = self else {
            return None;
        };
        Some(if let Value::Cell(values) = specification {
            &values.data()[0]
        } else {
            specification
        })
    }
    pub fn argument_count(&self, min: usize, max: usize) -> BuiltinResult<()> {
        if self.custom().is_some() || (min..=max).contains(&self.extra().len()) {
            Ok(())
        } else {
            Err(self
                .owner()
                .invalid("wrong number of built-in genetic operator parameters"))
        }
    }
    fn owner(&self) -> Owner {
        match self {
            Self::Builtin { owner, .. } | Self::Callback { owner, .. } => *owner,
        }
    }
    pub fn bind(&mut self, cooperation: &ScalarCooperation) -> BuiltinResult<()> {
        let selected = self
            .custom()
            .map(|target| super::super::callbacks::select(target, cooperation))
            .transpose()?;
        if let Self::Callback { selected: slot, .. } = self {
            *slot = selected.map(Box::new);
        }
        Ok(())
    }
    pub async fn call_with<I: super::super::callbacks::Invocation>(
        &self,
        arguments: impl IntoIterator<Item = Value>,
        outputs: usize,
        invocation: &mut I,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<std::ops::ControlFlow<I::Stop, crate::execution::RootedValueSequence>> {
        let Self::Callback {
            selected: Some(target),
            ..
        } = self
        else {
            return Err(super::super::callbacks::unavailable(
                "custom genetic operator has not been bound at its active invocation boundary",
            ));
        };
        let arguments = arguments.into_iter();
        let (minimum, maximum) = arguments.size_hint();
        let count = maximum
            .filter(|&count| count == minimum)
            .and_then(|n| n.checked_add(self.extra().len()))
            .ok_or_else(|| invalid("custom operator argument extent overflow"))?;
        let mut values = POLICY.capacity(count, cooperation.runtime())?;
        values.extend(arguments);
        for argument in self.extra() {
            values.push(argument.clone());
            cooperation.completed(1).await?;
        }
        let arguments =
            crate::call::BuiltinArguments::from_borrowed(&values, cooperation.runtime())?;
        let result = invocation
            .call(
                target.as_ref(),
                arguments,
                outputs,
                super::super::callbacks::Event::Work,
                cooperation,
            )
            .await;
        drop(values);
        result
    }
}
