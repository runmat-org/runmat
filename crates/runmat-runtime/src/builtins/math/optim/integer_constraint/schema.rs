//! The displayed value properties are the only domain-object authority.
use crate::execution::value_codec::{ValidationWorkspaceOwner, ValueCodecError};
use runmat_value::{ObjectInstance, Value};
pub(crate) const CLASS: runmat_types::StaticClassIdentity =
    runmat_types::standard::INTEGER_CONSTRAINT;
pub(crate) const TYPE_IDENTITY: &str = "runmat.optim.integer-constraint";
pub(crate) const FIELDS: &[&str] = &runmat_builtins::IntegerConstraintField::NAMES;
pub(crate) use runmat_builtins::IntegerConstraintField as Property;

/// Borrow the current generation, never an index cache from an earlier edit.
pub(crate) struct View<'a> {
    object: &'a ObjectInstance,
}

/// Hold the admitted COW generation itself, not a mirror of its indices.
/// No mutable access is exposed; later property publication cannot edit it.
pub(crate) struct ValidatedConstraint {
    object: ObjectInstance,
}
impl ValidatedConstraint {
    pub fn admit(
        object: &ObjectInstance,
        workspace: &ValidationWorkspaceOwner,
    ) -> Result<Self, ValueCodecError> {
        View::new(object, workspace)?;
        Ok(Self {
            object: object.clone(),
        })
    }
    pub fn indices(&self, key: Property) -> Result<Indices<'_>, String> {
        View {
            object: &self.object,
        }
        .indices(key)
    }
}
impl<'a> View<'a> {
    pub fn new(
        object: &'a ObjectInstance,
        workspace: &ValidationWorkspaceOwner,
    ) -> Result<Self, ValueCodecError> {
        validate_with_owner(object, workspace)?;
        Ok(Self { object })
    }
    pub fn indices(&self, key: Property) -> Result<Indices<'a>, String> {
        self.object
            .properties()
            .get(key.name())
            .and_then(Indices::new)
            .ok_or_else(|| {
                format!(
                    "IntegerConstraint.{} must be a full double vector",
                    key.name()
                )
            })
    }
}
fn structure(object: &ObjectInstance) -> Result<View<'_>, String> {
    if !object.class_name().is(CLASS)
        || object.has_native_payload()
        || object.selected_class().is_some()
        || object.dynamic_property_count() != 0
        || object.properties().len() != FIELDS.len()
    {
        return Err("expected a plain IntegerConstraint value with its three properties".into());
    }
    Ok(View { object })
}
pub(crate) fn validate_structure(object: &ObjectInstance) -> Result<(), String> {
    let view = structure(object)?;
    for key in Property::ALL {
        for index in view.indices(key)?.iter() {
            if !index.is_finite() || index < 1.0 || index.fract() != 0.0 {
                return Err(format!(
                    "{} must contain finite positive integer indices",
                    key.name()
                ));
            }
        }
    }
    Ok(())
}
pub(crate) fn scratch_len(object: &ObjectInstance) -> Result<usize, String> {
    let view = structure(object)?;
    Property::ALL.into_iter().try_fold(0usize, |count, key| {
        count
            .checked_add(view.indices(key)?.len())
            .ok_or_else(|| "IntegerConstraint index count overflow".into())
    })
}
/// Only caller-admitted temporary backing is sorted; displayed order is untouched.
pub(crate) fn validate_indices(object: &ObjectInstance, scratch: &mut [f64]) -> Result<(), String> {
    let view = structure(object)?;
    let mut cursor = 0;
    for key in Property::ALL {
        for index in view.indices(key)?.iter() {
            let slot = scratch
                .get_mut(cursor)
                .ok_or_else(|| "IntegerConstraint scratch is too short".to_owned())?;
            *slot = index;
            cursor += 1;
        }
    }
    if cursor != scratch.len() {
        return Err("IntegerConstraint scratch length mismatch".into());
    }
    scratch.sort_unstable_by(f64::total_cmp);
    if scratch.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(
            "IntegerConstraint indices must be distinct within and across properties".into(),
        );
    }
    Ok(())
}
fn invalid(message: String) -> ValueCodecError {
    ValueCodecError::Invalid {
        path: "$".into(),
        message,
    }
}
pub(crate) fn validate_with_owner(
    object: &ObjectInstance,
    workspace: &ValidationWorkspaceOwner,
) -> Result<(), ValueCodecError> {
    validate_structure(object).map_err(invalid)?;
    let len = scratch_len(object).map_err(invalid)?;
    if len == 0 {
        return validate_indices(object, &mut []).map_err(invalid);
    }
    workspace
        .with_f64_scratch(len, |scratch| validate_indices(object, scratch))
        .map_err(|source| ValueCodecError::Allocation {
            path: "$".into(),
            source,
        })?
        .map_err(invalid)
}
pub(crate) fn validate(
    object: &ObjectInstance,
    runtime: Option<&crate::context::RuntimeContext>,
) -> Result<(), ValueCodecError> {
    validate_with_owner(object, &ValidationWorkspaceOwner::for_runtime(runtime))
}

#[derive(Clone, Copy)]
pub(crate) struct Indices<'a> {
    scalar: Option<f64>,
    values: &'a [f64],
}
impl<'a> Indices<'a> {
    pub fn new(value: &'a Value) -> Option<Self> {
        match value {
            Value::Num(value) => Some(Self {
                scalar: Some(*value),
                values: &[],
            }),
            Value::Tensor(tensor)
                if tensor.is_empty()
                    || tensor.shape().len() <= 2 && (tensor.rows() == 1 || tensor.cols() == 1) =>
            {
                Some(Self {
                    scalar: None,
                    values: tensor.as_f64_slice()?,
                })
            }
            _ => None,
        }
    }
    pub fn len(self) -> usize {
        self.values.len() + usize::from(self.scalar.is_some())
    }
    pub fn iter(self) -> impl Iterator<Item = f64> + 'a {
        self.scalar.into_iter().chain(self.values.iter().copied())
    }
}
