use runmat_value::{ObjectInstance, Value};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use super::*;
use crate::class_registry::declaration::{
    immutable_properties, register_value_class, static_methods,
};

pub(super) const VALUE_FIELD: &[&str] = &["value"];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TypedStudyValue<T> {
    pub(super) value: T,
}

pub(super) fn to_typed_value<T: Serialize>(
    builtin: &'static str,
    class_name: &'static str,
    value: T,
) -> BuiltinResult<Value> {
    serializable_to_object(
        builtin,
        &ERROR_INTERNAL,
        class_name,
        &TypedStudyValue { value },
    )
}

pub(super) fn from_typed_value<T: DeserializeOwned>(
    builtin: &'static str,
    class_name: &'static str,
    value: &Value,
) -> BuiltinResult<T> {
    let wrapped: TypedStudyValue<T> = object_payload(builtin, value, class_name)?;
    Ok(wrapped.value)
}

/// One immutable study-value class: its portable codec schema and its static
/// constructor methods.
pub(super) struct ImmutableStudyClass {
    pub(super) class_name: &'static str,
    pub(super) type_identity: &'static str,
    pub(super) fields: &'static [&'static str],
    pub(super) methods: &'static [(&'static str, &'static str)],
}

/// Register the process-global portable codec schemas once per process.
pub(super) fn register_immutable_schemas(classes: &[ImmutableStudyClass]) {
    for class in classes {
        register_immutable_schema(class.class_name, class.type_identity, class.fields);
    }
}

/// Register the runtime classes in the current class registry. Callers guard
/// this with a `ClassRegistration` so every fresh registry receives them.
pub(super) fn register_immutable_runtime_classes(classes: &[ImmutableStudyClass]) {
    for class in classes {
        register_value_class(
            class.class_name,
            immutable_properties(class.fields),
            static_methods(class.methods.iter().copied()),
        );
    }
}

fn register_immutable_schema(
    class_name: &'static str,
    type_identity: &'static str,
    fields: &'static [&'static str],
) {
    crate::execution::value_codec::register_immutable_value_class(
        crate::execution::value_codec::ImmutableValueClassSchema {
            class_name,
            type_identity,
            schema_version: 1,
            allowed_fields: fields,
            required_fields: fields,
            validate: validate_immutable_value,
        },
    )
    .unwrap_or_else(|error| panic!("failed to register {class_name}: {error}"));
}

fn validate_immutable_value(object: &ObjectInstance) -> Result<(), String> {
    if object
        .properties
        .keys()
        .any(|name| name.starts_with("__runmat_"))
    {
        return Err(format!(
            "immutable study value `{}` contains an opaque runtime property",
            object.class_name
        ));
    }
    Ok(())
}

pub(super) fn object_or_cell<T: DeserializeOwned>(
    builtin: &'static str,
    value: &Value,
    class_name: &'static str,
) -> BuiltinResult<Vec<T>> {
    match value {
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|item| object_payload(builtin, item, class_name))
            .collect(),
        Value::Object(_) => Ok(vec![object_payload(builtin, value, class_name)?]),
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected `{class_name}` or a cell array of `{class_name}`; got {other:?}"),
        )),
    }
}

pub(super) fn nonempty_id(
    builtin: &'static str,
    field: &str,
    value: String,
) -> BuiltinResult<String> {
    if value.trim().is_empty() {
        Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{field} must not be empty"),
        ))
    } else {
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    /// One static constructor per FEA value-class family. Each family registers
    /// its runtime classes per class registry, so they must resolve on a thread
    /// whose registry is fresh even after another thread consumed the
    /// process-once codec schemas.
    const FAMILY_PROBES: &[(&str, &str)] = &[
        ("fea.StepObjectRef", "fea.StepObjectRef.constraint"),
        (
            "fea.ActivationLifecycle",
            "fea.ActivationLifecycle.activate",
        ),
        ("fea.StepObjectActivation", "fea.StepObjectActivation.new"),
        ("fea.ComputeBackend", "fea.ComputeBackend.cpu"),
        (
            "fea.FormulationRequirement",
            "fea.FormulationRequirement.experimental",
        ),
        ("fea.SolverOptions", "fea.SolverOptions.defaults"),
        ("fea.ExecutionStep", "fea.ExecutionStep.new"),
        ("fea.ExecutionPlan", "fea.ExecutionPlan.defaults"),
        ("fea.AccuracyTarget", "fea.AccuracyTarget.numericalResult"),
        ("fea.AccuracyTolerance", "fea.AccuracyTolerance.relative"),
        (
            "fea.NumericalEquivalencePolicy",
            "fea.NumericalEquivalencePolicy.deterministic",
        ),
        (
            "fea.AccuracyResourceCeiling",
            "fea.AccuracyResourceCeiling.new",
        ),
        ("fea.AccuracyCriterion", "fea.AccuracyCriterion.new"),
        ("fea.AccuracyPlan", "fea.AccuracyPlan.experimentalBaseline"),
        ("fea.ErrorContribution", "fea.ErrorContribution.algebraic"),
        (
            "fea.RequiredAccuracyEvidence",
            "fea.RequiredAccuracyEvidence.convergenceStudy",
        ),
        (
            "fea.UnavailableEvidencePolicy",
            "fea.UnavailableEvidencePolicy.fail",
        ),
        ("fea.AccuracyLimitAction", "fea.AccuracyLimitAction.refine"),
        (
            "fea.ObservationQuantity",
            "fea.ObservationQuantity.displacement",
        ),
        ("fea.FieldLocation", "fea.FieldLocation.node"),
        ("fea.SpatialReduction", "fea.SpatialReduction.moment"),
        ("fea.SeriesReduction", "fea.SeriesReduction.mean"),
        ("fea.DiagnosticKind", "fea.DiagnosticKind.convergence"),
        ("fea.StateKind", "fea.StateKind.thermal"),
        (
            "fea.TopologyTransitionPolicy",
            "fea.TopologyTransitionPolicy.requireIdentical",
        ),
        (
            "fea.IndependentDomain",
            "fea.IndependentDomain.staticDomain",
        ),
        ("fea.ObservationTarget", "fea.ObservationTarget.wholeModel"),
        ("fea.SignalDomain", "fea.SignalDomain.time"),
        ("fea.SignalSampling", "fea.SignalSampling.uniform"),
        ("fea.SignalInterpolation", "fea.SignalInterpolation.linear"),
        ("fea.SignalKnot", "fea.SignalKnot.jump"),
        (
            "fea.InitialConditionTarget",
            "fea.InitialConditionTarget.wholeModel",
        ),
        ("fea.ReferencePoint", "fea.ReferencePoint.fromCoordinates"),
        ("fea.PeriodicTransform", "fea.PeriodicTransform.identity"),
        ("fea.Direction", "fea.Direction.vector"),
        ("fea.PeriodicPairing", "fea.PeriodicPairing.matching"),
        ("fea.CoordinateFrame", "fea.CoordinateFrame.globalFrame"),
        ("fea.SpectrumAxis", "fea.SpectrumAxis.frequencyHertz"),
        ("fea.ResponseSpectrum", "fea.ResponseSpectrum.new"),
    ];

    fn assert_families_resolve() {
        super::super::ensure_value_classes_registered();
        for (class_name, builtin) in FAMILY_PROBES {
            let class = crate::class_registry::get_class(
                &runmat_types::ClassIdentity::new(*class_name).expect("class identity"),
            )
            .unwrap_or_else(|| panic!("{class_name} is registered in this class registry"));
            assert!(
                class
                    .methods
                    .values()
                    .any(|method| method.is_static && method.function_name == *builtin),
                "{builtin} is a static method of {class_name}"
            );
        }
    }

    #[test]
    fn every_fea_value_family_registers_in_each_fresh_class_registry() {
        assert_families_resolve();
        for _ in 0..2 {
            std::thread::spawn(assert_families_resolve)
                .join()
                .expect("a fresh registry resolves every FEA value family");
        }
    }
}
