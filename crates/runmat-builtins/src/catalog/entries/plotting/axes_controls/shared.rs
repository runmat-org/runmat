use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub const AXES_CONTROL_SETTER_OUTPUT_EXTENSION: crate::BuiltinExtensionDescriptor =
    crate::BuiltinExtensionDescriptor {
        id: "axes-control-setter-output",
        mode: crate::BuiltinExtensionMode::RunMatOnly,
        description: "Allow an assigned result from a graphics axes-control setter",
        error_identifier: Some("RunMat:compatibility:AxesControlSetterOutputExtension"),
    };

pub(super) fn infer_limit(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    name: &'static str,
    color_limits: bool,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-AXES-LIMIT-ARITY",
            format!("{name} accepts at most one control after an optional axes target"),
            2,
        ));
    }
    let has_target = request.arguments.len() == 2
        || request
            .arguments
            .first()
            .is_some_and(|argument| matches!(argument.kind, ValueKindFact::Object(_)));
    if has_target {
        let target = &request.arguments[0];
        let invalid = match &target.kind {
            ValueKindFact::Unknown => false,
            ValueKindFact::Numeric(numeric) => {
                numeric.domain != NumericDomain::Real || target.storage == StorageFact::Sparse
            }
            ValueKindFact::Object(_) => {
                crate::catalog::inference::known_invalid_scalar_object_handle(
                    target,
                    &[standard::AXES],
                )
            }
            _ => true,
        } || target.shape.element_count().is_some_and(|count| count != 1)
            || matches!(
                target.residency,
                ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
            );
        if invalid {
            diagnostics.push(argument_error(
                "RM-CATALOG-AXES-LIMIT-TARGET",
                format!("{name} requires a host scalar axes handle as its target"),
                0,
            ));
        }
    }
    let control_index = usize::from(has_target);
    if let Some(argument) = request.arguments.get(control_index) {
        let invalid = match argument.kind {
            ValueKindFact::Unknown => false,
            ValueKindFact::Numeric(numeric) => {
                numeric.domain == NumericDomain::Complex
                    || matches!(argument.storage, StorageFact::Sparse)
                    || argument
                        .shape
                        .element_count()
                        .is_some_and(|count| count != 2 && (has_target || count != 1))
            }
            ValueKindFact::String => {
                !matches!(argument.storage, StorageFact::Unknown | StorageFact::Scalar)
                    || crate::catalog::inference::known_invalid_text_scalar(argument)
            }
            // The existing style decoder flattens all character data, not only
            // a row. Shape alone cannot reject a potentially valid mode.
            ValueKindFact::Character => false,
            _ => true,
        };
        if invalid
            || matches!(
                argument.residency,
                ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-AXES-LIMIT-INPUT",
                format!("{name} requires a host two-element real numeric vector or mode token"),
                control_index,
            ));
        }
        if let Some(mode) = request.literals.literal_string_at(control_index) {
            let supported = if color_limits {
                matches!(mode.trim().to_ascii_lowercase().as_str(), "auto" | "manual")
            } else {
                matches!(
                    mode.trim().to_ascii_lowercase().as_str(),
                    "auto" | "manual" | "tight" | "padded" | "tickaligned" | "mode" | "method"
                )
            };
            if !supported {
                diagnostics.push(argument_error(
                    "RM-CATALOG-AXES-LIMIT-MODE",
                    format!("{name} does not support this mode"),
                    control_index,
                ));
            }
        }
        if let Some(values) = request.literals.literal_vector_at(control_index) {
            if values.len() == 2 {
                let lo = LiteralContext::numeric_from_literal(&values[0]);
                let hi = LiteralContext::numeric_from_literal(&values[1]);
                if lo
                    .zip(hi)
                    .is_some_and(|(lo, hi)| !lo.is_finite() || !hi.is_finite() || lo >= hi)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-AXES-LIMIT-ORDER",
                        format!("{name} limits must be finite and increasing"),
                        control_index,
                    ));
                }
            }
        }
    }
    let output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    let mut output = match request.literals.literal_string_at(control_index) {
        Some(mode)
            if matches!(mode.trim().to_ascii_lowercase().as_str(), "mode" | "method")
                && !color_limits =>
        {
            ValueFact::proven(
                ValueKindFact::String,
                ShapeFact::from(vec![Some(1), Some(1)]),
                StorageFact::Scalar,
            )
        }
        None if request
            .arguments
            .get(control_index)
            .is_some_and(|argument| {
                matches!(
                    argument.kind,
                    ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
                )
            })
            && !color_limits =>
        {
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
        _ => output,
    };
    output.residency = ResidencyFact::Host;
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}

#[derive(Clone, Copy)]
pub(super) enum AxisSettingForm {
    Scale,
    Angle,
    Format,
    Labels,
    Ticks,
}

pub(super) fn infer_axis_setting(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    name: &'static str,
    form: AxisSettingForm,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-AXES-CONTROL-ARITY",
            format!("{name} accepts at most two arguments"),
            2,
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        let target = index == 0
            && (request.arguments.len() == 2 || matches!(argument.kind, ValueKindFact::Object(_)));
        let invalid = if target {
            (match &argument.kind {
                ValueKindFact::Unknown => false,
                ValueKindFact::Numeric(numeric) => numeric.domain != NumericDomain::Real,
                ValueKindFact::Object(_) => {
                    crate::catalog::inference::known_invalid_scalar_object_handle(
                        argument,
                        &[standard::AXES],
                    )
                }
                _ => true,
            }) || {
                (!matches!(form, AxisSettingForm::Angle | AxisSettingForm::Format)
                    && argument
                        .shape
                        .element_count()
                        .is_some_and(|count| count != 1))
                    || matches!(argument.storage, StorageFact::Sparse)
            }
        } else {
            match argument.kind {
                ValueKindFact::Unknown => false,
                ValueKindFact::Numeric(numeric) => {
                    numeric.domain == NumericDomain::Complex
                        || matches!(form, AxisSettingForm::Scale | AxisSettingForm::Format)
                            && (request.arguments.len() != 1
                                || argument
                                    .shape
                                    .element_count()
                                    .is_some_and(|count| count != 1))
                        || matches!(form, AxisSettingForm::Labels)
                            && argument.shape.element_count().is_some_and(|count| {
                                if request.arguments.len() == 1 {
                                    !matches!(count, 0 | 1)
                                } else {
                                    count != 0
                                }
                            })
                        || matches!(form, AxisSettingForm::Angle)
                            && argument
                                .shape
                                .element_count()
                                .is_some_and(|count| count != 1)
                }
                ValueKindFact::String => {
                    matches!(form, AxisSettingForm::Angle)
                        || (!matches!(form, AxisSettingForm::Labels)
                            && (!matches!(
                                argument.storage,
                                StorageFact::Unknown | StorageFact::Scalar
                            ) || crate::catalog::inference::known_invalid_text_scalar(
                                argument,
                            )))
                }
                ValueKindFact::Character => matches!(form, AxisSettingForm::Angle),
                ValueKindFact::Cell(_) => !matches!(form, AxisSettingForm::Labels),
                _ => true,
            }
        };
        if invalid
            || matches!(
                argument.residency,
                ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-AXES-CONTROL-INPUT",
                format!("{name} requires host axes controls"),
                index,
            ));
        }
    }
    let control_index = request.arguments.len().checked_sub(1).filter(|index| {
        *index != 0 || !matches!(request.arguments[0].kind, ValueKindFact::Object(_))
    });
    if let Some(index) = control_index {
        if let Some(mode) = request.literals.literal_string_at(index) {
            let normalized = mode.trim().to_ascii_lowercase();
            let valid = match form {
                AxisSettingForm::Scale => matches!(normalized.as_str(), "linear" | "log"),
                AxisSettingForm::Angle => false,
                AxisSettingForm::Format => true,
                AxisSettingForm::Labels | AxisSettingForm::Ticks => {
                    matches!(normalized.as_str(), "auto" | "manual" | "mode")
                        || matches!(form, AxisSettingForm::Labels)
                }
            };
            if !valid {
                diagnostics.push(argument_error(
                    "RM-CATALOG-AXES-CONTROL-MODE",
                    format!("{name} does not accept this text control"),
                    index,
                ));
            }
        }
    }
    let mut output = match form {
        AxisSettingForm::Scale | AxisSettingForm::Format => {
            ValueFact::scalar(ValueKindFact::String)
        }
        AxisSettingForm::Angle => ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        AxisSettingForm::Labels => {
            let control = control_index.map(|index| &request.arguments[index]);
            let literal = control_index.and_then(|index| request.literals.literal_string_at(index));
            let mode_token = literal.as_deref().is_some_and(|mode| {
                matches!(
                    mode.trim().to_ascii_lowercase().as_str(),
                    "auto" | "manual" | "mode"
                )
            });
            // The label owner treats StringArray as label data, not a mode,
            // even for one element whose content is "auto". Unlike the shared
            // style decoder, only a row CharArray is a label-mode spelling.
            let array_labels = control.is_some_and(|argument| {
                matches!(argument.kind, ValueKindFact::String)
                    && argument.storage == StorageFact::Dense
            });
            let scalar_mode = control.is_some_and(|argument| match argument.kind {
                ValueKindFact::String => argument.storage == StorageFact::Scalar,
                ValueKindFact::Character => match &argument.shape {
                    ShapeFact::Scalar => true,
                    ShapeFact::Shaped { dims } => {
                        matches!(dims.first(), Some(DimensionFact::Known(1)))
                    }
                    _ => false,
                },
                _ => false,
            });
            if mode_token && scalar_mode {
                ValueFact::scalar(ValueKindFact::String)
            } else if mode_token && !array_labels {
                ValueFact::unknown(DynamicReason::RuntimeValue)
            } else if literal.is_some()
                || control.is_none()
                || control.is_some_and(|argument| {
                    matches!(
                        argument.kind,
                        ValueKindFact::Numeric(_) | ValueKindFact::Cell(_)
                    ) || matches!(argument.kind, ValueKindFact::String)
                        && matches!(argument.storage, StorageFact::Dense)
                })
            {
                // Successful label queries/setters return a cell row of
                // character rows. Padding depends on the selected axes ticks,
                // so a control's label count alone does not prove the result
                // extent. Unknown text may instead name a mode.
                let mut element = ValueFact::proven(
                    ValueKindFact::Character,
                    ShapeFact::from(vec![Some(1), None]),
                    StorageFact::Dense,
                );
                element.residency = ResidencyFact::Host;
                ValueFact::proven(
                    ValueKindFact::Cell(CellFact {
                        element: Box::new(element),
                        elements: Vec::new(),
                        elements_complete: false,
                    }),
                    ShapeFact::from(vec![Some(1), None]),
                    StorageFact::Dense,
                )
            } else {
                ValueFact::unknown(DynamicReason::RuntimeValue)
            }
        }
        AxisSettingForm::Ticks => {
            let control = control_index.map(|index| &request.arguments[index]);
            if control.is_some_and(|argument| {
                matches!(
                    argument.kind,
                    ValueKindFact::String | ValueKindFact::Character
                )
            }) {
                ValueFact::scalar(ValueKindFact::String)
            } else if control.is_none()
                || control
                    .is_some_and(|argument| matches!(argument.kind, ValueKindFact::Numeric(_)))
            {
                // A single numeric scalar may be an existing axes query or a
                // one-tick setter. Both return a host double row, but its length
                // is not proven from that ambiguous control alone.
                let length = control
                    .and_then(|argument| argument.shape.element_count())
                    .filter(|count| *count != 1);
                ValueFact::proven(
                    ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real,
                    }),
                    ShapeFact::from(vec![Some(1), length]),
                    StorageFact::Dense,
                )
            } else {
                let mut value = ValueFact::unknown(DynamicReason::RuntimeValue);
                value.residency = ResidencyFact::Host;
                value
            }
        }
    };
    output.residency = ResidencyFact::Host;
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}
