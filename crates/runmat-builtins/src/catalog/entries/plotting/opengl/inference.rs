use super::command::{OpenGlCommand, OpenGlSaveMode};
use crate::{
    catalog::inference::{argument_error, finish_fixed, known_invalid_text_scalar},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-OPENGL-ARITY",
            "expected at most a command and saved mode",
            2,
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        if known_invalid_text_scalar(argument) {
            diagnostics.push(argument_error(
                "RM-CATALOG-OPENGL-TEXT",
                "commands must be scalar text",
                index,
            ));
        }
    }
    let text = request.literals.literal_string_at(0);
    let command = text.as_deref().and_then(OpenGlCommand::parse);
    if text.is_some() && command.is_none() {
        diagnostics.push(argument_error(
            "RM-CATALOG-OPENGL-COMMAND",
            "unsupported renderer compatibility command",
            0,
        ));
    }
    match command {
        Some(OpenGlCommand::Save) => {
            if request
                .literals
                .literal_string_at(1)
                .is_some_and(|mode| OpenGlSaveMode::parse(&mode).is_none())
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-OPENGL-MODE",
                    "unsupported saved mode",
                    1,
                ));
            }
        }
        Some(_) if request.arguments.len() > 1 => diagnostics.push(argument_error(
            "RM-CATALOG-OPENGL-ARITY",
            "this command accepts no extra arguments",
            1,
        )),
        _ => {}
    }
    let output = if command == Some(OpenGlCommand::Report) {
        let mut fields = std::collections::BTreeMap::new();
        for name in [
            "Version",
            "Vendor",
            "Renderer",
            "HardwareSupportLevel",
            "Extensions",
        ] {
            fields.insert(name.into(), ValueFact::scalar(ValueKindFact::String));
        }
        for name in [
            "SupportsGraphicsSmoothing",
            "SupportsDepthPeelTransparency",
            "SupportsAlignVertexCenters",
            "Software",
        ] {
            fields.insert(name.into(), ValueFact::scalar(ValueKindFact::Logical));
        }
        fields.insert(
            "MaxTextureSize".into(),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        );
        ValueFact::proven(
            ValueKindFact::Struct(StructFact::scalar(fields, true)),
            ShapeFact::Scalar,
            StorageFact::Dense,
        )
    } else if request.arguments.is_empty() || command.is_some() {
        ValueFact::scalar(ValueKindFact::String)
    } else {
        // An unknown command can select string status or a scalar information struct.
        let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
        fact.shape = ShapeFact::Scalar;
        fact.residency = ResidencyFact::Host;
        fact
    };
    finish_fixed(entry, request, output, diagnostics)
}
