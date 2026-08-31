use super::{argument_error, finish_fixed, numeric_kind};
use crate::{BuiltinCatalogEntry, LogarithmBase};
use runmat_types::{
    infer_call, AliasFact, CallContract, CallInference, CallRequest, ContiguityFact, DynamicReason,
    LayoutFact, LiteralValue, MutationFact, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, StorageFact, ValueFact, ValueKindFact, ViewFact,
};

pub(super) fn infer_exp(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_exponential(request, entry, ExponentialKind::Exp)
}

pub(super) fn infer_expm1(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_exponential(request, entry, ExponentialKind::Expm1)
}

pub(super) fn infer_log1p(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOG1P-ARITY",
            "log1p requires exactly one input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOG1P-ARITY",
            "log1p accepts exactly one input",
            1,
        ));
    }
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOG1P-SPARSE",
            "log1p does not currently accept sparse input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let mut output = input.clone();
    let literal_domain = request
        .literals
        .literal_args
        .first()
        .and_then(log1p_literal_domain);
    let mut changes_class = false;
    match &input.kind {
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) if !matches!(class, NumericClass::Double | NumericClass::Single) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-LOG1P-COMPLEX-INTEGER",
                "log1p does not accept complex fixed-width integer input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) => {
            output.kind = numeric_kind(*class, NumericDomain::Complex);
            materialize_output(&mut output);
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }) => {
            let output_class = if matches!(class, NumericClass::Double | NumericClass::Single) {
                *class
            } else {
                changes_class = true;
                NumericClass::Double
            };
            let domain = literal_domain.or_else(|| {
                matches!(
                    class,
                    NumericClass::UInt8
                        | NumericClass::UInt16
                        | NumericClass::UInt32
                        | NumericClass::UInt64
                )
                .then_some(NumericDomain::Real)
            });
            if let Some(domain) = domain {
                output.kind = numeric_kind(output_class, domain);
                materialize_output(&mut output);
            } else {
                preserve_shape_on_dynamic_input(&mut output);
            }
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            changes_class = true;
            materialize_output(&mut output);
        }
        ValueKindFact::Object(_) | ValueKindFact::Unknown => {
            preserve_shape_on_dynamic_input(&mut output);
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-LOG1P-INPUT",
                "log1p requires numeric, logical, or character input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    if changes_class && matches!(output.residency, ResidencyFact::Device { .. }) {
        output.residency = ResidencyFact::Unknown;
    }
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn infer_logarithm(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    base: LogarithmBase,
) -> CallInference {
    let name = match base {
        LogarithmBase::Natural => "log",
        LogarithmBase::Common => "log10",
    };
    let (output, diagnostics) =
        infer_logarithm_value(request, name, matches!(base, LogarithmBase::Natural));
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn infer_log2(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let requested = request.outputs.requested.known_count();
    if requested == Some(2) {
        return infer_log2_dissection(request, entry);
    }

    let (value_output, mut diagnostics) = infer_logarithm_value(request, "log2", false);
    let exponent_output = log2_dissection_output(
        request.arguments.first(),
        &mut diagnostics,
        requested.is_none(),
    );
    let mut contract = CallContract::fixed(vec![value_output, exponent_output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

fn infer_log2_dissection(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = log2_dissection_output(request.arguments.first(), &mut diagnostics, true);
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOG2-ARITY",
            "log2 accepts exactly one input",
            1,
        ));
    }
    let mut contract = CallContract::fixed(vec![output.clone(), output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

fn log2_dissection_output(
    input: Option<&ValueFact>,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
    diagnose_invalid: bool,
) -> ValueFact {
    let Some(input) = input else {
        if diagnose_invalid {
            diagnostics.push(argument_error(
                "RM-CATALOG-LOG2-ARITY",
                "log2 requires exactly one input",
                0,
            ));
        }
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    if matches!(input.residency, ResidencyFact::Device { .. }) {
        if diagnose_invalid {
            diagnostics.push(argument_error(
                "RM-CATALOG-LOG2-GPU-DISSECTION",
                "two-output log2 does not support GPU-resident input",
                0,
            ));
        }
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    if matches!(input.storage, StorageFact::Sparse) {
        if diagnose_invalid {
            diagnostics.push(argument_error(
                "RM-CATALOG-LOG2-SPARSE",
                "log2 does not currently accept sparse input",
                0,
            ));
        }
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }

    let mut output = input.clone();
    match &mut output.kind {
        ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Complex => {
            if diagnose_invalid {
                diagnostics.push(argument_error(
                    "RM-CATALOG-LOG2-COMPLEX-DISSECTION",
                    "two-output log2 requires real input under the current compatibility pin",
                    0,
                ));
            }
            return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric) => {
            if !matches!(numeric.class, NumericClass::Double | NumericClass::Single) {
                numeric.class = NumericClass::Double;
            }
            numeric.domain = NumericDomain::Real;
            materialize_output(&mut output);
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize_output(&mut output);
        }
        ValueKindFact::Object(object)
            if object.runtime_class.as_ref().is_some_and(|class| {
                class.is(runmat_types::standard::TABLE)
                    || class.is(runmat_types::standard::TIMETABLE)
            }) =>
        {
            object.properties.clear();
            object.properties_complete = false;
            output.alias = AliasFact::Unique;
            output.mutation = MutationFact::ValueSemantics;
        }
        ValueKindFact::Object(_) | ValueKindFact::Unknown if !diagnose_invalid => {
            preserve_shape_on_dynamic_input(&mut output);
        }
        _ => {
            if diagnose_invalid {
                diagnostics.push(argument_error(
                    "RM-CATALOG-LOG2-DISSECTION-INPUT",
                    "two-output log2 requires real single, double, or supported tabular input",
                    0,
                ));
            }
            output = ValueFact::unknown(if diagnose_invalid {
                DynamicReason::UnsupportedRepresentation
            } else {
                DynamicReason::RuntimeValue
            });
        }
    }
    output
}

fn infer_logarithm_value(
    request: &CallRequest,
    name: &str,
    accepts_symbolic: bool,
) -> (ValueFact, Vec<runmat_types::InferenceDiagnostic>) {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOGARITHM-ARITY",
            format!("{name} requires exactly one input"),
            0,
        ));
        return (ValueFact::unknown(DynamicReason::RuntimeValue), diagnostics);
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOGARITHM-ARITY",
            format!("{name} accepts exactly one input"),
            1,
        ));
    }
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOGARITHM-SPARSE",
            format!("{name} does not currently accept sparse input"),
            0,
        ));
        return (
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let mut output = input.clone();
    let literal_domain = request
        .literals
        .literal_args
        .first()
        .and_then(logarithm_literal_domain);
    let mut changes_class = false;
    match &input.kind {
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) if !matches!(class, NumericClass::Double | NumericClass::Single) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-LOGARITHM-COMPLEX-INTEGER",
                format!("{name} does not accept complex fixed-width integer input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) => {
            output.kind = numeric_kind(*class, NumericDomain::Complex);
            materialize_output(&mut output);
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }) => {
            let output_class = if matches!(class, NumericClass::Double | NumericClass::Single) {
                *class
            } else {
                changes_class = true;
                NumericClass::Double
            };
            let domain = literal_domain.or_else(|| {
                matches!(
                    class,
                    NumericClass::UInt8
                        | NumericClass::UInt16
                        | NumericClass::UInt32
                        | NumericClass::UInt64
                )
                .then_some(NumericDomain::Real)
            });
            if let Some(domain) = domain {
                output.kind = numeric_kind(output_class, domain);
                materialize_output(&mut output);
            } else {
                preserve_shape_on_dynamic_input(&mut output);
            }
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            changes_class = true;
            materialize_output(&mut output);
        }
        ValueKindFact::Symbolic if accepts_symbolic => {
            materialize_output_preserving_storage(&mut output);
        }
        ValueKindFact::Object(object)
            if object.runtime_class.as_ref().is_some_and(|class| {
                class.is(runmat_types::standard::TABLE)
                    || class.is(runmat_types::standard::TIMETABLE)
            }) =>
        {
            let ValueKindFact::Object(object) = &mut output.kind else {
                unreachable!("tabular object branch preserves object fact")
            };
            object.properties.clear();
            object.properties_complete = false;
            output.alias = AliasFact::Unique;
            output.mutation = MutationFact::ValueSemantics;
        }
        ValueKindFact::Object(_) | ValueKindFact::Unknown => {
            preserve_shape_on_dynamic_input(&mut output);
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-LOGARITHM-INPUT",
                format!(
                    "{name} requires numeric, logical, character, or supported tabular input{}",
                    if accepts_symbolic {
                        ", or a symbolic expression"
                    } else {
                        ""
                    }
                ),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    if changes_class && matches!(output.residency, ResidencyFact::Device { .. }) {
        output.residency = ResidencyFact::Unknown;
    }
    (output, diagnostics)
}

fn logarithm_literal_domain(literal: &LiteralValue) -> Option<NumericDomain> {
    match literal {
        LiteralValue::Number(value) => Some(logarithm_real_domain(*value)),
        LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => {
            text.parse().ok().map(logarithm_real_domain)
        }
        LiteralValue::Complex { .. } => Some(NumericDomain::Complex),
        LiteralValue::Bool(_) | LiteralValue::Character(_) | LiteralValue::Empty => {
            Some(NumericDomain::Real)
        }
        LiteralValue::Vector(values) => logarithm_literal_sequence_domain(values.iter()),
        LiteralValue::Matrix(rows) => logarithm_literal_sequence_domain(rows.iter().flatten()),
        LiteralValue::String(_)
        | LiteralValue::Keyword(_)
        | LiteralValue::Symbolic(_)
        | LiteralValue::Unknown => None,
    }
}

fn logarithm_literal_sequence_domain<'a>(
    values: impl Iterator<Item = &'a LiteralValue>,
) -> Option<NumericDomain> {
    let mut domain = NumericDomain::Real;
    for value in values {
        match logarithm_literal_domain(value)? {
            NumericDomain::Complex => domain = NumericDomain::Complex,
            NumericDomain::Real => {}
        }
    }
    Some(domain)
}

fn logarithm_real_domain(value: f64) -> NumericDomain {
    if value < 0.0 {
        NumericDomain::Complex
    } else {
        NumericDomain::Real
    }
}

fn log1p_literal_domain(literal: &LiteralValue) -> Option<NumericDomain> {
    match literal {
        LiteralValue::Number(value) => Some(log1p_real_domain(*value)),
        LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => {
            text.parse().ok().map(log1p_real_domain)
        }
        LiteralValue::Complex { .. } => Some(NumericDomain::Complex),
        LiteralValue::Bool(_) | LiteralValue::Character(_) | LiteralValue::Empty => {
            Some(NumericDomain::Real)
        }
        LiteralValue::Vector(values) => log1p_literal_sequence_domain(values.iter()),
        LiteralValue::Matrix(rows) => log1p_literal_sequence_domain(rows.iter().flatten()),
        LiteralValue::String(_)
        | LiteralValue::Keyword(_)
        | LiteralValue::Symbolic(_)
        | LiteralValue::Unknown => None,
    }
}

fn log1p_literal_sequence_domain<'a>(
    values: impl Iterator<Item = &'a LiteralValue>,
) -> Option<NumericDomain> {
    let mut domain = NumericDomain::Real;
    for value in values {
        match log1p_literal_domain(value)? {
            NumericDomain::Complex => domain = NumericDomain::Complex,
            NumericDomain::Real => {}
        }
    }
    Some(domain)
}

fn log1p_real_domain(value: f64) -> NumericDomain {
    if value < -1.0 {
        NumericDomain::Complex
    } else {
        NumericDomain::Real
    }
}

#[derive(Debug, Clone, Copy)]
enum ExponentialKind {
    Exp,
    Expm1,
}

impl ExponentialKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Exp => "exp",
            Self::Expm1 => "expm1",
        }
    }

    const fn accepts_symbolic(self) -> bool {
        matches!(self, Self::Exp)
    }

    const fn preserves_sparse_zeros(self) -> bool {
        matches!(self, Self::Expm1)
    }
}

fn infer_exponential(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    operation: ExponentialKind,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-EXPONENTIAL-ARITY",
            format!("{} requires exactly one input", operation.name()),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-EXPONENTIAL-ARITY",
            format!("{} accepts exactly one input", operation.name()),
            1,
        ));
    }

    let mut output = input.clone();
    let mut changes_numeric_class = false;
    match &mut output.kind {
        ValueKindFact::Numeric(numeric) => {
            if numeric.domain == NumericDomain::Complex
                && !matches!(numeric.class, NumericClass::Double | NumericClass::Single)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-EXPONENTIAL-COMPLEX-INTEGER",
                    format!(
                        "{} does not accept complex fixed-width integer input",
                        operation.name()
                    ),
                    0,
                ));
                output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
                return finish_fixed(entry, request, output, diagnostics);
            }
            if !matches!(numeric.class, NumericClass::Double | NumericClass::Single) {
                numeric.class = NumericClass::Double;
                changes_numeric_class = true;
            }
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            changes_numeric_class = true;
        }
        ValueKindFact::Symbolic if operation.accepts_symbolic() => {
            return finish_fixed(entry, request, output, diagnostics);
        }
        ValueKindFact::Object(object) => {
            if object.runtime_class.as_ref().is_some_and(|class| {
                class.is(runmat_types::standard::TABLE)
                    || class.is(runmat_types::standard::TIMETABLE)
            }) {
                object.properties.clear();
                object.properties_complete = false;
                output.alias = AliasFact::Unique;
                output.mutation = MutationFact::ValueSemantics;
            } else {
                preserve_shape_on_dynamic_input(&mut output);
            }
            return finish_fixed(entry, request, output, diagnostics);
        }
        ValueKindFact::Unknown => {
            preserve_shape_on_dynamic_input(&mut output);
            return finish_fixed(entry, request, output, diagnostics);
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-EXPONENTIAL-INPUT",
                format!(
                    "{} requires numeric, logical, character, or supported tabular input{}",
                    operation.name(),
                    if operation.accepts_symbolic() {
                        ", or a symbolic expression"
                    } else {
                        ""
                    }
                ),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
            return finish_fixed(entry, request, output, diagnostics);
        }
    }

    if matches!(output.storage, StorageFact::Sparse) {
        output.residency = ResidencyFact::Host;
        if !operation.preserves_sparse_zeros() {
            output.storage = StorageFact::Dense;
        }
    } else if changes_numeric_class && matches!(output.residency, ResidencyFact::Device { .. }) {
        output.residency = ResidencyFact::Unknown;
    }
    materialize_output_preserving_storage(&mut output);
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn infer_phase_angle(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-PHASE-ANGLE-ARITY",
            "angle requires exactly one input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-PHASE-ANGLE-ARITY",
            "angle accepts exactly one input",
            1,
        ));
    }
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-PHASE-ANGLE-SPARSE",
            "angle does not currently accept sparse input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let mut output = input.clone();
    match &mut output.kind {
        ValueKindFact::Numeric(numeric)
            if matches!(numeric.class, NumericClass::Double | NumericClass::Single) =>
        {
            numeric.domain = NumericDomain::Real;
            materialize_output(&mut output);
        }
        ValueKindFact::Unknown => preserve_shape_on_dynamic_input(&mut output),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-PHASE-ANGLE-INPUT",
                "angle requires real or complex single- or double-precision input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn infer_signum(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-SIGNUM-ARITY",
            "sign requires exactly one input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SIGNUM-ARITY",
            "sign accepts exactly one input",
            1,
        ));
    }
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SIGNUM-SPARSE",
            "sign does not currently accept sparse input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let mut output = input.clone();
    let mut changes_class = false;
    match &output.kind {
        ValueKindFact::Numeric(NumericFact { class, domain })
            if domain == &NumericDomain::Complex
                && !matches!(class, NumericClass::Double | NumericClass::Single) =>
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-SIGNUM-COMPLEX-INTEGER",
                "sign does not accept complex fixed-width integer input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(_) => materialize_output(&mut output),
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            changes_class = true;
            materialize_output(&mut output);
        }
        ValueKindFact::Unknown => preserve_shape_on_dynamic_input(&mut output),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-SIGNUM-INPUT",
                "sign requires numeric, logical, or character input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    if changes_class && matches!(output.residency, ResidencyFact::Device { .. }) {
        output.residency = ResidencyFact::Unknown;
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn materialize_output(output: &mut ValueFact) {
    output.storage = if output.is_scalar() {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    output.layout = LayoutFact::ColumnMajor;
    output.contiguity = ContiguityFact::Contiguous;
    output.view = ViewFact::Materialized;
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
}

fn materialize_output_preserving_storage(output: &mut ValueFact) {
    if matches!(output.storage, StorageFact::Sparse) {
        output.view = ViewFact::Materialized;
        output.alias = AliasFact::Unique;
        output.mutation = MutationFact::ValueSemantics;
    } else {
        materialize_output(output);
    }
}

fn preserve_shape_on_dynamic_input(output: &mut ValueFact) {
    let shape = output.shape.clone();
    *output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.shape = shape;
}
