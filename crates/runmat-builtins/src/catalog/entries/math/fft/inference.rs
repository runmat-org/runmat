//! Static facts for the FFT family. Runtime option parsing remains authoritative.
use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Transform {
    Fft,
    Fft2,
    Fftn,
    Ifft,
    Ifft2,
    Ifftn,
    Fftshift,
    Ifftshift,
}

impl Transform {
    fn shift(self) -> bool {
        matches!(self, Self::Fftshift | Self::Ifftshift)
    }

    fn two_dimensional(self) -> bool {
        matches!(self, Self::Fft2 | Self::Ifft2)
    }

    fn multidimensional(self) -> bool {
        matches!(self, Self::Fftn | Self::Ifftn)
    }

    fn inverse(self) -> bool {
        matches!(self, Self::Ifft | Self::Ifft2 | Self::Ifftn)
    }

    fn maximum_arity(self) -> usize {
        match self {
            Self::Fft | Self::Fft2 => 3,
            Self::Ifft | Self::Ifft2 => 4,
            Self::Ifftn => 3,
            Self::Fftn | Self::Fftshift | Self::Ifftshift => 2,
        }
    }
}

pub(super) fn infer(
    transform: Transform,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() || request.arguments.len() > transform.maximum_arity() {
        diagnostics.push(argument_error(
            "RM-CATALOG-FFT-ARITY",
            format!(
                "{} received an invalid number of arguments",
                entry.identity.name
            ),
            request.arguments.len().saturating_sub(1),
        ));
    }
    let options = parse_options(transform, request, &mut diagnostics);
    let output = match request.arguments.first() {
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
        Some(input) => {
            let supported = matches!(
                input.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
            );
            if !supported || matches!(input.storage, StorageFact::Sparse | StorageFact::Opaque) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FFT-INPUT",
                    format!(
                        "{} requires dense numeric or logical input",
                        entry.identity.name
                    ),
                    0,
                ));
                ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
            } else {
                output_fact(transform, request, input, options)
            }
        }
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.dynamic_outputs = request.outputs.requested.known_count().is_none();
    inferred.diagnostics.extend(diagnostics);
    inferred
}

#[derive(Clone, Copy)]
struct Options {
    control_end: usize,
    symmetry: Option<bool>,
    uncertain_symmetry: bool,
}

fn parse_options(
    transform: Transform,
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Options {
    let mut control_end = request.arguments.len();
    let mut symmetry = None;
    let mut uncertain_symmetry = false;
    if transform.inverse() && control_end > 1 {
        let last = control_end - 1;
        if let Some(text) = literal_text(request, last) {
            symmetry = match text.trim().to_ascii_lowercase().as_str() {
                "symmetric" => Some(true),
                "nonsymmetric" => Some(false),
                _ => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FFT-SYMFLAG",
                        "unknown inverse FFT symmetry flag",
                        last,
                    ));
                    None
                }
            };
            control_end -= 1;
        } else if matches!(
            request.arguments[last].kind,
            ValueKindFact::String | ValueKindFact::Character
        ) {
            uncertain_symmetry = true;
            control_end -= 1;
        } else if matches!(request.arguments[last].kind, ValueKindFact::Unknown) {
            // This value may be a final flag or a numeric control. Neither
            // domain nor control geometry can be proven until dispatch.
            uncertain_symmetry = true;
        }
    }
    for index in 1..request.arguments.len() {
        if let Some(text) = literal_text(request, index) {
            if index + 1 != request.arguments.len() || !transform.inverse() {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FFT-SYMFLAG",
                    "symmetry flag must appear once at the end of an inverse FFT call",
                    index,
                ));
            } else if !text.trim().eq_ignore_ascii_case("symmetric")
                && !text.trim().eq_ignore_ascii_case("nonsymmetric")
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FFT-SYMFLAG",
                    "unknown inverse FFT symmetry flag",
                    index,
                ));
            }
        }
    }
    for index in 1..control_end {
        let fact = &request.arguments[index];
        let literal = request.literals.literal_args.get(index);
        let vector_control =
            (transform.two_dimensional() || transform.multidimensional() || transform.shift())
                && index == 1
                && control_end == 2;
        if matches!(
            fact.kind,
            ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Cell(_)
                | ValueKindFact::Struct(_)
                | ValueKindFact::Object(_)
        ) || matches!(
            fact.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Complex,
                ..
            })
        ) || (!vector_control
            && fact.shape.element_count().is_some_and(|count| count != 1)
            && !matches!(literal, Some(LiteralValue::Empty)))
            || (vector_control
                && fact.shape.known_dims().is_some_and(|dims| {
                    dims.len() > 2
                        || (dims.len() == 2
                            && dims[0].is_some_and(|size| size > 1)
                            && dims[1].is_some_and(|size| size > 1))
                }))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FFT-CONTROL",
                "FFT control requires real numeric or logical values of the documented shape",
                index,
            ));
        }
        let dimension_control = transform.shift()
            || (!transform.two_dimensional() && !transform.multidimensional() && index == 2);
        let tolerance = if dimension_control && !transform.shift() {
            // fft/ifft DIM uses the shared scalar-dimension decoder, while
            // lengths and the shift/ND vector parsers use binary64 epsilon.
            1e-6
        } else {
            f64::EPSILON
        };
        if let Some(literal) = literal {
            let invalid = if transform.shift() {
                literal_invalid_shift_dimension(literal)
            } else {
                literal_invalid_size(literal, tolerance)
            };
            if invalid {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FFT-CONTROL",
                    "FFT size and dimension values must be finite nonnegative integers",
                    index,
                ));
            }
        }
        if dimension_control && literal_nonnegative_integer(request, index, tolerance) == Some(0) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FFT-DIMENSION",
                "FFT dimensions are one-based and must be positive",
                index,
            ));
        }
    }
    Options {
        control_end,
        symmetry,
        uncertain_symmetry,
    }
}

fn literal_text(request: &CallRequest, index: usize) -> Option<&str> {
    match request.literals.literal_args.get(index)? {
        LiteralValue::String(text)
        | LiteralValue::Character(text)
        | LiteralValue::Keyword(text) => Some(text),
        _ => None,
    }
}

fn literal_invalid_size(literal: &LiteralValue, tolerance: f64) -> bool {
    match literal {
        LiteralValue::Number(_) | LiteralValue::Real { .. } | LiteralValue::Integer { .. } => {
            literal_unsigned(literal, tolerance).is_none()
        }
        LiteralValue::Vector(values) => values
            .iter()
            .any(|value| literal_invalid_size(value, tolerance)),
        LiteralValue::Matrix(rows) => rows
            .iter()
            .flatten()
            .any(|value| literal_invalid_size(value, tolerance)),
        _ => false,
    }
}

fn literal_invalid_shift_dimension(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Number(_) | LiteralValue::Real { .. } => {
            let Some(value) = LiteralContext::numeric_from_literal(literal) else {
                return true;
            };
            let rounded = value.round();
            // The runtime's shift parser saturates a finite floating index
            // above usize::MAX and ignores the resulting out-of-rank axis.
            !value.is_finite() || (rounded - value).abs() > f64::EPSILON || rounded < 1.0
        }
        LiteralValue::Integer { .. } => {
            literal_unsigned(literal, f64::EPSILON).is_none_or(|dimension| dimension == 0)
        }
        LiteralValue::Vector(values) => values.iter().any(literal_invalid_shift_dimension),
        LiteralValue::Matrix(rows) => rows.iter().flatten().any(literal_invalid_shift_dimension),
        _ => false,
    }
}

fn output_fact(
    transform: Transform,
    request: &CallRequest,
    input: &ValueFact,
    options: Options,
) -> ValueFact {
    let mut output = input.clone();
    if !transform.shift() {
        output.kind = if options.uncertain_symmetry {
            ValueKindFact::Unknown
        } else {
            match input.kind {
                ValueKindFact::Numeric(numeric) => ValueKindFact::Numeric(NumericFact {
                    class: if numeric.class == NumericClass::Single {
                        NumericClass::Single
                    } else {
                        NumericClass::Double
                    },
                    domain: if options.symmetry == Some(true) {
                        NumericDomain::Real
                    } else {
                        NumericDomain::Complex
                    },
                }),
                ValueKindFact::Logical => ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: if options.symmetry == Some(true) {
                        NumericDomain::Real
                    } else {
                        NumericDomain::Complex
                    },
                }),
                _ => ValueKindFact::Unknown,
            }
        };
        output.shape = if options.uncertain_symmetry {
            ShapeFact::Unknown
        } else {
            transformed_shape(transform, request, &input.shape, options.control_end)
        };
    }
    // Provider dispatch, fallback, and runtime-owned preparation may each
    // select a different physical allocation. Neither host nor device output
    // is provable from call facts alone.
    output.residency = ResidencyFact::Unknown;
    output.storage = StorageFact::Unknown;
    output.layout = LayoutFact::Unknown;
    output.contiguity = ContiguityFact::Unknown;
    output.alias = AliasFact::Unknown;
    output.view = ViewFact::Unknown;
    output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    output
}

fn transformed_shape(
    transform: Transform,
    request: &CallRequest,
    shape: &ShapeFact,
    control_end: usize,
) -> ShapeFact {
    let Some(mut dims) = shape.known_dims() else {
        return shape.clone();
    };
    dims.resize(dims.len().max(2), Some(1));
    if transform.two_dimensional() {
        if control_end > 1 {
            if control_end > 2 {
                dims[0] = length_or_existing(request, 1, dims[0]);
                dims[1] = length_or_existing(request, 2, dims[1]);
            } else if let Some(size) = literal_nonnegative_integer(request, 1, f64::EPSILON) {
                dims[0] = Some(size);
                dims[1] = Some(size);
            } else if let Some(sizes) = literal_size_vector(request, 1) {
                match sizes.as_slice() {
                    [] => {}
                    [size] => {
                        dims[0] = Some(*size);
                        dims[1] = Some(*size);
                    }
                    [rows, cols] => {
                        dims[0] = Some(*rows);
                        dims[1] = Some(*cols);
                    }
                    _ => return ShapeFact::Unknown,
                }
            } else if matches!(
                request.literals.literal_args.get(1),
                Some(LiteralValue::Empty)
            ) {
                // Empty SIZE leaves both lengths at their input extents.
            } else {
                // The shape of an unknown SIZE argument is not its contents.
                dims[0] = None;
                dims[1] = None;
            }
        }
    } else if transform.multidimensional() {
        if control_end > 1 {
            if let Some(sizes) = literal_size_vector(request, 1)
                .filter(|sizes| !sizes.is_empty() && sizes.iter().all(|size| *size > 0))
            {
                dims.resize(dims.len().max(sizes.len()), Some(1));
                for (axis, size) in sizes.into_iter().enumerate() {
                    dims[axis] = Some(size);
                }
            } else {
                return ShapeFact::Unknown;
            }
        }
    } else if !transform.shift() && control_end > 1 {
        let axis = if control_end > 2 {
            literal_positive_integer(request, 2).map(|dimension| dimension - 1)
        } else {
            default_axis(&dims)
        };
        if let Some(axis) = axis {
            if axis < 32 {
                dims.resize(dims.len().max(axis + 1), Some(1));
                if !matches!(
                    request.literals.literal_args.get(1),
                    Some(LiteralValue::Empty)
                ) {
                    dims[axis] = literal_nonnegative_integer(request, 1, f64::EPSILON);
                }
            } else {
                return ShapeFact::Unknown;
            }
        } else {
            return ShapeFact::Unknown;
        }
    }
    ShapeFact::from(dims)
}

fn length_or_existing(
    request: &CallRequest,
    index: usize,
    original: Option<usize>,
) -> Option<usize> {
    if matches!(
        request.literals.literal_args.get(index),
        Some(LiteralValue::Empty)
    ) {
        original
    } else {
        literal_nonnegative_integer(request, index, f64::EPSILON)
    }
}

fn default_axis(dims: &[Option<usize>]) -> Option<usize> {
    for (axis, dimension) in dims.iter().enumerate() {
        match dimension {
            Some(1) => {}
            Some(_) => return Some(axis),
            None => return None,
        }
    }
    Some(0)
}

fn literal_nonnegative_integer(
    request: &CallRequest,
    index: usize,
    tolerance: f64,
) -> Option<usize> {
    literal_unsigned(request.literals.literal_args.get(index)?, tolerance)
}

fn literal_unsigned(literal: &LiteralValue, tolerance: f64) -> Option<usize> {
    match literal {
        LiteralValue::Bool(value) => Some(usize::from(*value)),
        LiteralValue::Integer { .. } => usize::try_from(literal.integer_value_i128()?).ok(),
        LiteralValue::Number(_) | LiteralValue::Real { .. } => {
            let value = LiteralContext::numeric_from_literal(literal)?;
            if !value.is_finite() || value < 0.0 {
                return None;
            }
            let rounded = value.round();
            if (rounded - value).abs() > tolerance
                || rounded > usize::MAX as f64
                || (usize::BITS == 64 && rounded == usize::MAX as f64)
            {
                return None;
            }
            Some(rounded as usize)
        }
        _ => None,
    }
}

fn literal_positive_integer(request: &CallRequest, index: usize) -> Option<usize> {
    literal_nonnegative_integer(request, index, 1e-6).filter(|value| *value > 0)
}

fn literal_size_vector(request: &CallRequest, index: usize) -> Option<Vec<usize>> {
    match request.literals.literal_args.get(index)? {
        LiteralValue::Vector(values) => values
            .iter()
            .map(|value| literal_unsigned(value, f64::EPSILON))
            .collect(),
        LiteralValue::Matrix(rows) if rows.len() == 1 => rows[0]
            .iter()
            .map(|value| literal_unsigned(value, f64::EPSILON))
            .collect(),
        LiteralValue::Matrix(rows) if rows.iter().all(|row| row.len() == 1) => rows
            .iter()
            .map(|row| literal_unsigned(&row[0], f64::EPSILON))
            .collect(),
        _ => None,
    }
}
