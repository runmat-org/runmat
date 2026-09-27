//! Conditional successful-result geometry, not a second argument validator.
use super::Spectrum;
use crate::catalog::inference::argument_error;
use runmat_types::*;

const MAX_NFFT: usize = 1 << 22;

fn invalid_real(value: &LiteralValue) -> bool {
    match value {
        LiteralValue::Complex {
            real, imaginary, ..
        } => {
            real.parse::<f64>().is_ok_and(|n| !n.is_finite())
                || imaginary
                    .parse::<f64>()
                    .is_ok_and(|n| !n.is_finite() || n.abs() > 1.0e-12)
        }
        _ => LiteralContext::numeric_from_literal(value).is_some_and(|n| !n.is_finite()),
    }
}

fn proven_invalid_integer(value: &LiteralValue) -> bool {
    LiteralContext::numeric_from_literal(value).is_some() && integer(value).is_none()
}

pub(super) fn invalid_controls(
    request: &CallRequest,
    spectrum: Spectrum,
) -> Option<InferenceDiagnostic> {
    let args = request
        .literals
        .literal_args
        .get(1..request.arguments.len())?;
    let mut index = 0;
    let window = positional(args, &mut index)?;
    if let Some(value) = window {
        if matches!(value, LiteralValue::Vector(v) if v.iter().any(invalid_real))
            || (!matches!(value, LiteralValue::Vector(_) | LiteralValue::Empty)
                && invalid_real(value))
            || (matches!(spectrum, Spectrum::Spectrogram)
                && match value {
                    LiteralValue::Vector(v) if v.len() == 1 => proven_invalid_integer(&v[0]),
                    LiteralValue::Vector(_) | LiteralValue::Empty => false,
                    v => proven_invalid_integer(v),
                })
        {
            return Some(argument_error(
                "RM-CATALOG-SPECTRAL-WINDOW",
                "invalid spectral window",
                1,
            ));
        }
    }
    let signal_len = request
        .arguments
        .first()?
        .shape
        .known_dims()
        .and_then(|dims| match dims.as_slice() {
            [Some(1), Some(columns)] => Some(*columns),
            [Some(rows), _] => Some(*rows),
            _ => None,
        });
    let default_window = signal_len.and_then(|n| match spectrum {
        Spectrum::Periodogram => Some(n),
        Spectrum::Spectrogram => n.checked_mul(2).map(|twice| (twice / 9).max(1)),
    });
    let window_len = match window {
        None | Some(LiteralValue::Empty) => default_window,
        Some(LiteralValue::Vector(v)) if v.is_empty() => default_window,
        Some(LiteralValue::Vector(v))
            if v.len() == 1 && matches!(spectrum, Spectrum::Spectrogram) =>
        {
            integer(&v[0])
        }
        Some(LiteralValue::Vector(v)) => Some(v.len()),
        Some(v) if matches!(spectrum, Spectrum::Spectrogram) => integer(v),
        Some(_) => Some(1),
    };
    if matches!(spectrum, Spectrum::Periodogram) {
        if let (Some(actual), Some(expected)) = (window_len, signal_len) {
            if actual != expected {
                return Some(argument_error(
                    "RM-CATALOG-SPECTRAL-WINDOW",
                    "window length must match signal length",
                    1,
                ));
            }
        }
    } else {
        if window_len == Some(0) {
            return Some(argument_error(
                "RM-CATALOG-SPECTRAL-WINDOW",
                "window must be nonempty",
                1,
            ));
        }
        let overlap = positional(args, &mut index)?;
        if match overlap {
            Some(LiteralValue::Vector(v)) if v.len() == 1 => proven_invalid_integer(&v[0]),
            Some(LiteralValue::Vector(_) | LiteralValue::Empty) | None => false,
            Some(v) => proven_invalid_integer(v),
        } {
            return Some(argument_error(
                "RM-CATALOG-SPECTRAL-OVERLAP",
                "invalid overlap",
                2,
            ));
        }
        let overlap = match overlap {
            None | Some(LiteralValue::Empty) => window_len.map(|n| n / 2),
            Some(LiteralValue::Vector(v)) if v.is_empty() => window_len.map(|n| n / 2),
            Some(LiteralValue::Vector(v)) if v.len() == 1 => integer(&v[0]),
            Some(v) => integer(v),
        };
        if let (Some(overlap), Some(window)) = (overlap, window_len) {
            if overlap >= window {
                return Some(argument_error(
                    "RM-CATALOG-SPECTRAL-OVERLAP",
                    "overlap must be smaller than window",
                    2,
                ));
            }
        }
    }
    let grid_argument = index + 1;
    if let Some(LiteralValue::Vector(values)) = positional(args, &mut index)? {
        if values.len() >= 2 && values.iter().any(invalid_real) {
            return Some(argument_error(
                "RM-CATALOG-SPECTRAL-GRID",
                "frequency vector must be finite and real",
                grid_argument,
            ));
        }
    }
    None
}

pub(super) fn invalid_options(
    request: &CallRequest,
    spectrum: Spectrum,
) -> Option<InferenceDiagnostic> {
    if matches!(spectrum, Spectrum::Spectrogram) {
        if let Some([Some(rows), Some(cols)]) =
            request.arguments.first()?.shape.known_dims().as_deref()
        {
            if *rows > 1 && *cols > 1 {
                return Some(argument_error(
                    "RM-CATALOG-SPECTRAL-VECTOR",
                    "spectrogram requires a vector signal",
                    0,
                ));
            }
        }
    }
    let args = request
        .literals
        .literal_args
        .get(1..request.arguments.len())?;
    let mut index = 0;
    positional(args, &mut index)?;
    if matches!(spectrum, Spectrum::Spectrogram) {
        positional(args, &mut index)?;
    }
    let grid = positional(args, &mut index)?;
    positional(args, &mut index)?; // Fs
    let explicit_grid = matches!(grid, Some(LiteralValue::Vector(v)) if v.len() >= 2);
    let complex_signal = matches!(
        request.arguments.first()?.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Complex,
            ..
        })
    );
    for (offset, value) in args[index..].iter().enumerate() {
        let Some(option) = text(value) else {
            continue;
        };
        let invalid = (option == "onesided" && complex_signal)
            || (explicit_grid && matches!(option.as_str(), "onesided" | "twosided" | "centered"))
            || (matches!(spectrum, Spectrum::Spectrogram) && option == "reassigned");
        if invalid {
            return Some(argument_error(
                "RM-CATALOG-SPECTRAL-OPTION",
                "incompatible spectral option",
                index + offset + 1,
            ));
        }
    }
    None
}

/// Diagnose only a source-known grid in its proven positional slot. Dynamic
/// earlier controls may shift the slot, so they remain runtime-owned.
pub(super) fn invalid_grid(
    request: &CallRequest,
    spectrum: Spectrum,
) -> Option<InferenceDiagnostic> {
    let args = request
        .literals
        .literal_args
        .get(1..request.arguments.len())?;
    let mut index = 0;
    positional(args, &mut index)?;
    if matches!(spectrum, Spectrum::Spectrogram) {
        positional(args, &mut index)?;
    }
    let argument = index + 1;
    let grid = positional(args, &mut index)??;
    let invalid = match grid {
        LiteralValue::Empty => false,
        LiteralValue::Vector(v) if v.is_empty() => false,
        LiteralValue::Vector(v) if v.len() == 1 => {
            proven_invalid_integer(&v[0]) || integer(&v[0]).is_some_and(|n| n == 0 || n > MAX_NFFT)
        }
        LiteralValue::Vector(v) => {
            v.len()
                < if matches!(spectrum, Spectrum::Periodogram) {
                    1
                } else {
                    2
                }
                || v.len() > MAX_NFFT
        }
        v => proven_invalid_integer(v) || integer(v).is_some_and(|n| n == 0 || n > MAX_NFFT),
    };
    invalid.then(|| {
        argument_error(
            "RM-CATALOG-SPECTRAL-GRID",
            "frequency grid must have an admitted nonzero length",
            argument,
        )
    })
}

pub(super) fn output_shape(request: &CallRequest, spectrum: Spectrum, slot: usize) -> ShapeFact {
    let dims = request.arguments.first().and_then(|v| v.shape.known_dims());
    let (length, channels) = match dims.as_deref() {
        Some([Some(1), columns]) => (*columns, Some(1)),
        Some([rows, columns]) => (*rows, *columns),
        _ => (None, None),
    };
    let geometry = known_geometry(request, spectrum, length);
    let rows = geometry.map(|g| g.0);
    let columns = match spectrum {
        Spectrum::Periodogram => channels,
        Spectrum::Spectrogram => geometry.map(|g| g.1),
    };
    ShapeFact::from(match slot {
        1 => vec![rows, Some(1)],
        2 => vec![Some(1), columns],
        _ => vec![rows, columns],
    })
}

fn integer(value: &LiteralValue) -> Option<usize> {
    let n = LiteralContext::numeric_from_literal(value)?;
    (n.is_finite() && n >= 0.0 && n.fract() == 0.0 && n < usize::MAX as f64).then_some(n as usize)
}

fn text(value: &LiteralValue) -> Option<String> {
    match value {
        LiteralValue::Character(s) | LiteralValue::String(s) | LiteralValue::Keyword(s) => {
            Some(s.to_ascii_lowercase())
        }
        _ => None,
    }
}

// A known keyword leaves positional defaults in place. Unknown literals stop
// refinement: their runtime kind may select either a control or an option.
fn positional<'a>(args: &'a [LiteralValue], index: &mut usize) -> Option<Option<&'a LiteralValue>> {
    match args.get(*index) {
        None => Some(None),
        Some(LiteralValue::Unknown) => None,
        Some(v) if text(v).is_some() => Some(None),
        Some(v) => {
            *index += 1;
            Some(Some(v))
        }
    }
}

fn known_geometry(
    request: &CallRequest,
    spectrum: Spectrum,
    length: Option<usize>,
) -> Option<(usize, usize)> {
    let length = length.filter(|n| *n > 0)?;
    if request.literals.literal_args.len() < request.arguments.len() {
        return None;
    }
    let args = request
        .literals
        .literal_args
        .get(1..request.arguments.len())?;
    let mut index = 0;
    let default_window = match spectrum {
        Spectrum::Periodogram => length,
        Spectrum::Spectrogram => length.checked_mul(2)?.checked_div(9)?.max(1),
    };
    let window_control = positional(args, &mut index)?;
    let window = match (spectrum, window_control) {
        (Spectrum::Periodogram, _) => length, // Successful windows must match the signal.
        (_, None | Some(LiteralValue::Empty)) => default_window,
        (_, Some(LiteralValue::Vector(v))) if v.is_empty() => default_window,
        (_, Some(LiteralValue::Vector(v))) if v.len() == 1 => integer(&v[0])?,
        (_, Some(LiteralValue::Vector(v))) => v.len(),
        (_, Some(v)) => integer(v)?,
    };
    if window == 0 {
        return None;
    }
    let frames = match spectrum {
        Spectrum::Periodogram => 1,
        Spectrum::Spectrogram => {
            let overlap = match positional(args, &mut index)? {
                None | Some(LiteralValue::Empty) => window / 2,
                Some(LiteralValue::Vector(v)) if v.is_empty() => window / 2,
                Some(v) => integer(v)?,
            };
            let step = window.checked_sub(overlap).filter(|v| *v > 0)?;
            if length <= window {
                1
            } else {
                (length - window) / step + 1
            }
        }
    };
    let grid = positional(args, &mut index)?;
    let (nfft, explicit) = match grid {
        None | Some(LiteralValue::Empty) => (window.checked_next_power_of_two()?.max(256), false),
        Some(LiteralValue::Vector(v)) if v.is_empty() => {
            (window.checked_next_power_of_two()?.max(256), false)
        }
        Some(LiteralValue::Vector(v)) if v.len() == 1 => (integer(&v[0])?, false),
        Some(LiteralValue::Vector(v)) => (v.len(), true),
        Some(v) => (integer(v)?, false),
    };
    if nfft == 0
        || nfft > MAX_NFFT
        || (explicit && matches!(spectrum, Spectrum::Spectrogram) && nfft < 2)
    {
        return None;
    }
    positional(args, &mut index)?; // Sampling rate changes coordinates, not cardinality.
    let mut onesided = match &request.arguments.first()?.kind {
        ValueKindFact::Numeric(n) => n.domain == NumericDomain::Real,
        ValueKindFact::Logical => true,
        _ => return None,
    };
    for value in &args[index..] {
        match text(value)?.as_str() {
            "onesided" => onesided = true,
            "twosided" | "centered" => onesided = false,
            "psd" | "power" => {}
            _ => return None,
        }
    }
    Some((
        if explicit || !onesided {
            nfft
        } else {
            nfft / 2 + 1
        },
        frames,
    ))
}
