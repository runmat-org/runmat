use crate::catalog::inference::argument_error;
use runmat_types::{InferenceDiagnostic, ShapeFact};

pub(super) fn output(
    input: &ShapeFact,
    source_width: usize,
    target_width: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ShapeFact {
    let Some(count) = input.element_count() else {
        return ShapeFact::Unknown;
    };
    let Some(bytes) = count.checked_mul(source_width) else {
        return ShapeFact::Unknown;
    };
    if bytes % target_width != 0 {
        diagnostics.push(argument_error(
            "RM-CATALOG-TYPECAST-BYTE-COUNT",
            "input byte count is not divisible by the requested output element width",
            1,
        ));
        return ShapeFact::Unknown;
    }
    oriented(input, bytes / target_width)
}

fn oriented(input: &ShapeFact, length: usize) -> ShapeFact {
    let dims = input.known_dims().unwrap_or_default();
    if dims.contains(&Some(0)) {
        return if dims.first() == Some(&Some(1)) {
            vec![Some(1), Some(0)].into()
        } else if dims.get(1) == Some(&Some(1)) {
            vec![Some(0), Some(1)].into()
        } else {
            vec![Some(0), Some(0)].into()
        };
    }
    if dims.first() == Some(&Some(1)) || dims.iter().all(|dimension| *dimension == Some(1)) {
        vec![Some(1), Some(length)].into()
    } else {
        vec![Some(length), Some(1)].into()
    }
}
