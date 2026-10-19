//! Pagewise matrix-product geometry and literal transform projection.

use crate::catalog::inference::argument_error;
use runmat_types::{CallRequest, InferenceDiagnostic, ShapeFact, ValueFact};

#[derive(Clone, Copy)]
enum PageTransform {
    None,
    Swap,
}

fn page_transform(
    request: &CallRequest,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<PageTransform> {
    let selector = request.literals.literal_string_at(index)?;
    match selector.to_ascii_lowercase().as_str() {
        "none" => Some(PageTransform::None),
        "transpose" | "ctranspose" => Some(PageTransform::Swap),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-PAGEMTIMES-TRANSPOSE",
                "invalid page transpose selector",
                index,
            ));
            None
        }
    }
}

fn transformed_dims(shape: &ShapeFact, transform: PageTransform) -> Option<Vec<Option<usize>>> {
    let mut dims = shape.known_dims()?;
    if matches!(transform, PageTransform::Swap) && dims.len() >= 2 {
        dims.swap(0, 1);
    }
    Some(dims)
}

fn product_shape(
    left: &ValueFact,
    right: &ValueFact,
    lhs_transform: PageTransform,
    rhs_transform: PageTransform,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ShapeFact {
    if left.shape.element_count() == Some(1) {
        return transformed_dims(&right.shape, rhs_transform)
            .map_or(ShapeFact::Unknown, ShapeFact::from);
    }
    if right.shape.element_count() == Some(1) {
        return transformed_dims(&left.shape, lhs_transform)
            .map_or(ShapeFact::Unknown, ShapeFact::from);
    }
    let (Some(a), Some(b)) = (
        transformed_dims(&left.shape, lhs_transform),
        transformed_dims(&right.shape, rhs_transform),
    ) else {
        return ShapeFact::Unknown;
    };
    if a.len() < 2 || b.len() < 2 {
        return ShapeFact::Unknown;
    }
    if a[1].zip(b[0]).is_some_and(|(x, y)| x != y) {
        diagnostics.push(argument_error(
            "RM-CATALOG-PAGEMTIMES-INNER",
            "matrix page inner dimensions must agree",
            1,
        ));
    }
    let mut dims = vec![a[0], b[1]];
    for axis in 2..a.len().max(b.len()) {
        let x = a.get(axis).copied().unwrap_or(Some(1));
        let y = b.get(axis).copied().unwrap_or(Some(1));
        let extent = match (x, y) {
            (Some(1), other) | (other, Some(1)) => other,
            (Some(m), Some(n)) if m == n => Some(m),
            (Some(_), Some(_)) => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-PAGEMTIMES-PAGE",
                    "page dimensions are not broadcast-compatible",
                    1,
                ));
                None
            }
            _ => None,
        };
        dims.push(extent);
    }
    ShapeFact::from(dims)
}

pub(in crate::catalog::entries::math::linalg::ops) fn page_product_shape(
    request: &CallRequest,
    left: &ValueFact,
    right: &ValueFact,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ShapeFact {
    let transforms = if request.arguments.len() == 4 {
        page_transform(request, 1, diagnostics).zip(page_transform(request, 3, diagnostics))
    } else {
        Some((PageTransform::None, PageTransform::None))
    };
    transforms.map_or(ShapeFact::Unknown, |(lhs, rhs)| {
        product_shape(left, right, lhs, rhs, diagnostics)
    })
}
