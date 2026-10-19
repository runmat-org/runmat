//! A conservative positivity certificate for the cubic Jacobian determinant of a quadratic tet.
//!
//! The Tet10 map is quadratic, so each Jacobian row is affine in barycentric coordinates and
//! its determinant has degree three. Its degree-three Bernstein coefficients bound the entire
//! tetrahedron, unlike a finite collection of quadrature samples. Subdivision tightens that
//! bound; an inconclusive result is rejected rather than admitted as a valid element.

pub(super) type Jacobian = [[f64; 3]; 3];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CertificateFailure {
    InvertedOrDegenerate,
    Inconclusive,
}

const PERMUTATIONS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];
const MAX_SUBDIVISIONS: usize = 5;

pub(super) fn certify_positive_jacobian(vertices: [Jacobian; 4]) -> Result<(), CertificateFailure> {
    certify_cell(vertices, 0)
}

fn certify_cell(vertices: [Jacobian; 4], depth: usize) -> Result<(), CertificateFailure> {
    let mut lowest_coefficient = f64::INFINITY;
    let mut largest_magnitude = 0.0_f64;
    for first in 0..4 {
        for second in first..4 {
            for third in second..4 {
                let sources = [first, second, third];
                // The determinant is multilinear in its rows. Averaging the six assignments
                // gives the Bernstein coefficient for this degree-three multi-index, including
                // repeated indices (which appear with the right multiplicity in this average).
                let coefficient = PERMUTATIONS
                    .iter()
                    .map(|permutation| {
                        determinant([
                            vertices[sources[permutation[0]]][0],
                            vertices[sources[permutation[1]]][1],
                            vertices[sources[permutation[2]]][2],
                        ])
                    })
                    .sum::<f64>()
                    / 6.0;
                if !coefficient.is_finite() {
                    return Err(CertificateFailure::Inconclusive);
                }
                if first == third && coefficient <= 0.0 {
                    // A repeated-index coefficient is the actual determinant at a vertex.
                    return Err(CertificateFailure::InvertedOrDegenerate);
                }
                lowest_coefficient = lowest_coefficient.min(coefficient);
                largest_magnitude = largest_magnitude.max(coefficient.abs());
            }
        }
    }
    // Leave a small arithmetic margin: a coefficient at roundoff scale is not a certificate.
    if lowest_coefficient > 64.0 * f64::EPSILON * largest_magnitude {
        return Ok(());
    }
    if depth == MAX_SUBDIVISIONS {
        return Err(CertificateFailure::Inconclusive);
    }
    let center = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            vertices
                .iter()
                .map(|vertex| vertex[row][column])
                .sum::<f64>()
                / 4.0
        })
    });
    for omitted in 0..4 {
        let mut child = [center; 4];
        let mut next = 0;
        for (index, vertex) in vertices.into_iter().enumerate() {
            if index != omitted {
                child[next] = vertex;
                next += 1;
            }
        }
        certify_cell(child, depth + 1)?;
    }
    Ok(())
}

fn determinant(rows: Jacobian) -> f64 {
    rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        + rows[0][1] * (rows[1][2] * rows[2][0] - rows[1][0] * rows[2][2])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_positive_and_inverted_jacobians_are_distinguished() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert_eq!(certify_positive_jacobian([identity; 4]), Ok(()));
        let inverted = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert_eq!(
            certify_positive_jacobian([identity, identity, identity, inverted]),
            Err(CertificateFailure::InvertedOrDegenerate)
        );
    }
}
