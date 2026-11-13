//! Retained ordinary Schur factors and their block-preserving transformations.

use nalgebra::linalg::Schur;
use nalgebra::DMatrix;
use num_complex::Complex64;

#[derive(Clone, Debug)]
pub(crate) enum SchurForm {
    Real {
        vectors: DMatrix<f64>,
        triangular: DMatrix<f64>,
    },
    Complex {
        vectors: DMatrix<Complex64>,
        triangular: DMatrix<Complex64>,
    },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SchurBlock {
    pub start: usize,
    pub size: usize,
}

impl SchurForm {
    /// Admits finite, structurally valid factors. As with retained Schur
    /// factor APIs, the caller supplies orthogonal U; checking U'*U afresh
    /// would make a local block transformation cubic in the matrix order.
    pub(crate) fn from_real_parts(
        vectors: DMatrix<f64>,
        triangular: DMatrix<f64>,
    ) -> Result<Self, &'static str> {
        validate_parts(&vectors, &triangular)?;
        let blocks = real_blocks(&triangular);
        if blocks.iter().any(|block| {
            block.size == 2
                && block.start + 2 < triangular.nrows()
                && triangular[(block.start + 2, block.start + 1)] != 0.0
        }) {
            return Err("real Schur form contains overlapping 2-by-2 blocks");
        }
        Ok(Self::Real {
            vectors,
            triangular,
        })
    }

    pub(crate) fn from_complex_parts(
        vectors: DMatrix<Complex64>,
        triangular: DMatrix<Complex64>,
    ) -> Result<Self, &'static str> {
        validate_parts(&vectors, &triangular)?;
        if (1..triangular.nrows()).any(|i| triangular[(i, i - 1)] != Complex64::new(0.0, 0.0)) {
            return Err("complex Schur form must be upper triangular");
        }
        Ok(Self::Complex {
            vectors,
            triangular,
        })
    }

    pub(crate) fn real(matrix: DMatrix<f64>) -> Result<Self, &'static str> {
        let iterations = matrix.nrows().saturating_mul(1_000).max(1_000);
        let (vectors, triangular) = Schur::try_new(matrix, f64::EPSILON, iterations)
            .ok_or("real Schur iteration did not converge")?
            .unpack();
        Self::from_real_parts(vectors, triangular)
    }

    pub(crate) fn complex(matrix: DMatrix<Complex64>) -> Result<Self, &'static str> {
        let iterations = matrix.nrows().saturating_mul(1_000).max(1_000);
        let (vectors, triangular) = Schur::try_new(matrix, f64::EPSILON, iterations)
            .ok_or("complex Schur iteration did not converge")?
            .unpack();
        Self::from_complex_parts(vectors, triangular)
    }

    pub(crate) fn blocks(&self) -> Vec<SchurBlock> {
        match self {
            Self::Real { triangular, .. } => real_blocks(triangular),
            Self::Complex { triangular, .. } => (0..triangular.nrows())
                .map(|start| SchurBlock { start, size: 1 })
                .collect(),
        }
    }

    pub(crate) fn eigenvalues(&self) -> Vec<Complex64> {
        match self {
            Self::Real { triangular, .. } => real_eigenvalues(triangular),
            Self::Complex { triangular, .. } => (0..triangular.nrows())
                .map(|i| triangular[(i, i)])
                .collect(),
        }
    }

    pub(crate) fn into_complex(self) -> Result<Self, &'static str> {
        let (vectors, triangular) = match self {
            Self::Real {
                vectors,
                triangular,
            } => (vectors, triangular),
            complex @ Self::Complex { .. } => return Ok(complex),
        };
        let blocks = real_blocks(&triangular);
        let mut vectors = vectors.map(|x| Complex64::new(x, 0.0));
        let mut triangular = triangular.map(|x| Complex64::new(x, 0.0));
        for block in blocks {
            if block.size == 1 {
                continue;
            }
            let start = block.start;
            let local = triangular.view((start, start), (2, 2)).into_owned();
            let (local_vectors, _) = Schur::try_new(local, f64::EPSILON, 1_000)
                .ok_or("2-by-2 complex Schur conversion did not converge")?
                .unpack();
            apply_local_unitary(&mut vectors, &mut triangular, start, &local_vectors);
            let residual = triangular[(start + 1, start)].norm();
            let scale = triangular
                .view((start, start), (2, 2))
                .iter()
                .map(|value| value.norm())
                .fold(0.0_f64, f64::max);
            if !residual.is_finite()
                || !scale.is_finite()
                || residual > 1e-10 * scale.max(f64::MIN_POSITIVE)
            {
                return Err("complex Schur conversion did not triangularize a real block");
            }
            triangular[(start + 1, start)] = Complex64::new(0.0, 0.0);
        }
        Self::from_complex_parts(vectors, triangular)
    }

    pub(crate) fn reorder(&mut self, target_block_order: &[usize]) -> Result<(), &'static str> {
        let blocks = self.blocks();
        if target_block_order.len() != blocks.len() {
            return Err("Schur block ordering has the wrong length");
        }
        let mut seen = vec![false; blocks.len()];
        for &index in target_block_order {
            let Some(slot) = seen.get_mut(index) else {
                return Err("Schur block ordering references an absent block");
            };
            if *slot {
                return Err("Schur block ordering repeats a block");
            }
            *slot = true;
        }
        let mut current = (0..blocks.len()).collect::<Vec<_>>();
        let sizes = blocks.iter().map(|block| block.size).collect::<Vec<_>>();
        for (target, &original) in target_block_order.iter().enumerate() {
            let Some(mut position) = current.iter().position(|&id| id == original) else {
                return Err("Schur block ordering repeats or omits a block");
            };
            while position > target {
                let left = position - 1;
                let offset: usize = current[..left].iter().map(|&id| sizes[id]).sum();
                let p = sizes[current[left]];
                let q = sizes[current[position]];
                match self {
                    Self::Real {
                        vectors,
                        triangular,
                    } => {
                        super::reorder::exchange_real(vectors, triangular, offset, p, q)?;
                    }
                    Self::Complex {
                        vectors,
                        triangular,
                    } => {
                        super::reorder::exchange_complex(vectors, triangular, offset, p, q)?;
                    }
                }
                current.swap(left, position);
                position -= 1;
            }
        }
        Ok(())
    }
}

pub(crate) fn real_blocks(matrix: &DMatrix<f64>) -> Vec<SchurBlock> {
    let mut blocks = Vec::new();
    let mut index = 0;
    while index < matrix.nrows() {
        // The Schur backend explicitly deflates blocks by writing exact zeros.
        let size = if index + 1 < matrix.nrows() && matrix[(index + 1, index)] != 0.0 {
            2
        } else {
            1
        };
        blocks.push(SchurBlock { start: index, size });
        index += size;
    }
    blocks
}

pub(crate) fn real_eigenvalues(matrix: &DMatrix<f64>) -> Vec<Complex64> {
    let mut values = Vec::with_capacity(matrix.nrows());
    for block in real_blocks(matrix) {
        let i = block.start;
        if block.size == 1 {
            values.push(Complex64::new(matrix[(i, i)], 0.0));
            continue;
        }
        // Scale the 2x2 before evaluating its discriminant: off-diagonals
        // 1e20 and -1e-20 still encode a conjugate pair, not two real roots.
        let a = matrix[(i, i)];
        let b = matrix[(i, i + 1)];
        let c = matrix[(i + 1, i)];
        let d = matrix[(i + 1, i + 1)];
        let scale = a.abs().max(d.abs()).max(b.abs().sqrt() * c.abs().sqrt());
        if scale == 0.0 {
            values.extend([Complex64::new(0.0, 0.0); 2]);
            continue;
        }
        let center = (a / scale + d / scale) / 2.0;
        let delta = (a / scale - d / scale) / 2.0;
        let off_diagonal = (b.abs().sqrt() / scale.sqrt()) * (c.abs().sqrt() / scale.sqrt());
        let signed_product = if b.is_sign_negative() == c.is_sign_negative() {
            off_diagonal * off_diagonal
        } else {
            -off_diagonal * off_diagonal
        };
        let root = Complex64::new(delta * delta + signed_product, 0.0).sqrt();
        values.push((Complex64::new(center, 0.0) + root) * scale);
        values.push((Complex64::new(center, 0.0) - root) * scale);
    }
    values
}

fn validate_parts<T: nalgebra::ComplexField<RealField = f64> + Copy>(
    vectors: &DMatrix<T>,
    triangular: &DMatrix<T>,
) -> Result<(), &'static str> {
    let n = triangular.nrows();
    if triangular.ncols() != n || vectors.nrows() != n || vectors.ncols() != n {
        return Err("Schur factors must be equal-size square matrices");
    }
    if !vectors.iter().all(|value| value.modulus().is_finite())
        || !triangular.iter().all(|value| value.modulus().is_finite())
    {
        return Err("Schur factors must be finite");
    }
    for column in 0..n {
        for row in (column + 2)..n {
            if triangular[(row, column)] != T::from_real(0.0) {
                return Err("Schur form has entries below the first subdiagonal");
            }
        }
    }
    Ok(())
}

fn apply_local_unitary(
    vectors: &mut DMatrix<Complex64>,
    triangular: &mut DMatrix<Complex64>,
    start: usize,
    transform: &DMatrix<Complex64>,
) {
    let width = transform.nrows();
    let updated_columns = vectors.columns(start, width).into_owned() * transform;
    vectors
        .columns_mut(start, width)
        .copy_from(&updated_columns);
    let updated_rows = transform.adjoint() * triangular.rows(start, width).into_owned();
    triangular.rows_mut(start, width).copy_from(&updated_rows);
    let updated_columns = triangular.columns(start, width).into_owned() * transform;
    triangular
        .columns_mut(start, width)
        .copy_from(&updated_columns);
}
