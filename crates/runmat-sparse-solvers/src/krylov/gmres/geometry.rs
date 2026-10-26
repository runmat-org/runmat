use super::GmresWork;
use num_complex::Complex64;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GmresGeometry {
    pub n: usize,
    pub restart: usize,
    /// Eight n-vectors plus `restart+1` Arnoldi vectors.
    pub native_elements: usize,
    pub hessenberg_elements: usize,
    pub complex_small_elements: usize,
    pub real_small_elements: usize,
    pub history_elements: usize,
}

impl GmresGeometry {
    pub(super) fn validates<T>(&self, work: &GmresWork<'_, T>) -> bool {
        [
            &*work.x,
            &*work.best,
            &*work.candidate,
            &*work.previous,
            &*work.residual,
            &*work.product,
            &*work.preconditioned,
            &*work.intermediate,
        ]
        .iter()
        .all(|v| v.len() == self.n)
            && work.basis.len() >= self.native_elements - 8 * self.n
            && work.hessenberg.len() >= self.hessenberg_elements
            && work.sines.len() >= self.restart
            && work.cosines.len() >= self.restart
            && work.rotated_rhs.len() > self.restart
            && work.coefficients.len() >= self.restart
            && work.history.len() >= self.history_elements
    }
    /// Entries of the flexible direction basis `Z = M⁻¹ V` (flexible side only).
    pub fn flexible_elements(&self) -> usize {
        // `checked` already bounded (restart + 9) · n.
        self.restart * self.n
    }
    pub fn checked(n: usize, restart: usize, max_outer: usize) -> Option<Self> {
        if n == 0 || restart == 0 || max_outer == 0 {
            return None;
        }
        let native_elements = restart.checked_add(9)?.checked_mul(n)?;
        let hessenberg_elements = restart.checked_add(1)?.checked_mul(restart)?;
        // Sines, rotated RHS, and back-substitution coefficients.
        let complex_small_elements = restart.checked_mul(3)?.checked_add(1)?;
        let real_small_elements = restart;
        let history_elements = restart.checked_mul(max_outer)?.checked_add(1)?;
        Some(Self {
            n,
            restart,
            native_elements,
            hessenberg_elements,
            complex_small_elements,
            real_small_elements,
            history_elements,
        })
    }

    /// Project the three exact claimed backings into disjoint recurrence
    /// slices. No collection header, COW clone, or temporary vector appears
    /// at this handoff.
    pub fn project<'a, T>(
        self,
        native: &'a mut [T],
        complex: &'a mut [Complex64],
        real: &'a mut [f64],
    ) -> Option<GmresWork<'a, T>> {
        // Fields are inspectable for exact caller admission, not trusted input.
        let outer = self
            .history_elements
            .checked_sub(1)?
            .checked_div(self.restart)?;
        if Self::checked(self.n, self.restart, outer)? != self {
            return None;
        }
        let complex_count = self
            .hessenberg_elements
            .checked_add(self.complex_small_elements)?;
        let real_count = self
            .real_small_elements
            .checked_add(self.history_elements)?;
        if native.len() != self.native_elements
            || complex.len() != complex_count
            || real.len() != real_count
        {
            return None;
        }
        let (x, native) = native.split_at_mut(self.n);
        let (best, native) = native.split_at_mut(self.n);
        let (candidate, native) = native.split_at_mut(self.n);
        let (previous, native) = native.split_at_mut(self.n);
        let (residual, native) = native.split_at_mut(self.n);
        let (product, native) = native.split_at_mut(self.n);
        let (preconditioned, native) = native.split_at_mut(self.n);
        let (intermediate, basis) = native.split_at_mut(self.n);
        let (hessenberg, complex) = complex.split_at_mut(self.hessenberg_elements);
        let (sines, complex) = complex.split_at_mut(self.restart);
        let (rotated_rhs, coefficients) = complex.split_at_mut(self.restart + 1);
        let (cosines, history) = real.split_at_mut(self.restart);
        Some(GmresWork {
            x,
            best,
            candidate,
            previous,
            residual,
            product,
            preconditioned,
            intermediate,
            basis,
            hessenberg,
            cosines,
            sines,
            rotated_rhs,
            coefficients,
            history,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::krylov::pcg::PcgGeometry;

    #[test]
    fn workspace_geometry_is_checked_and_tracks_restart_not_dense_coefficient() {
        let layout = GmresGeometry::checked(40, 3, 7).unwrap();
        assert_eq!(layout.native_elements, 12 * 40);
        assert_eq!(layout.hessenberg_elements, 12);
        assert_eq!(layout.complex_small_elements, 10);
        assert_eq!(layout.history_elements, 22);
        let mut native = vec![0.0f32; layout.native_elements];
        let mut complex = vec![
            Complex64::new(0.0, 0.0);
            layout.hessenberg_elements + layout.complex_small_elements
        ];
        let mut real = vec![0.0f64; layout.real_small_elements + layout.history_elements];
        let work = layout
            .project(&mut native, &mut complex, &mut real)
            .unwrap();
        assert_eq!(work.basis.len(), 4 * 40);
        assert_eq!(work.history.len(), 22);
        assert!(GmresGeometry::checked(40, usize::MAX, 1).is_none());
        assert!(GmresGeometry::checked(40, 3, usize::MAX).is_none());
        let pcg = PcgGeometry::checked(40, 7, false).unwrap();
        assert_eq!(pcg.native_elements, 5 * 40);
        assert_eq!(pcg.history_elements, 8);
        assert_eq!(
            PcgGeometry::checked(40, 7, true).unwrap().native_elements,
            6 * 40
        );
        assert!(PcgGeometry::checked(usize::MAX, 7, false).is_none());
        assert!(PcgGeometry::checked(40, usize::MAX, false).is_none());
        assert!(PcgGeometry::checked(0, 7, false).is_none());
    }
}
