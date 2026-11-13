//! Exact row witnesses for contradictions already found by the singleton scan.
//! Original-source consumers must validate these rows against their own model.
use super::*;

pub(super) enum Contradiction {
    EmptyRow(usize),
    SingletonBounds {
        lower_entry: usize,
        upper_entry: usize,
    },
}

impl LinearCertificates<'_, '_> {
    pub(crate) async fn infeasibility_certificate(
        &self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<BigRational>>> {
        let Some(contradiction) = &self.contradiction else {
            return Ok(None);
        };
        let mut weights = runmat_value::exact_capacity_vec(self.model.b.len())
            .map_err(|source| allocation(source, self.model))?;
        while weights.len() < self.model.b.len() {
            let count = (self.model.b.len() - weights.len()).min(1024);
            weights.resize_with(weights.len() + count, BigRational::zero);
            cooperation.completed(count as u64).await?;
        }
        match *contradiction {
            Contradiction::EmptyRow(row) => {
                // Equality duals are free. Nonnegative-cone contradictions
                // have negative RHS, so their witness remains nonnegative.
                weights[row] =
                    BigRational::from_integer(if self.model.b[row] > 0. { -1 } else { 1 }.into());
            }
            Contradiction::SingletonBounds {
                lower_entry,
                upper_entry,
            } => {
                let a = self.model.a;
                let lower = BigRational::from_float(a.nzval[lower_entry])
                    .expect("validated finite singleton coefficient");
                let upper = BigRational::from_float(a.nzval[upper_entry])
                    .expect("validated finite singleton coefficient");
                weights[a.rowval[lower_entry]] -= lower.recip();
                weights[a.rowval[upper_entry]] += upper.recip();
            }
        }
        Ok(Some(weights))
    }
}
