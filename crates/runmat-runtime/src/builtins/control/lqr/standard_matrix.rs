//! One validated standard matrix request shared by lqr, care, and dlqr.
//!
//! `lqr.rs` delegates its matrix-input parser here. This method does not add
//! a second CARE/DARE numerical algorithm.

use super::{DMatrix, LqrDomain, LqrRequest, RealMatrix, RiccatiApi};
use crate::BuiltinResult;
use runmat_value::Value;

impl LqrRequest {
    pub(in crate::builtins::control) async fn prepare_standard_matrix(
        a: Value,
        b: Value,
        q: Value,
        r: Option<Value>,
        cross: Option<Value>,
        domain: LqrDomain,
        api: RiccatiApi,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<Self> {
        // All host class/shape checks precede any resident matrix gather.
        for (name, value) in [("A", &a), ("B", &b), ("Q", &q)] {
            RealMatrix::preflight(name, value, api, runtime)?;
        }
        if let Some(value) = r.as_ref() {
            RealMatrix::preflight("R", value, api, runtime)?;
        }
        if let Some(value) = cross.as_ref() {
            RealMatrix::preflight("N", value, api, runtime)?;
        }
        let a = RealMatrix::parse("A", a, api, runtime).await?;
        let b = RealMatrix::parse("B", b, api, runtime).await?;
        let q = RealMatrix::parse("Q", q, api, runtime).await?;
        let r = match r {
            Some(value) => RealMatrix::parse("R", value, api, runtime).await?,
            None => RealMatrix {
                matrix: DMatrix::<f64>::identity(b.cols(), b.cols()),
            },
        };
        let n = match cross {
            Some(value) => Some(RealMatrix::parse("N", value, api, runtime).await?),
            None => None,
        };
        Ok(Self {
            a,
            b,
            q,
            r,
            n,
            domain,
            api,
        })
    }
}
