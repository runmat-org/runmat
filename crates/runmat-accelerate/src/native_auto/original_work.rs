//! Keep historical calibrated scan/MAC rates distinct from cold publication work.
//! The existing element-stage rate is a cold proxy for publication/conversion,
//! not a claim to have measured those stages or the complete original operation.
use super::*;
use runmat_types::{reduction::SumWorkControls, MatrixWorkGeometry};

fn compose(scan: Option<f64>, stages: u64, publication_rate: f64) -> Option<f64> {
    let scan = scan?;
    let publication = cpu_estimate(publication_rate, usize::try_from(stages).ok()?)?;
    let total = scan + publication;
    (total.is_finite() && total >= 0.0).then_some(total)
}

pub(super) fn sum(
    auto: &NativeAutoOffload,
    value: &Value,
    route: &runmat_accelerate_api::ProviderSumWorkload,
) -> Option<f64> {
    let shape = value_shape(value)?;
    let work = route.invocation.original_work_geometry(SumWorkControls {
        shape,
        axes: &route.axes,
        input_class: route.input_class,
        domain: route.input_domain,
        accumulation_class: route.accumulation_class,
        publication_class: route.publication_class,
    })?;
    compose(
        cpu_estimate(
            auto.thresholds.cpu_reduction_per_elem,
            usize::try_from(work.scan_lanes).ok()?,
        ),
        work.publication_conversion_stages,
        auto.thresholds.cpu_elem_per_elem,
    )
}

pub(super) fn matrix(
    auto: &NativeAutoOffload,
    rows: usize,
    inner: usize,
    columns: usize,
) -> Option<f64> {
    let work = MatrixWorkGeometry::checked(
        u64::try_from(rows).ok()?,
        u64::try_from(inner).ok()?,
        u64::try_from(columns).ok()?,
    )?;
    compose(
        cpu_estimate(
            auto.thresholds.cpu_matmul_per_flop,
            usize::try_from(work.multiply_accumulates).ok()?,
        ),
        work.output_elements,
        auto.thresholds.cpu_elem_per_elem,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{NumericClass, NumericDomain, ReductionNaN, SumInvocation, SumOutputClass};
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct NoDevice {
        queries: AtomicUsize,
    }
    impl AccelProvider for NoDevice {
        fn device_info(&self) -> String {
            "no-device-work-prior".into()
        }
        fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            panic!("source prior cannot upload")
        }
        fn download<'a>(
            &'a self,
            _: &'a GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            panic!("source prior cannot download")
        }
        fn query_feasibility(
            &self,
            _query: &runmat_accelerate_api::ProviderFeasibilityQuery,
        ) -> runmat_accelerate_api::ProviderFeasibility {
            self.queries.fetch_add(1, Ordering::SeqCst);
            runmat_accelerate_api::ProviderFeasibility::rejected(
                runmat_accelerate_api::ProviderRejectionCode::UnsupportedOperation,
                "source prior fixture has no device operations",
            )
        }
    }
    static OWNER: NoDevice = NoDevice {
        queries: AtomicUsize::new(0),
    };
    fn route() -> runmat_accelerate_api::ProviderSumWorkload {
        runmat_accelerate_api::ProviderSumWorkload {
            invocation: SumInvocation::Sum,
            axes: vec![0],
            missing: ReductionNaN::Include,
            output: SumOutputClass::Default,
            input_class: NumericClass::Double,
            input_domain: NumericDomain::Real,
            accumulation_class: NumericClass::Double,
            publication_class: NumericClass::Double,
        }
    }
    #[test]
    fn zero_scan_or_inner_dimension_still_prices_nonempty_publication() {
        let auto = NativeAutoOffload::new(&OWNER, ThresholdConfig::default());
        let input = Value::Tensor(Tensor::new(vec![], vec![0, 7]).unwrap());
        assert_eq!(
            sum(&auto, &input, &route()),
            Some(7.0 * auto.thresholds.cpu_elem_per_elem)
        );
        assert_eq!(
            matrix(&auto, 3, 0, 7),
            Some(21.0 * auto.thresholds.cpu_elem_per_elem)
        );
        assert_eq!(matrix(&auto, 0, 3, 7), Some(0.0));
        assert!(matrix(&auto, usize::MAX, usize::MAX, 2).is_none());

        // Borrowing a carrier's geometry neither infers its class/domain nor
        // extends batch/provider admission. The already-admitted route owns it.
        for (value, domain, lanes) in [
            (Value::Complex(1.0, 2.0), NumericDomain::Complex, 2),
            (
                Value::ComplexTensor(
                    runmat_value::ComplexTensor::new(vec![], vec![0, 7, 2]).unwrap(),
                ),
                NumericDomain::Complex,
                28,
            ),
            (Value::Bool(true), NumericDomain::Real, 1),
            (
                Value::LogicalArray(
                    runmat_value::LogicalArray::new(vec![], vec![0, 7, 2]).unwrap(),
                ),
                NumericDomain::Real,
                14,
            ),
            (
                Value::CharArray(
                    runmat_value::CharArray::from_unicode_scalars_with_shape(vec![], vec![0, 7, 2])
                        .unwrap(),
                ),
                NumericDomain::Real,
                14,
            ),
        ] {
            let mut admitted = route();
            admitted.input_domain = domain;
            let scan = if value_shape(&value).unwrap().contains(&0) {
                0.0
            } else {
                lanes as f64 * auto.thresholds.cpu_reduction_per_elem
            };
            assert_eq!(
                sum(&auto, &value, &admitted),
                Some(scan + lanes as f64 * auto.thresholds.cpu_elem_per_elem)
            );
            assert_eq!(batch_dimension_from_value(&value), None);
        }
    }
    #[test]
    fn absent_rate_keeps_execution_ordinary_without_numeric_candidate_or_feedback() {
        let config = ThresholdConfig {
            cpu_reduction_per_elem: 0.0,
            ..ThresholdConfig::default()
        };
        let auto = NativeAutoOffload::new(&OWNER, config);
        let queries_before = OWNER.queries.load(Ordering::SeqCst);
        let evaluation = auto.evaluate_reduction(8);
        let (evaluation, rejection, plan) = auto
            .plan_operation(
                evaluation,
                "legacy.reduction",
                ProviderOperationFamily::Reduction,
                &[],
                ProviderWorkload::default(),
                Default::default(),
            )
            .expect("the ordinary host route remains legal without a price");
        assert!(!evaluation.recommend_gpu);
        assert_eq!(evaluation.reason, DecisionReason::ProviderRejected);
        assert!(evaluation.cpu_secs.is_none());
        assert!(plan.feedback.is_none());
        assert_eq!(rejection, Some(ProviderRejectionCode::UnsupportedOperation));
        // Absence of a host price no longer bypasses actual provider query/
        // resource admission. This context-free call gains no learning right.
        assert_eq!(OWNER.queries.load(Ordering::SeqCst), queries_before + 1);
        // A rejected provider has no execution observation to start. Its
        // feasibility query above is distinct from that execution authority.
        assert!(plan.provider_query.is_none());
        assert!(!plan.provider_observation);
    }
    #[test]
    fn cache_identity_and_rates_are_checked() {
        let info = OWNER.device_info_struct();
        let mut record = CalibrationRecord {
            version: CALIBRATION_VERSION,
            recorded_at: 0,
            precision: ProviderPrecision::F64,
            provider: CalibrationProviderDetails {
                name: info.name.clone(),
                vendor: info.vendor.clone(),
                backend: info.backend.clone(),
                device_id: info.device_id,
            },
            thresholds: ThresholdConfig::default(),
        };
        assert!(record.matches(&info, ProviderPrecision::F64));
        assert!(!record.matches(&info, ProviderPrecision::F32));
        record.thresholds.cpu_elem_per_elem = 0.0;
        assert!(!record.matches(&info, ProviderPrecision::F64));
    }
}
