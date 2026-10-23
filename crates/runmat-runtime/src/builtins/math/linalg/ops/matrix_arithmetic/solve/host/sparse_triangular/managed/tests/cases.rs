use super::*;
#[derive(Clone, Copy)]
pub(super) enum Class {
    RealF32,
    RealF64,
    ComplexF32,
    ComplexF64,
}
impl Class {
    pub(super) fn inputs(self, lower: bool) -> (Value, Value) {
        let (pointers, rows) = if lower {
            (vec![0, 2, 3], vec![0, 1, 1])
        } else {
            (vec![0, 1, 3], vec![0, 0, 1])
        };
        let rhs = if lower {
            vec![2.0, 11.0, 4.0, 2.0]
        } else {
            vec![8.0, 8.0, 1.0, -4.0]
        };
        match self {
            Self::RealF32 => (
                Value::SparseTensor(
                    SparseTensor::new_f32(2, 2, pointers, rows, vec![2.0, 3.0, 4.0]).unwrap(),
                ),
                Value::Tensor(
                    Tensor::from_f32(rhs.into_iter().map(|x| x as f32).collect(), vec![2, 2])
                        .unwrap(),
                ),
            ),
            Self::RealF64 => (
                Value::SparseTensor(
                    SparseTensor::new(2, 2, pointers, rows, vec![2.0, 3.0, 4.0]).unwrap(),
                ),
                Value::Tensor(Tensor::new(rhs, vec![2, 2]).unwrap()),
            ),
            Self::ComplexF32 => (
                Value::SparseTensor(
                    SparseTensor::new_complex_f32(
                        2,
                        2,
                        pointers,
                        rows,
                        vec![(2.0, 0.5), (3.0, 0.75), (4.0, 1.0)],
                    )
                    .unwrap(),
                ),
                Value::ComplexTensor(
                    ComplexTensor::from_f32(
                        rhs.into_iter()
                            .map(|v| (v as f32, (v * 0.25) as f32))
                            .collect(),
                        vec![2, 2],
                    )
                    .unwrap(),
                ),
            ),
            Self::ComplexF64 => (
                Value::SparseTensor(
                    SparseTensor::new_complex(
                        2,
                        2,
                        pointers,
                        rows,
                        vec![(2.0, 0.5), (3.0, 0.75), (4.0, 1.0)],
                    )
                    .unwrap(),
                ),
                Value::ComplexTensor(
                    ComplexTensor::new(
                        rhs.into_iter().map(|v| (v, v * 0.25)).collect(),
                        vec![2, 2],
                    )
                    .unwrap(),
                ),
            ),
        }
    }
    pub(super) fn quote(self) -> u64 {
        match self {
            Self::RealF32 => HostAllocationPlan::retained_tensor_bytes(NumericDType::F32, &[2, 2]),
            Self::RealF64 => HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[2, 2]),
            Self::ComplexF32 => {
                HostAllocationPlan::retained_complex_tensor_bytes(NumericDType::F32, &[2, 2])
            }
            Self::ComplexF64 => {
                HostAllocationPlan::retained_complex_tensor_bytes(NumericDType::F64, &[2, 2])
            }
        }
        .unwrap()
    }
    pub(super) fn verify(self, value: &Value) {
        let expected = [1.0, 2.0, 2.0, -1.0];
        match (self, value) {
            (Self::RealF32, Value::Tensor(t)) => {
                assert_eq!(t.shape(), &[2, 2]);
                for (&v, e) in t.as_f32_slice().unwrap().iter().zip(expected) {
                    assert!((f64::from(v) - e).abs() < 1e-5);
                }
            }
            (Self::RealF64, Value::Tensor(t)) => {
                assert_eq!(t.shape(), &[2, 2]);
                for (&v, e) in t.as_f64_slice().unwrap().iter().zip(expected) {
                    assert!((v - e).abs() < 1e-12);
                }
            }
            (Self::ComplexF32, Value::ComplexTensor(t)) => {
                assert_eq!(t.shape(), &[2, 2]);
                for (&v, e) in t.as_f32_slice().unwrap().iter().zip(expected) {
                    assert!((f64::from(v.0) - e).hypot(f64::from(v.1)) < 1e-5);
                }
            }
            (Self::ComplexF64, Value::ComplexTensor(t)) => {
                assert_eq!(t.shape(), &[2, 2]);
                for (&v, e) in t.as_f64_slice().unwrap().iter().zip(expected) {
                    assert!((v.0 - e).hypot(v.1) < 1e-12);
                }
            }
            _ => panic!("triangular native class changed"),
        }
    }
}
