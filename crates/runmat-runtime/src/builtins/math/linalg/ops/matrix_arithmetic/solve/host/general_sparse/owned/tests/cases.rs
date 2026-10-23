use crate::context::host_allocation::HostAllocationPlan;
use runmat_value::{ComplexTensor, NumericDType, SparseTensor, Tensor, Value};

#[derive(Clone, Copy)]
pub(super) enum Class {
    RealF32,
    RealF64,
    ComplexF32,
    ComplexF64,
}
impl Class {
    pub(super) fn inputs(self) -> (Value, Value) {
        let pointers = vec![0, 1, 3];
        let rows = vec![1, 0, 1];
        match self {
            Self::RealF32 => (
                Value::SparseTensor(
                    SparseTensor::new_f32(2, 2, pointers, rows, vec![1.0, 2.0, 3.0]).unwrap(),
                ),
                Value::Tensor(Tensor::from_f32(vec![10.0, 19.0, -2.0, -1.0], vec![2, 2]).unwrap()),
            ),
            Self::RealF64 => (
                Value::SparseTensor(
                    SparseTensor::new(2, 2, pointers, rows, vec![1.0, 2.0, 3.0]).unwrap(),
                ),
                Value::Tensor(Tensor::new(vec![10.0, 19.0, -2.0, -1.0], vec![2, 2]).unwrap()),
            ),
            Self::ComplexF32 => (
                Value::SparseTensor(
                    SparseTensor::new_complex_f32(
                        2,
                        2,
                        pointers,
                        rows,
                        vec![(1.0, 0.25), (2.0, 0.5), (3.0, 0.75)],
                    )
                    .unwrap(),
                ),
                Value::ComplexTensor(
                    ComplexTensor::from_complex_storage(
                        runmat_value::ComplexStorage::F32(
                            vec![
                                runmat_value::ComplexElement(10.0, 2.5),
                                runmat_value::ComplexElement(19.0, 4.75),
                                runmat_value::ComplexElement(-2.0, -0.5),
                                runmat_value::ComplexElement(-1.0, -0.25),
                            ]
                            .into(),
                        ),
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
                        vec![(1.0, 0.25), (2.0, 0.5), (3.0, 0.75)],
                    )
                    .unwrap(),
                ),
                Value::ComplexTensor(
                    ComplexTensor::new(
                        vec![(10.0, 2.5), (19.0, 4.75), (-2.0, -0.5), (-1.0, -0.25)],
                        vec![2, 2],
                    )
                    .unwrap(),
                ),
            ),
        }
    }
    pub(super) fn quote(self) -> u64 {
        let (dtype, complex) = match self {
            Self::RealF32 => (NumericDType::F32, false),
            Self::RealF64 => (NumericDType::F64, false),
            Self::ComplexF32 => (NumericDType::F32, true),
            Self::ComplexF64 => (NumericDType::F64, true),
        };
        if complex {
            HostAllocationPlan::retained_complex_tensor_bytes(dtype, &[2, 2]).unwrap()
        } else {
            HostAllocationPlan::retained_tensor_bytes(dtype, &[2, 2]).unwrap()
        }
    }
    pub(super) fn verify(self, value: &Value) {
        let expected = [4.0, 5.0, 2.0, -1.0];
        match (self, value) {
            (Self::RealF32, Value::Tensor(tensor)) => {
                assert_eq!(tensor.shape(), &[2, 2]);
                for (&v, e) in tensor.as_f32_slice().unwrap().iter().zip(expected) {
                    assert!((f64::from(v) - e).abs() < 1e-5);
                }
            }
            (Self::RealF64, Value::Tensor(tensor)) => {
                assert_eq!(tensor.shape(), &[2, 2]);
                for (&v, e) in tensor.as_f64_slice().unwrap().iter().zip(expected) {
                    assert!((v - e).abs() < 1e-12);
                }
            }
            (Self::ComplexF32, Value::ComplexTensor(tensor)) => {
                assert_eq!(tensor.shape(), &[2, 2]);
                for (&v, e) in tensor.as_f32_slice().unwrap().iter().zip(expected) {
                    assert!((f64::from(v.0) - e).hypot(f64::from(v.1)) < 1e-5);
                }
            }
            (Self::ComplexF64, Value::ComplexTensor(tensor)) => {
                assert_eq!(tensor.shape(), &[2, 2]);
                for (&v, e) in tensor.as_f64_slice().unwrap().iter().zip(expected) {
                    assert!((v.0 - e).hypot(v.1) < 1e-12);
                }
            }
            _ => panic!("class changed during solve publication"),
        }
    }
}
