use super::*;
use runmat_value::{Tensor, Value};

#[test]
fn sobol_creation_streams_the_selected_direction_sequence_without_uniform_fallback() {
    futures::executor::block_on(async {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let mut cooperation =
            super::super::ScalarCooperation::for_builtin(super::super::IDENTITY, Some(&runtime));
        let mut draft = Draft::new(None, 4, 2, false, &mut cooperation)
            .await
            .unwrap();
        fill_sobol(&mut draft, &[(0., 1.), (0., 1.)], &mut cooperation)
            .await
            .unwrap();
        let population = draft.finish();
        assert_eq!(
            population.positions.as_f64_slice().unwrap(),
            [0., 0.5, 0.75, 0.25, 0., 0.5, 0.25, 0.75]
        );
    });
}

#[test]
fn partial_numeric_creation_preserves_prefix_and_bitstrings_validate_supplied_rows() {
    futures::executor::block_on(async {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let mut cooperation =
            super::super::ScalarCooperation::for_builtin(super::super::IDENTITY, Some(&runtime));
        let initial = Value::Tensor(Tensor::new(vec![-0., 3.], vec![1, 2]).unwrap());
        let mut draft = Draft::new(Some(initial), 3, 2, false, &mut cooperation)
            .await
            .unwrap();
        assert_eq!(draft.missing(), 1..3);
        draft
            .fill_uniform(&[(1., 1.), (4., 4.)], false, &mut cooperation)
            .await
            .unwrap();
        let population = draft.finish();
        let values = population.positions.as_f64_slice().unwrap();
        assert_eq!(values, [-0., 1., 1., 3., 4., 4.]);
        assert_eq!(values[0].to_bits(), (-0.0_f64).to_bits());
        let original = Tensor::new(vec![2., 4.], vec![2, 1]).unwrap();
        let original_data = original.as_f64_slice().unwrap().as_ptr();
        let mut full = Draft::new(
            Some(Value::Tensor(original.clone())),
            2,
            1,
            false,
            &mut cooperation,
        )
        .await
        .unwrap();
        full.set(0, &[2.], false, &mut cooperation).await.unwrap();
        let shared = full.finish();
        assert_eq!(
            shared.positions.as_f64_slice().unwrap().as_ptr(),
            original_data,
            "complete unchanged initial population must share backing"
        );
        let mut modified = Draft::new(Some(shared.value()), 2, 1, false, &mut cooperation)
            .await
            .unwrap();
        modified
            .set(0, &[3.], false, &mut cooperation)
            .await
            .unwrap();
        let modified = modified.finish();
        assert_ne!(
            modified.positions.as_f64_slice().unwrap().as_ptr(),
            original_data
        );
        assert_eq!(modified.positions.as_f64_slice().unwrap(), [3., 4.]);
        assert_eq!(original.as_f64_slice().unwrap(), [2., 4.]);
        let invalid = Draft::new(
            Some(Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).unwrap())),
            1,
            1,
            true,
            &mut cooperation,
        )
        .await
        .err()
        .expect("nonbinary supplied genome must fail before creation");
        assert!(invalid.to_string().contains("bitstrings"));
    });
}
