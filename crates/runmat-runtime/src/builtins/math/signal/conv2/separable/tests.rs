use futures::executor::block_on;

use super::*;

fn real_matrix(rows: usize, cols: usize, data: &[f64]) -> Matrix {
    Matrix {
        rows,
        cols,
        data: data
            .iter()
            .copied()
            .map(|real| Complex::new(real, 0.0))
            .collect(),
    }
}

#[test]
fn two_pass_full_same_and_valid_use_the_original_signal_geometry() {
    let signal = real_matrix(2, 2, &[1.0, 3.0, 2.0, 4.0]);
    let column = [Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
    let row = [Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0)];
    for (mode, shape, expected) in [
        (
            Conv2Mode::Full,
            (3, 3),
            vec![1.0, 5.0, 6.0, 1.0, 3.0, 2.0, -2.0, -8.0, -8.0],
        ),
        (Conv2Mode::Same, (2, 2), vec![3.0, 2.0, -8.0, -8.0]),
        (Conv2Mode::Valid, (1, 1), vec![3.0]),
    ] {
        let mut cooperation = SignalCooperation::new(None, "conv2");
        let output = block_on(convolve_separable(
            &signal,
            &column,
            &row,
            mode,
            false,
            &mut cooperation,
        ))
        .expect("separable convolution");
        assert_eq!((output.rows, output.cols), shape);
        assert_eq!(
            output.data.iter().map(|value| value.re).collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn empty_and_short_valid_cases_do_not_allocate_full_intermediates() {
    let empty = real_matrix(0, 3, &[]);
    let column = [Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
    let row = [Complex::new(1.0, 0.0)];
    let mut cooperation = SignalCooperation::new(None, "conv2");
    let same = block_on(convolve_separable(
        &empty,
        &column,
        &row,
        Conv2Mode::Same,
        false,
        &mut cooperation,
    ))
    .expect("empty Same");
    assert_eq!((same.rows, same.cols), (0, 3));
    let full = block_on(convolve_separable(
        &empty,
        &column,
        &row,
        Conv2Mode::Full,
        false,
        &mut cooperation,
    ))
    .expect("empty Full");
    assert_eq!((full.rows, full.cols), (0, 0));
    let short = real_matrix(1, 1, &[2.0]);
    let valid = block_on(convolve_separable(
        &short,
        &column,
        &row,
        Conv2Mode::Valid,
        false,
        &mut cooperation,
    ))
    .expect("short Valid");
    assert_eq!((valid.rows, valid.cols), (0, 0));
}

#[test]
fn balanced_factors_keep_finite_extreme_products_finite() {
    let signal = real_matrix(1, 1, &[2.0]);
    for (column_value, row_value) in [(1e308, 1e-308), (1e-308, 1e308)] {
        let column = [Complex::new(column_value, 0.0)];
        let row = [Complex::new(row_value, 0.0)];
        let mut cooperation = SignalCooperation::new(None, "conv2");
        let output = block_on(convolve_separable(
            &signal,
            &column,
            &row,
            Conv2Mode::Full,
            false,
            &mut cooperation,
        ))
        .expect("finite extreme product");
        let expected = 2.0 * (column_value * row_value);
        assert!(output.data[0].re.is_finite());
        assert!((output.data[0].re - expected).abs() <= 1e-12);
    }
}

#[test]
fn balanced_complex_and_single_passes_preserve_output_domain() {
    let signal = real_matrix(1, 1, &[2.0]);
    let column = [Complex::new(1e308, 0.0)];
    let row = [Complex::new(0.0, 1e-308)];
    let mut cooperation = SignalCooperation::new(None, "conv2");
    let complex = block_on(convolve_separable(
        &signal,
        &column,
        &row,
        Conv2Mode::Full,
        false,
        &mut cooperation,
    ))
    .expect("finite complex product");
    assert!(complex.data[0].im.is_finite());
    assert!((complex.data[0].im - 2.0).abs() <= 1e-12);

    let column = [Complex::new(1e30, 0.0)];
    let row = [Complex::new(1e-30, 0.0)];
    let single = block_on(convolve_separable(
        &signal,
        &column,
        &row,
        Conv2Mode::Full,
        true,
        &mut cooperation,
    ))
    .expect("finite single product");
    assert!(single.data[0].re.is_finite());
    assert!((single.data[0].re - 2.0).abs() <= 1e-6);
}

#[test]
fn unsafe_intermediate_uses_general_kernel_before_two_pass_work() {
    let signal = real_matrix(2, 1, &[1e308, 1e308]);
    let column = [Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
    let row = [Complex::new(1e-308, 0.0)];
    let mut cooperation = SignalCooperation::new(None, "conv2");
    let output = block_on(convolve_separable(
        &signal,
        &column,
        &row,
        Conv2Mode::Full,
        false,
        &mut cooperation,
    ))
    .expect("finite general-kernel output");
    assert_eq!((output.rows, output.cols), (3, 1));
    for (actual, expected) in output.data.iter().zip([1.0, 2.0, 1.0]) {
        assert!(actual.re.is_finite());
        assert!((actual.re - expected).abs() <= 1e-12);
    }

    let tiny = real_matrix(1, 1, &[f64::from_bits(1)]);
    let tiny_column = [Complex::new(1e-308, 0.0)];
    let huge_row = [Complex::new(1e308, 0.0)];
    let output = block_on(convolve_separable(
        &tiny,
        &tiny_column,
        &huge_row,
        Conv2Mode::Full,
        false,
        &mut cooperation,
    ))
    .expect("subnormal general-kernel output");
    assert_eq!(output.data[0].re.to_bits(), f64::from_bits(1).to_bits());
}
