use super::*;
use clarabel::algebra::CscMatrix;

fn make_point(n: usize) -> Point {
    Point {
        x: vec![0.0; n],
        f: 0.0,
        inequality: Vec::new().into(),
        equality: Vec::new().into(),
    }
}
fn parameters() -> super::super::mesh::Parameters {
    super::super::mesh::Parameters {
        initial: 1.0,
        maximum: f64::INFINITY,
        expansion: 2.0,
        contraction: 0.5,
        tolerance: 0.9,
        rotate: false,
        accelerate: false,
    }
}
#[tokio::test]
async fn original_half_polls_complete_a_spanning_pair_without_redrawing_random_frame() {
    let mut cooperation = ScalarCooperation::for_builtin(super::super::IDENTITY, None);
    let geometry = Linear::new(
        CscMatrix::new(0, 2, vec![0, 0, 0], vec![], vec![]),
        vec![],
        0,
        &mut cooperation,
    )
    .await
    .unwrap();
    let scaling = Scaling::unit(2);
    let point = make_point(2);
    let mut driver = Driver::new(OptimPatternAlgorithm::NupsMads, false, 1.0, 1e-3, 1e-6).unwrap();
    let mut mesh = Mesh::nonuniform(parameters()).unwrap();
    let first = {
        let frame = driver
            .frame(
                &point,
                &geometry,
                mesh.size(),
                &scaling,
                1e-3,
                &mut cooperation,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(frame.count().unwrap(), 3);
        frame.direction(0, &mut cooperation).await.unwrap().unwrap()
    };
    assert_eq!(
        driver.completed(None, 3, &mut mesh).unwrap(),
        mesh::Update::Continue
    );
    assert!(
        !driver.covered(),
        "a positive half is not a convergence certificate"
    );
    let second = {
        let frame = driver
            .frame(
                &point,
                &geometry,
                mesh.size(),
                &scaling,
                1e-3,
                &mut cooperation,
            )
            .await
            .unwrap()
            .unwrap();
        frame.direction(0, &mut cooperation).await.unwrap().unwrap()
    };
    assert_eq!(second, first.iter().map(|v| -*v).collect::<Vec<_>>());
    assert_eq!(
        driver.completed(None, 3, &mut mesh).unwrap(),
        mesh::Update::BelowTolerance
    );
    assert!(driver.covered());
    driver.configure(1e-2, 1e-6, false);
    assert!(
        !driver.covered(),
        "changed boundary geometry invalidates coverage"
    );
}

#[tokio::test]
async fn axes_include_positive_sum_and_original_equalities_and_narrow_bounds_remain_authoritative()
{
    let mut cooperation = ScalarCooperation::for_builtin(super::super::IDENTITY, None);
    let scaling = Scaling::unit(2);
    let point = make_point(2);
    let free = Linear::new(
        CscMatrix::new(0, 2, vec![0, 0, 0], vec![], vec![]),
        vec![],
        0,
        &mut cooperation,
    )
    .await
    .unwrap();
    for positive in [true, false] {
        let frame = Frame::new(
            &point,
            &free,
            1.0,
            &scaling,
            frame::Orientation::Gps,
            positive,
            1e-3,
            &mut cooperation,
        )
        .await
        .unwrap()
        .unwrap();
        let sum = frame.direction(2, &mut cooperation).await.unwrap().unwrap();
        assert_eq!(
            sum,
            if positive {
                vec![1.0, 1.0]
            } else {
                vec![-1.0, -1.0]
            }
        );
    }
    let equality = Linear::new(
        CscMatrix::new(1, 2, vec![0, 1, 1], vec![0], vec![1.0]),
        vec![0.0],
        1,
        &mut cooperation,
    )
    .await
    .unwrap();
    let frame = Frame::new(
        &point,
        &equality,
        1.0,
        &scaling,
        frame::Orientation::Gps,
        true,
        1e-3,
        &mut cooperation,
    )
    .await
    .unwrap()
    .unwrap();
    assert!(frame.spans_pair());
    assert_eq!(frame.count().unwrap(), 2);
    for index in 0..2 {
        let ray = frame
            .direction(index, &mut cooperation)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(ray[0], 0.0);
        assert!(equality
            .candidate_with_step(&point.x, &ray, 1.0, 1e-12, &mut cooperation)
            .await
            .unwrap()
            .is_ok());
    }
    let narrow = Linear::new(
        CscMatrix::new(2, 1, vec![0, 2], vec![0, 1], vec![-1.0, 1.0]),
        vec![5e-7, 5e-7],
        0,
        &mut cooperation,
    )
    .await
    .unwrap();
    let scaling = Scaling::unit(1);
    let point = make_point(1);
    let frame = Frame::new(
        &point,
        &narrow,
        1.0,
        &scaling,
        frame::Orientation::Gps,
        true,
        1e-3,
        &mut cooperation,
    )
    .await
    .unwrap()
    .unwrap();
    let ray = frame.direction(0, &mut cooperation).await.unwrap().unwrap();
    let prefix = narrow
        .candidate_with_step(&point.x, &ray, 1.0, 1e-12, &mut cooperation)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(prefix.point, [5e-7]);
}
