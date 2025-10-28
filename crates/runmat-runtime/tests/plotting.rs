use runmat_builtins::Value;
use runmat_runtime::call_builtin;

fn tensor(data: &[f64], rows: usize, cols: usize) -> Value {
    let t = runmat_builtins::Tensor { data: data.to_vec(), shape: vec![rows, cols], rows, cols };
    Value::Tensor(t)
}

#[test]
fn test_errorbar_builtin_variants() {
    std::env::set_var("RUNMAT_PLOT_MODE", "headless");
    // errorbar(Y)
    let y = tensor(&[1.0, 2.0, 1.5], 3, 1);
    let r = runmat_plot::show_plot_unified({
        let mut f = runmat_plot::plots::Figure::new();
        let yv = if let Value::Tensor(t) = y { t.data } else { vec![] };
        let xv: Vec<f64> = (1..=yv.len()).map(|k| k as f64).collect();
        let el = vec![0.0; yv.len()];
        let mut p = runmat_plot::plots::ErrorBar::new(xv, yv.clone(), el.clone(), el).unwrap();
        p = p.with_style(glam::Vec4::new(0.0,0.0,0.0,1.0), 1.0, 0.02);
        f.add_errorbar(p);
        f
    }, Some("/tmp/test_errorbar1.png"));
    assert!(r.is_ok());

    // errorbar(X,Y,E)
    let x = tensor(&[0.0, 1.0, 2.0], 3, 1);
    let y = tensor(&[1.0, 2.0, 1.5], 3, 1);
    let e = tensor(&[0.2, 0.1, 0.3], 3, 1);
    let r = runmat_plot::show_plot_unified({
        let mut f = runmat_plot::plots::Figure::new();
        let xv = if let Value::Tensor(t) = x { t.data } else { vec![] };
        let yv = if let Value::Tensor(t) = y { t.data } else { vec![] };
        let ev = if let Value::Tensor(t) = e { t.data } else { vec![] };
        let mut p = runmat_plot::plots::ErrorBar::new(xv, yv, ev.clone(), ev).unwrap();
        p = p.with_style(glam::Vec4::new(0.0,0.0,0.0,1.0), 1.0, 0.02);
        f.add_errorbar(p);
        f
    }, Some("/tmp/test_errorbar2.png"));
    assert!(r.is_ok());
}

#[test]
fn test_stairs_and_stem_builtins() {
    std::env::set_var("RUNMAT_PLOT_MODE", "headless");
    let y = tensor(&[1.0, 2.0, 1.5, 2.5], 4, 1);
    {
        let mut f = runmat_plot::plots::Figure::new();
        let yv = if let Value::Tensor(t) = y.clone() { t.data } else { vec![] };
        let xv: Vec<f64> = (1..=yv.len()).map(|k| k as f64).collect();
        let mut p = runmat_plot::plots::StairsPlot::new(xv.clone(), yv.clone()).unwrap();
        p = p.with_style(glam::Vec4::new(0.0,0.5,1.0,1.0), 1.0);
        f.add_stairs_plot(p);
        assert!(runmat_plot::show_plot_unified(f, Some("/tmp/test_stairs.png")).is_ok());
    }
    {
        let mut f = runmat_plot::plots::Figure::new();
        let yv = if let Value::Tensor(t) = y { t.data } else { vec![] };
        let xv: Vec<f64> = (1..=yv.len()).map(|k| k as f64).collect();
        let mut p = runmat_plot::plots::StemPlot::new(xv, yv).unwrap();
        p = p.with_style(glam::Vec4::new(0.0,0.0,0.0,1.0), glam::Vec4::new(0.0,0.5,1.0,1.0), 0.0);
        f.add_stem_plot(p);
        assert!(runmat_plot::show_plot_unified(f, Some("/tmp/test_stem.png")).is_ok());
    }
}

#[test]
fn test_area_builtin_stacked_grouped() {
    std::env::set_var("RUNMAT_PLOT_MODE", "headless");
    // Single series
    let y = tensor(&[1.0, 2.0, 0.5], 3, 1);
    {
        let mut f = runmat_plot::plots::Figure::new();
        let yv = if let Value::Tensor(t) = y { t.data } else { vec![] };
        let xv: Vec<f64> = (1..=yv.len()).map(|k| k as f64).collect();
        let mut p = runmat_plot::plots::AreaPlot::new(xv, yv).unwrap();
        p = p.with_style(glam::Vec4::new(0.0,0.5,1.0,0.4), 0.0);
        f.add_area_plot(p);
        assert!(runmat_plot::show_plot_unified(f, Some("/tmp/test_area_single.png")).is_ok());
    }

    // Stacked: Y is matrix [rows=3, cols=2]
    let ymat = tensor(&[1.0, 2.0, 0.5,  0.5, 0.5, 0.5], 3, 2);
    {
        let mut f = runmat_plot::plots::Figure::new();
        let t = if let Value::Tensor(t) = ymat { t } else { panic!("expected tensor") };
        let rows = t.rows; let cols = t.cols;
        let xv: Vec<f64> = (1..=rows).map(|k| k as f64).collect();
        let mut acc = vec![0.0f64; rows];
        for c in 0..cols {
            let y: Vec<f64> = (0..rows).map(|r| acc[r] + t.data[r + c*rows]).collect();
            let mut p = runmat_plot::plots::AreaPlot::new(xv.clone(), y.clone()).unwrap();
            p = p.with_style(glam::Vec4::new(0.0,0.5,1.0,0.4), 0.0);
            f.add_area_plot(p);
            acc = y;
        }
        assert!(runmat_plot::show_plot_unified(f, Some("/tmp/test_area_stacked.png")).is_ok());
    }

    // Grouped
    let ymat = tensor(&[1.0, 2.0, 0.5,  0.5, 0.5, 0.5], 3, 2);
    {
        let mut f = runmat_plot::plots::Figure::new();
        let t = if let Value::Tensor(t) = ymat { t } else { panic!("expected tensor") };
        let rows = t.rows; let cols = t.cols;
        let xv: Vec<f64> = (1..=rows).map(|k| k as f64).collect();
        for c in 0..cols {
            let y: Vec<f64> = (0..rows).map(|r| t.data[r + c*rows]).collect();
            let mut p = runmat_plot::plots::AreaPlot::new(xv.clone(), y).unwrap();
            p = p.with_style(glam::Vec4::new(0.0,0.5,1.0,0.4), 0.0);
            f.add_area_plot(p);
        }
        assert!(runmat_plot::show_plot_unified(f, Some("/tmp/test_area_grouped.png")).is_ok());
    }
}

#[test]
fn test_quiver_and_pie_end_to_end() {
    std::env::set_var("RUNMAT_PLOT_MODE", "headless");
    // Quiver
    {
        let mut f = runmat_plot::plots::Figure::new();
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 0.0];
        let u = vec![1.0, 0.0, -1.0];
        let v = vec![0.0, 1.0, 0.0];
        let mut q = runmat_plot::plots::QuiverPlot::new(x, y, u, v).unwrap();
        q = q.with_style(glam::Vec4::new(0.0,0.0,0.0,1.0), 1.5, 1.0, 0.1);
        f.add_quiver_plot(q);
        assert!(runmat_plot::show_plot_unified(f, Some("/tmp/test_quiver.png")).is_ok());
    }

    // Pie
    {
        let mut f = runmat_plot::plots::Figure::new();
        let vals = vec![1.0, 2.0, 3.0, 4.0];
        let p = runmat_plot::plots::PieChart::new(vals, None).unwrap();
        f.add_pie_chart(p);
        assert!(runmat_plot::show_plot_unified(f, Some("/tmp/test_pie.png")).is_ok());
    }
}

#[test]
fn test_runtime_scatter_varargs_and_legend() {
    std::env::set_var("RUNMAT_PLOT_MODE", "headless");
    use runmat_builtins::Value;
    let x = tensor(&[0.0, 1.0, 2.0, 3.0], 4, 1);
    let y = tensor(&[0.1, 0.2, 0.3, 0.4], 4, 1);
    // scatter(X,Y,S,C,'filled','DisplayName','Series A')
    let s = tensor(&[16.0, 25.0, 36.0, 49.0], 4, 1);
    let c = tensor(&[1.0, 0.0, 0.0,  0.0,1.0,0.0,  0.0,0.0,1.0,  1.0,1.0,1.0], 4, 3);
    // Call builtin via plotting module directly is not exposed here; instead, build a figure using API
    let mut f = runmat_plot::plots::Figure::new();
    let xv = if let Value::Tensor(t) = x { t.data } else { vec![] };
    let yv = if let Value::Tensor(t) = y { t.data } else { vec![] };
    let sv = if let Value::Tensor(t) = s { t.data } else { vec![] };
    let cv = if let Value::Tensor(t) = c { t } else { panic!("expected tensor for colors") };
    let mut sp = runmat_plot::plots::ScatterPlot::new(xv, yv).unwrap().with_style(glam::Vec4::new(0.5,0.5,0.5,1.0), 5.0, runmat_plot::plots::scatter::MarkerStyle::Circle);
    sp.set_sizes(sv.into_iter().map(|v| (v as f32).sqrt()).collect());
    let rows = cv.rows; let mut cols_vec = Vec::with_capacity(rows);
    for r in 0..rows { cols_vec.push(glam::Vec4::new(cv.data[r] as f32, cv.data[r+rows] as f32, cv.data[r+2*rows] as f32, 1.0)); }
    sp.set_colors(cols_vec);
    sp.set_filled(true);
    sp = sp.with_label("Series A");
    f.add_scatter_plot(sp);
    // Legend enabled by default
    assert!(runmat_plot::show_plot_unified(f.clone(), Some("/tmp/test_runtime_scatter.png")).is_ok());
    let legend = f.legend_entries();
    assert_eq!(legend.len(), 1);
    assert_eq!(legend[0].label, "Series A");
}

#[test]
fn test_subplot_assignments() {
    std::env::set_var("RUNMAT_PLOT_MODE", "headless");
    let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| v.sin()).collect();
    let y2: Vec<f64> = x.iter().map(|v| v.cos()).collect();
    let mut f = runmat_plot::plots::Figure::new();
    f.set_subplot_grid(2, 1);
    // First axes
    let lp1 = runmat_plot::plots::LinePlot::new(x.clone(), y.clone()).unwrap().with_style(glam::Vec4::new(0.0,0.5,1.0,1.0), 1.5, runmat_plot::plots::line::LineStyle::Solid);
    let i1 = f.add_line_plot(lp1);
    let _ = f.assign_plot_to_axes(i1, 0);
    // Second axes
    let lp2 = runmat_plot::plots::LinePlot::new(x.clone(), y2.clone()).unwrap().with_style(glam::Vec4::new(1.0,0.5,0.0,1.0), 1.5, runmat_plot::plots::line::LineStyle::Solid);
    let i2 = f.add_line_plot(lp2);
    let _ = f.assign_plot_to_axes(i2, 1);
    let (rows, cols) = f.axes_grid();
    assert_eq!((rows, cols), (2,1));
    let indices = f.plot_axes_indices();
    assert_eq!(indices.len(), 2);
    assert_ne!(indices[0], indices[1]);
    assert!(runmat_plot::show_plot_unified(f, Some("/tmp/test_subplot_assign.png")).is_ok());
}

#[test]
fn test_hist_and_histogram_options() {
    std::env::set_var("RUNMAT_PLOT_MODE", "headless");
    use runmat_builtins::Tensor;
    // hist delegates to histogram with defaults
    let v = tensor(&[1.0,2.0,2.5,3.0,4.0], 5, 1);
    let tv: Tensor = std::convert::TryInto::try_into(&v).unwrap();
    assert!(call_builtin("hist", &[Value::Tensor(tv)]).is_ok());
    // histogram name-value coverage
    let edges = tensor(&[1.0,2.0,3.0,4.0], 4, 1);
    assert!(call_builtin("histogram", &[v.clone(), Value::String("NumBins".to_string()), Value::from(5.0)]).is_ok());
    assert!(call_builtin("histogram", &[v.clone(), Value::String("BinWidth".to_string()), Value::from(0.5)]).is_ok());
    assert!(call_builtin("histogram", &[v.clone(), Value::String("BinEdges".to_string()), edges]).is_ok());
    let limits = tensor(&[1.0, 4.0], 2, 1);
    assert!(call_builtin("histogram", &[v.clone(), Value::String("BinLimits".to_string()), limits]).is_ok());
    assert!(call_builtin("histogram", &[v.clone(), Value::String("Normalization".to_string()), Value::String("pdf".to_string())]).is_ok());
    assert!(call_builtin("histogram", &[v, Value::String("FaceColor".to_string()), Value::String("r".to_string()), Value::String("EdgeColor".to_string()), Value::String("k".to_string()), Value::String("DisplayName".to_string()), Value::String("H1".to_string())]).is_ok());
}


#[cfg(test)]
mod bar_builtin_tests {
	use runmat_builtins::Value;
	use runmat_runtime::call_builtin;
	use std::sync::Once;

	static INIT: Once = Once::new();
	fn init_headless() {
		INIT.call_once(|| {
			std::env::set_var("RUNMAT_PLOT_MODE", "headless");
		});
	}

	fn tensor(data: &[f64], shape: &[usize]) -> Value {
		let t = runmat_builtins::Tensor { data: data.to_vec(), shape: shape.to_vec(), rows: shape.get(0).copied().unwrap_or(0), cols: shape.get(1).copied().unwrap_or(0) };
		Value::Tensor(t)
	}

	#[test]
	fn bar_vector_default() {
		init_headless();
		let y = tensor(&[1.0, 3.0, 2.0], &[3, 1]);
		let res = call_builtin("bar", &[y]);
		assert!(res.is_ok());
	}

	#[test]
	fn bar_with_numeric_x() {
		init_headless();
		let x = tensor(&[10.0, 20.0, 30.0], &[3, 1]);
		let y = tensor(&[5.0, 6.0, 7.0], &[3, 1]);
		let res = call_builtin("bar", &[x, y]);
		assert!(res.is_ok());
	}

	#[test]
	fn bar_categorical_x() {
		init_headless();
		// Cell array of strings emulated via Value::Cell with Value::String elements
		let cats = runmat_builtins::CellArray::new(
			vec![
				Value::String("A".to_string()),
				Value::String("B".to_string()),
				Value::String("C".to_string()),
			],
			1,
			3,
		).unwrap();
		let x = Value::Cell(cats);
		let y = tensor(&[2.0, 1.0, 3.0], &[3, 1]);
		let res = call_builtin("bar", &[x, y]);
		assert!(res.is_ok());
	}

	#[test]
	fn bar_grouped_matrix() {
		init_headless();
		let y = tensor(&[1.0, 2.0, 3.0, 2.0, 1.0, 0.0], &[3, 2]);
		let res = call_builtin("bar", &[y, Value::String("grouped".to_string())]);
		assert!(res.is_ok());
	}

	#[test]
	fn bar_stacked_matrix() {
		init_headless();
		let y = tensor(&[1.0, -2.0, 3.0, 2.0, -1.0, 1.5], &[3, 2]);
		let res = call_builtin("bar", &[y, Value::String("stacked".to_string())]);
		assert!(res.is_ok());
	}

	#[test]
	fn barh_alias_and_name_values() {
		init_headless();
		let y = tensor(&[4.0, 1.0], &[2, 1]);
		let res = call_builtin("barh", &[
			y,
			Value::String("DisplayName".to_string()), Value::String("Series".to_string()),
			Value::String("FaceColor".to_string()), Value::String("r".to_string()),
			Value::String("EdgeColor".to_string()), Value::String("k".to_string()),
		]);
		assert!(res.is_ok());
	}
}
