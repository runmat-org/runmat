//! Measure's item labels name each component occurrence by its full path, so two occurrences
//! that share a name (one part in two copies of a subassembly) read apart.

use super::measure;
use crate::runtime::Files;
use runmat_cad::regen::kernel::TessellationOptions;
use serde_json::{json, Map};

const HUB: &str = "runmat 0.7\nstd 1\n\npart hub {\n    sketch s1 on origin.xy {\n        c = circle(center = [0 mm, 0 mm], radius = 10 mm)\n    }\n    feature body = extrude(profile = sketch.s1, distance = 8 mm)\n}\n";

const STATION: &str = "runmat 0.7\nstd 1\n\nassembly station {\n    use part hub as hub\n}\n";

const DRONE: &str = "runmat 0.7\nstd 1\n\nassembly drone {\n    use assembly station as left\n    use assembly station as right {\n        placement = { placement = offset, offset = { translation = [100 mm, 0 mm, 0 mm] } }\n    }\n}\n";

fn files() -> Files {
    [
        ("/p/hub.part", HUB),
        ("/p/station.assembly", STATION),
        ("/p/drone.assembly", DRONE),
    ]
    .map(|(p, t)| (p.to_owned(), t.to_owned()))
    .into()
}

fn uuid(path: &str) -> String {
    let (index, _, _) =
        crate::runtime::compile_assembly("/p/drone.assembly", DRONE, &files()).unwrap();
    index["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["path"] == path)
        .unwrap_or_else(|| panic!("no occurrence {path}: {index}"))["uuid"]
        .as_str()
        .unwrap()
        .to_owned()
}

#[test]
fn occurrences_that_share_a_name_read_by_their_paths() {
    let hub = crate::runtime::regenerate_part_in(
        "/p/hub.part",
        HUB,
        None,
        &TessellationOptions::default(),
    )
    .unwrap_or_else(|d| panic!("{:?}", d.iter().map(|d| &d.message).collect::<Vec<_>>()))
    .display;
    let body = &hub.bodies[0];
    let rim = format!("edge:{}:{}", body.index, body.edges[0].index);
    let (left, right) = (uuid("left/hub"), uuid("right/hub"));
    let mut values = Map::new();
    values.insert("mode".into(), json!("distance"));
    values.insert(
        "items".into(),
        json!([
            format!("occurrence:{left}/{rim}"),
            format!("occurrence:{right}/{rim}")
        ]),
    );
    let a = measure("/p/drone.assembly", DRONE, &files(), &values).unwrap();
    assert_eq!(a.settings.len(), 2, "{:?}", a.settings);
    assert!(a.settings[0].starts_with("left/hub · "), "{:?}", a.settings);
    assert!(a.settings[1].starts_with("right/hub · "), "{:?}", a.settings);
    assert_ne!(a.settings[0], a.settings[1]);
    // A whole component reads by its path too.
    values.insert(
        "items".into(),
        json!([format!("occurrence:{left}"), format!("occurrence:{right}")]),
    );
    let b = measure("/p/drone.assembly", DRONE, &files(), &values).unwrap();
    assert_eq!(b.settings, vec!["left/hub".to_owned(), "right/hub".to_owned()]);
}
