# Writes disc.part beside it (python3 gen_disc.py disc.part): a ventilated brake disc with 36 straight radial vanes.
import math, sys
N = 36
vane_t = 5.0      # mm
r_in, r_out = 95.0, 148.0
lines = []
for k in range(N):
    a = 2 * math.pi * k / N
    c, s = math.cos(a), math.sin(a)
    h = vane_t / 2
    pts = [(r_in, -h), (r_out, -h), (r_out, h), (r_in, h)]
    w = [(x * c - y * s, x * s + y * c) for x, y in pts]
    f = lambda p: f"[{p[0]:.4f} mm, {p[1]:.4f} mm]"
    for i, name in enumerate(["a", "b", "c", "d"]):
        lines.append(f"        v{k}_{name} = line(start = {f(w[i])}, end = {f(w[(i + 1) % 4])})")
vanes = "\n".join(lines)
vane_faces = ",\n".join(
    [f"            feature.vanes.face.side(vane_s.v{k}_{n})" for k in range(N) for n in "abcd"]
    + ["            feature.inboard.face.side(inboard_s.back)", "            feature.hat.face.side(hat_s.back)"])
pcd = 114.3 / 2
bolts = ",\n".join(
    f"            b{k + 1} = point({pcd * math.cos(2 * math.pi * k / 5):.4f} mm, {pcd * math.sin(2 * math.pi * k / 5):.4f} mm, input.hat_height)"
    for k in range(5))
text = f'''runmat 0.7
std 1

// A ventilated front brake disc in grey cast iron EN-GJL-250, 300 x 28 mm: an inboard and an
// outboard friction plate, 9 mm each, joined by 36 straight radial vanes, 5 mm thick and 10 mm
// tall, and a 45 mm deep hat that carries the outboard plate to the wheel hub (five 14 mm holes
// on a 114.3 mm pitch circle, a 66 mm centre bore). Air enters the vane channels at the inboard
// plate's 184 mm eye and leaves at the rim. "Vanes" names the channel walls: the vanes and the
// plates' inner faces between them.
//
// The disc's axis is z; the inboard friction face lies on z = 0 and the hub face on z = 60 mm.
// The plate, hat and bore sizes are inputs; the vane count and thickness come from the generator
// (runmat-cad/wasm/tests/sim/brake-disc, gen_disc.py), which writes the vane sketch.
part disc {{
    input outer_radius : length = 150 mm
    input plate : length = 9 mm
    input channel : length = 10 mm
    input eye_radius : length = 92 mm
    input hat_radius : length = 86 mm
    input hat_wall : length = 6 mm
    input hat_height : length = 60 mm
    input hub_face : length = 7 mm
    input bore_radius : length = 33 mm
    input bolt_hole : length = 14 mm

    let ring = input.plate * 2 + input.channel
    let outboard = input.plate + input.channel
    let hat_in = input.hat_radius - input.hat_wall
    let hub_under = input.hat_height - input.hub_face

    // The sections in the plane through the axis, (z, r) on it.
    sketch inboard_s on origin.zx {{
        face = line(start = [0 mm, input.eye_radius], end = [0 mm, input.outer_radius])
        rim = line(start = [0 mm, input.outer_radius], end = [input.plate, input.outer_radius])
        back = line(start = [input.plate, input.outer_radius], end = [input.plate, input.eye_radius])
        eye = line(start = [input.plate, input.eye_radius], end = [0 mm, input.eye_radius])
    }}
    feature inboard = revolve(profile = sketch.inboard_s, axis = origin.z, angle = 360 deg)
    sketch hat_s on origin.zx {{
        back = line(start = [let.outboard, let.hat_in], end = [let.outboard, input.outer_radius])
        rim = line(start = [let.outboard, input.outer_radius], end = [let.ring, input.outer_radius])
        face = line(start = [let.ring, input.outer_radius], end = [let.ring, input.hat_radius])
        wall = line(start = [let.ring, input.hat_radius], end = [input.hat_height, input.hat_radius])
        mount = line(start = [input.hat_height, input.hat_radius], end = [input.hat_height, input.bore_radius])
        bore = line(start = [input.hat_height, input.bore_radius], end = [let.hub_under, input.bore_radius])
        under = line(start = [let.hub_under, input.bore_radius], end = [let.hub_under, let.hat_in])
        inner = line(start = [let.hub_under, let.hat_in], end = [let.outboard, let.hat_in])
    }}
    feature hat = revolve(profile = sketch.hat_s, axis = origin.z, angle = 360 deg)
    feature vane_plane = plane.offset(base = origin.xy, distance = input.plate)
    sketch vane_s on plane.vane_plane {{
{vanes}
    }}
    feature vanes = extrude(profile = sketch.vane_s, distance = input.channel)
    feature rotor = boolean.union(target = feature.hat.body, tools = [feature.inboard.body, feature.vanes.body])
    feature wheel_bolts = hole(
        face = feature.hat.face.side(hat_s.mount),
        at = {{
{bolts}
        }},
        diameter = input.bolt_hole,
        through_all
    )

    region friction_inboard : face = {{ select = [feature.inboard.face.side(inboard_s.face)], expect = exactly_one, label = "Friction face inboard" }}
    region friction_outboard : face = {{ select = [feature.hat.face.side(hat_s.face)], expect = exactly_one, label = "Friction face outboard" }}
    region vane_faces : face = {{
        select = [
{vane_faces}
        ],
        expect = at_least_one,
        label = "Vanes"
    }}
    region bolt_holes : face = {{
        select = [
            feature.wheel_bolts.face.wall(b1),
            feature.wheel_bolts.face.wall(b2),
            feature.wheel_bolts.face.wall(b3),
            feature.wheel_bolts.face.wall(b4),
            feature.wheel_bolts.face.wall(b5)
        ],
        expect = at_least_one,
        label = "Bolt holes"
    }}
    region exterior : face = {{
        select = [
            feature.inboard.face.side(inboard_s.rim),
            feature.inboard.face.side(inboard_s.eye),
            feature.hat.face.side(hat_s.rim),
            feature.hat.face.side(hat_s.wall),
            feature.hat.face.side(hat_s.mount),
            feature.hat.face.side(hat_s.bore),
            feature.hat.face.side(hat_s.under),
            feature.hat.face.side(hat_s.inner)
        ],
        expect = at_least_one,
        label = "Exterior"
    }}

    output rotor = {{ body = feature.hat.body, label = "Rotor", part_number = "BD-300V", material = "grey_iron_en_gjl_250", color = "#6F7378" }}
}}
'''
open(sys.argv[1], "w").write(text)
