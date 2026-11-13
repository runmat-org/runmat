---
description: Build a product family in RunMat MCAD. Give one part several configurations with their own sizes, suppressed features and part numbers, make left- and right-hand versions, place each configuration in an assembly, and export and draw every member of the family.
keywords: [product family, configurations, design table, dash numbers, part numbers, suppress feature, left hand, right hand, opposite hand, mirror, family of parts]
---
# Product families and configurations

A product family is one design sold or built in several versions: lengths, thicknesses, with or without a feature, left and right hand. In RunMat MCAD the family lives in one `.part` file as a list of configurations, and every assembly, drawing, BOM and export can name a member by its configuration.

This guide makes a family of angle brackets: a standard size, a long version, a thick version, a version without its stiffening rib, and a left-hand version.

## 1. One part with inputs

Start from a part whose sizes are inputs. Each configuration then sets inputs instead of copying geometry.

```part file=angle_bracket.part
runmat 0.7
std 1

part angle_bracket {
    input length : length = 60 mm
    input width : length = 30 mm
    input thickness : length = 4 mm
    input bolt_at : length = 12 mm

    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [input.length, 0 mm])
        l2 = line(start = [input.length, 0 mm], end = [input.length, input.width])
        l3 = line(start = [input.length, input.width], end = [0 mm, input.width])
        l4 = line(start = [0 mm, input.width], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s, distance = input.thickness)
    sketch r on origin.xy {
        b1 = line(start = [10 mm, 13 mm], end = [50 mm, 13 mm])
        b2 = line(start = [50 mm, 13 mm], end = [50 mm, 17 mm])
        b3 = line(start = [50 mm, 17 mm], end = [10 mm, 17 mm])
        b4 = line(start = [10 mm, 17 mm], end = [10 mm, 13 mm])
    }
    feature rib = extrude(profile = sketch.r, distance = 8 mm, mode = add)
    feature bolt = hole(face = feature.body.face.end, at = { a = point(input.bolt_at, 6 mm, input.thickness) }, diameter = 5.5 mm, through_all)

    output angle_bracket = { body = feature.body.body, part_number = "AB-060", description = "Angle bracket, 60 mm" }

    configuration long = { length = 90 mm, part_number = "AB-090", description = "Angle bracket, 90 mm" }
    configuration thick = { features = { body = { distance = 6 mm } }, part_number = "AB-060-T", description = "Angle bracket, 6 mm plate" }
    configuration plain = { suppress = [feature.rib], part_number = "AB-060-P", description = "Angle bracket, no rib" }
    configuration left = { bolt_at = 48 mm, part_number = "AB-060-L", description = "Angle bracket, left hand" }
}
```

## 2. What each configuration changes

- `long` sets an input: the length is 90 mm, and everything built from `input.length` follows.
- `thick` sets a feature's own parameter, the extrude distance, without adding an input for it.
- `plain` suppresses the rib. Features after a suppressed feature rebuild without it.
- `left` moves the bolt hole to the other end, which makes it the right-hand bracket's mirror image about its mid-plane (x = 30 mm), and gives it its own part number. An assembly's mirror places it as the other hand (step 4).

Every configuration sets its own `part_number` and `description`, and these flow to the BOM, the title blocks and STEP product names. Fields a configuration leaves out keep the part's defaults. A configuration that names an input, a feature or a parameter the part doesn't have is an error on the configuration.

In the Desktop, the Configurations list activates a member and recompiles the part in it. An edit made while a member is active can apply to all configurations, to the active one only, or to members you pick, and the Outline marks the features a member suppresses or sets.

## 3. Place members in an assembly

An occurrence selects a member by name, or sets inputs directly for a one-off size:

```assembly file=frame.assembly
runmat 0.7
std 1

assembly frame {
    use part angle_bracket as standard
    use part angle_bracket as long_one {
        configuration = { select = named, name = long }
        placement = { placement = offset, offset = { translation = [0 mm, 50 mm, 0 mm] } }
    }
    use part angle_bracket as custom {
        configuration = { select = inputs, values = { length = 75 mm } }
        placement = { placement = offset, offset = { translation = [0 mm, 100 mm, 0 mm] } }
    }
}
```

The Outline labels each occurrence with its member or its input values in display units (`length 75 mm`), and the BOM lists each part number with its count.

## 4. Left and right hand

A mirror in an assembly reflects occurrences about a plane. For a part that is its own mirror image, `hand = symmetric(…)` reuses the same design. For a handed part, `hand = opposite(…)` places the opposite-hand member, so the BOM lists both part numbers. Its `point` and `normal` give the plane, in the part's own coordinates, about which that member mirrors the seed:

```assembly file=pair.assembly
runmat 0.7
std 1

assembly pair {
    use part angle_bracket as right
    mirror left_side = {
        seeds = [right],
        plane = { plane = explicit, point = [0 mm, 0 mm, 0 mm], normal = [1, 0, 0] },
        hand = opposite(source = { definition = { kind = part, name = angle_bracket } }, configuration = { select = named, name = left }, point = [30 mm, 0 mm, 0 mm], normal = [1, 0, 0])
    }
}
```

## 5. Export and draw the family

```sh
runmat-design export angle_bracket.part --format step --configuration long --out step
runmat-design export angle_bracket.part --format step --all-configurations --out step
```

`--all-configurations` writes one STEP file per member. A drawing's title block reads the part number and description of the configuration it shows, so the same `.drawing` exported as DXF with `--all-configurations` writes every sheet once per member, each with its own part number.

To try a size without adding a member, use a run instead: `design.inspect` with `set` regenerates the part at the given input values and leaves the file unchanged. See [Scripting and sweeps](/docs/mcad/scripting).

## Next

- [Configurations](/docs/mcad/part/configurations) lists every field a configuration takes.
- [Patterns, mirrors and fasteners](/docs/mcad/assembly/patterns-and-mirrors#mirrors) covers mirrors in full.
