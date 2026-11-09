---
description: Reference for RunMat part configurations. Override inputs, suppress features, set a feature's own parameters, and give each configuration its own part number and description, for product families and left- and right-hand parts.
keywords: [configurations, product family, design table, part number, suppress feature, variants, left hand, right hand, dash numbers, family of parts]
---
# Configurations

A configuration is a named version of a part: different input values, features left out, a feature's own parameters changed, and its own part number. One file then describes a whole product family, and the BOM, title blocks and STEP exports carry each configuration's own identity.

```part
runmat 0.7
std 1

part bracket {
    input width : length = 80 mm
    input depth : length = 50 mm
    input thickness : length = 10 mm

    sketch s1 on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])
        l2 = line(start = [input.width, 0 mm], end = [input.width, input.depth])
        l3 = line(start = [input.width, input.depth], end = [0 mm, input.depth])
        l4 = line(start = [0 mm, input.depth], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s1, distance = input.thickness)
    feature round = fillet(edges = [edges_between(feature.body.face.side(s1.l1), feature.body.face.end)], radius = 1.5 mm)

    output bracket = { body = feature.body.body, part_number = "BR-100" }

    configuration long = { width = 120 mm, part_number = "BR-120", description = "Long bracket" }
    configuration plain = { suppress = [feature.round], part_number = "BR-100-P" }
    configuration thick = { features = { body = { distance = 12 mm } }, part_number = "BR-100-T" }
}
```

## What a configuration sets

| Field | Sets |
|---|---|
| `<input> = <value>` | An input's value: `width = 120 mm`. |
| `suppress = [feature.<name>, …]` | Features left out of the history in this configuration. |
| `features = { <feature> = { <parameter> = <value> } }` | A feature's own parameters: `features = { body = { distance = 12 mm } }`. |
| `part_number` | The part number in this configuration: a dash number, or the other hand of a mirrored part. |
| `description` | The description in this configuration. |
| `properties` | Other properties in this configuration. |

A configuration that names an input, a feature or a parameter the part does not have is an error on the configuration. Fields it leaves out keep the part's defaults.

## Using a configuration

- **In an assembly**, an occurrence selects one: `configuration = { select = named, name = long }`, or sets inputs directly with `configuration = { select = inputs, values = { width = 120 mm } }`. See [Occurrences and mates](/docs/mcad/assembly/mates).
- **In the editor**, activating a configuration recompiles the part in it, and edits made while it is active can apply to all configurations, to this one, or to ones you pick. The Outline marks features suppressed or set in the active configuration.
- **From the command line**, `runmat-design export bracket.part --format step --configuration long` exports one configuration, and `--all-configurations` exports each in turn.
- **In a drawing**, title blocks read the configuration's part number and description.

## Configurations and runs

A configuration is part of the design. To try input values without adding one, use a run: `design.inspect` with `set` regenerates the part with input values for one run and leaves the file unchanged. See [Agent tools and the command line](/docs/design-language/agent-and-cli#inputs-for-one-run).
