---
description: Reference for build variants in a RunMat circuit. Fitted and not-fitted components, alternate parts and values, variants built on other variants, components in channels, and how ERC, the BOM and the assembly outputs follow each variant.
keywords: [variants, assembly variants, not fitted, DNP, do not populate, alternate parts, BOM variants, build options]
---
# Variants

A variant changes how components are fitted: `variant lite = { overrides = { D1 = not_fitted } }`. An override is `fitted`, `not_fitted`, or an alternate part or value. `base = <variant>` builds on another variant, and a component inside a channel is named by its path (`"/ch[0]/CD2"`). ERC, the BOM and the assembly outputs follow the variant, so a not-fitted part neither raises ERC findings nor appears in that variant's pick-and-place file.

```circuit fragment
variant lite = { overrides = { D1 = not_fitted } }
variant field = { base = lite, overrides = { R1 = fitted } }
```

The fabrication outputs write the BOM and pick-and-place file of each variant you choose. See [Fabrication outputs and netlists](/docs/ecad/fabrication).
