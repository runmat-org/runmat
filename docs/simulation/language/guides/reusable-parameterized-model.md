# Reusable parameterized model

Use model inputs for values or regions that change between studies while the model's physical
structure stays the same.

```fea
model cantilever(region body : region<solid>, region clamp : region<face>, region tip : region<face>) {
    input body_material : material
    input tip_load : force

    assign input.body_material to input.body

    constraint fixed_support {
        structural.fixed { target = input.clamp }
    }
    load service_force {
        force on input.tip = [0, -input.tip_load, 0]
    }
    load_case service { include load.service_force }
}
```

Bind every required parameter when the study selects the model:

```fea
model = model.cantilever(
    body = geometry.bracket.region.body,
    clamp = geometry.bracket.region.support,
    tip = geometry.bracket.region.load_face,
    body_material = material.aluminum,
    tip_load = input.service_load,
)
```

Region parameters include topology in their type. Material parameters retain capability checking;
supplying a thermal-only material to structural elasticity is still invalid. Named arguments make
the binding independent of parameter order.

Put stable mounting behavior in an imported model and compose it with `include model.mounting`.
Bind the included model's inputs explicitly. Composition rejects cycles and overlapping physical
ownership; it does not use last-declaration-wins semantics.

Keep geometry parameters on the geometry declaration. A width or hole-radius change must create a
new geometry and mesh identity, while a force-only model input may reuse them.
