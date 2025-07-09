"""
Delay the computation to where it is used. This is used to reduce the number of register consumption.

Example:
```
  let a = create(...)
  let b = create(...)
  let c = a + 1
  let d = b + 1
```

will be transformed to

```
  let a = create(...)
  let c = a + 1
  let b = create(...)
  let d = b + 1
```

For now, we only consider the case of let-chain and only move the let definitions in the same let statement.

"""
