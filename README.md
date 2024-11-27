## Minimal instant-meshes with aarch64 support.

This fork provides a minimal (non-gui) build with aarch64 support for [instant-meshes](https://github.com/wjakob/instant-meshes).

## Build Instructions

1. Clone the repository.
```
git clone --depth 1 git@github.com:Matter-and-Form/instant-meshes.git

```

2. Update submodules.
```
cd instant-meshes
git submodule update --init --recursive
```

3. Native build: Run the native build script.
```
scripts/build
```

4. Cross compile aarch64 build: Run the cross-compile aarch64 build script.
```
scripts/build-aarch64
```

