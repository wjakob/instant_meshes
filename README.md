## Minimal instant-meshes with aarch64 support.

This fork provides a minimal (non-gui) build with aarch64 support for [instant-meshes](https://github.com/wjakob/instant-meshes).

## Build Instructions

1. Clone the repository.
```
git clone --depth 1 git@github.com:Matter-and-Form/instant-meshes.git

```

1. Update submodules.
```
cd instant-meshes
git submodule update --init --recursive
```

3. Run the build script to build instant-meshes.
```
scripts/build-instant-meshes
```

4. Cross compile aarch64 in Raspbian-11 docker container. Run Raspbian-11 docker container with a shared volume to instant-meshes.
```
docker run -v .:/instant-meshes -it ghcr.io/matter-and-form/debian-bullseye-cc-raspbian-11-aarch64
```
In the docker container, go to the instant-meshes directory and run the build script for aarch64 cross-compile.
```
cd instant-meshes/
scripts/build aarch64
```
