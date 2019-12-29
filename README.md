# Minecraft Floating Islands Generator
This simple python script generate a map with floating islands

![An example of generated land](example/floating-islands.gif)

## Usage

A minecraft map is divided in regions (512 x 256 x 512 blocks).
Each region is divided into Chunk (32x256x32 blocks),
and each chunk is devided into
(vertical) sections (32x32x32 blocks).

Each region of coordinates `(rx, rz)` is stored in a region file
named `r.rx.rz.mca`. This python script generate such
files in his execution folder.

To generate a map, create a world with minecraft,
 move to the `world/region/` folder and call the
script `~/minecraft-floating-islands-generator/generate_world.py`.
This will generate files with name `r.rx.ry.mca` where `rx` and `ry`
are integers corresponding to the coordinates of a region.
This files will contain the new generated land.

You can select which region should be generated by editing the lines
```python
X_AMPLITUDE = 2
Z_AMPLITUDE = 2
```
from the script. By default, it generates the regions
 with `rx in [-2, -1, 0, 1, 2]` and `rx in [-2, -1, 0, 1, 2]`.
It is a total of 25 regions.
Other region of your map won't be affected by this script.

## Generator

The generator create rock/dirt island, populate them with trees and few ores
(metal, diamons, gold, coal).

## Improvement
The noise generator is very fast, thanks to the awesome `pyfastnoisesimd`
library.

Yet, this script is very slow.
The actual bottleneck is the usage of
the python anvil library that use list of `Block` object istead of
simple binary chunk. Using directly the NBT library and
converting numpy array into directly into a binary array
could reduce the saving process to only few seconds.

The second bottlebeck is the manipulation of a `np.array` of python
`Block` object insted of simple integers. Using integers and a palet
(block <-> integer dictionnary) would improve the speed of populating
dramatically, and simplify drastically the code.
