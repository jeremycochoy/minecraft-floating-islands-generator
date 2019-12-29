#!/usr/bin/env python3

import anvil
import pyfastnoisesimd as fns
import numpy as np
import time

def to_block(name):
    return anvil.Block('minecraft', name)

# Usefull blocks type
grass = anvil.Block('minecraft', 'grass_block')
sand = anvil.Block('minecraft', 'sand')
bedrock = anvil.Block('minecraft', 'bedrock')
dirt = anvil.Block('minecraft', 'dirt')
air = anvil.Block('minecraft', 'air')
glass = anvil.Block('minecraft', 'glass')
pumpkin = anvil.Block('minecraft', 'pumpkin')
diorite = anvil.Block('minecraft', 'diorite')
stone = anvil.Block('minecraft', 'stone')
oak_log_up = anvil.Block('minecraft', 'oak_log', properties={'axis': 'y'})
oak_leaves = anvil.Block('minecraft', 'oak_leaves', properties={'persistent': True})
coal = anvil.Block('minecraft', 'coal_ore')
iron = anvil.Block('minecraft', 'iron_ore')
gold = anvil.Block('minecraft', 'gold_ore')
diamond = anvil.Block('minecraft', 'diamond_ore')
redstone = anvil.Block('minecraft', 'redstone_ore')
emerald = anvil.Block('minecraft', 'emerald_ore')
lapis_lazuli = anvil.Block('minecraft', 'lapis_ore')
water = to_block('water')
lava = to_block('lava')

oak_log = anvil.Block('minecraft', 'oak_log')
spruce_log = anvil.Block('minecraft', 'spruce_log')
dark_oak_log = anvil.Block('minecraft', 'dark_oak_log')
acacia_log = anvil.Block('minecraft', 'acacia_log')

oak_leaves = anvil.Block('minecraft', 'oak_leaves')
spruce_leaves = anvil.Block('minecraft', 'spruce_leaves')
dark_oak_leaves = anvil.Block('minecraft', 'dark_oak_leaves')
acacia_leaves = anvil.Block('minecraft', 'acacia_leaves')

logs_list = [oak_log, spruce_log, dark_oak_log, acacia_log]
leaves_map = {
    oak_log: oak_leaves,
    spruce_log: spruce_leaves,
    dark_oak_log: dark_oak_leaves,
    acacia_log: acacia_leaves,
}

grass_plant = to_block('grass')
fern = to_block('fern')
tall_grass = to_block('tall_grass')
large_fern = to_block('large_fern')
dandelion = to_block('dandelion')
poppy = to_block('poppy')
blue_orchid = to_block('blue_orchid')
sunflower = to_block('sunflower')
rose_bush = to_block('rose_bush')
flowers_list = [grass_plant, fern, tall_grass, large_fern] * 5 + [dandelion, poppy, blue_orchid, sunflower, rose_bush]



# Log timing of script
original_timer_start = timer_start = time.time()


def log(string):
    global timer_start
    print(string, "-", time.time() - timer_start, "seconds")
    timer_start = time.time()


def numpy_matrix_to_region(matrix, region_x=0, region_z=0):
    """
    Transform a 512x256x512 matrix into a region object

    :param matrix: A 512x256x512 matrix containing `anvil.Block`
    :param region_x: The x coordinate of the region
    :param region_y: The y coordinate of the region
    :return: An anvil.Region object
    """
    # Assert the matrix is well shaped x, y, z block
    assert tuple(matrix.shape) == (512, 256, 512)

    # Create a new region
    region = anvil.EmptyRegion(region_x, region_z)

    # Chunk X
    for cx in range(0, 32):
        # Chunk Y
        for cz in range(0, 32):
            # Create a new chunk
            chunk = anvil.EmptyChunk(region_x * 32 + cx, region_z * 32 + cz)
            region.add_chunk(chunk)
            for cy in range(0, 10):
                # Create a new section
                section = anvil.EmptySection(cy)
                chunk.add_section(section=section, replace=True)
                # Set data in this chunk
                x = cx * 16
                y = cy * 16
                z = cz * 16
                data = matrix[x: x + 16, y:y + 16, z:z + 16]
                data = data.transpose(1, 2, 0)  # Move to YZX ordering
                data = data.reshape(16*16*16)
                assert(len(data) == 4096)
                section.blocks[0:4096] = data

    return region


def generate_region(rx, rz):
    """
    Create a new region block
    :param rx: Coordinate x of the region
    :param rz: Coordinate z of the region
    :return: A new region
    """
    matrix = generate_matrix(rx, rz)
    log("Noise Matrix generated")
    block_matrix, top_layers = generate_ground_and_rock(matrix)
    log("Matrix filled with ground")
    block_matrix = populate_with_ores(block_matrix, matrix)
    log("Ground populated with ores")
    block_matrix = populate_pools(block_matrix, top_layers[0])
    log("Lava and water pool created")
    block_matrix = populate_with_trees(block_matrix, top_layers[0])
    log("Ground populated with trees")
    # log_ores(block_matrix, matrix)
    # log("Log matrix blocks")
    region = numpy_matrix_to_region(block_matrix, rx, rz)
    log("Matrix converted to region")
    return region


def get_noise_block(shape, rx=0, rz=0):
    """
    Fill a matrix with a nice perlin noise.

    :param shape:
    :return:
    """
    perlin = fns.Noise(seed=42, numWorkers=2)
    perlin.frequency = 0.02
    perlin.noiseType = fns.NoiseType.Perlin
    perlin.fractal.octaves = 4
    perlin.fractal.lacunarity = 2.1
    perlin.fractal.gain = 0.15
    perlin.perturb.perturbType = fns.PerturbType.NoPerturb
    return perlin.genAsGrid(shape, start=[rx * 512, 0, rz * 512])


def get_noise_block_trees(shape, seed=27):
    """
    Fill a matrix with a nice perlin noise.

    :param shape:
    :return:
    """
    perlin = fns.Noise(seed=seed, numWorkers=2)
    perlin.frequency = 0.1
    perlin.noiseType = fns.NoiseType.Perlin
    perlin.fractal.octaves = 4
    perlin.fractal.lacunarity = 2.1
    perlin.fractal.gain = 0.30
    perlin.perturb.perturbType = fns.PerturbType.NoPerturb
    block = perlin.genAsGrid(shape)
    block -= block.min()
    block /= block.std()
    return block


def log_ores(block_matrix, matrix):
    """
    Display the proportion of diverse blocks types.

    :param block_matrix: A matrix containing blocks
    """
    non_zero = matrix != 0
    total_non_zero_blocks = non_zero.sum()

    def block_percent(block):
        return (block_matrix[non_zero] == block).sum() / total_non_zero_blocks * 100

    print("--- Blocks percents: ---")
    print("Dirt: ", block_percent(dirt), "%")
    print("Grass: ", block_percent(grass), "%")
    print("Stone: ", block_percent(stone), "%")
    print("Coal: ", block_percent(coal), "%")
    print("Iron: ", block_percent(iron), "%")
    print("Gold: ", block_percent(gold), "%")
    print("Redstone: ", block_percent(redstone), "%")
    print("Lapis: ", block_percent(lapis_lazuli), "%")
    print("Diamond: ", block_percent(diamond), "%")
    print("Emerald: ", block_percent(emerald), "%")


def matrix_log(matrix):
    print("Display some properties of the generated matrix:")
    print("\tmin  =", matrix.min())
    print("\tmax  =", matrix.max())
    print("\tmean =", matrix.mean())
    print("\tstd  =", matrix.std())


def generate_ground_and_rock(matrix):
    matrix[matrix < 0.7] = 0
    # Renormalize
    non_zero_indices = matrix > 0
    matrix[non_zero_indices] -= matrix[non_zero_indices].min()
    matrix /= matrix[non_zero_indices].std()

    # Fill with ground and stone
    block_matrix = np.ndarray(tuple(matrix.shape), dtype=anvil.Block)
    block_matrix[matrix > 0] = diorite
    block_matrix[matrix > 0.5] = stone

    # Create an enumeration of Y coordinate
    enumerated_y = np.arange(256)[None, :, None]
    enumerated_y = np.repeat(enumerated_y, 512, axis=0)
    enumerated_y = np.repeat(enumerated_y, 512, axis=2)
    # Compute the top layer (higher height)
    enumerated_y[matrix <= 0] = 0
    top_layer = enumerated_y == enumerated_y.max(axis=1, keepdims=True)
    top_layer = np.logical_and(top_layer, enumerated_y > 0)
    # Compute the layer right below the top layer
    enumerated_y[top_layer] = 0
    second_layer = enumerated_y == enumerated_y.max(axis=1, keepdims=True)
    second_layer = np.logical_and(second_layer, enumerated_y > 0)

    block_matrix[top_layer] = grass
    block_matrix[second_layer] = dirt

    # Add sand at the bottom of the world
    block_matrix[:, 1:20, :] = sand
    block_matrix[:, 0:1, :] = bedrock

    return block_matrix, [top_layer, second_layer]


def populate_with_ores(block_matrix, matrix):
    # Apply a growth process based on seeds
    def apply_seeds(seeds, block, filter_fct=always_true, contagion=0.5, block_constrains=0):
        """
        This grows ore from seeds. The `contagion` factor control the probability for a neighbor of a seed
        to be replaced by `block`. The contagious process happen in all direction, each time with a new
        contagion probability squared (`contagion= contagion_previous_step ** 2`.

        :param seeds: A block_matrix.shape boolean array that represent seeds for the growth process
        :param block: A block type to place
        :param filter_fct: A function that tell if a certain block type can be replaced.
        :param contagion: Probability of a seed to convert its neighbors.
        :param block_constrains: A block_matrix.shape boolean array that constrain on where the seeds can start
        :return:
        """
        assert seeds.shape == matrix.shape
        # Brows array as a 1d vector, because life is tough
        non_zero_coordinates = np.argwhere(matrix != 0)
        if block_constrains is None:
            block_constrains = np.ones(seeds[matrix != 0].shape) == 1
        for location in np.argwhere(np.logical_and(block_constrains, seeds[matrix != 0])):
            seed_coordinates = tuple(non_zero_coordinates[location[0]])
            block_matrix[seed_coordinates] = block
            ore_growth_process(block, block_matrix, seed_coordinates, contagion=contagion, condition_fct=filter_fct)

    def is_in_list(list):
        def tmp(block):
            return block in list

        return tmp

    def populate_ore(ore, poisson=0.001, contagion=0.5, can_replace=[], block_constrains=None):
        # Generate seeds with a poisson process
        seeds = np.random.poisson(poisson, (512, 256, 512)) > 0
        # Start the growth process from seeds
        can_replace = can_replace + [ore]
        apply_seeds(seeds=seeds, block=ore, filter_fct=is_in_list(can_replace),
                    contagion=contagion, block_constrains=block_constrains)

    # Blocks that can be replaced by ores
    assert block_matrix.shape == matrix.shape
    stone_blocks = block_matrix[matrix != 0] == stone  # Stone blocks
    stone_and_diorite_blocks = np.logical_or(stone_blocks, block_matrix[matrix != 0] == diorite)  # Stone + diorite
    center_island = np.logical_or(stone_blocks, matrix[matrix != 0] > 2)  # Stone in the heart of the map (value > 2)

    # Generate ores
    populate_ore(ore=coal, poisson=0.004, contagion=0.7, can_replace=[stone, diorite], block_constrains=stone_and_diorite_blocks)
    populate_ore(ore=iron, poisson=0.0035, contagion=0.6, can_replace=[stone, diorite, coal], block_constrains=stone_blocks)
    populate_ore(ore=gold, poisson=0.002, contagion=0.4, can_replace=[stone, diorite, coal, iron], block_constrains=stone_blocks)
    populate_ore(ore=redstone, poisson=0.003, contagion=0.5, can_replace=[stone, diorite, coal], block_constrains=stone_blocks)
    populate_ore(ore=diamond, poisson=0.0009, contagion=0.4, can_replace=[stone, diorite, coal, gold, iron], block_constrains=center_island)
    populate_ore(ore=emerald, poisson=0.0003, contagion=0, can_replace=[stone, diorite, coal, gold, iron], block_constrains=center_island)
    populate_ore(ore=lapis_lazuli, poisson=0.002, contagion=0.3, can_replace=[stone, diorite, coal, iron], block_constrains=center_island)

    return block_matrix


def populate_pools(block_matrix, top_layer):
    # Create seeds for pools
    seeds_poisson = np.random.poisson(0.0005, block_matrix.shape) > 0
    for coord in np.argwhere(np.logical_and(top_layer, seeds_poisson)):
        block = np.random.choice([water, lava, water])  # Add lava or water pool
        pool_growth_process(block, block_matrix, tuple(coord))
    return block_matrix


def is_invalid_coordinate(center):
    return center[0] < 0 or center[1] < 0 or center[2] < 0 or center[0] >= 512 or center[1] >= 256 or center[2] >= 512


def pool_growth_process(new_block, block_matrix, base, contagion=0.9):
    def growth(center=base, contagion=contagion):
        if is_invalid_coordinate(center):
            return
        if np.random.uniform() < contagion:
            if block_matrix[center] in [dirt, grass, stone]:
                block_matrix[center] = new_block

            # Neighbors
            left = (center[0] - 1, center[1], center[2])
            right = (center[0] + 1, center[1], center[2])
            forward = (center[0], center[1], center[2] - 1)
            backward = (center[0], center[1], center[2] + 1)

            # Put stone around lava to prevent fires
            if new_block == lava:
                def stonify(pos):
                    if is_invalid_coordinate(pos):
                        return
                    if block_matrix[pos] in [dirt, grass]:
                        block_matrix[pos] = stone
                for x in [-1, 0, 1]:
                    for y in [-1, 0, 1]:
                        for z in [-1, 0, 1]:
                            stonify((center[0] + x, center[1] + y, center[2] + z))

            # Propagate
            growth(center, contagion=contagion ** 2 * 0.9)
            growth(left, contagion=contagion ** 2 * 0.9)
            growth(right, contagion=contagion ** 2 * 0.9)
            growth(forward, contagion=contagion ** 2 * 0.9)
            growth(backward, contagion=contagion ** 2 * 0.9)
    growth()
    return block_matrix


def populate_with_trees(block_matrix, top_layer):
    # Create seeds for trees
    seeds_poisson = np.random.poisson(0.015, block_matrix.shape) > 0
    seeds_perlin = get_noise_block_trees(block_matrix.shape, seed=np.random.randint(424242))
    seeds_perlin = np.logical_and(seeds_perlin > 0.1, seeds_perlin < 4)
    seeds = np.logical_and(seeds_perlin, seeds_poisson)
    for coord in np.argwhere(np.logical_and(top_layer, seeds)):
        if block_matrix[tuple(coord)] in [lava, water]:  # Skip lava or water pools
            continue
        wood_log = np.random.choice(logs_list)
        coord[1] += 1  # Move above floor
        block_matrix = tree_growth_process(wood_log, block_matrix, tuple(coord))

    # Create seeds for plants
    seeds_poisson = np.random.poisson(0.1, block_matrix.shape) > 0
    seeds_perlin = get_noise_block_trees(block_matrix.shape, seed=np.random.randint(424242))
    seeds_perlin = np.logical_and(seeds_perlin > 0.1, seeds_perlin < 4)
    seeds = np.logical_and(seeds_perlin, seeds_poisson)
    for coord in np.argwhere(np.logical_and(top_layer, seeds)):
        if block_matrix[tuple(coord)] in [lava, water, stone]:  # Skip lava or water pools
            continue
        coord[1] += 1  # Move above floor
        if block_matrix[tuple(coord)] is None or block_matrix[tuple(coord)] == air:
            block_matrix[tuple(coord)] = np.random.choice(flowers_list)
    return block_matrix


def tree_growth_process(new_block, block_matrix, base, contagion=1):
    height = np.random.randint(3, 6)

    # Put trunk
    block_matrix[base] = new_block
    trunk = np.array(base)
    for i in range(height):
        trunk[1] += 1
        block_matrix[tuple(trunk)] = new_block
    # Replace block by leaves
    new_block = leaves_map[new_block]
    block_matrix[tuple(trunk)] = new_block

    def growth(center=tuple(trunk), contagion=contagion):
        if center[0] < 0 or center[1] < 0 or center[2] < 0 or center[0] >= 512 or center[1] >= 256 or center[2] >= 512:
            return
        if np.random.uniform() < contagion:
            # print("growth at", center, "with", new_block)
            if block_matrix[center] is None or block_matrix[center] == air:
                block_matrix[center] = new_block
            # Propagate
            left = (center[0] - 1, center[1], center[2])
            right = (center[0] + 1, center[1], center[2])
            top = (center[0], center[1] + 1, center[2])
            down = (center[0], center[1] - 1, center[2])
            forward = (center[0], center[1], center[2] - 1)
            backward = (center[0], center[1], center[2] + 1)
            #growth(center, contagion=contagion ** 3 * 0.9)
            growth(left, contagion=contagion ** 3 * 0.9)
            growth(right, contagion=contagion ** 3 * 0.9)
            growth(top, contagion=contagion ** 3 * 0.9)
            growth(down, contagion=contagion ** 10 * 0.9)
            growth(forward, contagion=contagion ** 3 * 0.9)
            growth(backward, contagion=contagion ** 3 * 0.9)
    growth()
    return block_matrix


def always_true(_):
    return True


def ore_growth_process(new_block, block_matrix, center, contagion=0.5, condition_fct=None):
    if condition_fct is None:
        condition_fct = always_true

    if center[0] < 0 or center[1] < 0 or center[2] < 0 or center[0] >= 512 or center[1] >= 256 or center[2] >= 512:
        return
    if np.random.uniform() < contagion:
        if condition_fct(block_matrix[center]):
            block_matrix[center] = new_block
            # Propagate
            left = (center[0] - 1, center[1], center[2])
            right = (center[0] + 1, center[1], center[2])
            top = (center[0], center[1] + 1, center[2])
            down = (center[0], center[1] - 1, center[2])
            forward = (center[0], center[1], center[2] - 1)
            backward = (center[0], center[1], center[2] + 1)
            ore_growth_process(new_block, block_matrix, left, contagion=contagion ** 2, condition_fct=condition_fct)
            ore_growth_process(new_block, block_matrix, right, contagion=contagion ** 2, condition_fct=condition_fct)
            ore_growth_process(new_block, block_matrix, top, contagion=contagion ** 2, condition_fct=condition_fct)
            ore_growth_process(new_block, block_matrix, down, contagion=contagion ** 2, condition_fct=condition_fct)
            ore_growth_process(new_block, block_matrix, forward, contagion=contagion ** 2, condition_fct=condition_fct)
            ore_growth_process(new_block, block_matrix, backward, contagion=contagion ** 2, condition_fct=condition_fct)


def generate_matrix(rx, rz):
    """
    Generate a noise matrix with values in [0, 1].
    Gradients are used to "cut" the top and bottom smoothly.

    :param rx: X offset
    :param rz: Z offset
    :return: A 512x512x256 noise matrix
    """
    scale = 0.01
    # TODO: Handle rx/ry for blocks generation
    data = get_noise_block(shape=(512, 256, 512), rx=rx, rz=rz)
    # Move into [0, 1]
    data -= data.min()
    data /= data.max()
    # Add negative gradient on y in [20, 50] and [145, 160]
    enumerated_y = np.arange(256)[None, :, None]  # Enumerate the Y (height) value along Y axis
    data += (np.minimum(130 - enumerated_y, 0)) / 15  # Negative gradient on top
    data += (np.minimum(enumerated_y - 50, 0)) / 30  # Negative gradient on bottom

    return data


X_AMPLITUDE = 2
Z_AMPLITUDE = 2

for region_x in range(-X_AMPLITUDE, X_AMPLITUDE):
    for region_z in range(-Z_AMPLITUDE, Z_AMPLITUDE):
        region = generate_region(region_x, region_z)
        log("Region generated")
        region.save(f"r.{region_x}.{region_z}.mca")
        log("Region saved")

print("Script took:", time.time() - original_timer_start, "seconds")
