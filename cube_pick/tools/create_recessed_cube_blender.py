"""Rebuild the measured object and export cube_recessed.obj.

Run in Blender's Scripting workspace, or headless:
    blender -b --python create_recessed_cube_blender.py
Geometry and the exported OBJ use metres, centred on the origin, Z up.
The pockets are separate, with flat bottoms and a solid middle section.
"""

from pathlib import Path
import math

import bpy
import bmesh
from mathutils import Vector


WIDTH_MM = 40.0            # length and breadth (X, Y)
HEIGHT_MM = 45.0           # Z, the axis of the circular pockets
HOLE_DIAMETER_MM = 33.0
RECESS_DEPTH_MM = 20.0
BOTTOM_POCKET = True       # same pocket on the underside; False = top pocket only
OUTER_BEVEL_MM = 0.6
RECESS_BEVEL_MM = 0.3
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'assets'  # cube_pick/assets


def apply_modifier(obj, modifier):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def aim_at(obj, point=(0, 0, 0)):
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def area_light(scene, name, location, energy, size):
    data = bpy.data.lights.new(name, 'AREA')
    data.energy = energy
    data.shape = 'DISK'
    data.size = size
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = location
    aim_at(obj)


def main():
    assert WIDTH_MM > HOLE_DIAMETER_MM > 0
    pockets = [(1, 'Top')] + ([(-1, 'Bottom')] if BOTTOM_POCKET else [])
    assert 0 < RECESS_DEPTH_MM * len(pockets) < HEIGHT_MM, 'Pockets would meet or break through'
    width, height = WIDTH_MM / 1000, HEIGHT_MM / 1000
    radius, depth = HOLE_DIAMETER_MM / 2000, RECESS_DEPTH_MM / 1000
    side = max(width, height)  # framing for the preview camera / viewport

    if bpy.context.window is not None:
        # Interactive: a new scene preserves the scene that was already open in Blender.
        scene = bpy.data.scenes.new('Cube with circular recesses')
        bpy.context.window.scene = scene
    else:
        # Headless (blender -b): work in the default scene, emptied.
        scene = bpy.context.scene
        for obj in list(scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.0

    bpy.ops.mesh.primitive_cube_add(size=1.0)
    cube = bpy.context.object
    cube.scale = (width, width, height)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    cube.name = 'Cube_Recessed'
    outer = cube.modifiers.new('Rounded outer edges', 'BEVEL')
    outer.width = OUTER_BEVEL_MM / 1000
    outer.segments = 4
    apply_modifier(cube, outer)

    overshoot = 0.002
    for sign, label in pockets:
        z = sign * (height / 2 - depth / 2 + overshoot / 2)
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=128, radius=radius, depth=depth + overshoot,
            location=(0, 0, z),
        )
        cutter = bpy.context.object
        cutter.name = label + '_Pocket_Cutter'
        cut = cube.modifiers.new(label + ' circular recess', 'BOOLEAN')
        cut.operation = 'DIFFERENCE'
        cut.solver = 'EXACT'
        cut.object = cutter
        apply_modifier(cube, cut)
        bpy.data.objects.remove(cutter, do_unlink=True)

    rim = cube.modifiers.new('Soft pocket rims and floors', 'BEVEL')
    rim.width = RECESS_BEVEL_MM / 1000
    rim.segments = 3
    rim.limit_method = 'ANGLE'
    rim.angle_limit = math.radians(40)
    apply_modifier(cube, rim)

    for face in cube.data.polygons:
        face.use_smooth = True
    weighted = cube.modifiers.new('Surface normals', 'WEIGHTED_NORMAL')
    weighted.keep_sharp = True
    apply_modifier(cube, weighted)

    material = bpy.data.materials.new('Warm ivory plastic')
    material.use_nodes = True
    shader = next(node for node in material.node_tree.nodes if node.type == 'BSDF_PRINCIPLED')
    shader.inputs['Base Color'].default_value = (0.80, 0.76, 0.62, 1.0)
    shader.inputs['Roughness'].default_value = 0.42
    material.diffuse_color = (0.91, 0.89, 0.81, 1.0)
    cube.data.materials.append(material)
    for key, value in {
        'width_mm': WIDTH_MM, 'height_mm': HEIGHT_MM,
        'hole_diameter_mm': HOLE_DIAMETER_MM, 'recess_depth_mm': RECESS_DEPTH_MM,
        'pocket_count': len(pockets),
        'solid_below_pockets_mm': HEIGHT_MM - len(pockets) * RECESS_DEPTH_MM,
    }.items():
        cube[key] = value

    # Validate the finished solid and both recess floors before exporting.
    bm = bmesh.new()
    bm.from_mesh(cube.data)
    assert all(edge.is_manifold for edge in bm.edges), 'Mesh has open or non-manifold edges'
    assert bm.calc_volume(signed=True) > 0, 'Mesh normals are inverted'
    volume_mm3 = bm.calc_volume() * 1e9
    bm.free()
    expected_floor = height / 2 - depth
    for sign, _ in pockets:
        hit, position, _, _ = cube.ray_cast(
            Vector((0, 0, sign * height)), Vector((0, 0, -sign)),
        )
        assert hit and abs(position.z - sign * expected_floor) < 1e-6, 'Wrong pocket depth'

    # Triangulate a copy for a single watertight, metre-scale OBJ.
    mesh = cube.data.copy()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj_path = OUTPUT_DIR / 'cube_recessed.obj'
    with obj_path.open('w') as obj_file:
        obj_file.write(f'# Blender block: metres, Z up; {len(pockets)} circular pocket(s)\n')
        obj_file.write(f'# {WIDTH_MM} x {WIDTH_MM} x {HEIGHT_MM} mm; bore {HOLE_DIAMETER_MM} mm; '
                       f'pocket depth {RECESS_DEPTH_MM} mm\n')
        obj_file.write('o Cube_Recessed\n')
        for vertex in mesh.vertices:
            x, y, z = vertex.co
            obj_file.write(f'v {x:.10f} {y:.10f} {z:.10f} 0.91 0.89 0.81\n')
        for face in mesh.polygons:
            obj_file.write('f ' + ' '.join(str(i + 1) for i in face.vertices) + '\n')
    triangle_count = len(mesh.polygons)
    bpy.data.meshes.remove(mesh)

    world = bpy.data.worlds.new('Soft grey studio')
    world.use_nodes = True
    background = next(node for node in world.node_tree.nodes if node.type == 'BACKGROUND')
    background.inputs['Color'].default_value = (0.14, 0.17, 0.21, 1)
    background.inputs['Strength'].default_value = 0.35
    scene.world = world
    area_light(scene, 'Key softbox', (0.035, -0.065, 0.12), 0.35, 0.05)
    area_light(scene, 'Fill softbox', (-0.09, -0.025, 0.045), 0.08, 0.08)
    area_light(scene, 'Rim softbox', (0.045, 0.07, 0.065), 0.18, 0.07)
    area_light(scene, 'Bottom softbox', (-0.025, -0.055, -0.10), 0.15, 0.08)

    camera_data = bpy.data.cameras.new('Preview camera')
    camera = bpy.data.objects.new('Preview camera', camera_data)
    scene.collection.objects.link(camera)
    camera_data.type = 'ORTHO'
    camera_data.ortho_scale = side * 1.9
    camera_data.clip_start = 0.001
    camera_data.clip_end = 10
    camera.location = (0.09, -0.11, 0.115)
    aim_at(camera)
    scene.camera = camera
    scene.render.resolution_x = 1000
    scene.render.resolution_y = 1000
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    scene.render.filepath = str(OUTPUT_DIR / 'cube_recessed.png')

    bpy.ops.object.select_all(action='DESELECT')
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube
    for area in (bpy.context.screen.areas if bpy.context.screen else []):
        if area.type == 'VIEW_3D':
            area.spaces.active.clip_start = 0.0001
            region = area.spaces.active.region_3d
            region.view_location = (0, 0, 0)
            region.view_distance = side * 3
            region.view_rotation = camera.rotation_euler.to_quaternion()

    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / 'cube_recessed.blend'))
    print(f'CREATED: {obj_path}')
    print(f'DIMENSIONS: {WIDTH_MM} x {WIDTH_MM} x {HEIGHT_MM} mm; {len(pockets)} x '
          f'{HOLE_DIAMETER_MM} mm diameter x {RECESS_DEPTH_MM} mm deep pocket(s)')
    print(f'VALIDATION: manifold, outward normals, two separate pocket floors; volume {volume_mm3:.1f} mm^3; {triangle_count} OBJ triangles')


if __name__ == '__main__':
    main()
