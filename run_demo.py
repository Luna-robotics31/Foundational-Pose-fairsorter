# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
import json
from pathlib import Path
import time
from cube_pick.vision.pose_coordinates import read_transform, pose_record



if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str,
                      default=f'{code_dir}/debug/run_demo_{time.time_ns()}')
  parser.add_argument('--T_robot_camera', type=str,
                      help='Measured camera optical frame to robot base 4x4 matrix, in metres')
  parser.add_argument('--T_object_grasp', type=str,
                      help='Desired gripper TCP pose relative to OBJ frame, 4x4 in metres')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)
  T_robot_camera = read_transform(args.T_robot_camera) if args.T_robot_camera else None
  T_object_grasp = read_transform(args.T_object_grasp) if args.T_object_grasp else None
  model_center = mesh.bounds.mean(axis=0)
  coordinate_records = []

  debug = args.debug
  debug_dir = args.debug_dir
  output = Path(debug_dir)
  if output.exists() and (not output.is_dir() or any(output.iterdir())):
    raise ValueError(f'Output directory must be new or empty: {output}')
  (output / 'track_vis').mkdir(parents=True, exist_ok=True)
  (output / 'ob_in_cam').mkdir(exist_ok=True)
  print(f'Pose output: {output.resolve()}')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if i==0:
      mask = reader.get_mask(0).astype(bool)
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    # FoundationPose returns T_camera_object. The camera optical axes are
    # X right, Y down, Z forward; the values here are metres.
    record = pose_record(reader.id_strs[i], pose, model_center,
                         T_robot_camera, T_object_grasp)
    coordinate_records.append(record)
    center_camera = record['center_camera_m']
    print(f'{reader.id_strs[i]} camera center XYZ (mm): '
          f'{center_camera[0]*1000:.1f}, {center_camera[1]*1000:.1f}, '
          f'{center_camera[2]*1000:.1f}')

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)



    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

  with open(f'{debug_dir}/cube_coordinates.jsonl', 'w') as coordinates:
    for record in coordinate_records:
      coordinates.write(json.dumps(record) + '\n')
