import numpy as np
import torch
import random

from metadrive.type import MetaDriveType


def _static_obbs_overlap(car_a, car_b):
    """
    Check if two static vehicle OBBs overlap (2D SAT).
    car_a, car_b: dicts with "begin" (x, y), "heading" (rad), "length", "width".
    begin is center; half-extents are length/2, width/2.
    """
    def _get_corners(car):
        cx, cy = car["begin"][0], car["begin"][1]
        h = float(car["heading"])
        L2 = float(car["length"]) / 2.0
        W2 = float(car["width"]) / 2.0
        ux, uy = np.cos(h), np.sin(h)
        vx, vy = -np.sin(h), np.cos(h)
        return np.array([
            [cx + L2 * ux + W2 * vx, cy + L2 * uy + W2 * vy],
            [cx + L2 * ux - W2 * vx, cy + L2 * uy - W2 * vy],
            [cx - L2 * ux - W2 * vx, cy - L2 * uy - W2 * vy],
            [cx - L2 * ux + W2 * vx, cy - L2 * uy + W2 * vy],
        ])

    def _get_axes(car):
        h = float(car["heading"])
        return [
            np.array([np.cos(h), np.sin(h)]),
            np.array([-np.sin(h), np.cos(h)]),
        ]

    corners_a = _get_corners(car_a)
    corners_b = _get_corners(car_b)
    axes = _get_axes(car_a) + _get_axes(car_b)

    for axis in axes:
        proj_a = corners_a @ axis
        proj_b = corners_b @ axis
        min_a, max_a = proj_a.min(), proj_a.max()
        min_b, max_b = proj_b.min(), proj_b.max()
        if max_a < min_b or max_b < min_a:
            return False
    return True


def _deduplicate_background_by_collision(background_vehicles):
    """
    Merge static tracks that collide (same physical vehicle). Build collision graph,
    find connected components, keep one representative per component (min show_time, then scenario_id).
    """
    if not background_vehicles:
        return background_vehicles
    items = list(background_vehicles.items())
    n = len(items)
    # Build adjacency by index
    parent = list(range(n))

    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(n):
        for j in range(i + 1, n):
            if _static_obbs_overlap(items[i][1], items[j][1]):
                union(i, j)

    # Representative per component: index with min (show_time, scenario_id)
    comp_rep = {}
    for i in range(n):
        r = find(i)
        sid, car = items[i][0], items[i][1]
        key = (car.get("show_time", 0), sid)
        if r not in comp_rep or key < comp_rep[r][0]:
            comp_rep[r] = (key, sid, car)

    return {sid: car for (_, sid, car) in comp_rep.values()}


def is_on_lane(pos, map_manager, threshold=2.0):
    """Check if a position is on a valid lane (within lateral tolerance)."""
    if map_manager is None or map_manager.current_map is None:
        return True
    try:
        lane, _ = map_manager.current_map.road_network.get_closest_lane_index(pos, return_lane=True)
        if lane is None:
            return False
        long, lat = lane.local_coordinates(pos)
        width = lane.width
        if abs(lat) <= (width / 2 + threshold):
            return True
        return False
    except Exception:
        return False


def filter_traffic_tracks_to_birth_lists(
    current_traffic_data,
    sdc_scenario_id,
    map_manager,
    *,
    lane_threshold=5.0,
    static_displacement_threshold=5.0,
    static_speed_threshold=1.0,
    return_stats=False,
    deduplicate_static_by_collision=True,
):
    """
    Filter traffic tracks into controlled (car_birth_info_list) and background lists.

    - controlled (car_birth_info_list): 非 SDC、类型 VEHICLE、至少一帧 valid、在车道内、且非静态
      （位移/速度超过阈值）。用于策略控制或专家回放，spawn 时机为 show_time == round。
    - background (background_vehicles): 同上但在车道内且判定为静态（位移 < 5m、速度 < 1 m/s）。
      仅作场景占位与观测邻居，spawn 时机为 show_time == round，按 valid 在 step 中移除。

    Returns (car_birth_info_list, background_vehicles, obj_to_clean) or, if return_stats=True,
    (car_birth_info_list, background_vehicles, obj_to_clean, stats_dict).
    stats_dict: n_total, n_no_valid, n_off_lane, n_static, n_controlled.
    """
    car_birth_info_list = []
    background_vehicles = {}
    obj_to_clean = []
    n_total = 0
    n_no_valid = 0
    n_off_lane = 0
    n_static = 0

    for scenario_id, track in current_traffic_data.items():
        if scenario_id == sdc_scenario_id:
            continue
        if track["type"] != MetaDriveType.VEHICLE:
            continue

        n_total += 1
        obj_to_clean.append(scenario_id)
        valid = track["state"]["valid"]
        if not valid.any():
            n_no_valid += 1
            continue

        first_show = int(np.argmax(valid))
        last_show = len(valid) - 1 - int(np.argmax(valid[::-1]))
        mid_show = (first_show + last_show) // 2

        start_pos = track["state"]["position"][first_show]
        is_valid_track = True
        if not is_on_lane(start_pos, map_manager, threshold=lane_threshold):
            mid_pos = track["state"]["position"][mid_show]
            if not is_on_lane(mid_pos, map_manager, threshold=lane_threshold):
                is_valid_track = False

        if not is_valid_track:
            n_off_lane += 1
            continue

        positions = track["state"]["position"][valid.astype(bool)]
        velocities = track["state"]["velocity"][valid.astype(bool)]
        total_displacement = 0.0
        max_speed = 0.0
        if len(positions) > 1:
            total_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
            max_speed = float(np.max(np.linalg.norm(velocities, axis=1)))
        is_static = total_displacement < static_displacement_threshold and max_speed < static_speed_threshold

        if is_static:
            n_static += 1
            background_vehicles[scenario_id] = {
                "id": track["metadata"]["object_id"],
                "show_time": first_show,
                "begin": (
                    float(track["state"]["position"][first_show, 0]),
                    float(track["state"]["position"][first_show, 1]),
                ),
                "heading": float(track["state"]["heading"][first_show]),
                "end": (
                    float(track["state"]["position"][last_show, 0]),
                    float(track["state"]["position"][last_show, 1]),
                ),
                "scenario_id": scenario_id,
                "length": track["state"]["length"][first_show],
                "width": track["state"]["width"][first_show],
                "valid": valid,
            }
            continue

        car_birth_info_list.append({
            "id": track["metadata"]["object_id"],
            "show_time": first_show,
            "begin": (
                float(track["state"]["position"][first_show, 0]),
                float(track["state"]["position"][first_show, 1]),
            ),
            "heading": float(track["state"]["heading"][first_show]),
            "end": (
                float(track["state"]["position"][last_show, 0]),
                float(track["state"]["position"][last_show, 1]),
            ),
            "scenario_id": scenario_id,
            "length": track["state"]["length"][first_show],
            "width": track["state"]["width"][first_show],
        })

    if deduplicate_static_by_collision and background_vehicles:
        background_vehicles = _deduplicate_background_by_collision(background_vehicles)

    if return_stats:
        stats = {
            "n_total": n_total,
            "n_no_valid": n_no_valid,
            "n_off_lane": n_off_lane,
            "n_static": n_static,
            "n_controlled": len(car_birth_info_list),
        }
        return car_birth_info_list, background_vehicles, obj_to_clean, stats
    return car_birth_info_list, background_vehicles, obj_to_clean


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print('Random seed: {}'.format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)