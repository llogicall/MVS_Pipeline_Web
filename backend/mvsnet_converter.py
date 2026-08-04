import collections
import math
import shutil
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


class Image(BaseImage):
    def qvec2rotmat(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)


class ConversionError(RuntimeError):
    pass


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = {camera_model.model_id: camera_model for camera_model in CAMERA_MODELS}

PARAM_TYPES = {
    "SIMPLE_PINHOLE": ["f", "cx", "cy"],
    "PINHOLE": ["fx", "fy", "cx", "cy"],
    "SIMPLE_RADIAL": ["f", "cx", "cy", "k"],
    "SIMPLE_RADIAL_FISHEYE": ["f", "cx", "cy", "k"],
    "RADIAL": ["f", "cx", "cy", "k1", "k2"],
    "RADIAL_FISHEYE": ["f", "cx", "cy", "k1", "k2"],
    "OPENCV": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"],
    "OPENCV_FISHEYE": ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"],
    "FULL_OPENCV": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"],
    "FOV": ["fx", "fy", "cx", "cy", "omega"],
    "THIN_PRISM_FISHEYE": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"],
}


def read_next_bytes(fid, num_bytes: int, format_char_sequence: str, endian_character: str = "<"):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise ConversionError("Unexpected end of COLMAP binary model file.")
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_text(path: Path) -> Dict[int, Camera]:
    cameras = {}
    with Path(path).open("r", encoding="utf-8") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            camera_id = int(elems[0])
            cameras[camera_id] = Camera(
                id=camera_id,
                model=elems[1],
                width=int(elems[2]),
                height=int(elems[3]),
                params=np.array(tuple(map(float, elems[4:]))),
            )
    return cameras


def read_cameras_binary(path: Path) -> Dict[int, Camera]:
    cameras = {}
    with Path(path).open("rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            if model_id not in CAMERA_MODEL_IDS:
                raise ConversionError(f"Unsupported COLMAP camera model id: {model_id}")
            model = CAMERA_MODEL_IDS[model_id]
            params = read_next_bytes(fid, 8 * model.num_params, "d" * model.num_params)
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model.model_name,
                width=camera_properties[2],
                height=camera_properties[3],
                params=np.array(params),
            )
    return cameras


def read_images_text(path: Path) -> Dict[int, Image]:
    images = {}
    with Path(path).open("r", encoding="utf-8") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            image_id = int(elems[0])
            qvec = np.array(tuple(map(float, elems[1:5])))
            tvec = np.array(tuple(map(float, elems[5:8])))
            camera_id = int(elems[8])
            image_name = elems[9]
            points_line = fid.readline().split()
            xys = np.column_stack(
                [tuple(map(float, points_line[0::3])), tuple(map(float, points_line[1::3]))]
            )
            point3d_ids = np.array(tuple(map(int, points_line[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3d_ids,
            )
    return images


def read_images_binary(path: Path) -> Dict[int, Image]:
    images = {}
    with Path(path).open("rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2d = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points2d, "ddq" * num_points2d)
            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
            )
            point3d_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3d_ids,
            )
    return images


def read_points3d_text(path: Path) -> Dict[int, Point3D]:
    points3d = {}
    with Path(path).open("r", encoding="utf-8") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            point3d_id = int(elems[0])
            points3d[point3d_id] = Point3D(
                id=point3d_id,
                xyz=np.array(tuple(map(float, elems[1:4]))),
                rgb=np.array(tuple(map(int, elems[4:7]))),
                error=float(elems[7]),
                image_ids=np.array(tuple(map(int, elems[8::2]))),
                point2D_idxs=np.array(tuple(map(int, elems[9::2]))),
            )
    return points3d


def read_points3d_binary(path: Path) -> Dict[int, Point3D]:
    points3d = {}
    with Path(path).open("rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            properties = read_next_bytes(fid, 43, "QdddBBBd")
            point3d_id = properties[0]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track_elems = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            points3d[point3d_id] = Point3D(
                id=point3d_id,
                xyz=np.array(properties[1:4]),
                rgb=np.array(properties[4:7]),
                error=float(properties[7]),
                image_ids=np.array(tuple(map(int, track_elems[0::2]))),
                point2D_idxs=np.array(tuple(map(int, track_elems[1::2]))),
            )
    return points3d


def read_model(model_dir: Path, ext: str):
    model_dir = Path(model_dir)
    if ext == ".txt":
        return (
            read_cameras_text(model_dir / "cameras.txt"),
            read_images_text(model_dir / "images.txt"),
            read_points3d_text(model_dir / "points3D.txt"),
        )
    if ext == ".bin":
        return (
            read_cameras_binary(model_dir / "cameras.bin"),
            read_images_binary(model_dir / "images.bin"),
            read_points3d_binary(model_dir / "points3D.bin"),
        )
    raise ConversionError(f"Unsupported COLMAP model extension: {ext}")


def detect_model_ext(model_dir: Path) -> str:
    model_dir = Path(model_dir)
    if (model_dir / "cameras.bin").is_file() and (model_dir / "images.bin").is_file() and (model_dir / "points3D.bin").is_file():
        return ".bin"
    if (model_dir / "cameras.txt").is_file() and (model_dir / "images.txt").is_file() and (model_dir / "points3D.txt").is_file():
        return ".txt"
    raise ConversionError(f"No complete COLMAP sparse model found in {model_dir}")


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def camera_intrinsics(cameras: Dict[int, Camera]) -> Dict[int, np.ndarray]:
    intrinsics = {}
    for camera_id, camera in cameras.items():
        if camera.model not in PARAM_TYPES:
            raise ConversionError(f"Unsupported COLMAP camera model: {camera.model}")
        params_dict = {key: value for key, value in zip(PARAM_TYPES[camera.model], camera.params)}
        if "f" in params_dict:
            params_dict["fx"] = params_dict["f"]
            params_dict["fy"] = params_dict["f"]
        intrinsics[camera_id] = np.array(
            [
                [params_dict["fx"], 0, params_dict["cx"]],
                [0, params_dict["fy"], params_dict["cy"]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
    return intrinsics


def remap_images(images: Dict[int, Image]) -> Dict[int, Image]:
    return {index + 1: images[image_id] for index, image_id in enumerate(sorted(images.keys()))}


def image_extrinsics(images: Dict[int, Image]) -> Dict[int, np.ndarray]:
    extrinsics = {}
    for image_id, image in images.items():
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = qvec2rotmat(image.qvec)
        extrinsic[:3, 3] = image.tvec
        extrinsics[image_id] = extrinsic
    return extrinsics


def depth_ranges_for_images(
    images: Dict[int, Image],
    points3d: Dict[int, Point3D],
    extrinsics: Dict[int, np.ndarray],
    interval_scale: float,
    max_d: int,
) -> Dict[int, Tuple[float, float, float, float]]:
    depth_ranges = {}
    for image_id, image in images.items():
        zs = []
        for point3d_id in image.point3D_ids:
            if point3d_id == -1 or point3d_id not in points3d:
                continue
            xyz1 = np.append(points3d[int(point3d_id)].xyz, 1.0)
            transformed = np.matmul(extrinsics[image_id], xyz1)
            z = float(transformed[2])
            if math.isfinite(z) and z > 0:
                zs.append(z)
        if not zs:
            raise ConversionError(f"Image {image.name} has no valid positive sparse point depths.")

        zs_sorted = sorted(zs)
        num_max = max(1, min(len(zs_sorted), int(len(zs_sorted) * 0.1) or 1))
        num_min = max(1, min(len(zs_sorted), int(len(zs_sorted) * 0.03) or 1))
        depth_min = float(np.mean(zs_sorted[:num_min]))
        depth_max = float(np.mean(zs_sorted[-num_max:]))
        if depth_max <= depth_min:
            depth_max = depth_min + 1.0

        depth_num = float(max_d)
        depth_interval = (depth_max - depth_min) / (depth_num - 1) / interval_scale
        depth_ranges[image_id] = (depth_min, depth_interval, depth_num, depth_max)
    return depth_ranges


def view_selection(
    images: Dict[int, Image],
    points3d: Dict[int, Point3D],
    extrinsics: Dict[int, np.ndarray],
    max_src_views: int,
    theta0: float,
    sigma1: float,
    sigma2: float,
) -> List[List[Tuple[int, float]]]:
    image_ids = sorted(images.keys())
    point_sets = {
        image_id: {int(point_id) for point_id in images[image_id].point3D_ids if int(point_id) != -1}
        for image_id in image_ids
    }
    centers = {
        image_id: -np.matmul(extrinsics[image_id][:3, :3].transpose(), extrinsics[image_id][:3, 3:4])[:, 0]
        for image_id in image_ids
    }
    score = np.zeros((len(image_ids), len(image_ids)), dtype=np.float64)

    for i, image_id_i in enumerate(image_ids):
        for j, image_id_j in enumerate(image_ids[i + 1 :], start=i + 1):
            shared = point_sets[image_id_i].intersection(point_sets[image_id_j])
            pair_score = 0.0
            for point_id in shared:
                if point_id not in points3d:
                    continue
                point = points3d[point_id].xyz
                v1 = centers[image_id_i] - point
                v2 = centers[image_id_j] - point
                denom = np.linalg.norm(v1) * np.linalg.norm(v2)
                if denom == 0:
                    continue
                cosine = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
                theta = (180.0 / np.pi) * np.arccos(cosine)
                sigma = sigma1 if theta <= theta0 else sigma2
                pair_score += float(np.exp(-((theta - theta0) ** 2) / (2 * sigma**2)))
            score[i, j] = pair_score
            score[j, i] = pair_score

    selections = []
    for i in range(len(image_ids)):
        sorted_score = np.argsort(score[i])[::-1]
        selections.append([(int(k), float(score[i, k])) for k in sorted_score[:max_src_views] if int(k) != i])
    return selections


def write_cam_file(path: Path, extrinsic: np.ndarray, intrinsic: np.ndarray, depth_range: Tuple[float, float, float, float]) -> None:
    with Path(path).open("w", encoding="utf-8") as fid:
        fid.write("extrinsic\n")
        for row in extrinsic:
            fid.write(" ".join(str(float(value)) for value in row) + " \n")
        fid.write("\nintrinsic\n")
        for row in intrinsic:
            fid.write(" ".join(str(float(value)) for value in row) + " \n")
        fid.write("\n%f %f %f %f\n" % depth_range)


def write_pair_file(path: Path, view_sel: List[List[Tuple[int, float]]]) -> None:
    with Path(path).open("w", encoding="utf-8") as fid:
        fid.write(f"{len(view_sel)}\n")
        for ref_view, sorted_score in enumerate(view_sel):
            fid.write(f"{ref_view}\n{len(sorted_score)} ")
            for image_id, score in sorted_score:
                fid.write(f"{image_id} {score:f} ")
            fid.write("\n")


def convert_images(images: Dict[int, Image], image_dir: Path, output_dir: Path) -> None:
    for index, image_id in enumerate(sorted(images.keys())):
        source = Path(image_dir) / images[image_id].name
        destination = output_dir / f"{index:08d}.jpg"
        if not source.is_file():
            raise ConversionError(f"COLMAP image file does not exist: {source}")
        if source.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.copyfile(source, destination)
            continue
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ConversionError(f"Could not read image for conversion: {source}")
        if not cv2.imwrite(str(destination), image):
            raise ConversionError(f"Could not write converted image: {destination}")


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def convert_colmap_dense_to_mvsnet(
    dense_folder: Path,
    save_folder: Path,
    interval_scale: float = 1.0,
    max_d: int = 192,
    theta0: float = 5.0,
    sigma1: float = 1.0,
    sigma2: float = 10.0,
    model_ext: str = "auto",
    max_src_views: int = 10,
) -> None:
    dense_folder = Path(dense_folder)
    save_folder = Path(save_folder)
    image_dir = dense_folder / "images"
    model_dir = dense_folder / "sparse"
    cam_dir = save_folder / "cams"
    image_converted_dir = save_folder / "images_post"

    if not image_dir.is_dir():
        raise ConversionError(f"COLMAP dense images directory does not exist: {image_dir}")
    if not model_dir.is_dir():
        raise ConversionError(f"COLMAP dense sparse model directory does not exist: {model_dir}")
    if model_ext == "auto":
        model_ext = detect_model_ext(model_dir)

    save_folder.mkdir(parents=True, exist_ok=True)
    _ensure_clean_dir(cam_dir)
    _ensure_clean_dir(image_converted_dir)

    cameras, images, points3d = read_model(model_dir, model_ext)
    if not cameras:
        raise ConversionError(f"No cameras found in COLMAP sparse model: {model_dir}")
    if not images:
        raise ConversionError(f"No registered images found in COLMAP sparse model: {model_dir}")

    images = remap_images(images)
    intrinsics = camera_intrinsics(cameras)
    extrinsics = image_extrinsics(images)
    depth_ranges = depth_ranges_for_images(images, points3d, extrinsics, interval_scale, max_d)
    view_sel = view_selection(images, points3d, extrinsics, max_src_views, theta0, sigma1, sigma2)

    for index, image_id in enumerate(sorted(images.keys())):
        image = images[image_id]
        write_cam_file(
            cam_dir / f"{index:08d}_cam.txt",
            extrinsics[image_id],
            intrinsics[image.camera_id],
            depth_ranges[image_id],
        )
    write_pair_file(save_folder / "pair.txt", view_sel)
    convert_images(images, image_dir, image_converted_dir)
