"""
SegRap2023 Dataset Loader.

Actual directory layout (verified):
  <data_root>/segrap_XXXX/
      image.nii.gz          <- ncCT
      image_contrast.nii.gz <- ceCT (unused by default)
      Cochlea_L.nii.gz
      Cochlea_R.nii.gz
      ... (one file per OAR)

Default data_root:
  data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases/

Split (seed=42):
  train : segrap_0000 – segrap_0083  (84 cases, 70%)
  val   : segrap_0084 – segrap_0101  (18 cases, 15%)
  test  : segrap_0102 – segrap_0119  (18 cases, 15%)
"""
import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path

THIN_WALL_OARS = [
    'Cochlea_L', 'Cochlea_R',
    'VestibulSemi_L', 'VestibulSemi_R',
    'IAC_L', 'IAC_R',
    'TympanicCavity_L', 'TympanicCavity_R',
    'MiddleEar_L', 'MiddleEar_R',
]

# All 45 pure OARs (no overlap structures); name matches actual .nii.gz filenames
TASK001_PURE_OARS = [
    'Brain', 'BrainStem', 'Chiasm',
    'TemporalLobe_L', 'TemporalLobe_R',
    'Hippocampus_L', 'Hippocampus_R',
    'Eye_L', 'Eye_R', 'Lens_L', 'Lens_R',
    'OpticNerve_L', 'OpticNerve_R',
    'MiddleEar_L', 'MiddleEar_R',
    'IAC_L', 'IAC_R',
    'TympanicCavity_L', 'TympanicCavity_R',
    'VestibulSemi_L', 'VestibulSemi_R',
    'Cochlea_L', 'Cochlea_R',
    'ETbone_L', 'ETbone_R',
    'Pituitary', 'OralCavity',
    'Mandible_L', 'Mandible_R',
    'Submandibular_L', 'Submandibular_R',
    'Parotid_L', 'Parotid_R',
    'Mastoid_L', 'Mastoid_R',
    'TMjoint_L', 'TMjoint_R',
    'SpinalCord', 'Esophagus',
    'Larynx', 'Larynx_Glottic', 'Larynx_Supraglot',
    'PharynxConst', 'Thyroid', 'Trachea',
]

# Fixed split boundaries (70 / 15 / 15 %)
_TRAIN_END = 84    # 0000–0083
_VAL_END = 102     # 0084–0101
_TEST_END = 120    # 0102–0119


def _build_split(split: str):
    all_ids = [f'segrap_{i:04d}' for i in range(120)]
    if split == 'train':
        return all_ids[:_TRAIN_END]
    if split == 'val':
        return all_ids[_TRAIN_END:_VAL_END]
    if split == 'test':
        return all_ids[_VAL_END:_TEST_END]
    raise ValueError(f"split must be 'train'/'val'/'test', got '{split}'")


class SegRapDataset(Dataset):
    """
    mode='thin_wall'  : each sample = one (CT, binary mask) for one thin-wall OAR.
                        len = n_cases × len(THIN_WALL_OARS).
    mode='all_oars'   : each sample = one CT with a full 46-class mask
                        (0=bg, 1..45=TASK001_PURE_OARS indices).
                        len = n_cases.
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        mode: str = 'thin_wall',
        target_size: tuple = (128, 128, 128),
        seed: int = 42,
        oar_subset=None,       # override oar list for thin_wall mode
        max_samples: int = 0,  # 0 = use all; >0 truncates for quick smoke tests
        cache_dir: str = '',   # '' = no caching; path = cache preprocessed .npz
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.mode = mode
        self.target_size = target_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.case_ids = _build_split(split)

        if mode == 'thin_wall':
            self.oars = oar_subset if oar_subset is not None else THIN_WALL_OARS
            self.samples = [
                (case_id, oar)
                for case_id in self.case_ids
                for oar in self.oars
                if self._label_path(case_id, oar).exists()
            ]
        elif mode == 'all_oars':
            self.oars = TASK001_PURE_OARS
            self.samples = [
                (case_id, None)
                for case_id in self.case_ids
            ]
        else:
            raise ValueError(f"mode must be 'thin_wall' or 'all_oars', got '{mode}'")

        if max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_id, oar_name = self.samples[idx]

        if self.mode == 'thin_wall':
            if self.cache_dir is not None:
                cache_path = self.cache_dir / f'{case_id}_{oar_name}.npz'
                cached = None
                if cache_path.exists():
                    d = np.load(cache_path)
                    if 'spacing' in d:   # guards against pre-spacing cache files
                        cached = (d['image'], d['label'], tuple(float(s) for s in d['spacing']))
                if cached is not None:
                    image, label, spacing = cached
                else:
                    image, label, spacing = self._load_thin_wall_pair(case_id, oar_name)
                    image, label = self._resize_and_normalise(image, label)
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(str(cache_path), image=image, label=label,
                                        spacing=np.array(spacing, dtype=np.float64))
            else:
                image, label, spacing = self._load_thin_wall_pair(case_id, oar_name)
                image, label = self._resize_and_normalise(image, label)
            return {
                'image':    torch.from_numpy(image).float().unsqueeze(0),  # (1,D,H,W)
                'label':    torch.from_numpy(label).long(),                # (D,H,W)
                'case_id':  case_id,
                'oar_name': oar_name,
                'spacing':  spacing,   # effective (D,H,W) mm/voxel after ROI-crop + resize
            }
        else:
            image = self._load_ct(case_id)
            native_spacing = self.get_spacing(case_id)
            pre_resize_shape = image.shape
            # Build multi-class label: 0=bg, class_idx = OAR position + 1
            label = np.zeros(image.shape, dtype=np.int64)
            for cls_idx, oar in enumerate(self.oars, start=1):
                p = self._label_path(case_id, oar)
                if p.exists():
                    lbl = nib.load(str(p)).get_fdata().astype(np.float32)
                    label[lbl > 0.5] = cls_idx
            image, label = self._preprocess_whole_volume(image, label)
            spacing = self._effective_spacing(pre_resize_shape, native_spacing)
            return {
                'image':    torch.from_numpy(image).float().unsqueeze(0),
                'label':    torch.from_numpy(label).long(),
                'case_id':  case_id,
                'oar_name': '',
                'spacing':  spacing,   # effective (D,H,W) mm/voxel after whole-volume resize
            }

    # ------------------------------------------------------------------ #
    def _case_dir(self, case_id: str) -> Path:
        return self.data_root / case_id

    def _label_path(self, case_id: str, oar_name: str) -> Path:
        return self._case_dir(case_id) / f'{oar_name}.nii.gz'

    def _load_ct(self, case_id: str) -> np.ndarray:
        """Load full CT volume in float32 (used by all_oars mode)."""
        ct_path = self._case_dir(case_id) / 'image.nii.gz'
        img = nib.load(str(ct_path))
        data = img.get_fdata(dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]
        return data

    def _effective_spacing(self, pre_resize_shape, native_spacing) -> tuple:
        """mm/voxel after a (crop or whole-volume) array of `pre_resize_shape`
        is resized to `self.target_size`. HD95 / surface-dice need this — the
        native CT spacing no longer applies once the ROI has been resampled."""
        target = self.target_size
        return tuple(
            float(native_spacing[i]) * pre_resize_shape[i] / target[i]
            for i in range(3)
        )

    def _load_thin_wall_pair(self, case_id: str, oar_name: str,
                             margin: int = 32):
        """Load label first, compute ROI bbox, then load only the CT crop.

        Strategy:
          1. Load the binary label mask (uint8, ~130 MB for full volume).
          2. Compute the padded bounding box from the mask.
          3. Load the CT in its native dtype (int16 for SegRap2023 = ~267 MB)
             and immediately slice to the bbox before converting to float32.

        Peak RAM drops from ~1.3 GB (float64 full CT + float32 CT) to
        ~400 MB (int16 full CT + small float32 crop).
        Falls back to whole-volume load when the OAR mask is empty.

        Returns (image, label, effective_spacing) — effective_spacing is the
        mm/voxel of the *resized* (target_size) array, not the native CT.
        """
        native_spacing = self.get_spacing(case_id)  # header-only read, cheap

        # --- 1. label: native uint8, NO float32 cast ---
        # Requesting dtype=np.float32 forces nibabel to allocate a 508 MB float32
        # copy even when slope/inter are None — avoid by keeping native uint8.
        lbl_img = nib.load(str(self._label_path(case_id, oar_name)))
        lbl_raw = np.asanyarray(lbl_img.dataobj)   # uint8 ~133 MB, no upcast
        label = (lbl_raw > 0).astype(np.uint8)
        del lbl_img, lbl_raw

        iD, iH, iW = label.shape
        coords = np.where(label > 0)
        if len(coords[0]) == 0:
            # OAR absent – load full CT and return so whole-volume resize can run
            image = self._load_ct(case_id)
            spacing = self._effective_spacing(image.shape, native_spacing)
            return image, label, spacing

        # --- 2. padded bbox ---
        d0, d1 = int(coords[0].min()), int(coords[0].max())
        h0, h1 = int(coords[1].min()), int(coords[1].max())
        w0, w1 = int(coords[2].min()), int(coords[2].max())
        d0 = max(0, d0 - margin);  d1 = min(iD - 1, d1 + margin)
        h0 = max(0, h0 - margin);  h1 = min(iH - 1, h1 + margin)
        w0 = max(0, w0 - margin);  w1 = min(iW - 1, w1 + margin)

        # --- 3. CT: native int16 (no float64), crop first, then convert ---
        # Same trick: no dtype arg → nibabel returns int16 without float64 upcasting.
        ct_img = nib.load(str(self._case_dir(case_id) / 'image.nii.gz'))
        slope, inter = ct_img.header.get_slope_inter()
        raw = np.asanyarray(ct_img.dataobj)             # int16 ~267 MB
        raw_crop = raw[d0:d1 + 1, h0:h1 + 1, w0:w1 + 1].copy()
        del raw, ct_img                                  # free ~267 MB immediately

        image = raw_crop.astype(np.float32)              # tiny crop → no pressure
        if slope is not None and slope != 1.0:
            image = image * float(slope)
        if inter is not None and inter != 0.0:
            image = image + float(inter)

        spacing = self._effective_spacing(image.shape, native_spacing)
        return image, label[d0:d1 + 1, h0:h1 + 1, w0:w1 + 1], spacing

    def _load_label(self, case_id: str, oar_name: str) -> np.ndarray:
        lbl = nib.load(str(self._label_path(case_id, oar_name)))
        # Use uint8 (1 byte) not int64 (8 bytes) — binary label for a 1024³ CT
        # would otherwise allocate ~992 MB before the ROI crop.
        # Conversion to int64 happens in _resize_and_normalise on the tiny crop.
        data = np.asarray(lbl.dataobj, dtype=np.float32)
        return (data > 0.5).astype(np.uint8)

    def _preprocess(self, image: np.ndarray, label: np.ndarray):
        """Dispatch to ROI-crop (thin_wall) or whole-volume resize (all_oars)."""
        if self.mode == 'thin_wall':
            return self._preprocess_roi_crop(image, label)
        return self._preprocess_whole_volume(image, label)

    def _preprocess_roi_crop(self, image: np.ndarray, label: np.ndarray, margin: int = 32):
        """
        Crop a padded bounding box around the OAR mask, then resize to target_size.

        At 0.53 mm in-plane spacing, Cochlea diameter ~10 mm ≈ 19 voxels.
        A 32-voxel margin gives ~17 mm context on each side — enough for the
        model to see surrounding bone landmarks without including the whole head.
        Falls back to whole-volume resize when the mask is empty (label absent).
        """
        iD, iH, iW = image.shape
        coords = np.where(label > 0)

        if len(coords[0]) == 0:
            # OAR absent for this case — fall back to whole-volume so the
            # sample is still valid (all-background label)
            return self._preprocess_whole_volume(image, label)

        d0, d1 = int(coords[0].min()), int(coords[0].max())
        h0, h1 = int(coords[1].min()), int(coords[1].max())
        w0, w1 = int(coords[2].min()), int(coords[2].max())

        # Expand by margin, clamp to volume boundary
        d0 = max(0, d0 - margin);  d1 = min(iD - 1, d1 + margin)
        h0 = max(0, h0 - margin);  h1 = min(iH - 1, h1 + margin)
        w0 = max(0, w0 - margin);  w1 = min(iW - 1, w1 + margin)

        img_crop = image[d0:d1 + 1, h0:h1 + 1, w0:w1 + 1]
        lbl_crop = label[d0:d1 + 1, h0:h1 + 1, w0:w1 + 1]

        return self._resize_and_normalise(img_crop, lbl_crop)

    def _preprocess_whole_volume(self, image: np.ndarray, label: np.ndarray):
        """Whole-volume resize to target_size (used for all_oars mode)."""
        return self._resize_and_normalise(image, label)

    def _resize_and_normalise(self, image: np.ndarray, label: np.ndarray):
        """Shared: trilinear resize → z-score normalise image."""
        D, H, W = self.target_size
        img_t = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        lbl_t = torch.from_numpy(label).float().unsqueeze(0).unsqueeze(0)
        img_t = F.interpolate(img_t, size=(D, H, W), mode='trilinear', align_corners=False)
        lbl_t = F.interpolate(lbl_t, size=(D, H, W), mode='nearest')
        image = img_t.squeeze().numpy()
        label = lbl_t.squeeze().numpy().astype(np.int64)
        mu, sigma = image.mean(), image.std() + 1e-8
        image = (image - mu) / sigma
        return image, label

    def get_spacing(self, case_id: str):
        """Return voxel spacing (mm) for a given case, useful for HD95 computation."""
        ct_path = self._case_dir(case_id) / 'image.nii.gz'
        img = nib.load(str(ct_path))
        return tuple(float(v) for v in img.header.get_zooms()[:3])
