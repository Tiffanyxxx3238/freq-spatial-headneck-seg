"""
Quick sanity check for SegRapDataset.
Run from segrap_research/:
    python test_dataset.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data.segrap_dataset import SegRapDataset

DATA_ROOT = '../data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases'

ds = SegRapDataset(
    DATA_ROOT,
    split='train',
    mode='thin_wall',
)
print('Dataset size:', len(ds))

sample = ds[0]
print('image shape:', sample['image'].shape)
print('label unique:', sample['label'].unique())
print('case_id:', sample['case_id'])
print('oar_name:', sample['oar_name'])

# 確認 Cochlea 不是全零
for i in range(len(ds)):
    s = ds[i]
    if s['oar_name'] == 'Cochlea_L':
        fg = s['label'].sum().item()
        print(f'  {s["case_id"]} Cochlea_L fg voxels: {fg}')
        if i >= 5:
            break

# Verify spacing is returned correctly
spacing = ds.get_spacing(ds.case_ids[0])
print(f'\nSpacing for {ds.case_ids[0]}: {spacing}')
