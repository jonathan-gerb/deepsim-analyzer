import os
from pathlib import Path
fp = os.path.dirname(os.path.realpath(__file__))
fp = Path(fp)
model_name = "dino_deitsmall8_pretrain_full_checkpoint.pth"
full_path = fp.parents[2] / "data" / "raw_immutable" / "models" / model_name
print(os.path.exists(full_path))