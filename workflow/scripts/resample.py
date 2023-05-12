from nilearn import image
import nibabel as nib
import numpy as np


def main() -> None:
    orig = nib.load(snakemake.input.raw)
    resampled = image.resample_img(
        orig, target_affine=np.eye(3), interpolation="linear"
    )
    nib.save(resampled, snakemake.output.resampled)


if __name__ == "__main__":
    main()
