# PartGraph MVP Guide

This is the Minimum Viable Product (MVP) codebase for the PartGraph project.

## 1. Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 2. Project Structure

- `src/model/`: Contains the core model definitions.
  - `backbone.py`: ResNet18 wrapper.
  - `slot_attention.py`: The **Part-Aware Slot Attention** module.
  - `part_autoencoder.py`: Combines backbone and slots for the reconstruction task.
- `src/utils/`: Helper functions (positional encoding).
- `src/mvp_run.py`: A script to verify the architecture.

## 3. Running the Test

Run the following command to verify that the model architecture is correct and can perform a forward/backward pass:

```bash
python src/mvp_run.py
```

## 4. Next Steps (Moving to Colab)

1. Upload this entire folder to Google Drive.
2. Open a Colab notebook.
3. Mount Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/path/to/PartGraph-FewShot
   ```
4. Install requirements.
5. Run `src/mvp_run.py`.
6. Modify `src/mvp_run.py` to load the CUB-200 dataset instead of dummy data.
