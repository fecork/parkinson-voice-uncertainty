# SpecAugment Fix for LSTM Temporal Coherence

## Summary of Changes

### Problem
The previous implementation applied SpecAugment frame-by-frame with independent random masks, breaking temporal coherence required by LSTM models to learn temporal dependencies.

### Solution
Applied SpecAugment to the FULL spectrogram BEFORE segmenting into frames, ensuring all frames in a sequence share the same masked regions.

## Files Modified

### 1. `modules/data/augmentation.py`
- **Rewritten `spec_augment()` function** (lines 121-172):
  - Added `p` parameter for probabilistic augmentation
  - Added `rng` parameter for reproducible results
  - Improved bounds handling with `rng.integers()`
  - Now operates on full spectrograms ensuring temporal coherence

- **Updated `preprocess_audio_with_augmentation()`**:
  - Changed `spec_augment_before_segment` default from `False` to `True`
  - Updated to use `rng` parameter when calling `spec_augment()`
  - Improved docstring to emphasize LSTM requirements

- **Updated `create_augmented_dataset()`**:
  - Changed `spec_augment_before_segment` default from `False` to `True`
  - Updated docstring to emphasize temporal coherence

### 2. `modules/core/sequence_dataset.py`
- **Enhanced `normalize_sequence()` documentation**:
  - Added explicit comment about LSTM requirements
  - Clarified that normalization uses ALL frames together
  - Improved code comments for clarity

### 3. `data_preprocessing.ipynb`
- **Updated configuration (Cell 16)**:
  - Changed title to "CONFIGURACIÓN PARA LSTM (COHERENCIA TEMPORAL)"
  - Set `FORCE_REGENERATE = True` to regenerate cache with new method
  - Added `SPEC_AUGMENT_BEFORE_SEGMENT = True`
  - Added `SKIP_NORMALIZATION = True` (normalize by sequence later)
  - Updated print messages to explain the new approach

- **Updated dataset generation (Cells 17-18)**:
  - Removed obsolete `augmentation_types` parameter
  - Added `spec_augment_before_segment=SPEC_AUGMENT_BEFORE_SEGMENT`
  - Added `skip_normalization=SKIP_NORMALIZATION`

- **Added verification cells (Cells 21-22)**:
  - New markdown cell explaining verification purpose
  - New code cell with `verify_specaugment_coherence()` function
  - Automatically checks if masks are consistent across frames
  - Provides clear pass/fail feedback

## Key Improvements

1. **Temporal Coherence**: All frames in a sequence now share the same mask pattern
2. **LSTM Optimization**: Model can learn temporal dependencies without mask noise
3. **Reproducibility**: Added RNG parameter for consistent augmentation
4. **Better Documentation**: Clear explanation of why this matters for LSTM
5. **Verification**: Built-in check to validate coherence after generation

## Expected Results

After regenerating cache with `FORCE_REGENERATE=True`:
- **155 original samples** (13 files)
- **310 augmented samples** (155 × 2 versions)
- **Total: ~465 samples** per class

All augmented sequences will have consistent masking across frames, preserving temporal relationships for LSTM learning.

## Usage

1. Run `data_preprocessing.ipynb` from Cell 16 onwards
2. Cache will regenerate with correct SpecAugment
3. Verification cell will confirm temporal coherence
4. Use generated sequences for LSTM training

## Validation

The verification cell checks that:
- Masks are identical across all frames in a sequence
- Frequency bands masked are consistent
- Time segments masked are consistent
- Overall temporal coherence is maintained

Status indicators:
- ✅ COHERENTE: Correct for LSTM
- ❌ INCONSISTENTE: Incorrect, needs regeneration
