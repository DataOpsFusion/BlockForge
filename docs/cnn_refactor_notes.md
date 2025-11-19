# CNN encoder / classifier split

Leave a breadcrumb for tomorrow so we remember how the vision-side CNN pieces should evolve alongside the text CharCNN models.

## Goals
- Separate generic convolutional feature extractors (1D/2D/3D) from classifier-specific heads.
- Make CharCNN inherit from the general 1D CNN classifier so it automatically benefits from future infra improvements.
- Keep the option to reuse CharCNN as a pure feature encoder by inheriting from the encoder variant instead.

## Proposed class hierarchy
1. `CNN1dEncoder`, `CNN2dEncoder`, `CNN3dEncoder`
   - Subclass the current `vision.cnn.CNN` but stop after the convolutional / pooling stack.
   - Accept `nums_class=None` or defer classifier creation entirely.
   - Expose a `build_core`/`forward_features` style hook so downstream classes can plug in custom heads.
2. `CNN1dClassifier(CNN1dEncoder)` (same for 2d/3d)
   - Adds a standard classifier head (e.g., global pooling + MLP) on top of the encoder outputs.
   - Keeps constructor parity with the old `CNN` signature so existing usages simply switch inheritance.

## CharCNN follow-up
- `BaseCharCNN` should inherit from `CNN1dClassifier`; `build_core` remains the text-specific convolutional stack.
- `CharCNNBasic` and `CharCNNAdvance` stay thin wrappers that only override `build_core`.
- If we later want a CharCNN-based feature extractor, just inherit from `CNN1dEncoder` instead and skip classifier wiring.

## Action items
- Implement the encoder / classifier split under `src/vision/cnn/`.
- Update `src/nlp/charcnn/model.py` so `BaseCharCNN` extends the new `CNN1dClassifier`.
- Add docs/tests covering the new inheritance tree before refactoring the advanced CharCNN variant.
