# Full Retrieval Results — ViT-L/14 + RoBERTa-Large

COCO: Karpathy 5K test split. Flickr30k: Karpathy 1K test split.

| Method | Flickr I2T | Flickr T2I | COCO I2T | COCO T2I |
|---|---:|---:|---:|---:|
| CLS Linear | 60.4 | 47.0 | 39.8 | 29.5 |
| CLS Linear+STR | 62.8 | 48.0 | 40.3 | 29.4 |
| CLS MLP | 60.4 | 46.2 | 40.1 | 29.8 |
| CLS MLP+STR | 61.3 | 47.6 | 40.4 | 29.4 |
| CLS SAIL | 46.1 | 33.5 | - | - |
| CLS SAIL+STR | 45.3 | 32.7 | - | - |
| | | | | |
| SAIL Concat | 51.7 | 40.9 | 38.3 | 29.1 |
| | | | | |
| Token Linear | 63.8 | 50.3 | 44.0 | 33.8 |
| Token Linear+STR | 66.2 | 49.6 | 45.9 | 32.0 |
| Token MLP | 64.2 | 51.6 | 45.0 | 33.5 |
| Token MLP+STR | 64.3 | 49.7 | 45.2 | 31.6 |
| Token FA | 62.9 | 48.9 | 43.5 | 32.8 |
| Token FA+STR | 64.8 | 49.0 | 44.3 | 31.3 |
| Token BA K=128 | 69.2 | 54.8 | 49.7 | 37.2 |
| Token BA K=256 | 72.2 | 58.0 | 52.4 | 39.2 |
| Token BA K=512 | 75.5 | 60.2 | 54.6 | 41.2 |
| Token BA K=1024 | 77.3 | 61.2 | 55.3 | 42.1 |
