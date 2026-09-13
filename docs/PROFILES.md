# Preprocessing profiles

The two legacy code folders have been merged so the only intended implementation difference is selected through `--profile` or `UIKB_PROFILE`.

| Behavior | `baseline` | `hazm` |
|---|---|---|
| Dense query normalization | none beyond shared light query cleanup | Hazm `Normalizer` |
| Dense passage normalization | none | Hazm `Normalizer` |
| Model query prefixes | E5/BGE/Qwen: `query: `; Matina: none | same model-prefix table |
| Model passage prefixes | E5/BGE/Qwen: `passage: `; Matina: none | same model-prefix table |
| BM25 normalization | light Arabic/Persian character cleanup | Hazm `Normalizer` |
| BM25 tokenization | whitespace | Hazm `word_tokenize` |
| BM25 digits | unchanged after shared corpus cleanup | Persian digits mapped to Latin after Hazm normalization |
| Index tree | `indices/baseline` | `indices/hazm` |
| Paper role | final Dev-150 and Test-60 | pre-freeze Dev-130 ablation |

## Why a single codebase?

Keeping two near-duplicate repositories makes it difficult to know whether a score difference is caused by preprocessing or unrelated code drift. The merged implementation isolates the intended treatment in `configs/baseline.json` and `configs/hazm.json`, while benchmark logic, metrics, RRF, reranking, and data loading remain shared.

## Matina and Hazm

Matina was added after the historical Hazm experiment. The merged code can technically apply the Hazm profile to Matina if an index is built, but **such a run is not one of the reported historical ablation results**. The repository therefore ships no prebuilt `indices/hazm/matina` index.
