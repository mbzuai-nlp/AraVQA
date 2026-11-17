# AraVQA

AraVQA is an Arabic factoid visual question answering dataset built from Arabic Wikipedia. Each example pairs an image reference with an Arabic question, four answer choices, and a gold answer. The training examples were generated automatically. The test split was manually reviewed.

This repository accompanies [*AraVQA: Building a New Arabic Factoid Visual Question Answering Dataset from Wikipedia*](https://aclanthology.org/2026.acl-long.91/) (ACL 2026).

## Dataset files

| File | Questions | Intended use |
| --- | ---: | --- |
| `AraVQA-Train.csv` | 45,966 | Training |
| `AraVQA-Test.csv` | 4,791 | Evaluation |
| **Total** | **50,757** | |

The counts reflect the CSVs in this release. Section 4.1 of the paper gives the same total and test count but lists 45,969 training questions.

## CSV schema

Both splits use the same UTF-8 CSV schema.

| Fields | Meaning |
| --- | --- |
| `raw_id` | Example identifier |
| `question`, `A`, `B`, `C`, `D` | Arabic question and four options |
| `answer_letter`, `answer` | Gold option letter and text |
| `article_title`, `article_id`, `section_title`, `section_text` | Arabic Wikipedia article and section metadata and source text |
| `img_url`, `img_path`, `img_desc`, `img_license` | Image URL, source filename, description, and recorded image license |
| `topic` | Topic label |

## Loading an example

```python
import csv

with open("AraVQA-Test.csv", encoding="utf-8-sig", newline="") as source:
    row = next(csv.DictReader(source))

model_input = {
    "image_url": row["img_url"],
    "question": row["question"],
    "choices": {letter: row[letter] for letter in "ABCD"},
}
gold_answer = row["answer_letter"]
```

For image-grounded evaluation, fetch the image at `img_url` and provide it alongside the question and choices. Compare the predicted option letter with `answer_letter`. Image files are hosted externally. The source metadata and gold fields can be used for analysis.

## Training and evaluation script

`sft_vision_phi.py` is the accompanying Phi vision-language experiment script. It looks for both CSVs in the working directory. Its entry point controls whether training and evaluation run. The data loader fetches images from their URLs.

## Sources and licenses

The CSVs retain Arabic Wikipedia source metadata and the recorded per-image terms in `img_license`. Consult the source page for the image's attribution and reuse terms. A dataset-wide license and a software license were not specified in the supplied release files.

## Citation

Sultan Alrowili, Younes Samih, Abed Alhakim Freihat, and Mathan Kumar Eswaran. 2026. *AraVQA: Building a New Arabic Factoid Visual Question Answering Dataset from Wikipedia*. Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2026–2042. [ACL Anthology](https://aclanthology.org/2026.acl-long.91/).
