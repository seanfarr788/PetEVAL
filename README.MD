---
license: apache-2.0
task_categories:
- token-classification
- text-classification
language:
- en
tags:
- Biomedical
- HIPAA
- Biology
size_categories:
- 1K<n<10K
---
# PetEVAL

We introduce PetEVAL, the first benchmark dataset derived from real-world, free-text veterinary electronic health records (EHRs). PetEVAL comprises 17,600 professionally annotated EHRs from first-opinion veterinary practices across the UK, partitioned into training (11,000), evaluation (1,600), and test (5,000) sets with distinct clinic distributions to assess model generalizability. Each record is annotated with International Classification of Disease 11 (ICD-11) syndromic chapter labels (20,408 labels), disease Named Entity Recognition (NER) tags (429 labels), and anonymisation NER tags (8,244 labels). PetEVAL enables evaluating Natural Language Processing (NLP) tools across applications, including syndrome surveillance and disease outbreak detection. We implement a multistage anonymisation protocol, replacing identifiable information with clinically relevant pseudonyms while establishing the first definition of identifiers in veterinary free text. PetEVAL introduces three core tasks: syndromic classification, disease entity recognition, and de-identification. We provide baseline results using BERT-base, PetBERT, and Llama 3.1 8B generative models. Our experiments demonstrate the unique challenges of veterinary text, showcasing the importance of domain-specific approaches. By fostering advancements in veterinary informatics and epidemiology, we envision PetEVAL catalysing innovations in veterinary care, animal health, and comparative biomedical research through access to real-world, annotated veterinary clinical data.

[Link to Paper](https://aclanthology.org/2025.bionlp-1.29/)

## The SAVSNET Dataset

We utilise data from the Small Animal Veterinary Surveillance Network (SAVSNET), a sentinel network of 253 volunteer first-opinion veterinary practices across the United Kingdom that have collected electronic health records (EHRs) since March 2014. This network has accumulated over 12 million EHRs, with participating practices selected based on their practice management software compatibility with the SAVSNET data exchange system. During each consultation with a clinician or nurse, comprehensive data includes species, breed, sex, neuter status, age, owner's postcode, insurance and microchipping status, and a detailed free-text clinical narrative. These narratives may contain information about symptoms, diagnoses, treatments, procedures, or other clinical matters. Owners can opt out of data collection during any consultation. The SAVSNET group operates under ethical approval from the University of Liverpool Ethics Committee (RETH001081), ensuring adherence to established ethical standards.

## Tasks

### Task 1 - Anonymisation
Ensuring the privacy and security of EHRs is crucial for safeguarding the personal information of pet owners and facilitating the easy sharing of data use in clinical and academic research. The dataset is labelled with NER entities and spans applied to pseudo-anonymised contextual placeholders. The objective is to maintain the integrity and utility of clinical information within the EHR while effectively anonymising various types of personal data. This includes names (both animal and human), location details (such as city, town, and addresses), organisation names (including attending veterinary practices, referral hospitals, kennels, and laboratories), contact details (emails, phone numbers), id-numbers (passport numbers, insurance policy numbers, MRCVS codes), and any other explicit identifiers. The anonymisation is compliant with the HIPAA Safe Harbour.


### Task 2 - Syndromic Disease Classification
Given the critical role of monitoring national disease outbreaks in public health, effective surveillance systems can provide invaluable insights, such as in informing clinicians of key symptoms to observe, enabling researchers to identify aetiological agents, and establishing an automated reporting mechanism for public health agencies to facilitate swift notification of changes in disease occurrence. However, the task is not straightforward, particularly when dealing with novel diseases or syndromes with unknown symptoms. Effective outbreak reduction strategies hinge on the ability to detect outbreaks with minimal cases. To address these challenges, the dataset is provided with International Classification of Disease 11 (ICD-11) chapters, which includes contextual discussions such as symptoms and diagnoses. The task is structured as a multi-label classification problem, as a consult or condition may cover a range of presenting symptoms. Performance is evaluated using multi-label classification metrics, including precision and recall, macro-average F1-Score, and weighted F1-Score.

#### Train
```
python run_model_classification.py --problem_type multi_label_classification --mode train --pretrained_model "SAVSNET/PetBERT" --dataset "datasets/1.annonymisation/arrow" --output_dir "models/1.annonymisation/PetBERT/"
```
#### Eval
```
python run_model_classification.py --problem_type multi_label_classification --mode eval --pretrained_model "models/1.annonymisation/PetBERT/" --dataset "datasets/1.annonymisation/arrow" --output_dir "projects/PetEVAL/syndromic/report.csv"

```


### Task 3 - Disease Extraction
Identifying specific diseases is critical for downstream epidemiological studies, which aim to reveal novel risk factors, seasonality, and other trends. This task is particularly challenging due to the private healthcare nature of veterinary practices in the UK and much of the world. Confirmation diagnostic tests are rare, as owners often wish to avoid the inherent costs, opting instead to take the advice of clinicians or due to the lack of available resources or expertise not found in first opinion practice. Additionally, the presence of negations or listing of differential diagnoses complicates the task further. In our study, the dataset is labelled with the diagnostic disease contained within it. This process is framed as NER task using the IOB2 format, wherein the entity of 'disease' and its spans are provided. Evaluation utilises SeqEval for precision, recall, and F1-score.

## Reference

    Sean Farrell, Alan Radford, Noura Al Moubayed, and Peter-John Noble.
    2025. [PetEVAL: A veterinary free text electronic health records benchmark](https://aclanthology.org/2025.bionlp-1.29/). In
    _Proceedings of the 24th Workshop on Biomedical Language Processing_, pages 341–353, Viena, Austria. Association for Computational Linguistics.