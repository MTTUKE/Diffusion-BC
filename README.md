# Diffusion Model for Coronal Hole Segmentation

Tento projekt obsahuje implementáciu difúzneho modelu pre binárnu segmentáciu koronálnych dier na snímkach SDO/AIA 193 Å. Projekt bol vytvorený v rámci bakalárskej práce Využitie difúznych modelov pre segmentáciu koronálnych dier.

## Inštalácia

Odporúčané je vytvoriť virtuálne prostredie:

    python -m venv .venv

Aktivácia vo Windows PowerShell:

    .\.venv\Scripts\activate

Inštalácia potrebných knižníc:

    pip install -r requirements.txt

## Štruktúra projektu

    train.py              - trénovanie modelu
    sample.py             - inferencia / generovanie masiek
    config.py             - konfigurácia hlavného experimentu
    config_clean.py       - konfigurácia doplnkového cleaned experimentu

    data/                 - dáta a dátové triedy
    models/               - U-Net architektúra
    diffusion/            - difúzny proces, DDPM/DDIM samplovanie
    utils/                - metriky, seed, EMA a pomocné funkcie
    eval_reiss/           - skripty pre Reiss benchmark
    scripts/              - pomocné skripty pre prípravu dát

    checkpoints/          - checkpoint hlavného modelu
    checkpoints_clean/    - checkpoint modelu trénovaného na cleaned dátach
    comparison_29/        - vstupy pre Reiss benchmark
    reiss_run/            - výsledky Reiss benchmarku pre hlavný model
    reiss_run_clean/      - výsledky Reiss benchmarku pre cleaned model

## Dáta

Hlavný experiment používa dáta v štýle SCSS-Net:

    data/193_train/
    data/193_test/

Doplnkový cleaned experiment používa:

    data/cleaned_256/

Pre cleaned experiment musí byť dostupný manifest:

    data/cleaned_256/manifest.csv

Reiss benchmark vstupy sú uložené v priečinku:

    comparison_29/

## Trénovanie modelu

Hlavný experiment sa spustí príkazom:

    python train.py --mode old

Tento režim používa konfiguráciu zo súboru:

    config.py

Doplnkový cleaned experiment sa spustí príkazom:

    python train.py --mode clean

Tento režim používa konfiguráciu zo súboru:

    config_clean.py

Rýchla kontrola spustenia tréningu:

    python train.py --mode old --epochs 1

    python train.py --mode clean --epochs 1

## Inferencia

Na generovanie segmentačných masiek sa používa:

    python sample.py

Finálne nastavenie hlavného modelu použité v práci:

    DDIM steps: 100
    n_samples: 6
    threshold: 0.72
    TTA: flip4
    kernel size: 5
    min area: 200 px

Finálne nastavenie pre Reiss benchmark:

    DDIM steps: 100
    n_samples: 6
    threshold: 0.68
    TTA: flip4
    kernel size: 5
    min area: 120 px

## Reiss benchmark

Spustenie Reiss pipeline pre hlavný model:

    python eval_reiss/run_reiss_pipeline.py --source comparison_29 --ckpt checkpoints/best.pt --work-dir reiss_run --method-name Diffusion-BC --ddim-steps 100 --n-samples 6 --thr 0.68 --tta flip4 --kernel-size 5 --min-area 120

Spustenie Reiss pipeline pre cleaned model:

    python eval_reiss/run_reiss_pipeline.py --source comparison_29 --ckpt checkpoints_clean/best.pt --work-dir reiss_run_clean --method-name Diffusion-BC-Clean --ddim-steps 100 --n-samples 6 --thr 0.68 --tta flip4 --kernel-size 5 --min-area 120

Ak výstupný priečinok už existuje, treba ho pred opätovným spustením odstrániť alebo použiť parameter --overwrite, ak ho daný skript podporuje.

## Kontrola projektu

Kontrola syntaxe všetkých Python súborov:

    python -m compileall .

Kontrola dostupných argumentov:

    python train.py --help
    python sample.py --help
    python eval_reiss/run_reiss_pipeline.py --help

Kontrola Reiss vstupov:

    python eval_reiss/prepare_manifest.py --source comparison_29 --out-dir tmp_reiss_dataset --overwrite

Očakávaný výsledok:

    Matched cases: 29
    Unmatched inputs: 0
    Unused annotations: 0

## Výsledky použité v práci

Hlavný testovací dataset:

    Dice:      0.8597
    IoU:       0.7624
    Precision: 0.8510
    Recall:    0.8904

Reiss benchmark pre hlavný model:

    CH TPR:       0.767
    Filament FPR: 0.096
    Other:        7

Reiss benchmark pre cleaned model:

    CH_TPR:  0.4767
    Fil_FPR: 0.0000
    Other:   0
